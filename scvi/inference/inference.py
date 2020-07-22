import logging
import copy
from typing import Union

import matplotlib.pyplot as plt
import torch
import anndata
from numpy import ceil

from scvi.inference import Trainer
from scvi.models.utils import one_hot
from scvi.dataset._constants import (
    _X_KEY,
    _BATCH_KEY,
    _LOCAL_L_MEAN_KEY,
    _LOCAL_L_VAR_KEY,
    _LABELS_KEY,
)
from scvi.models import Classifier

plt.switch_backend("agg")
logger = logging.getLogger(__name__)


class UnsupervisedTrainer(Trainer):
    """Class for unsupervised training of an autoencoder.

    Parameters
    ----------
    model
        A model instance from class ``VAE``, ``VAEC``, ``SCANVI``, ``AutoZIVAE``
    gene_dataset
        A gene_dataset instance like ``CortexDataset()``
    train_size
        The train size, a float between 0 and 1 representing proportion of dataset to use for training
        to use Default: ``0.9``.
    test_size
        The test size,  a float between 0 and 1 representing proportion of dataset to use for testing
        to use Default: ``None``, which is equivalent to data not in the train set. If ``train_size`` and ``test_size``
        do not add to 1 then the remaining samples are added to a ``validation_set``.
    **kwargs
        Other keywords arguments from the general Trainer class.

    Other Parameters
    ----------------
    n_epochs_kl_warmup
        Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
        the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
        improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
        Be aware that large datasets should avoid this mode and rely on n_iter_kl_warmup. If this parameter is not
        None, then it overrides any choice of `n_iter_kl_warmup`.
    n_iter_kl_warmup
        Number of iterations for warmup (useful for bigger datasets)
        int(128*5000/400) is a good default value.
    normalize_loss
        A boolean determining whether the loss is divided by the total number of samples used for
        training. In particular, when the global KL divergence is equal to 0 and the division is performed, the loss
        for a minibatchis is equal to the average of reconstruction losses and KL divergences on the minibatch.
        Default: ``None``, which is equivalent to setting False when the model is an instance from class
        ``AutoZIVAE`` and True otherwise.

    Examples
    --------
    >>> gene_dataset = CortexDataset()
    >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
    ... n_labels=gene_dataset.n_labels)

    >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
    >>> infer.train(n_epochs=20, lr=1e-3)

    Notes
    -----
    Two parameters can help control the training KL annealing
    If your applications rely on the posterior quality,
    (i.e. differential expression, batch effect removal), ensure the number of total
    epochs (or iterations) exceed the number of epochs (or iterations) used for KL warmup

    """

    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        gene_dataset: anndata.AnnData,
        train_size: Union[int, float] = 0.9,
        test_size: Union[int, float] = None,
        n_iter_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        normalize_loss: bool = None,
        augmented_lagrangian_lr: float = 1e-1,
        lambda0: float = 1.0,
        n_grid_z: int = 50,
        temperature_start_end: list = [1.0, 0.2],
        use_adversarial_loss: bool = False,
        discriminator: Classifier = None,
        kappa: float = None,
        **kwargs
    ):
        train_size = float(train_size)
        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError(
                "train_size needs to be greater than 0 and less than or equal to 1"
            )
        super().__init__(model, gene_dataset, **kwargs)

        self.augmented_lagrangian_lr = augmented_lagrangian_lr
        self.lambda0 = lambda0
        self.temperature_start_end = temperature_start_end
        self.use_adversarial_loss = use_adversarial_loss
        self.kappa = kappa

        # Set up number of warmup iterations
        self.n_iter_kl_warmup = n_iter_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.normalize_loss = (
            not (
                hasattr(self.model, "reconstruction_loss")
                and self.model.reconstruction_loss == "autozinb"
            )
            if normalize_loss is None
            else normalize_loss
        )

        # Total size of the dataset used for training
        # (e.g. training set in this class but testing set in AdapterTrainer).
        # It used to rescale minibatch losses (cf. eq. (8) in Kingma et al., Auto-Encoding Variational Bayes, ICLR 2013)
        self.n_samples = 1.0

        if type(self) is UnsupervisedTrainer:
            (
                self.train_set,
                self.test_set,
                self.validation_set,
            ) = self.train_test_validation(model, gene_dataset, train_size, test_size)
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]
            self.validation_set.to_monitor = ["elbo"]
            self.n_samples = len(self.train_set.indices)

        if hasattr(self.model, "neural_decomposition_decoder"):
            if self.model.neural_decomposition_decoder is True:
                dim = self.model.n_input
                dev = torch.device("cuda") if self.use_cuda is True else None
                self.Lambda_z = self.lambda0 * torch.ones(1, dim, device=dev)
                self.Lambda_c = self.lambda0 * torch.ones(1, dim, device=dev)
                self.Lambda_cz_1 = self.lambda0 * torch.ones(n_grid_z, dim, device=dev)
                self.Lambda_cz_2 = self.lambda0 * torch.ones(
                    self.model.n_batch, dim, device=dev
                )
                self.grid_z = torch.randn((n_grid_z, self.model.n_latent), device=dev)

        if use_adversarial_loss is True and discriminator is None:
            discriminator = Classifier(
                n_input=self.model.n_latent,
                n_hidden=32,
                n_labels=self.gene_dataset.uns["scvi_summary_stats"]["n_batch"],
                n_layers=2,
                logits=True,
            )

        self.discriminator = discriminator
        if self.use_cuda and self.discriminator is not None:
            self.discriminator.cuda()

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors: dict, feed_labels: bool = True):
        sample_batch = tensors[_X_KEY]
        local_l_mean = tensors[_LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_LOCAL_L_VAR_KEY]
        batch_index = tensors[_BATCH_KEY]
        y = tensors[_LABELS_KEY]

        # The next lines should not be modified, because scanVI's trainer inherits
        # from this class and should NOT include label information to compute the ELBO by default
        if not feed_labels:
            y = None
        reconst_loss, kl_divergence_local, kl_divergence_global = self.model(
            sample_batch, local_l_mean, local_l_var, batch_index, y
        )
        loss = (
            self.n_samples
            * torch.mean(reconst_loss + self.kl_weight * kl_divergence_local)
            + kl_divergence_global
        )
        if self.normalize_loss:
            loss = loss / self.n_samples

        if hasattr(self.model, "neural_decomposition_decoder"):
            if self.model.neural_decomposition_decoder is True:
                (int_z, int_s, int_zs_ds, int_zs_dz) = self.model._calculate_integrals(
                    self.grid_z
                )

                # penalty with fixed lambda0
                penalty0 = self.lambda0 * (
                    int_z.abs().mean()
                    + int_s.abs().mean()
                    + int_zs_ds.abs().mean()
                    + int_zs_dz.abs().mean()
                )

                penalty_BDMM = (
                    (self.Lambda_z * int_z).mean()
                    + (self.Lambda_c * int_s).mean()
                    + (self.Lambda_cz_1 * int_zs_ds).mean()
                    + (self.Lambda_cz_2 * int_zs_dz).mean()
                )

                self.int_z = int_z
                self.int_s = int_s
                self.int_zs_ds = int_zs_ds
                self.int_zs_dz = int_zs_dz
                penalty = penalty_BDMM + penalty0
            else:
                penalty = 0
        else:
            penalty = 0

        return loss + penalty

    @property
    def kl_weight(self):
        epoch_criterion = self.n_epochs_kl_warmup is not None
        iter_criterion = self.n_iter_kl_warmup is not None
        if epoch_criterion:
            kl_weight = min(1.0, self.epoch / self.n_epochs_kl_warmup)
        elif iter_criterion:
            kl_weight = min(1.0, self.n_iter / self.n_iter_kl_warmup)
        else:
            kl_weight = 1.0
        return kl_weight

    def loss_discriminator(
        self, z, batch_index, predict_true_class=True, return_details=True
    ):

        n_classes = self.gene_dataset.uns["scvi_summary_stats"]["n_batch"]
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.discriminator(z))

        if predict_true_class:
            cls_target = one_hot(batch_index, n_classes)
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            cls_target = torch.zeros_like(one_hot_batch)
            # place zeroes where true label is
            cls_target.masked_scatter_(
                ~one_hot_batch.bool(), torch.ones_like(one_hot_batch) / (n_classes - 1)
            )

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss

    def _get_z(self, tensors):
        sample_batch = tensors[_X_KEY]

        z = self.model.sample_from_posterior_z(sample_batch, give_mean=False)

        return z

    def on_training_loop(self, tensors_dict):
        if self.use_adversarial_loss:
            if self.kappa is None:
                kappa = 1 - self.kl_weight
            else:
                kappa = self.kappa
            batch_index = tensors_dict[0][_BATCH_KEY]
            if kappa > 0:
                z = self._get_z(*tensors_dict)
                # Train discriminator
                d_loss = self.loss_discriminator(z.detach(), batch_index, True)
                d_loss *= kappa
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generative model to fool discriminator
                fool_loss = self.loss_discriminator(z, batch_index, False)
                fool_loss *= kappa

            # Train generative model
            self.optimizer.zero_grad()
            self.current_loss = loss = self.loss(*tensors_dict)
            if kappa > 0:
                (loss + fool_loss).backward()
            else:
                loss.backward()
            self.optimizer.step()

        else:
            self.current_loss = loss = self.loss(*tensors_dict)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def on_training_begin(self):
        epoch_criterion = self.n_epochs_kl_warmup is not None
        iter_criterion = self.n_iter_kl_warmup is not None
        if epoch_criterion:
            log_message = "KL warmup for {} epochs".format(self.n_epochs_kl_warmup)
            if self.n_epochs_kl_warmup > self.n_epochs:
                logger.info(
                    "KL warmup phase exceeds overall training phase"
                    "If your applications rely on the posterior quality, "
                    "consider training for more epochs or reducing the kl warmup."
                )
        elif iter_criterion:
            log_message = "KL warmup for {} iterations".format(self.n_iter_kl_warmup)
            n_iter_per_epochs_approx = ceil(
                self.gene_dataset.uns["scvi_summary_stats"]["n_cells"] / self.batch_size
            )
            n_total_iter_approx = self.n_epochs * n_iter_per_epochs_approx
            if self.n_iter_kl_warmup > n_total_iter_approx:
                logger.info(
                    "KL warmup phase may exceed overall training phase."
                    "If your applications rely on posterior quality, "
                    "consider training for more epochs or reducing the kl warmup."
                )
        else:
            log_message = "Training without KL warmup"
        logger.info(log_message)

        if hasattr(self.model, "neural_decomposition_decoder"):
            if self.model.neural_decomposition_decoder is True:
                self.temperature_schedule = torch.linspace(
                    self.temperature_start_end[0],
                    self.temperature_start_end[1],
                    self.n_epochs,
                )
                self.model._set_temperature_for_mask(self.temperature_start_end[0])

    def on_training_end(self):
        if self.kl_weight < 0.99:
            logger.info(
                "Training is still in warming up phase. "
                "If your applications rely on the posterior quality, "
                "consider training for more epochs or reducing the kl warmup."
            )

    def on_iteration_end(self):
        super().on_iteration_end()

        if hasattr(self.model, "neural_decomposition_decoder"):
            if self.model.neural_decomposition_decoder is True:
                with torch.no_grad():
                    self.Lambda_z += self.augmented_lagrangian_lr * self.int_z
                    self.Lambda_c += self.augmented_lagrangian_lr * self.int_s
                    self.Lambda_cz_1 += self.augmented_lagrangian_lr * self.int_zs_ds
                    self.Lambda_cz_2 += self.augmented_lagrangian_lr * self.int_zs_dz

                    self.lambda0 += 1.0
                    self.lambda0 = min(self.lambda0, 50000)

                    self.model._set_temperature_for_mask(
                        self.temperature_schedule[self.epoch]
                    )


class AdapterTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, posterior_test, frequency=5):
        super().__init__(model, gene_dataset, frequency=frequency)
        self.test_set = posterior_test
        self.test_set.to_monitor = ["elbo"]
        self.params = list(self.model.z_encoder.parameters()) + list(
            self.model.l_encoder.parameters()
        )
        self.z_encoder_state = copy.deepcopy(model.z_encoder.state_dict())
        self.l_encoder_state = copy.deepcopy(model.l_encoder.state_dict())
        self.n_scale = len(self.test_set.indices)

    @property
    def posteriors_loop(self):
        return ["test_set"]

    def train(self, n_path=10, n_epochs=50, **kwargs):
        for i in range(n_path):
            # Re-initialize to create new path
            self.model.z_encoder.load_state_dict(self.z_encoder_state)
            self.model.l_encoder.load_state_dict(self.l_encoder_state)
            super().train(n_epochs, params=self.params, **kwargs)

        return min(self.history["elbo_test_set"])
