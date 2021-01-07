import logging
from typing import Union

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import anndata
from scvi import _CONSTANTS
from scvi.data._treedataset import TreeDataset
from scvi.core.modules.treevae import TreeVAE
from scvi.core.trainers.trainer import Trainer
from scvi.core.data_loaders import ScviDataLoader

logger = logging.getLogger(__name__)

plt.switch_backend("agg")

class SequentialCladeSampler(SubsetRandomSampler):
    """ A sampler that is used to feed observations to the VAE for model fitting.
    A `SequentiaCladeSampler` instance is instantiated with a subtree, which has had leaves
    collapsed to form 'clades', which are groups of observations (leaves) that we assume are
    drawn iid. A single iteration using the SequentailCladeSampler instance will randomly sample
    a single observation from each clade, which we will use as a batch for training our VAE.
    :param data_source: A list of 'clades', each of which corresponding to a 'leaf' of the model's tree.
    :param args: a set of arguments to be passed into ``SubsetRandomSampler``
    :param kwargs: Keyword arguments to be passed into ``SubsetRandomSampler``
    """

    def __init__(self, data_source, *args, **kwargs):
        super().__init__(data_source, *args, **kwargs)
        self.clades = data_source

    def __iter__(self):
        # randomly draw a cell from each clade (i.e. bunch of leaves)
        return iter([np.random.choice(l) for l in self.clades if len(l) > 0])

class Treetrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder
    with a latent tree structure.
    Args:
        :model: A model instance from class ``TreeVAE``
        :gene_dataset: A TreeDataset instance
        :train_size: The train size, either a float between 0 and 1 or an integer for the number of training samples
         to use Default: ``0.8``.
        :test_size: The test size, either a float between 0 and 1 or an integer for the number of training samples
         to use Default: ``None``, which is equivalent to data not in the train set. If ``train_size`` and ``test_size``
         do not add to 1 or the length of the dataset then the remaining samples are added to a ``validation_set``.
        :n_epochs_kl_warmup: Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
            the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
            improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.
    Examples:
        #>>> tree_dataset = TreeDataset(GeneExpressionDataset, tree) ??
        >>> treevae = treeVAE(tree_dataset.nb_genes, tree = tree_dataset.tree
        ... n_batch =tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer = TreeTrainer(treevae, tree_dataset)
        >>> trainer.train(n_epochs=400)
    """
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        adata: anndata.AnnData,
        train_size: Union[int, float] = 0.9,
        test_size: Union[int, float] = None,
        n_iter_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        normalize_loss: bool = None,
        **kwargs
    ):

        train_size = float(train_size)
        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError(
                "train_size needs to be greater than 0 and less than or equal to 1"
            )

        super().__init__(model, adata, **kwargs)

        # Set up number of warmup iterations
        self.n_iter_kl_warmup = n_iter_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.normalize_loss = (
            not (
                hasattr(self.model, "gene_likelihood")
                and self.model.gene_likelihood == "autozinb"
            )
            if normalize_loss is None
            else normalize_loss
        )
        # Clades
        self.clades = []
        self.barcodes = model.barcodes

        self.train_set, self.test_set = self.train_test_validation(
            model, adata, train_size
        )

        self.train_set.to_monitor = ["elbo"]
        self.test_set.to_monitor = ["elbo"]
        self.validation_set.to_monitor = ["elbo"]

        # loss function
        self.history_train, self.history_eval = {}, {}
        self.history_train['elbo'], self.history_train['Reconstruction'], self.history_train['MP_lik'], \
        self.history_train['Gaussian pdf'] = [], [], [], []
        self.history_eval['elbo'], self.history_eval['Reconstruction'], self.history_eval['MP_lik'], self.history_eval[
            'Gaussian pdf'] = [], [], [], []

    @property
    def scvi_data_loaders_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        """ Computes the loss of the model after a specific iteration.
        Computes the mean reconstruction loss, which is derived after a forward pass
        of the model
        :param tensors: Observations to be passed through model
        :return: Mean reconstruction loss.
        """
        sample_batch = tensors[_CONSTANTS.X_KEY]
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]

        # Reconstruction Loss
        reconst_loss, qz, mp_lik, kl_l = self.model.forward(
            x=sample_batch,
            local_l_mean=local_l_mean,
            local_l_var=local_l_var,
            batch_index=batch_index,
            barcodes=self.barcodes,
        )

        loss_1 = torch.mean(reconst_loss)
        self.history_train['Reconstruction'].append(loss_1.item())
        loss_2 = self.kl_weight * torch.mean(qz)
        self.history_train['Gaussian pdf'].append(self.kl_weight * loss_2.item())
        loss_3 = -1 * self.kl_weight * (mp_lik / reconst_loss.shape[0])
        self.history_train['MP_lik'].append(self.kl_weight * loss_3.item())
        self.history_train['elbo'].append(loss_1.item() + loss_2.item() + loss_3.item())

        return loss_1 + loss_2 + loss_3

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    def train_test_validation(
        self,
        model: TreeVAE = None,
        adata = None,
        train_size: float = 0.9,
        test_size: float = None,
        type_class=ScviDataLoader,
    ):
        """Creates posteriors ``train_set``, ``test_set``, ``validation_set``.
        If ``train_size + test_size < 1`` then ``validation_set`` is non-empty.
        This works a bit differently for a TreeTrainer - in order to respect the
        tree prior we need to draw our observations from within sets of cells related
        to one another (i.e in a clade).  One can think of this analagously to
        identifying clusters from the hierarchical ordering described by the tree, and splitting
        each cluster into train/test/validation.
        The procedure of actually clustering the tree into clades that contain several
        iid observations is done in the constructor function for TreeVAE (scvi.models.treevae).
        This procedure below will simply split the clades previously identified into
        train/test/validation sets according to the train_size specified.
        :param model: A ``TreeVAE` model.
        :param gene_dataset: A ``TreeDataset`` instance.
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        :param type_class: Type of Posterior object to create (here, TreePosterior)
        """

        def get_indices_in_dataset(_subset, _subset_indices, master_list):

            _cells = np.array(_subset)[np.array(_subset_indices)]
            filt = np.array(list(map(lambda x: x in _cells, master_list)))

            return list(np.where(filt == True)[0])

        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if adata is None and hasattr(self, "model")
            else adata
        )

        barcodes = gene_dataset.barcodes

        # this is where we need to shuffle within the tree structure
        train_indices, test_indices, validate_indices = [], [], []

        # for each clade induced by an internal node at a given depth split into
        # train, test, and validation and append these indices to the master list

        # introduce an index for each leaf in the tree
        for l in model.tree.get_leaves():
            c = l.cells
            indices = get_indices_in_dataset(c, list(range(len(c))), barcodes)
            l.indices = np.array(indices)
            self.clades.append(indices)

        # randomly split leaves into test, train, and validation sets
        for l in model.tree.get_leaves():
            leaf_bunch = l.indices

            if len(leaf_bunch) == 1:
                x = random.random()
                if x < train_size:
                    train_indices.append([leaf_bunch[0]])
                else:
                    test_indices.append([leaf_bunch[0]])
                    #train_indices.append([leaf_bunch[0]])

            else:
                n_train, n_test = _validate_shuffle_split(
                    len(leaf_bunch), test_size, train_size
                )

                random_state = np.random.RandomState(seed=self.seed)
                permutation = random_state.permutation(leaf_bunch)
                test_indices.append(list(permutation[:n_test]))
                train_indices.append(list(permutation[n_test: (n_test + n_train)]))
                # split test set in two
                validate_indices.append(list(permutation[(n_test + n_train):]))

        # some print statement to ensure test/train/validation sets created correctly
        print("train_leaves: ", train_indices)
        print("test_leaves: ", test_indices)
        print("validation leaves: ", validate_indices)

        return (
            self.create_scvi_dl(
                model, adata, indices=train_indices, type_class=type_class
            ),
            self.create_scvi_dl(
                model, adata, indices=test_indices, type_class=type_class
            ),
            self.create_scvi_dl(
                model, adata, indices=validate_indices, type_class=type_class
            ),
        )

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        super().train(n_epochs=n_epochs, lr=lr, eps=eps, params=params)
