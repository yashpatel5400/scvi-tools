from typing import Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from scvi._compat import Literal
from scvi.module.base import BaseModuleClass
from scvi.nn import FCLayers
from scvi.train import TrainingPlan


class CPATrainingPlan(TrainingPlan):
    def __init__(
        self,
        module: BaseModuleClass,
        lr=1e-3,
        weight_decay=1e-6,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversary_steps: int = 3,
        reg_adversary: int = 5,
        penalty_adversary: int = 3,
        adversary_width: float = 64,
        adversary_lr=1e-3,
        # dosers_lr=1e-3,
        # autoencoder_wd=1e-3,
        adversary_depth: float = 2,
        step_size_lr: int = 15,
        adversary_wd=1e-2,
        autoencoder_wd=1e-2,
    ):
        """Training plan for the CPA model"""
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
        )

        adversarial_models_kwargs = dict(
            n_hidden=adversary_width,
            n_layers=adversary_depth,
        )
        self.adversary_lr = adversary_lr
        self.adversary_wd = adversary_wd
        self.autoencoder_wd = autoencoder_wd
        # self.dosers_lr = dosers_lr
        # self.autoencoder_wd = autoencoder_wd

        # Adversarial modules and hparams
        self.covariates_adv_nn = nn.ModuleDict(
            {
                key: FCLayers(
                    n_in=module.n_latent, n_out=n_cats, **adversarial_models_kwargs
                )
                for key, n_cats in module.cat_to_ncats.items()
            }
        )
        self.treatments_adv_nn = FCLayers(
            n_in=module.n_latent, n_out=module.n_treatments, **adversarial_models_kwargs
        )
        self.adv_loss_covariates = nn.CrossEntropyLoss()
        self.adv_loss_treatments = nn.BCEWithLogitsLoss()
        self.reg_adversary = reg_adversary
        self.penalty_adversary = penalty_adversary
        self.automatic_optimization = False
        self.iter_count = 0
        self.adversary_steps = adversary_steps
        self.step_size_lr = step_size_lr

    def _adversarial_classifications(self, z_basal):
        """Computes adversarial classifier predictions

        Parameters
        ----------
        z_basal : tensor
            Basal states
        """
        pred_treatments = self.treatments_adv_nn(z_basal)
        pred_covariates = dict()
        for cat_cov_name in self.module.cat_to_ncats:
            pred_covariates[cat_cov_name] = self.covariates_adv_nn[cat_cov_name](
                z_basal
            )
        return pred_treatments, pred_covariates

    def adversarial_losses(self, tensors, inference_outputs, generative_outputs):
        """Computes adversarial classification losses and regularizations"""
        z_basal = inference_outputs["z_basal"]
        treatments = tensors["treatments"]
        c_dict = inference_outputs["c_dict"]
        pred_treatments, pred_covariates = self._adversarial_classifications(z_basal)

        # Classification losses
        adv_cats_loss = 0.0
        for cat_cov_name in self.module.cat_to_ncats:
            adv_cats_loss += self.adv_loss_covariates(
                pred_covariates[cat_cov_name],
                c_dict[cat_cov_name].long().squeeze(-1),
            )
        adv_t_loss = self.adv_loss_treatments(pred_treatments, (treatments > 0).float())
        adv_loss = adv_t_loss + adv_cats_loss

        # Penalty losses
        adv_penalty_cats = 0.0
        for cat_cov_name in self.module.cat_to_ncats:
            cat_penalty = (
                torch.autograd.grad(
                    pred_covariates[cat_cov_name].sum(), z_basal, create_graph=True
                )[0]
                .pow(2)
                .mean()
            )
            adv_penalty_cats += cat_penalty

        adv_penalty_treatments = (
            torch.autograd.grad(
                pred_treatments.sum(),
                z_basal,
                create_graph=True,
            )[0]
            .pow(2)
            .mean()
        )
        adv_penalty = adv_penalty_cats + adv_penalty_treatments

        return dict(
            adv_loss=adv_loss,
            adv_penalty=adv_penalty,
        )

    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = torch.optim.Adam(
            params1, lr=self.lr, eps=self.autoencoder_wd, weight_decay=self.weight_decay
        )
        params2 = filter(
            lambda p: p.requires_grad,
            list(self.covariates_adv_nn.parameters())
            + list(self.treatments_adv_nn.parameters()),
        )
        optimizer2 = torch.optim.Adam(
            params2,
            lr=self.adversary_lr,
            eps=self.adversary_wd,
            weight_decay=self.weight_decay,
        )
        optims = [optimizer1, optimizer2]
        if self.step_size_lr is not None:
            scheduler1 = StepLR(optimizer1, step_size=self.step_size_lr)
            scheduler2 = StepLR(optimizer2, step_size=self.step_size_lr)
            schedulers = [scheduler1, scheduler2]
            return optims, schedulers
        else:
            return optims

    def training_step(self, batch, batch_idx):
        opt, adv_opt = self.optimizers()

        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False)
        reconstruction_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        losses = self.adversarial_losses(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        adv_loss = losses["adv_loss"]
        adv_penalty = losses["adv_penalty"]
        # Adversarial update
        if self.iter_count % self.adversary_steps:
            adv_opt.zero_grad()
            adv_loss = adv_loss + self.penalty_adversary * adv_penalty
            self.manual_backward(adv_loss)
            adv_opt.step()

        # Model update
        else:
            opt.zero_grad()
            loss = reconstruction_loss.mean() - self.reg_adversary * adv_loss
            self.manual_backward(loss)
            opt.step()

        self.iter_count += 1
        return dict(
            reconstruction_loss=reconstruction_loss.mean(),
            adv_loss=adv_loss,
            adv_penalty=adv_penalty,
        )

    def training_epoch_end(self, outputs):
        reconstruction_loss, adv_loss, adv_penalty = 0, 0, 0
        for tensors in outputs:
            reconstruction_loss += tensors["reconstruction_loss"]
            adv_loss += tensors["adv_loss"]
            adv_penalty += tensors["adv_penalty"]
            # n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        self.log("reconstruction_loss_train", reconstruction_loss.mean())
        self.log("adv_loss_train", adv_penalty)
        self.log("adv_penalty_train", adv_penalty)
        if self.step_size_lr:
            sch, adv_sch = self.lr_schedulers()
            sch.step()
            adv_sch.step()

    def validation_step(self, batch, batch_idx):
        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False)
        reconstruction_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        losses = self.adversarial_losses(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        adv_loss = losses["adv_loss"]
        adv_penalty = losses["adv_penalty"]
        return dict(
            reconstruction_loss=reconstruction_loss.mean(),
            adv_loss=adv_loss,
            adv_penalty=adv_penalty,
            # n_obs=reconstruction_loss.shape[0],
        )

    def validation_epoch_end(self, outputs):
        reconstruction_loss, adv_loss, adv_penalty = 0, 0, 0
        for tensors in outputs:
            reconstruction_loss += tensors["reconstruction_loss"]
            adv_loss += tensors["adv_loss"]
            adv_penalty += tensors["adv_penalty"]
            # n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        self.log("reconstruction_loss_validation", reconstruction_loss.mean())
        self.log("adv_loss_validation", adv_penalty)
        self.log("adv_penalty_validation", adv_penalty)
