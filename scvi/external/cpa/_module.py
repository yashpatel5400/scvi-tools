import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import FCLayers

from ._utils import _CE_CONSTANTS, DecoderNB


class CPAModule(BaseModuleClass):
    """
    CPA module using NEGATIVE BINOMIAL likelihood
    To align with the original paper, we use COVARIATE-specific shifts
    """

    def __init__(
        self,
        n_genes: int,
        batch_keys_to_dim: dict,
        n_latent: int = 10,
        nn_kwargs=None,
        lambd=None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.lambd = lambd
        cat_dim = sum(batch_keys_to_dim.values())
        cat_to_ncats = {
            key: val
            for key, val in batch_keys_to_dim.items()
            if key != "cat_continuous"
        }
        self.cat_to_ncats = cat_to_ncats
        self.batch_keys_to_dim = batch_keys_to_dim
        self.px_r = torch.nn.Parameter(torch.randn(n_genes))
        self.decoder_x = DecoderNB(
            n_input=n_latent,
            n_output=n_genes,
            **nn_kwargs,
        )

        self.covs_mat = torch.nn.Parameter(torch.randn(cat_dim, n_latent))

        self.treatment_mat = torch.nn.Parameter(torch.randn(1, n_latent))

        n_hidden = nn_kwargs.get("n_hidden", 128)
        n_layers = nn_kwargs.get("n_layers", 1)
        self.encoder = FCLayers(
            n_in=n_genes,
            n_out=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_encoder = FCLayers(
            n_in=n_genes,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.adv_covs = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(n_latent, 128),
                    nn.ReLU(),
                    nn.Linear(128, _ndim),
                    # nn.Softmax(),
                )
                for key, _ndim in cat_to_ncats.items()
            }
        )
        self.adv_treatment = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.adv_loss = nn.CrossEntropyLoss()
        self.adv_loss_bin = nn.BCEWithLogitsLoss()

    def _get_inference_input(self, tensors):
        x = tensors[_CE_CONSTANTS.X_KEY]
        treatment = tensors[_CE_CONSTANTS.TREATMENT]
        c_dict = dict()
        c_oh_dict = dict()
        for key, n_classes in self.batch_keys_to_dim.items():
            val = tensors[key]
            c_dict[key] = val
            if key in self.cat_to_ncats.keys():
                val_oh = one_hot(val.long().squeeze(), num_classes=n_classes)
            else:
                val_oh = val
            c_oh_dict[key] = val_oh
        input_dict = dict(
            x=x,
            treatment=treatment,
            c_dict=c_dict,
            c_oh_dict=c_oh_dict,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        treatment,
        c_dict,
        c_oh_dict,
        n_samples: int = 1,
    ):
        x_ = torch.log1p(x)
        z_basal = self.encoder(x_)

        concats = []
        for cat, _ in self.batch_keys_to_dim.items():
            concats.append(c_oh_dict[cat])
        concats = torch.cat(concats, -1)
        z_covs = concats.float() @ self.covs_mat
        z_treatment = treatment @ self.treatment_mat

        z = z_basal + z_covs + z_treatment
        library = self.l_encoder(x_)

        return dict(
            z=z,
            library=library,
            x=x,
            treatment=treatment,
            c_dict=c_dict,
            c_oh_dict=c_oh_dict,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        treatment = tensors[_CE_CONSTANTS.TREATMENT]

        c_dict = dict()
        c_oh_dict = dict()
        for key, n_classes in self.batch_keys_to_dim.items():
            val = tensors[key]
            c_dict[key] = val
            if key in self.cat_to_ncats.keys():
                val_oh = one_hot(val.long().squeeze(), num_classes=n_classes)
            else:
                val_oh = val
            c_oh_dict[key] = val_oh

        input_dict = {
            "z": z,
            "library": library,
            "treatment": treatment,
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        z,
        library,
        treatment,
    ):
        pred_treatment = self.adv_treatment(z)

        pred_cat_covs_dict = dict()
        for cat_cov_name in self.cat_to_ncats:
            pred_cat_covs_dict[cat_cov_name] = self.adv_covs[cat_cov_name](z)

        dist_px = self.decoder_x(inputs=z, library=library, px_r=self.px_r, t=treatment)

        return dict(
            pred_treatment=pred_treatment,
            dist_px=dist_px,
            pred_cat_covs_dict=pred_cat_covs_dict,
        )

    def get_reconstruction(self, tensors, n_samples=1):
        inference_outputs, gen_outputs = self.forward(
            tensors, inference_kwargs=dict(n_samples=n_samples), compute_loss=False
        )
        x = inference_outputs["x"]
        dist_px = gen_outputs["dist_px"]
        log_px = dist_px.log_prob(x).sum(-1)
        return log_px

    def loss(self, tensors, inference_outputs, generative_outputs):
        inference_outputs, gen_outputs = self.forward(
            tensors, inference_kwargs=dict(n_samples=1), compute_loss=False
        )

        treatment = inference_outputs["treatment"]
        c_dict = inference_outputs["c_dict"]
        x = inference_outputs["x"]

        dist_px = gen_outputs["dist_px"]
        pred_treatment = gen_outputs["pred_treatment"]
        pred_cat_covs_dict = gen_outputs["pred_cat_covs_dict"]
        log_px = dist_px.log_prob(x).sum(-1)

        # adv_t_loss = self.adv_loss(pred_treatment, treatment.long())
        adv_t_loss = self.adv_loss_bin(pred_treatment, treatment)
        adv_cats_loss = 0.0
        for cat_cov_name in self.cat_to_ncats:
            adv_cats_loss += self.adv_loss(
                pred_cat_covs_dict[cat_cov_name],
                c_dict[cat_cov_name].long().squeeze(-1),
            )

        adv_loss = adv_t_loss + adv_cats_loss
        penalty = self.lambd * adv_loss
        loss = -log_px.mean() + penalty

        kl_local = torch.tensor(penalty)
        kl_global = torch.tensor(0.0)
        return LossRecorder(
            loss=loss,
            reconstruction_loss=-log_px,
            kl_local=kl_local,
            kl_global=kl_global,
        )

    def get_log_scale(self, tensors, inference_kwargs):
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            compute_loss=False,
            inference_kwargs=inference_kwargs,
        )
        log_px_rate = generative_outputs["dist_px"].mu.log2()
        px_scale = log_px_rate - inference_outputs["library"].exp().log2()
        return px_scale
