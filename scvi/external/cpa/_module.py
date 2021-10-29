import torch
import torch.distributions as db
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.nn.functional import one_hot

from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, FCLayers

from ._utils import _CE_CONSTANTS, DecoderGauss, DecoderNB, TreatmentEmbedder


class CPAModule(BaseModuleClass):
    """
    CPA module using NEGATIVE BINOMIAL likelihood

    batch_keys_to_dim:
    treatments
        list of names of the form treatment_i
    covariates
        dict of keys to ndim
        except cat continuous which are considered overalll
    """

    def __init__(
        self,
        n_genes: int,
        n_treatments: int,
        batch_keys_to_dim: dict,
        n_latent: int = 256,
        loss_ae="gauss",
        doser="linear",
        autoencoder_width=256,
        autoencoder_depth=2,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        variational: bool = False,
    ):
        super().__init__()
        shared_norm_hparams = dict(
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        shared_ae_hparams = dict(
            n_hidden=autoencoder_width,
            n_layers=autoencoder_depth,
        )

        cat_to_ncats = {
            key: val
            for key, val in batch_keys_to_dim.items()
            if key != "cat_continuous"
        }
        self.cat_to_ncats = cat_to_ncats
        self.batch_keys_to_dim = batch_keys_to_dim
        self.n_treatments = n_treatments
        self.n_latent = n_latent

        # Embedders
        self.treatments_embed = TreatmentEmbedder(
            n_cats=n_treatments, n_latent=n_latent, nonlin=doser
        )
        self.covariates_embed = nn.ModuleDict(
            {
                key: torch.nn.Embedding(n_cats, n_latent)
                for key, n_cats in cat_to_ncats.items()
            }
        )

        self.variational = variational
        if variational:
            self.encoder = Encoder(
                n_genes,
                n_latent,
                var_activation=nn.Softplus(),
                **shared_ae_hparams,
                **shared_norm_hparams,
            )
        else:
            self.encoder = FCLayers(
                n_in=n_genes,
                n_out=n_latent,
                **shared_ae_hparams,
                **shared_norm_hparams,
            )
        self.l_encoder = FCLayers(
            n_in=n_genes,
            n_out=1,
            **shared_ae_hparams,
            **shared_norm_hparams,
        )

        # Decoder components
        self.px_r = torch.nn.Parameter(torch.randn(n_genes))
        self.loss_ae = loss_ae
        if loss_ae == "gauss":
            self.decoder_x = DecoderGauss(
                n_input=n_latent,
                n_output=n_genes,
                **shared_ae_hparams,
                **shared_norm_hparams,
            )
        else:
            self.decoder_x = DecoderNB(
                n_input=n_latent,
                n_output=n_genes,
                **shared_ae_hparams,
                **shared_norm_hparams,
            )

    def _get_inference_input(self, tensors):
        x = tensors[_CE_CONSTANTS.X_KEY]
        treatments = tensors[_CE_CONSTANTS.TREATMENTS]
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
            treatments=treatments,
            c_dict=c_dict,
            c_oh_dict=c_oh_dict,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        treatments,
        c_dict,
        c_oh_dict,
    ):
        x_ = torch.log1p(x)
        if self.variational:
            qz_m, qz_v, z_basal = self.encoder(x_)
            dist_qzbasal = db.Normal(qz_m, qz_v.sqrt())
        else:
            dist_qzbasal = None
            z_basal = self.encoder(x_)
        library = self.l_encoder(x_)

        z_covariates = []
        for cat, _ in self.batch_keys_to_dim.items():
            z_cat_i = self.covariates_embed[cat](c_oh_dict[cat].argmax(-1))
            z_covariates.append(z_cat_i[None])
        z_covariates = torch.cat(z_covariates, 0).sum(
            0
        )  # Summing all covariates representations
        z_treatment = self.treatments_embed(treatments)
        z = z_basal + z_covariates + z_treatment
        return dict(
            z=z,
            z_basal=z_basal,
            dist_qz=dist_qzbasal,
            library=library,
            x=x,
            treatments=treatments,
            c_dict=c_dict,
            c_oh_dict=c_oh_dict,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
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
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        z,
        library,
    ):
        dist_px = self.decoder_x(inputs=z, library=library, px_r=self.px_r)
        return dict(
            dist_px=dist_px,
        )

    def loss(self, tensors, inference_outputs, generative_outputs):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        x = inference_outputs["x"]
        # Reconstruction loss & regularizations
        dist_px = generative_outputs["dist_px"]
        log_px = dist_px.log_prob(x).sum(-1)
        # Compute reconstruction
        reconstruction_loss = -log_px
        if self.loss_ae == "gauss":
            # This condition is here to be consistent with the original codebase
            # TODO: Remove this if needed
            variance = dist_px.scale ** 2
            mean = dist_px.loc
            term1 = variance.log().div(2)
            term2 = (x - mean).pow(2).div(variance.mul(2))
            reconstruction_loss = (term1 + term2).mean()

        # TODO: Add KL annealing if needed
        if self.variational:
            dist_qz = inference_outputs["dist_qz"]
            dist_pz = db.Normal(
                torch.zeros_like(dist_qz.loc), torch.ones_like(dist_qz.scale)
            )
            kl_z = kl_divergence(dist_qz, dist_pz).sum(-1)
            loss = -log_px + kl_z
        else:
            loss = reconstruction_loss

        return loss

    def get_expression(self, tensors, **inference_kwargs):
        """Computes gene expression means and std.

        Only implemented for the gaussian likelihood.

        Parameters
        ----------
        tensors : dict
            Considered inputs

        """
        _, generative_outputs, = self.forward(
            tensors,
            compute_loss=False,
            inference_kwargs=inference_kwargs,
        )
        if self.loss_ae == "gauss":
            mus = generative_outputs["dist_px"].loc
            stds = generative_outputs["dist_px"].scale
            return mus, stds
        else:
            raise ValueError
