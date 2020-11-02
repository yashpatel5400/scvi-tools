import logging

from anndata import AnnData
from typing import List, Optional, Sequence, Union
import torch
import numpy as np
import pandas as pd
from torch.distributions import Normal

from scvi._compat import Literal
from scvi.core.data_loaders import ScviDataLoader
from scvi.core.models import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin
from scvi.core.modules import SPLITVAE
from scvi.core.trainers import UnsupervisedTrainer

logger = logging.getLogger(__name__)


Number = Union[int, float]


class SPLITVI(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        nuisance_genes_mask: List = [],
        n_latentA: int = 10,
        n_latentB: int = 10,
        n_hidden: int = 128,
        n_hidden_split: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_cuda: bool = True,
        **model_kwargs,
    ):
        super(SPLITVI, self).__init__(adata, use_cuda=use_cuda)
        self.model = SPLITVAE(
            n_input=self.summary_stats["n_vars"],
            nuisance_genes_mask=nuisance_genes_mask,
            n_latentA=n_latentA,
            n_latentB=n_latentB,
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_hidden_split=n_hidden_split,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self.nuisance_genes_mask_ = nuisance_genes_mask
        self._model_summary_string = (
            "SPLITVAE Model with the following params: \n Non-Nuisance Latent Size: {},"
            "Nuisance Latent Size: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_latentA,
            n_latentB,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @property
    def _trainer_class(self):
        return UnsupervisedTrainer

    @property
    def _scvi_dl_class(self):
        return ScviDataLoader

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        scdl = self._make_scvi_dl(adata=adata, indices=indices, batch_size=batch_size)
        latent_z = []
        latent_u = []
        for tensors in scdl:
            inference_inputs = self.model._get_inference_input(tensors)
            outputs = self.model.inference(**inference_inputs)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            qu_m = outputs["qu_m"]
            qu_v = outputs["qu_v"]
            z = outputs["z"]
            u = outputs["u"]

            if give_mean:
                # does each model need to have this latent distribution param?
                if self.model.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                    z = self.z_encoder.z_transformation(samples)
                    z = z.mean(dim=0)
                    samples = Normal(qu_m, qu_v.sqrt()).sample([mc_samples])
                    u = self.z_encoder.z_transformation(samples)
                    u = u.mean(dim=0)
                else:
                    z = qz_m
                    u = qu_m

            latent_z += [z.cpu()]
            latent_u += [u.cpu()]
        latent_z = np.array(torch.cat(latent_z))
        latent_u = np.array(torch.cat(latent_u))
        return latent_z, latent_u

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        non_nuisance_gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        exprs = super().get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=non_nuisance_gene_list,
            library_size=library_size,
            n_samples=n_samples,
            batch_size=batch_size,
            return_mean=return_mean,
            return_numpy=True,
        )

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[self.nuisance_genes_mask_ == 0],
                index=adata.obs_names[indices],
            )
        else:
            return exprs
