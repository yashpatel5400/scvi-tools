import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import _CONSTANTS
from scvi.module import DemuxVAE

from ._totalvi import _get_totalvi_protein_priors
from .base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin

logger = logging.getLogger(__name__)


class DEMUXVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    demux variational inference

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_latent
        Dimensionality of the latent space.
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    **model_kwargs
        Keyword args for :class:`~scvi.module.DemuxVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key="protein_expression")
    >>> vae = scvi.model.TOTALVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_totalVI"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 5,
        empirical_protein_background_prior: Optional[bool] = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats["n_vars"] > 10)
        )
        if emp_prior:
            prior_mean, prior_scale = _get_totalvi_protein_priors(
                adata, protein_key=_CONSTANTS.X_KEY
            )
            print(prior_mean.shape)
        else:
            prior_mean, prior_scale = None, None

        self.module = DemuxVAE(
            n_input=self.summary_stats["n_vars"],
            n_latent=n_latent,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            **model_kwargs,
        )
        self._model_summary_string = (
            "DemuxVI Model with the following params: \nn_latent: {}, "
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @torch.no_grad()
    def predict_foreground_proba(self, n_samples_mc: int = 10000) -> np.ndarray:
        adata = self._validate_anndata(None)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=None)
        pi = []
        for tensors in scdl:
            p = 1 - self.module.get_pi(
                tensors[_CONSTANTS.X_KEY], n_samples_mc=n_samples_mc
            )
            pi.append(p.cpu().numpy())

        return np.concatenate(pi)

    def predict_identity(self) -> pd.DataFrame:

        foreground_proba = self.predict_foreground_proba()
        background_proba = 1 - foreground_proba

        df = pd.DataFrame(index=self.adata.obs_names)
        df["Negative"] = np.exp(np.sum(np.log(background_proba), axis=1))

        for i, h in enumerate(self.adata.var_names):
            df[f"Singlet_{h}"] = np.exp(
                np.log(foreground_proba[:, i])
                + np.sum(np.delete(np.log(background_proba), i, 1), axis=1)
            )

        df["Multiplet"] = 1 - df.sum(1)

        return df
