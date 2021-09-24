from typing import Optional, Sequence

import numpy as np
import torch
from anndata import AnnData

from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin

from ._module import CPAModule


class CPA(UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata,
        batch_keys_to_dim,
        nn_kwargs,
        lambd=1e-3,
    ):
        super().__init__(adata)
        self.n_genes = self.summary_stats["n_vars"]
        self.module = CPAModule(
            n_genes=self.n_genes,
            batch_keys_to_dim=batch_keys_to_dim,
            n_latent=10,
            nn_kwargs=nn_kwargs,
            lambd=lambd,
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        n_samples=200,
    ):
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        reco = []
        for tensors in scdl:
            _reco = self.module.get_reconstruction(tensors, n_samples=n_samples)
            reco.append(_reco.cpu())
        reco = torch.cat(reco, dim=0)
        return reco.mean()

    @torch.no_grad()
    def get_counterfactual_scale(
        self,
        treatment,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples_overall: int = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
    ):
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        log_scales = []
        for tensors in scdl:
            ce_t = treatment * torch.ones_like(tensors["treatment"])
            tensors["treatment"] = ce_t
            log_scale = self.module.get_log_scale(
                tensors, inference_kwargs=dict(n_samples=n_samples)
            )
            log_scales.append(log_scale.cpu())
        return torch.cat(log_scales, dim=1).numpy()
