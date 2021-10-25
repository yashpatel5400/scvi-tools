from typing import Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData

from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.train import TrainRunner

from ._module import CPAModule
from ._task import CPATrainingPlan


class CPA(UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(self, adata, batch_keys_to_dim, **nn_kwargs):
        super().__init__(adata)
        self.n_genes = self.summary_stats["n_vars"]
        n_treatments = adata.obsm["treatments"].shape[-1]
        self.module = CPAModule(
            n_genes=self.n_genes,
            n_treatments=n_treatments,
            batch_keys_to_dim=batch_keys_to_dim,
            **nn_kwargs,
        )

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = CPATrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

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
    ):
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        reco = []
        for tensors in scdl:
            _, _, _reco = self.module(tensors)
            reco.append(_reco.cpu())
        reco = torch.cat(reco, dim=0)
        return reco.mean()

    @torch.no_grad()
    def predict(
        self,
        treatments: torch.Tensor = None,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        log_scales = []
        n_treatments = self.module.n_treatments
        assert (treatments.shape[0] == n_treatments) & (treatments.ndim == 1)
        for tensors in scdl:
            device = tensors["treatments"].device
            n_obs = tensors["treatments"].shape[0]
            ce_t = (
                treatments.clone()[None].expand(n_obs, n_treatments).to(device=device)
            )
            tensors["treatment"] = ce_t
            log_scale = self.module.get_log_scale(tensors)
            log_scales.append(log_scale.cpu())
        return torch.cat(log_scales, dim=0).numpy()

    def get_embeddings(self, dose=1.0):
        treatments = dose * torch.eye(5, device=self.module.device)
        embeds = self.module.treatments_embed(treatments).detach().cpu().numpy()
        return embeds
