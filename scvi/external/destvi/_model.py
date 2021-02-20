import logging
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, isspmatrix


import numpy as np
import pandas as pd
from typing import Optional, List, OrderedDict
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import TrainingPlan
from scvi.external.destvi._module import HSTDeconv
from scvi.external.condscvi._model import CondSCVI
from torch.utils.data import TensorDataset, DataLoader
from scvi.data import register_tensor_from_anndata

from scvi.model.base import BaseModelClass

logger = logging.getLogger(__name__)


class DestVI(BaseModelClass):
    """
    Hierarchical DEconvolution of Spatial Transcriptomics data (DestVI).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    st_adata
        spatial transcriptomics AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    state_dict
        state_dict from the CondSCVI model 
    use_gpu
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scvi.modules.VAEC`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.external.CondSCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        st_adata: AnnData,
        cell_type_mapping: np.ndarray,
        sc_state_dict: List[OrderedDict],
        n_latent: int,
        n_layers: int,
        n_hidden: int,
        use_gpu: bool = True,
        **module_kwargs,
    ):
        st_adata.obs["_indices"] = np.arange(st_adata.n_obs)
        register_tensor_from_anndata(st_adata, "ind_x", "obs", "_indices")
        super(DestVI, self).__init__(st_adata, use_gpu=use_gpu)
        self.module = HSTDeconv(
            n_spots=st_adata.n_obs,
            n_labels=cell_type_mapping.shape[0],
            sc_state_dict=sc_state_dict,
            n_genes=st_adata.n_vars,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            **module_kwargs,
        )
        self.cell_type_mapping = cell_type_mapping
        self._model_summary_string = ("DestVI Model")
        self.init_params_ = self._get_init_params(locals())  


    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model: CondSCVI,
        use_gpu: bool = True,
        **model_kwargs,
    ):
        """
        Alternate constructor for exploiting a pre-trained model on RNA-seq data.

        Parameters
        ----------
        st_adata
            registed anndata object
        sc_model
            trained RNADeconv model
        use_gpu
            Use the GPU or not.
        **model_kwargs
            Keyword args for :class:`~scvi.external.DestVI`
        """
        state_dict = (sc_model.module.decoder.state_dict(), sc_model.module.px_decoder.state_dict(), sc_model.module.px_r.detach().cpu().numpy())

        return cls(
            st_adata,
            sc_model.scvi_setup_dict_["categorical_mappings"]["_scvi_labels"][
                "mapping"
            ],
            state_dict,
            sc_model.module.n_latent,
            sc_model.module.n_layers,
            sc_model.module.n_hidden,
            use_gpu=use_gpu,
            **model_kwargs,
        )
 

    @property
    def _plan_class(self):
        return TrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader

    def get_proportions(self, dataset=None, keep_noise=False) -> np.ndarray:
        """
        Returns the estimated cell type proportion for the spatial data.

        Shape is n_cells x n_labels OR n_cells x (n_labels + 1) if keep_noise.

        Parameters
        ----------
        keep_noise
            whether to account for the noise term as a standalone cell type in the proportion estimate.
        """
        column_names = self.cell_type_mapping
        if keep_noise:
            column_names = np.append(column_names, "noise_term")

        if self.module.amortization in ["both", "proportion"]:
            data = dataset.X
            if isspmatrix(data):
                data = data.A
            dl = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=128) # create your dataloader
            prop_ = []
            for tensors in dl:
                prop_local = self.module.get_proportions(x=tensors[0])
                prop_ += [prop_local.cpu()]
            data = np.array(torch.cat(prop_))
        else:
            data = self.module.get_proportions(keep_noise=keep_noise)
        return pd.DataFrame(
            data=data,
            columns=column_names,
            index=self.adata.obs.index,
        )

    def get_gamma(self, dataset=None) -> np.ndarray:
        """
        Returns the estimated cell-type specific latent space for the spatial data.

        Shape is n_cells x n_latent x n_labels.
        """
        if self.module.amortization in ["both", "latent"]:
            data = dataset.X
            if isspmatrix(dataset.X):
                data = dataset.X.A
            dl = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=128) # create your dataloader
            gamma_ = []
            for tensors in dl:
                gamma_local = self.module.get_gamma(x=tensors[0])
                gamma_ += [gamma_local.cpu()]
            data = np.array(torch.cat(gamma_, dim=-1))
        else:
            data = self.module.get_gamma()
        return np.transpose(data, (2, 0, 1))

    @torch.no_grad()
    def get_scale_for_ct(
        self,
        x: Optional[np.ndarray] = None,
        ind_x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Return the scaled parameter of the NB for every cell in queried cell types.

        Parameters
        ----------
        x
            gene expression data
        ind_x
            indices
        y
            cell types

        Returns
        -------
        gene_expression
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        dl = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), 
                    torch.tensor(ind_x, dtype=torch.long), 
                    torch.tensor(y, dtype=torch.long)), batch_size=128) # create your dataloader
        scale = []
        for tensors in dl:
            px_scale = self.module.get_ct_specific_expression(tensors[0], tensors[1], tensors[2])
            scale += [px_scale.cpu()]
        return np.array(torch.cat(scale))
    
