import logging
import torch
from anndata import AnnData

import numpy as np
from typing import Optional, Sequence
from scvi._compat import Literal
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import TrainingPlan
from scvi.external.condscvi._module import VAEC
from scvi import _CONSTANTS
from torch.utils.data import TensorDataset, DataLoader

from scvi.model.base import BaseModelClass

logger = logging.getLogger(__name__)


class DestVI(BaseModelClass):
    """
    Hierarchical DEconvolution of Spatial Transcriptomics data (DestVI)

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
        state_dict: List[OrderedDict],
        amortized: bool = False,
        use_gpu: bool = True,
        **module_kwargs,
    ):
        st_adata.obs["_indices"] = np.arange(st_adata.n_obs)
        register_tensor_from_anndata(st_adata, "ind_x", "obs", "_indices")
        super(DestVI, self).__init__(st_adata, use_gpu=use_gpu)
        self.module = hstDeconv(
            n_spots=st_adata.n_obs,
            n_labels=cell_type_mapping.shape[0],
            state_dict=state_dict,
            n_genes=st_adata.n_vars,
            use_gpu=True,
            **module_kwargs,
        )
        self.cell_type_mapping = cell_type_mapping
        self._model_summary_string = ("DestVI Model")
        self.init_params_ = self._get_init_params(locals())       

    @property
    def _plan_class(self):
        return TrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader

    def get_proportions(self, keep_noise=False) -> np.ndarray:
        """
        Returns the estimated cell type proportion for the spatial data. Shape is n_cells x n_labels OR n_cells x (n_labels + 1) if keep_noise

        Parameters:
        -----------
        keep_noise
            whether to account for the noise term as a standalone cell type in the proportion estimate.
        """
        column_names = self.cell_type_mapping
        if keep_noise:
            column_names = column_names.append("noise_term")
        return pd.DataFrame(
            data=self.module.get_proportions(keep_noise),
            columns=column_names,
            index=self.adata.obs.index,
        )

        #TODO: Get the gamma variables as well and fix format
        #TODO: Get variables for amortized versions as well

    @torch.no_grad()
    def get_scale_for_ct(
        self,
        x: Optional[np.ndarray] = None,
        ind_x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Return the scaled parameter of the NB for every cell in queried cell types

        Parameters
        ----------

        Returns
        -------
        gene_expression
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        dl = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), 
                    torch.tensor(ind_x, dtype=torch.long), 
                    torch.tensor(y, dtype=torch.long)), batch_size=128) # create your dataloader

        rate = []
        for tensors in dl:
            # TODO: use inference and generative to auto-move data
            px_rate = self.module.get_ct_specific_expression(tensors[0], tensors[1], tensors[2])
            rate += [px_rate.cpu()]
        return np.array(torch.cat(rate))
    
