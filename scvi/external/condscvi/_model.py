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

from scvi.model.base import BaseModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)


class CondSCVI(RNASeqMixin, VAEMixin, BaseModelClass):
    """
    Conditional version of single-cell Variational Inference, used for hierarchical deconvolution of spatial transcriptomics data.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
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
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: str = "normal",
        use_gpu: bool = True,
        **module_kwargs,
    ):
        super(CondSCVI, self).__init__(adata, use_gpu=use_gpu)

        self.module = VAEC(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_labels=self.summary_stats["n_labels"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **module_kwargs,
        )
        self._model_summary_string = (
            "Conditional SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @property
    def _plan_class(self):
        return TrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader

    @torch.no_grad()
    def get_vamp_prior(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
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
        mean = []
        var = []
        for tensors in scdl:
            x = tensors[_CONSTANTS.X_KEY]
            y = tensors[_CONSTANTS.LABELS_KEY]
            out = self.module.inference(x, y)
            mean_, var_  = out["qz_m"], out["qz_v"]
            mean += [mean_.cpu()]
            var += [var_.cpu()]
        return np.array(torch.cat(mean)), np.array(torch.cat(var))

    @torch.no_grad()
    def generate_from_latent(
        self,
        z: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the scaled parameter of the NB for every cell.

        Parameters
        ----------
        z
            Numpy array with latent space
        labels
            Numpy array with labels
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        gene_expression
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        dl = DataLoader(TensorDataset(torch.tensor(z), torch.tensor(labels, dtype=torch.long)), batch_size=128) # create your dataloader

        rate = []
        for tensors in dl:
            px_rate = self.module.generative(tensors[0], torch.ones((tensors[0].shape[0], 1)), tensors[1])["px_scale"]
            rate += [px_rate.cpu()]
        return np.array(torch.cat(rate))
