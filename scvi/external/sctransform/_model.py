from typing import Optional, Union

import numpy as np
import pyro
from anndata import AnnData
from pytorch_lightning.callbacks import Callback

from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass
from scvi.train import PyroTrainingPlan, TrainRunner

from ._module import SCTransformModule


class SCTransform(BaseModelClass):
    """
    Reimplementation of SCTransform.


    Parameters
    ----------
    sc_adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    **model_kwargs
        Keyword args for :class:`~scvi.external.stereoscope.RNADeconv`

    Examples
    --------
    >>> sc_adata = anndata.read_h5ad(path_to_sc_anndata)
    >>> scvi.data.setup_anndata(sc_adata, labels_key="labels")
    >>> stereo = scvi.external.stereoscope.RNAStereoscope(sc_adata)
    >>> stereo.train()
    """

    def __init__(
        self,
        sc_adata: AnnData,
        **model_kwargs,
    ):
        super().__init__(sc_adata)
        self.n_genes = self.summary_stats["n_vars"]
        self.module = SCTransformModule(
            in_features=self.summary_stats["n_continuous_covs"],
            out_features=self.n_genes,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCTransform Model with params: \nn_genes: {}"
        ).format(
            self.n_genes,
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        lr: float = 1e-3,
        use_jit: bool = True,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 1024,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for :class:`~scvi.lightning.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.lightning.Trainer`.
        """
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
        optim = pyro.optim.ClippedAdam({"lr": lr})
        if use_jit:
            loss_fn = pyro.infer.JitTrace_ELBO()
            dl = self._make_data_loader(self.adata)
            trainer_kwargs["callbacks"] = [PyroJitGuideWarmup(dl)]
        else:
            loss_fn = pyro.infer.Trace_ELBO()
        pyro.clear_param_store()
        training_plan = PyroTrainingPlan(
            self.module, optim=optim, loss_fn=loss_fn, **plan_kwargs
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


class PyroJitGuideWarmup(Callback):
    def __init__(self, train_dl) -> None:
        super().__init__()
        self.dl = train_dl

    def on_train_start(self, trainer, pl_module):
        """
        Way to warmup Pyro Guide in an automated way.

        Also device agnostic.
        """

        # warmup guide for JIT
        pyro_model = pl_module.module.model
        dev = pyro_model.linear.weight.device
        pyro_guide = pl_module.module.guide
        for tensors in self.dl:
            tens = {k: t.to(dev) for k, t in tensors.items()}
            args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
            pyro_guide(*args, **kwargs)
            break
