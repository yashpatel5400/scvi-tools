import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.infer.autoguide import AutoDelta
from pyro.nn import PyroModule, PyroParam, PyroSample
from pytorch_lightning.callbacks import Callback
from torch.distributions import constraints
from torch.nn.functional import softplus

from scvi import _CONSTANTS
from scvi.module.base import PyroBaseModuleClass


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


class SCTransformPyroModel(PyroModule):
    def __init__(self, in_features, out_features, scale_factor=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_factor = scale_factor

        self.register_buffer("one", torch.tensor(1.0))

        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(
            lambda prior: dist.Uniform(-5 * self.one, 10 * self.one)
            .expand([self.out_features, self.in_features])
            .to_event(2)
        )
        self.linear.bias = PyroSample(
            lambda prior: dist.Uniform(-50 * self.one, 10 * self.one)
            .expand([self.out_features])
            .to_event(1)
        )
        # self.theta = PyroParam(
        #     lambda: 0.01 * self.one * torch.ones(self.out_features),
        #     constraint=constraints.positive,
        # )
        self.theta_unsoft = PyroParam(
            lambda: -4.5 * self.one * torch.ones(self.out_features),
        )
        self.epsilon = 1e-6

    def forward(self, x, covariates):
        if self.scale_factor is None:
            scale_factor = 1 / (x.shape[0] * x.shape[1])
        else:
            scale_factor = self.scale_factor
        log_mean = self.linear(covariates)
        with pyro.plate("data", x.shape[0]):
            theta = softplus(self.theta_unsoft)
            nb_logits = log_mean - theta.log()
            pyro.sample(
                "obs",
                dist.NegativeBinomial(total_count=theta, logits=nb_logits).to_event(1),
                # dist.Poisson(log_mean.exp()).to_event(1),
                obs=x,
            )
        return log_mean


class SCTransformModule(PyroBaseModuleClass):
    def __init__(self, in_features, out_features):

        super().__init__()
        self._model = SCTransformPyroModel(in_features, out_features)
        self._guide = AutoDelta(self.model)
        # def _passguide(*args, **kwargs):
        #     pass

        # self._guide = _passguide

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict[_CONSTANTS.X_KEY]
        covariates = tensor_dict[_CONSTANTS.CONT_COVS_KEY]

        return (x, covariates), {}
