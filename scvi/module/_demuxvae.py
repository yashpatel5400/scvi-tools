from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

torch.backends.cudnn.benchmark = True


class DemuxVAE(BaseModuleClass):
    """
    DemuxVAE.

    Parameters
    ----------
    n_input
        Number of antibodies
    n_hidden
        Number of hidden neurons
    dropout_rate
        Dropout rate
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 15,
        dropout_rate: float = 0.2,
        pi_prior_scale: float = 0.05,
        pi_prior_mean: float = 0,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.pi_prior_mean = pi_prior_mean
        self.pi_prior_scale = pi_prior_scale
        self.latent_distribution = "normal"

        # hidden shared layer
        self.fc1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # background mean
        self.fc3 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.fc41 = nn.Linear(n_hidden, n_input)
        self.fc42 = nn.Linear(n_hidden, n_input)

        # pi
        self.fc5 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.fc61 = nn.Linear(n_hidden, n_input)
        self.fc62 = nn.Linear(n_hidden, n_input)

        self.increment = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )

        self.pi_logits = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )

        self.zi_pi_logits = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )

        # each protein/hashtag gets a dispersion parameter
        self.log_disp = nn.Parameter(torch.ones(n_input))

        if protein_background_prior_mean is None:
            self.background_prior_mean = torch.nn.Parameter(2 * torch.ones(n_input))
            self.background_prior_log_scale = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input), -10, 1)
            )
        else:
            init_mean = protein_background_prior_mean.ravel()
            init_scale = protein_background_prior_scale.ravel()
            self.background_prior_mean = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_prior_log_scale = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]

        return dict(x=x)

    def _get_generative_input(self, tensors, inference_outputs):
        x = tensors[_CONSTANTS.X_KEY]

        return dict(x=x)

    @auto_move_data
    def inference(self, x):

        x_ = torch.log(1 + x)
        output = {}
        h1 = self.fc1(x_)

        h2 = self.fc3(h1)
        output["beta"] = self.fc41(h2).exp()

        h3 = self.fc5(h1)
        output["pi_logits"] = self.fc61(h3)

        mean_increment = torch.relu(self.increment(h1)) + 1
        output["mean_increment"] = mean_increment

        return output

    @auto_move_data
    def generative(self, x):
        output = {}
        return output

    def get_pi(self, x):

        e_out = self.inference(x)
        pi = torch.sigmoid(e_out["pi_logits"])
        return pi

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight=1.0, eps=1e-8
    ):

        inf_out = inference_outputs

        mean_foreground = inf_out["mean_increment"] * inf_out["beta"]
        theta = torch.exp(self.log_disp)
        pi_logits = inf_out["pi_logits"]

        # reconst = -log_mixture_nb(
        #     x,
        #     inf_out["beta"],
        #     mean_foreground,
        #     theta,
        #     pi_logits,
        #     gen_out["zi_pi_logits"],
        # ).sum(dim=-1)

        px_conditional = NegativeBinomialMixture(
            mu1=inf_out["beta"],
            mu2=mean_foreground,
            theta1=theta,
            mixture_logits=pi_logits,
        )
        reconst = -px_conditional.log_prob(tensors[_CONSTANTS.X_KEY]).sum(dim=-1)

        prior = (
            -Normal(
                self.background_prior_mean,
                torch.exp(0.5 * self.background_prior_log_scale),
            )
            .log_prob(torch.log(inf_out["beta"]))
            .sum(dim=-1)
        )

        pi_prior = -Normal(0, self.pi_prior_scale).log_prob(pi_logits).sum(dim=-1)

        kl_div = prior + pi_prior

        loss = torch.mean(reconst + kl_weight * kl_div)

        return LossRecorder(loss, reconst, kl_div)
