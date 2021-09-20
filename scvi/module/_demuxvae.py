from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from scvi import _CONSTANTS
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
        pi_prior_scale: float = 25,
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
            # nn.Linear(n_hidden, n_hidden),
            # nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )

        self.log_mean_background = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )
        self.log_mean_foreground = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )
        self.pi_logits = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_input),
        )
        self.zi_pi_logits = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
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
        fc1 = self.fc1(x_)

        output["beta"] = F.softplus(self.log_mean_background(fc1))
        output["mean_increment"] = F.softplus(self.log_mean_foreground(fc1)) + 1
        output["pi_logits"] = self.pi_logits(fc1)
        output["zi_pi_logits"] = self.zi_pi_logits(fc1)

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

        mean_foreground = inf_out["mean_increment"] + inf_out["beta"]
        theta = torch.exp(self.log_disp)
        pi_logits = inf_out["pi_logits"]

        reconst = -log_mixture_nb(
            tensors[_CONSTANTS.X_KEY],
            inf_out["beta"],
            mean_foreground,
            theta,
            pi_logits,
            inf_out["zi_pi_logits"],
        ).sum(dim=-1)

        # px_conditional = NegativeBinomialMixture(
        #     mu1=inf_out["beta"],
        #     mu2=mean_foreground,
        #     theta1=theta,
        #     mixture_logits=pi_logits,
        # )
        # reconst = -px_conditional.log_prob(tensors[_CONSTANTS.X_KEY]).sum(dim=-1)

        # prior = (
        #     -Normal(
        #         self.background_prior_mean,
        #         torch.exp(0.5 * self.background_prior_log_scale),
        #     )
        #     .log_prob(torch.log(inf_out["beta"]))
        #     .sum(dim=-1)
        # )
        prior = 0

        pi_prior = -Normal(0, self.pi_prior_scale).log_prob(pi_logits).sum(dim=-1)

        kl_div = prior + pi_prior

        loss = torch.mean(reconst + kl_div)

        return LossRecorder(loss, reconst, kl_div)


def log_mixture_nb(x, mu_1, mu_2, theta, pi, zi_pi, eps=1e-8):
    """
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a mixture nb model.
    pi is the probability to be in the first component.

    For totalVI, the first component should be background.

    Variables:
    mu1: mean of the first negative binomial component (has to be positive support) (shape: minibatch x genes)
    mu2: mean of the second negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
    lgamma_x_theta = torch.lgamma(x + theta)
    lgamma_theta = torch.lgamma(theta)

    lgamma_x_plus_1 = torch.lgamma(x + 1)

    log_nb_1 = _log_zinb_positive(
        x,
        mu_1,
        theta,
        zi_pi,
        lgamma_x_theta=lgamma_x_theta,
        lgamma_theta=lgamma_theta,
        lgamma_x_plus_1=lgamma_x_plus_1,
    )
    log_nb_2 = (
        theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
        + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
        + lgamma_x_theta
        - lgamma_theta
        - lgamma_x_plus_1
    )

    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi)), dim=0)
    softplus_pi = F.softplus(-pi)

    log_mixture_nb = logsumexp - softplus_pi

    return log_mixture_nb


def _log_zinb_positive(
    x,
    mu,
    theta,
    pi,
    eps=1e-8,
    lgamma_x_theta=None,
    lgamma_theta=None,
    lgamma_x_plus_1=None,
):
    """Note: All inputs are torch Tensors
    log likelihood (scalar) of a minibatch according to a zinb model.
    Notes:
    We parametrize the bernoulli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant

    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    if lgamma_x_theta is None:
        lgamma_x_theta = torch.lgamma(x + theta)
    if lgamma_theta is None:
        lgamma_theta = torch.lgamma(theta)
    if lgamma_x_plus_1 is None:
        lgamma_x_plus_1 = torch.lgamma(x + 1)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + lgamma_x_theta
        - lgamma_theta
        - lgamma_x_plus_1
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res
