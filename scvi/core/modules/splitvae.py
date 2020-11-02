import torch
import numpy as np

import torch.nn.functional as F
from typing import Iterable, Dict, List
from scvi._compat import Literal

from torch import nn as nn
from torch.distributions import Normal, kl_divergence as kl
from scvi.core.modules._base import FCLayers
from scvi.core.modules.vae import VAE
from scvi.core.modules.utils import one_hot
from scvi import _CONSTANTS
from scvi.core.modules._base._base_module import SCVILoss


class SplitDecoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers

    Returns
    -------
    """

    def __init__(
        self,
        n_input_x: int,
        n_input_s: int,
        n_output_x: int,
        n_output_s: int,
        n_cat_list: Iterable[int] = None,
        n_layers_x: int = 1,
        n_layers_s: int = 1,
        n_hidden_x: int = 128,
        n_hidden_s: int = 128,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input_x + n_input_s,
            n_out=n_hidden_x,
            n_cat_list=n_cat_list,
            n_layers=n_layers_x,
            n_hidden=n_hidden_x,
            dropout_rate=0,
        )
        # mean gamma
        # removed softmax from here
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden_x, n_output_x))
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden_x, n_output_x)
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden_x, n_output_x)

        # S DECODER
        self.ps_decoder = FCLayers(
            n_in=n_input_s,
            n_out=n_hidden_s,
            n_cat_list=n_cat_list,
            n_layers=n_layers_s,
            n_hidden=n_hidden_s,
            dropout_rate=0,
        )
        # mean gamma
        # removed softmax from here
        self.ps_scale_decoder = nn.Sequential(nn.Linear(n_hidden_s, n_output_s))
        # dispersion: here we only deal with gene-cell dispersion case
        self.ps_r_decoder = nn.Linear(n_hidden_s, n_output_s)
        # dropout
        self.ps_dropout_decoder = nn.Linear(n_hidden_s, n_output_s)
        self.use_softmax = use_softmax

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        u: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input_x,)``
        u :
            tensor with shape ``(n_input_s,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        latent = torch.cat([z, u], dim=1)

        px = self.px_decoder(latent, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        # px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        ps = self.ps_decoder(u, *cat_list)
        ps_scale = self.ps_scale_decoder(ps)
        ps_dropout = self.ps_dropout_decoder(ps)

        if self.use_softmax:
            px_scale = nn.Softmax(dim=-1)(px_scale)
            ps_scale = nn.Softmax(dim=-1)(ps_scale)
        else:
            px_scale = torch.exp(px_scale)
            ps_scale = torch.exp(ps_scale)

        scale = torch.cat([px_scale, ps_scale], dim=1)
        rate = torch.exp(library) * scale

        px_rate = rate[:, : px_scale.shape[1]]
        ps_rate = rate[:, px_scale.shape[1] :]

        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        # ps_rate = torch.exp(library) * ps_scale  # torch.clamp( , max=12)
        ps_r = self.ps_r_decoder(ps) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout, ps_scale, ps_r, ps_rate, ps_dropout


class SplitEncoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Different from standard encoder in that the input and output is split into
    two groups
    :param n_input: The total dimensionality of the input (data space)
    :param n_inputA: The size of the first group (input)
    :param n_output: The total dimensionality of the output (latent space)
    :param n_outputA: The dimensionality of the first output group (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_inputA: int,
        n_outputA: int,
        n_inputB: int,
        n_outputB: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_hidden_split: int = 128,
    ):
        super(SplitEncoder, self).__init__()
        self.n_outputA = n_outputA
        self.encoderA = FCLayers(
            n_in=n_inputA + n_inputB,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_split,
            dropout_rate=dropout_rate,
        )

        self.mean_encoderA = nn.Linear(n_hidden, n_outputA + n_outputB)
        self.var_encoderA = nn.Linear(n_hidden, n_outputA + n_outputB)

        # self.encoderB = FCLayers(
        #     n_in=n_inputB,
        #     n_out=n_hidden,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden,
        #     dropout_rate=dropout_rate,
        # )

        # self.mean_encoderB = nn.Linear(n_hidden, n_outputB)
        # self.var_encoderB = nn.Linear(n_hidden, n_outputB)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x: torch.Tensor, s: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)
        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        # added this for single encoder. shouldnt split x into x_ and s_ in SplitVAE
        x = torch.cat([x, s], dim=1)
        q = self.encoderA(x, *cat_list)
        q_m = self.mean_encoderA(q)
        q_v = torch.exp(
            self.var_encoderA(q)
        )  # (computational stability safeguard)torch.clamp(, -5, 5)
        latent = self.reparameterize(q_m, q_v)

        q_mA = q_m[:, : self.n_outputA]
        q_mB = q_m[:, self.n_outputA :]
        q_vA = q_v[:, : self.n_outputA]
        q_vB = q_v[:, self.n_outputA :]
        latentA = latent[:, : self.n_outputA]
        latentB = latent[:, self.n_outputA :]

        # for two encoders
        # qB = self.encoderB(s, *cat_list)
        # q_mB = self.mean_encoderB(qB)
        # q_vB = torch.exp(
        #     self.var_encoderB(qB)
        # )  # (computational stability safeguard)torch.clamp(, -5, 5)
        # latentB = self.reparameterize(q_mB, q_vB)

        return q_mA, q_mB, q_vA, q_vB, latentA, latentB


class SPLITVAE(VAE):
    r"""Variational auto-encoder model.
    Same as VAE but adds a split to partition the latent space
    Adds two new parameters: n_inputA, and n_latentA
        -- Yes, I know the names are bad...
    :param n_input: Number of input genes
    :param n_inputA: Number of input genes in the first Group
    :param n_latentA: Number of latent factors in the first Group
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    :param log_variational: Log variational distribution
    :param reconstruction_loss:  One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)
    """

    def __init__(
        self,
        n_input: int,
        nuisance_genes_mask: List = [],
        n_latentA: int = 10,
        n_latentB: int = 10,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        n_hidden_split: int = 128,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
    ):

        super(SPLITVAE, self).__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden,
            n_latentA + n_latentB,
            n_layers,
            dropout_rate,
            dispersion,
            log_variational,
            gene_likelihood,
            latent_distribution,
        )
        self.nuisance_genes_idx = np.where(nuisance_genes_mask == 1)[0]
        self.non_nuisance_genes_idx = np.where(nuisance_genes_mask == 0)[0]

        self.n_inputA = n_input - np.sum(nuisance_genes_mask)
        self.n_inputB = np.sum(nuisance_genes_mask)
        self.n_latentA = n_latentA
        self.n_latentB = n_latentB

        self.px_r = torch.nn.Parameter(torch.randn(self.n_inputA))
        self.ps_r = torch.nn.Parameter(torch.randn(self.n_inputB))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = SplitEncoder(
            n_inputA=self.n_inputA,
            n_outputA=self.n_latentA,
            n_inputB=self.n_inputB,
            n_outputB=self.n_latentB,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            n_hidden_split=n_hidden_split,
        )

        self.decoder = SplitDecoder(
            n_input_x=self.n_latentA,
            n_input_s=self.n_latentB,
            n_output_x=self.n_inputA,
            n_output_s=self.n_inputB,
            n_cat_list=[n_batch],
            n_layers_x=n_layers,
            n_layers_s=n_layers,
            n_hidden_x=n_hidden,
            n_hidden_s=n_hidden,
        )

    def _get_inference_input(self, tensors, transform_batch=None):
        inference_input = super()._get_inference_input(tensors, transform_batch)
        gene_expression = inference_input["x"]
        # assert self.n_inputA + self.n_inputB == gene_expression.shape[1]
        x = gene_expression[:, self.non_nuisance_genes_idx]
        s = gene_expression[:, self.nuisance_genes_idx]
        inference_input["x"] = x
        inference_input["s"] = s
        return inference_input

    def inference(
        self, x, s, batch_index=None, n_samples=1, transform_batch=None
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        s_ = s
        if self.log_variational:
            x_ = torch.log(1 + x_)
            s_ = torch.log(1 + s_)

        # Sampling
        qz_m, qu_m, qz_v, qu_v, z, u = self.z_encoder(x_, s_)

        latent = torch.cat([x_, s_], dim=1)
        ql_m, ql_v, library = self.l_encoder(latent)

        outputs = dict()

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

            qu_m = qu_m.unsqueeze(0).expand((n_samples, qu_m.size(0), qu_m.size(1)))
            qu_v = qu_v.unsqueeze(0).expand((n_samples, qu_v.size(0), qu_v.size(1)))
            # when z is normal, untran_z == z
            untran_u = Normal(qu_m, qu_v.sqrt()).sample()
            u = self.z_encoder.z_transformation(untran_u)

            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            qu_m=qu_m,
            qu_v=qu_v,
            u=u,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )
        return outputs

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        generative_input = super()._get_generative_input(
            tensors, inference_outputs, transform_batch=transform_batch
        )
        generative_input["u"] = inference_outputs["u"]
        return generative_input

    def generative(self, z, u, library, batch_index, y=None):
        (
            px_scale,
            px_r,
            px_rate,
            px_dropout,
            ps_scale,
            ps_r,
            ps_rate,
            ps_dropout,
        ) = self.decoder(self.dispersion, z, u, library, batch_index)

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
            ps_r = self.ps_r

        px_r = torch.exp(px_r)
        ps_r = torch.exp(ps_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            ps_scale=ps_scale,
            ps_r=ps_r,
            ps_rate=ps_rate,
            ps_dropout=ps_dropout,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        scale_loss: float = 1.0,
    ):
        """Returns the reconstruction loss and the KL divergences

        Parameters
        ----------
        x
            tensor of values with shape (batch_size, n_input)
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variancess of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape (batch_size, n_labels) (Default value = None)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences

        """
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qu_m = inference_outputs["qu_m"]
        qu_v = inference_outputs["qu_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]

        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        ps_rate = generative_outputs["ps_rate"]
        ps_r = generative_outputs["ps_r"]
        ps_dropout = generative_outputs["ps_dropout"]

        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]
        gene_expression = tensors[_CONSTANTS.X_KEY]

        x = gene_expression[:, self.non_nuisance_genes_idx]
        s = gene_expression[:, self.nuisance_genes_idx]

        # KL Divergence of Z
        meanz = torch.zeros_like(qz_m)
        scalez = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(meanz, scalez)).sum(
            dim=1
        )
        # kl Divergence of U
        means = torch.zeros_like(qu_m)
        scales = torch.ones_like(qu_v)
        kl_divergence_u = kl(Normal(qu_m, torch.sqrt(qu_v)), Normal(means, scales)).sum(
            dim=1
        )

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        kl_divergence_l = kl_weight * kl_divergence_l
        kl_divergence = kl_divergence_z + kl_divergence_u + kl_divergence_l

        reconst_loss_x = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        reconst_loss_s = self.get_reconstruction_loss(s, ps_rate, ps_r, ps_dropout)

        reconst_loss = reconst_loss_s + reconst_loss_x
        loss = torch.mean(reconst_loss + kl_divergence)

        return SCVILoss(loss, reconst_loss, kl_divergence, 0.0)

    # def get_sample_rate(
    #     self, d, batch_index=None, y=None, n_samples=1, transform_batch=None
    # ) -> torch.Tensor:
    #     """Returns the tensor of means of the negative binomial distribution

    #     Parameters
    #     ----------
    #     x
    #         tensor of values with shape ``(batch_size, n_input)``
    #     y
    #         tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
    #     batch_index
    #         array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
    #     n_samples
    #         number of samples (Default value = 1)
    #     transform_batch
    #         int of batch to transform samples into (Default value = None)

    #     Returns
    #     -------
    #     type
    #         tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``

    #     """
    #     x = d[:, self.non_nuisance_genes_idx]
    #     s = d[:, self.nuisance_genes_idx]

    #     x_rate = self.inference(
    #         x,
    #         s,
    #         batch_index=batch_index,
    #         y=y,
    #         n_samples=n_samples,
    #         transform_batch=transform_batch,
    #     )["px_rate"]

    #     s_rate = self.inference(
    #         x,
    #         s,
    #         batch_index=batch_index,
    #         y=y,
    #         n_samples=n_samples,
    #         transform_batch=transform_batch,
    #     )["ps_rate"]

    #     all_genes = torch.zeros(x_rate.shape[0], x_rate.shape[1] + s_rate.shape[1])
    #     all_genes = all_genes.to(x_rate.device)
    #     all_genes[:, self.non_nuisance_genes_idx] = x_rate
    #     all_genes[:, self.nuisance_genes_idx] = s_rate
    #     return all_genes

    # def get_sample_scale(
    #     self, d, batch_index=None, y=None, n_samples=1, transform_batch=None
    # ) -> torch.Tensor:
    #     """Returns the tensor of predicted frequencies of expression

    #     Parameters
    #     ----------
    #     x
    #         tensor of values with shape ``(batch_size, n_input)``
    #     batch_index
    #         array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
    #     y
    #         tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
    #     n_samples
    #         number of samples (Default value = 1)
    #     transform_batch
    #         int of batch to transform samples into (Default value = None)

    #     Returns
    #     -------
    #     type
    #         tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``

    #     """
    #     x = d[:, self.non_nuisance_genes_idx]
    #     s = d[:, self.nuisance_genes_idx]

    #     x_scale = self.inference(
    #         x,
    #         s,
    #         batch_index=batch_index,
    #         y=y,
    #         n_samples=n_samples,
    #         transform_batch=transform_batch,
    #     )["px_scale"]

    #     s_scale = self.inference(
    #         x,
    #         s,
    #         batch_index=batch_index,
    #         y=y,
    #         n_samples=n_samples,
    #         transform_batch=transform_batch,
    #     )["ps_scale"]

    #     all_genes = torch.zeros(x_scale.shape[0], x_scale.shape[1] + s_scale.shape[1])
    #     all_genes = all_genes.to(x_scale.device)
    #     all_genes[:, self.non_nuisance_genes_idx] = x_scale
    #     all_genes[:, self.nuisance_genes_idx] = s_scale
    #     return all_genes

    # def sample_from_posterior_z(
    #     self, d, y=None, give_mean=False, n_samples=5000
    # ) -> torch.Tensor:
    #     """Samples the tensor of latent values from the posterior

    #     Parameters
    #     ----------
    #     x
    #         tensor of values with shape ``(batch_size, n_input)``
    #     y
    #         tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
    #     give_mean
    #         is True when we want the mean of the posterior  distribution rather than sampling (Default value = False)
    #     n_samples
    #         how many MC samples to average over for transformed mean (Default value = 5000)

    #     Returns
    #     -------
    #     type
    #         tensor of shape ``(batch_size, n_latent)``

    #     """
    #     x = d[:, self.non_nuisance_genes_idx]
    #     s = d[:, self.nuisance_genes_idx]
    #     qz_m, qu_m, qz_v, qu_v, z, u = model.z_encoder(x, s, None)
    #     return z, u
