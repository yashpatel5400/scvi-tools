import torch
import torch.nn as nn
from torch.distributions import Normal

from scvi.data import register_tensor_from_anndata
from scvi.distributions import NegativeBinomial
from scvi.nn import FCLayers


def register_dataset(
    adata,
    treatments_key,
    cat_keys,
):
    """TEMPORARY.

    Quick and dirty way to construct the dataloader for the CPA model.
    This function will be replaced once the AnnData refactor is completed within
    scvi-tools.

    Parameters
    ----------
    adata : AnnData
    treatments_key : str
        Obsm key for the treatments
    cat_keys : list
        List of categorical covariates
    """
    register_tensor_from_anndata(adata, "treatments", "obsm", treatments_key)
    batch_keys_to_dim = dict()
    for cat in cat_keys:
        new_cat_key = "cat_{}".format(cat)
        register_tensor_from_anndata(
            adata, new_cat_key, "obs", cat, is_categorical=True
        )
        batch_keys_to_dim[new_cat_key] = len(adata.obs[cat].unique())
    return batch_keys_to_dim


class _CE_CONSTANTS:
    X_KEY = "X"
    TREATMENTS = "treatments"
    C_KEY = "covariates"
    CAT_COVS_KEY = "cat_covs"
    CONT_COVS_KEY = "cont_covs"
    BATCH_KEY = "batch_indices"
    LOCAL_L_MEAN_KEY = "local_l_mean"
    LOCAL_L_VAR_KEY = "local_l_var"
    LABELS_KEY = "labels"
    PROTEIN_EXP_KEY = "protein_expression"


class DecoderNB(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_layers,
        use_layer_norm=True,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidd = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_output,
                n_layers=n_layers,
                n_hidden=n_hidden,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
            ),
            nn.Softmax(-1),
        )

    def forward(self, inputs, library, px_r):
        px_scale = self.hidd(inputs)
        px_rate = library.exp() * px_scale
        return NegativeBinomial(mu=px_rate, theta=px_r.exp())


class DecoderGauss(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_layers,
        use_layer_norm=True,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidd = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
        )
        self.var_ = nn.Linear(n_hidden, n_output)
        self.mean_ = nn.Linear(n_hidden, n_output)

    def forward(self, inputs, library, px_r):
        hidd_ = self.hidd(inputs)
        locs = self.mean_(hidd_)
        variances = self.var_(hidd_).exp().add(1).log().add(1e-3)
        return Normal(loc=locs, scale=variances.sqrt())


class TreatmentEmbedder(nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, n_latent, n_cats, nonlin="sigmoid"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super().__init__()
        # self.embeddings = nn.Embedding(n_cats, n_latent)
        self.cats_to_latent_map = nn.Linear(n_cats, n_latent, bias=False)
        self.nonlin = nonlin
        self.beta = nn.Parameter(torch.ones(1, n_cats), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, n_cats), requires_grad=True)

    def get_responses(self, x):
        if self.nonlin == "logsigm":
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def forward(self, x):
        ys = self.get_responses(x)  # shape (n_batch, n_treatments)
        return self.cats_to_latent_map(ys)
