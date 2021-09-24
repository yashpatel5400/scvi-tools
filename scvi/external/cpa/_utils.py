import torch.nn as nn

from scvi.data import register_tensor_from_anndata
from scvi.distributions import NegativeBinomial
from scvi.nn import FCLayers


def register_dataset(
    adata,
    treatment_key,
    cont_key,
    cat_keys,
):
    register_tensor_from_anndata(adata, "treatment", "obs", treatment_key)
    batch_keys_to_dim = dict()
    if cont_key is not None:
        register_tensor_from_anndata(adata, "cat_continuous", "obsm", cont_key)
        batch_keys_to_dim = {"cat_continuous": adata.obsm[cont_key].shape[-1]}
    for cat in cat_keys:
        new_cat_key = "cat_{}".format(cat)
        register_tensor_from_anndata(
            adata, new_cat_key, "obs", cat, is_categorical=True
        )
        batch_keys_to_dim[new_cat_key] = len(adata.obs[cat].unique())
    return batch_keys_to_dim


class _CE_CONSTANTS:
    X_KEY = "X"
    TREATMENT = "treatment"
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

    def forward(self, inputs, library, px_r, t):
        px_scale = self.hidd(inputs)
        px_rate = library.exp() * px_scale
        return NegativeBinomial(mu=px_rate, theta=px_r.exp())
