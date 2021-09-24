import anndata as ad
import numpy as np
import pandas as pd

from scvi.data import setup_anndata
from scvi.external.cpa._model import CPA
from scvi.external.cpa._utils import register_dataset


def generate_synth():
    n_cells = 2000
    n_genes = 500
    X = np.random.randint(low=0, high=1000, size=(n_cells, n_genes))
    obs = pd.DataFrame(
        dict(
            c0=np.random.randn(n_cells),
            c1=np.random.randn(n_cells),
            treatment=np.random.rand(n_cells) >= 0.5,
            Annotation=np.array(["v1", "v2"])[np.random.randint(2, size=n_cells)],
            cat2=np.random.randint(10, size=n_cells),
        )
    )
    obs.loc[:, "Annotation"] = obs.loc[:, "Annotation"].astype("category")
    obs.loc[:, "cat2"] = obs.loc[:, "cat2"].astype("category")
    dataset = ad.AnnData(
        X=X,
        obs=obs,
    )
    dataset.obsm["cont1"] = dataset.obs.loc[:, ["c0", "c1"]].values

    setup_anndata(dataset)
    batch_keys_to_dim = register_dataset(
        dataset,
        treatment_key="treatment",
        cont_key="cont1",
        cat_keys=["Annotation", "cat2"],
    )
    return dict(dataset=dataset, batch_keys_to_dim=batch_keys_to_dim)


def test_cpa():
    data = generate_synth()
    dataset = data["dataset"]
    batch_keys_to_dim = data["batch_keys_to_dim"]
    model = CPA(
        adata=dataset,
        batch_keys_to_dim=batch_keys_to_dim,
        nn_kwargs=dict(n_layers=1, n_hidden=32),
        lambd=1.0,
    )
    model.train(max_epochs=3, plan_kwargs=dict(lr=1e-4))
    model.get_reconstruction_error(adata=dataset, indices=[1, 2, 3, 4], n_samples=50)
    model.get_latent_representation()
    # model.get_counterfactual_scale(treatment=0, n_samples=25)
