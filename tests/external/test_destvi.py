import numpy as np

from scvi.data import synthetic_iid
from scvi.external import DestVI, CondSCVI


def test_destVI(save_path):

    # Step1 learn CondSCVI
    n_latent=2
    n_labels=5
    n_layers=2
    dataset = synthetic_iid(n_labels=n_labels)
    sc_model = CondSCVI(dataset, n_latent=n_latent, n_layers=n_layers)
    sc_model.train(1, train_size=1)
    z = sc_model.get_latent_representation()
    sc_model.get_vamp_prior(dataset)

    z = np.random.random_sample((100, n_latent)).astype(np.float32)
    labels = np.zeros((100,1), np.long)
    sc_model.generate_from_latent(z, labels)

    # step 2 learn destVI with multiple amortization scheme

    for amor_scheme in ["both", "none", "proportion", "latent"]:
        spatial_model = DestVI.from_rna_model(dataset, sc_model, amortization=amor_scheme)
        spatial_model.train(max_epochs=10)
        assert(not np.any(np.isnan(spatial_model.history["elbo_train"])))
        
        assert(spatial_model.get_proportions(dataset).shape == (dataset.n_obs, n_labels))
        assert(spatial_model.get_gamma(dataset).shape == (dataset.n_obs, n_latent, n_labels))

        x = dataset.X[:50]
        ind_x = np.arange(50)[:, np.newaxis].astype(np.long)
        y = np.zeros((50,1), np.long)
        assert(spatial_model.get_scale_for_ct(x, ind_x, y).shape == (50, dataset.n_vars))