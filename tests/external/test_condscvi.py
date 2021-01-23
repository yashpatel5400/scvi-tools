from scvi.external import CondSCVI
from scvi.data import synthetic_iid
import numpy as np

def test_condscvi(save_path):
    dataset = synthetic_iid(n_labels=5)
    model = CondSCVI(dataset)
    model.train(1, train_size=1)
    z = model.get_latent_representation()
    model.get_vamp_prior(dataset)
    model.generate_from_latent(z, np.ones((z.shape[0],1)))
