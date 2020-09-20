import time
import numpy as np
from numpy.random import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
sys.path.append(os.path.realpath('.'))
from utils.precision_matrix import precision_matrix

class Poisson_GLM:
    def __init__(self, tree_name, dim, latent, vis):
        self.tree_name = tree_name
        self.latent = latent
        self.dim = dim
        self.covariance = None
        self.n_leaves = None
        # simulated latent space
        self.z = None
        self.X = None
        self.vis = vis

    def simulate_latent(self):
        self.covariance, _ = precision_matrix(self.tree_name, self.latent)
        self.n_leaves = int(self.covariance.shape[0] / 2)

        # Define epsilon.
        epsilon = 0.0001

        # Add small perturbation for numerical stability.
        K = self.covariance + epsilon * np.identity(self.n_leaves * self.latent)

        #  Cholesky decomposition.
        L = np.linalg.cholesky(K)

        # sanity check
        assert (np.dot(L, np.transpose(L)).all() == K.all())

        # Number of samples.
        u = np.random.normal(loc=0, scale=1, size=self.latent* self.n_leaves).reshape(self.latent, self.n_leaves)

        t = time.time()
        # scale samples with Cholesky factor
        self.z = L @ u.flatten()
        self.z = self.z.reshape((-1, 2))

        print("Sampling with Cholesky took {} seconds".format(time.time() - t))

        if self.vis:
            sns.jointplot(x=self.z[:, 0],
                          y=self.z[:, 1],
                          kind="kde",
                          space=0);
            plt.show()

    def simulate_ge(self):
        # dimension of initial space (i.e number of genes)

        W = np.random.normal(size=(self.latent, self.dim))
        beta = np.random.normal(size=self.dim)

        print(W.shape, beta.shape, self.z.shape)

        mu = np.exp(self.z @ W + beta)

        self.X = poisson(mu)

        if self.vis:
            ## Poissson distribution
            fig, axes = plt.subplots(1, 1, figsize=(14, 3), sharey=True)

            bins = np.arange(0, 30, 5)
            axes.hist(self.X, bins=bins)
            axes.set_title('simulated gene expressiond data')
            plt.show()














