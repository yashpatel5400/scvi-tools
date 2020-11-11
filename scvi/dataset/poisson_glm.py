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
    def __init__(self, tree_name, dim, latent, vis, only):
        self.tree_name = tree_name
        self.latent = latent
        self.dim = dim
        self.covariance = None
        self.n_nodes = None
        # simulated latent space
        self.z = None
        self.X = None
        self.mu = None
        self.vis = vis
        self.leaves_only = only

        leaves_covariance, full_covariance = precision_matrix(self.tree_name, self.latent)
        if self.leaves_only:
            self.covariance = leaves_covariance
        else:
            self.covariance = full_covariance
        self.n_nodes = int(self.covariance.shape[0] / 2)

    def simulate_latent(self):
        # Define epsilon.
        epsilon = 0.0001

        # Add small perturbation for numerical stability.
        K = self.covariance + epsilon * np.identity(self.n_nodes * self.latent)

        #  Cholesky decomposition.
        L = np.linalg.cholesky(K)

        # sanity check
        assert (np.dot(L, np.transpose(L)).all() == K.all())

        # Number of samples.
        u = np.random.normal(loc=0, scale=1, size=self.latent* self.n_nodes).reshape(self.latent, self.n_nodes)

        t = time.time()
        # scale samples with Cholesky factor
        self.z = L @ u.flatten()
        self.z = self.z.reshape((-1, 2))

        print("Sampling with Cholesky took {} seconds".format(time.time() - t))

        if self.vis:
            sns.jointplot(x=self.z[:, 0],
                          y=self.z[:, 1],
                          kind="kde",
                          space=0,
                          color='green');
            plt.title('Latent Space')
            plt.xlabel('x axis')
            plt.ylabel('y axis')
            plt.show()

    def simulate_ge(self):
        # dimension of initial space (i.e number of genes)

        W = np.random.normal(size=(self.latent, self.dim))
        beta = np.random.normal(size=self.dim)

        print(W.shape, beta.shape, self.z.shape)

        self.mu = np.exp(self.z @ W + beta)

        self.X = poisson(self.mu)

        if self.vis:
            ## Poissson distribution
            fig, axes = plt.subplots(1, 1,
                                     figsize=(14, 8),
                                     sharey=True,
                                     )

            bins = np.arange(0, 30, 5)

            cm = plt.cm.get_cmap('RdYlBu_r')

            n, binss, patches = axes.hist(self.X,
                      bins=bins,
                      edgecolor='black',
                      )
            # set color of patches
            # scale values to interval [0,1]
            bin_centers = 0.5 * (binss[:-1] + binss[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))

            axes.set_title('Histogram of simulated gene expression data')
            plt.ylabel('Counts')
            plt.xlabel('Gene Expression value')
            plt.legend(['gene_' + str(i) for i in list(range(self.dim))], loc='best')
            plt.show()














