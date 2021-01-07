from unittest import TestCase
from anndata import AnnData
import numpy as np
import scvi
from scvi.data._treedataset import TreeDataset
from scvi.core.modules.treevae import TreeVAE
from scvi.utils.precision_matrix import precision_matrix, marginalize_covariance
import torch
from ete3 import Tree
from scipy.stats import multivariate_normal
import unittest


class TestMessagePassing(TestCase):
    def test_mp_inference(self):
        tree_name = "../data/tree_test.txt"
        with open(tree_name, "r") as myfile:
            tree_string = myfile.readlines()

        tree = Tree(tree_string[0], 1)
        leaves = tree.get_leaves()

        # dimension of latent space
        d = 2
        N = 0
        for idx, node in enumerate(tree.traverse("levelorder")):
            N += 1
            # set node index
            node.add_features(index=idx)

        leaves_index = [(tree & n.name).index for n in leaves]
        leaves_index.sort()

        # create toy Anndata gene expression dataset
        x = np.random.randint(1, 100, (len(leaves), 10))
        adata = AnnData(x)
        adata.obs_names = [n.name for n in tree.get_leaves()]
        adata.obs["batch"] = 0
        adata.layers["counts"] = adata.X.copy()

        #register Anndata dataset
        scvi.data.setup_anndata(adata, layer="counts", batch_key="batch")

        # No batches beacause of the message passing
        use_batches = False

        treevae = TreeVAE(cas_dataset.nb_genes,
                      tree=cas_dataset.tree,
                      n_batch=cas_dataset.n_batches * use_batches,
                      n_latent=d)

        evidence_leaves = np.hstack([np.array([1.0] * d)] * len(leaves))
        evidence = torch.from_numpy(np.array([np.array([1.0] * d)] * len(leaves))).type(torch.DoubleTensor)

        #####################
        print("")
        print("|>>>>>>> Test 1: Message Passing Likelihood <<<<<<<<|")
        treevae.initialize_visit()
        treevae.initialize_messages(
            evidence,
            cas_dataset.barcodes,
            d
        )
        treevae.perform_message_passing((vae.tree & vae.root), d, False)
        mp_lik = vae.aggregate_messages_into_leaves_likelihood(
            d,
            add_prior=True
        )
        print("Test 1: Message passing output O(nd): ", mp_lik)

        # likelihood via  covariance matrix Marginalization + inversion
        leaves_covariance, full_covariance = precision_matrix(tree_name, d, 1)
        leaves_mean = np.array([0] * len(leaves) * d)
        pdf_likelihood = multivariate_normal.logpdf(evidence_leaves,
                                                    leaves_mean,
                                                    leaves_covariance)

        print("Test 1: Gaussian marginalization + inversion output O(n^3d^3): ", pdf_likelihood)
        self.assertTrue((np.abs(mp_lik - pdf_likelihood) < 1e-10))

        #####################
        print("")
        print("Test 2: Message Passing Posterior Predictive Density at internal node")

        #query_node = '0|0|0|0|9|0|3|7|0|-|-|-|16|0|0|0|2|0|0|2|0|0|0|0|8|0|2|0|0|-|-|2|0|0|0|0|0|-|-|-|0|0|0|2|2|2|0|0' # --> internal node
        query_node = "0|0|0|0|5|10|0|0|0|2|0|0|0|0|0|0|0|0|2|0|0|0|0|0|0|0|0|0|0"

        # evidence
        evidence_leaves = np.hstack([np.array([1.0] * d)] * (len(leaves)))

        # MP call from the query node

        # Gaussian conditioning formula
        to_delete_idx_ii = [i for i in list(range(N)) if (i not in leaves_index)]
        to_delete_idx_ll = [i for i in list(range(N)) if (i in leaves_index)]

        # partition index
        I = [idx for idx in list(range(N)) if idx not in to_delete_idx_ii]
        L = [idx for idx in list(range(N)) if idx not in to_delete_idx_ll]

        # covariance marginalization
        cov_ii = marginalize_covariance(full_covariance, [to_delete_idx_ii], d)
        cov_ll = marginalize_covariance(full_covariance, [to_delete_idx_ll], d)
        cov_il = marginalize_covariance(full_covariance, [to_delete_idx_ii, to_delete_idx_ll], d)
        cov_li = np.copy(cov_il.T)

        internal_post_mean_transform = np.dot(cov_li, np.linalg.inv(cov_ii))
        internal_post_covar = cov_ll - np.dot(np.dot(cov_li, np.linalg.inv(cov_ii)), cov_il)

        # Message Passing
        vae.initialize_visit()
        vae.initialize_messages(
            evidence,
            cas_dataset.barcodes,
            d
        )
        vae.perform_message_passing((vae.tree & query_node), d, True)
        query_idx = (tree & query_node).index
        query_idx = L.index(query_idx)

        post_mean = np.dot(internal_post_mean_transform, np.hstack(evidence_leaves))[query_idx * d: query_idx * d + 2]
        print("Test2: Gaussian conditioning formula O(n^3d^3): ",
              post_mean,
              internal_post_covar[query_idx * d, query_idx * d])

        print("Test2: Message passing output O(nd): ",
              (vae.tree & query_node).mu, (vae.tree & query_node).nu
              )

        self.assertTrue((np.abs((vae.tree & query_node).nu - internal_post_covar[query_idx * d, query_idx * d])) < 1e-8)
        self.assertTrue((np.abs((vae.tree & query_node).mu[0] - post_mean[0])) < 1e-8)
        self.assertTrue((np.abs((vae.tree & query_node).mu[1] - post_mean[1])) < 1e-8)

        print("")

if __name__ == '__main__':
    unittest.main()
