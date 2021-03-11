from unittest import TestCase
from anndata import AnnData
import numpy as np
from scvi.dataset.tree import TreeDataset
from scvi.dataset.anndataset import AnnDatasetFromAnnData
from scvi.models.treevae import TreeVAE
from scvi.utils.precision_matrix import precision_matrix, marginalize_covariance
import copy
from ete3 import Tree
import torch
from scipy.stats import multivariate_normal
import unittest
import pdb


class TestMessagePassing(TestCase):
    def test_mp_inference(self):
        # Import Tree
        #tree_name = "../data/lg7_tree_hybrid_priors.alleleThresh.processed.txt"
        #tree_name = "../data/tree_test.txt"
        #tree_name = "../data/toy.nw"

        # with open(tree_name, "r") as myfile:
        #     tree_string = myfile.readlines()
        # tree = Tree(tree_string[0], 1)

        tree = Tree()
        tree.populate(4)

        print(tree)

        # Indexing nodes
        # dimension of latent space
        d = 2
        N = 0
        for idx, node in enumerate(tree.traverse("levelorder")):
            N += 1
            node.name = str(idx)
            # set node index
            node.add_features(index=idx)
        leaves = tree.get_leaves()
        leaves_index = [n.index for n in leaves]
        #print(tree)

        # fixed branch length
        var = 1.0

        #create toy Gene Expression dataset
        x = np.random.randint(1, 100, (len(leaves), 10))
        adata = AnnData(x)
        gene_dataset = AnnDatasetFromAnnData(adata)
        barcodes = [l.name for l in leaves]
        gene_dataset.initialize_cell_attribute('barcodes', barcodes)

        #create tree dataset
        tree_bis = copy.deepcopy(tree)
        cas_dataset = TreeDataset(gene_dataset, tree=tree_bis)

        use_batches = False

        vae = TreeVAE(cas_dataset.nb_genes,
                      tree=cas_dataset.tree,
                      n_batch=cas_dataset.n_batches * use_batches,
                      n_latent=d,
                      prior_t=var
                    )

        # Trivial evidence
        #rand = np.random.rand()
        #evidence_leaves = np.hstack([np.array([rand] * d)] * len(leaves))
        #evidence = np.array([np.array([rand] * d)] * len(leaves))

        # Gaussian evidence
        evidence = np.random.randn(len(leaves), 2)
        evidence_leaves = evidence.flatten()

        # test
        #evidence =  np.array([[0.39631839, 0.39631839],
                           #  [0.88514786, 0.88514786],
                           #  [0.02791887, 0.02791887],
                           #  [0.19234958, 0.19234958],
                           #  [0.36469069, 0.36469069],
                           #  [0.05106667, 0.05106667],
                           #  [0.05682752, 0.05682752],
                           #  [0.6672685, 0.6672685]]
                           # )
        #evidence_leaves = evidence.flatten()

        #####################
        print("")
        print("|>>>>>>> Test 1: Message Passing Likelihood <<<<<<<<|")
        vae.initialize_visit()
        vae.initialize_messages(
            evidence,
            cas_dataset.barcodes,
            d
        )
        vae.perform_message_passing((vae.tree & vae.root), d, False)
        mp_lik = vae.aggregate_messages_into_leaves_likelihood(
            d,
            add_prior=True
        )
        print("Test 1: Message passing output O(nd): ", mp_lik)

        # likelihood via  covariance matrix Marginalization + inversion
        leaves_covariance, full_covariance = precision_matrix(tree=tree,
                                                              d=d,
                                                              branch_length=var)
        leaves_mean = np.array([0] * len(leaves) * d)
        pdf_likelihood = multivariate_normal.logpdf(evidence_leaves,
                                                    leaves_mean,
                                                    leaves_covariance)

        print("Test 1: Gaussian marginalization + inversion output O(n^3d^3): ", pdf_likelihood)
        #self.assertTrue((np.abs(mp_lik - pdf_likelihood) < 1e-10))

        # #####################
        print("")
        print("Test 2: Message Passing Posterior Predictive Density at internal nodes ")
        do_internal = True
        if do_internal:
            for n in tree.traverse():

                if n.is_leaf():
                    continue

                query_node = n.name
                print("Query node:", query_node)

                # evidence
                #rand = np.random.rand()
                #evidence_leaves0 = np.hstack([np.array([rand] * d)] * (len(leaves)))
                #evidence0 = torch.from_numpy(np.array([np.array([rand] * d)] * len(leaves))).type(torch.DoubleTensor)
                evidence = np.random.randn(len(leaves), 2)
                evidence_leaves = evidence.flatten()

                # evidence = np.array([[ 88.27739235, 823.81583794],
                #                               [26.88869607, 29.37755672],
                #                               [ 92.94421904, 646.59218223],
                #                               [ 59.09533284, 291.41799639],
                #                               [43.20778628, 78.49412141],
                #                               [ 34.69196824, 860.644678  ],
                #                               [ 61.63106644, 201.88305813],
                #                               [ 17.24895897, 398.2992055 ]])
                #
                # evidence_leaves = np.array([88.27739235, 823.81583794, 26.88869607, 29.37755672,
                #                             92.94421904, 646.59218223, 59.09533284, 291.41799639,
                #                             43.20778628, 78.49412141, 34.69196824, 860.644678,
                #                             61.63106644, 201.88305813, 17.24895897, 398.2992055])

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

                #pdb.set_trace()

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
            #
            #     self.assertTrue((np.abs((vae.tree & query_node).nu - internal_post_covar[query_idx * d, query_idx * d])) < 1e-8)
            #     self.assertTrue((np.abs((vae.tree & query_node).mu[0] - post_mean[0])) < 1e-8)
            #     self.assertTrue((np.abs((vae.tree & query_node).mu[1] - post_mean[1])) < 1e-8)
            #
                print("")

if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    unittest.main()





