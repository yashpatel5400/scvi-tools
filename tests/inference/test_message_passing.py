import os
from unittest import TestCase
from anndata import AnnData
import numpy as np
from scvi.dataset.tree import TreeDataset
from scvi.dataset.anndataset import AnnDatasetFromAnnData
from scvi.inference.tree_inference import TreeTrainer
from scvi.models.treevae import TreeVAE
from utils.precision_matrix import precision_matrix
import torch
from ete3 import Tree
from scipy.stats import multivariate_normal
import copy
import unittest

class TestMessagePassing(TestCase):
    def test_mp_inference(self, tree_name):
        with open(tree_name, "r") as myfile:
            tree_string = myfile.readlines()

        tree = Tree(tree_string[0], 1)
        leaves = tree.get_leaves()

        #create Gene Expression dataset
        # toy data
        x = np.random.randint(1, 100, (len(leaves), 10))
        adata = AnnData(x)
        gene_dataset = AnnDatasetFromAnnData(adata)
        barcodes = [l.name for l in tree.get_leaves()]
        gene_dataset.initialize_cell_attribute('barcodes', barcodes)

        #create tree dataset
        cas_dataset = TreeDataset(gene_dataset, tree_name=tree_name)

        # No batches beacause of the message passing
        use_batches = False
        use_cuda = False

        d = 2
        vae = TreeVAE(cas_dataset.nb_genes,
                      tree=cas_dataset.tree,
                      n_batch=cas_dataset.n_batches * use_batches,
                      n_latent=d)

        trainer = TreeTrainer(
            model=vae,
            gene_dataset=cas_dataset,
            train_size=0.8,
            use_cuda=use_cuda,
            frequency=5,
        )

        posterior = trainer.create_posterior(trainer.model, cas_dataset, trainer.clades)

        data = [x for x in posterior.data_loader]
        x = data[0]

        batch_index = torch.from_numpy(np.array([[0]] * 67))
        y = copy.deepcopy(batch_index)

        outputs = vae.inference(x[0], batch_index, y)
        z = outputs["z"]

        # message passing likelihood
        vae.initialize_visit()
        # torch.cat((torch.FloatTensor([0] * (N - len(leaves)) * d).reshape(-1, d), z), 0)
        vae.initialize_messages(
            z,
            cas_dataset.barcodes,
            z.shape[1]
        )
        vae.perform_message_passing((vae.tree & vae.root), z.shape[1], False)
        mp_lik = vae.aggregate_messages_into_leaves_likelihood(
            z.shape[1],
            add_prior=False
        )
        print("Message passing output O(nd): ", mp_lik)

        # likelihood via matrix inversion
        leaves_covariance = precision_matrix(tree_name, d)
        d = z.shape[1]
        leaves_mean = np.array([0] * len(leaves) * d)
        pdf_likelihood = multivariate_normal.logpdf(np.hstack(z.detach().numpy()),
                                                    leaves_mean,
                                                    leaves_covariance)
        print("Gaussian marginalization output O(n^3d^3): ", pdf_likelihood)

        self.assertEqual(mp_lik, pdf_likelihood)


if __name__ == '__main__':
    unittest.main()




