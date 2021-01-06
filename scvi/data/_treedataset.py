import logging
from scvi.data._scvidataset import ScviDataset
from ete3 import Tree
import numpy as np

logger = logging.getLogger(__name__)

class TreeDataset(ScviDataset):
    """Forms a ``ScviDataset`` with a corresponding Tree structure (phylogeny) relating
    every cell.
    This is the dataset class that will be used to interact with the TreeVAE model. It's
    important to observe here that this function does not take in expression data from a CSV
    or sparse matrix or a Anndata object, for example, but rather assumes that an scVI ScviDataset has
    already been created. The resulting API of the dataset remains very similar to that of a
    typical ScviDataset but with the addition of a tree (of class `ete3.Tree`) that
    will be used as a prior during model fitting.

    :param expr: ``scvi.data._scvidataset.ScviDataset`` instance.
    :param tree_name: file path to tree to read in from.
    :param tree: ``ete3.Tree`` instance.
    """

    def __init__(self, scvidataset: ScviDataset, tree_name: str = None, tree: Tree = None):

        if tree_name is not None:
            self.tree = Tree(tree_name, 1)
            # polytomy is not a problem anymore: message passing deals with general trees
            # self.tree.resolve_polytomy(recursive=True)
        else:
            self.tree = tree

        if self.tree is None:
            logger.error(
                "Must provide a tree file path or a loaded ete3 tree object if you're using TreeDataset."
            )

            # assert we have barcode labels for cells
            if "barcodes" not in scvidataset.attributes_and_types:
                logger.error("Must provide cell barcode, or names, as a cell attribute.")

        super().__init__()

        self.set_distance()
        if self.adata.obs_names:
            self.filter_cells_by_tree()

    def set_distance(self):
        # set distance
        for n in self.tree.traverse():
            n.distance = self.tree.get_distance(n)

    def filter_cells_by_tree(self):
        """
        Prunes away cells that don't appear consistently between the tree object and the
        RNA expression dataset.
        """

        leaves = self.tree.get_leaf_names()
        keep_barcodes = np.intersect1d(leaves, list(self.adata.obs_names))
        self.tree.prune(keep_barcodes)

        #return self.filter_cells_by_attribute(keep_barcodes, on="barcodes")











