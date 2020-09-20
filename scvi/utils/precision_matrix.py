import time
import numpy as np
from ete3 import Tree


def precision_matrix(tree_name, d):
    """
    :param tree_name: path of the ete3 tree file
    :param d: dimension of latent space
    :return: the covariance matrix of the gaussian vector induced by the tree,
     after inversion and post processing of the constructed precision matrix
    """
    # load tree
    with open(tree_name, "r") as myfile:
        tree_string = myfile.readlines()

    tree = Tree(tree_string[0], 1)
    leaves = tree.get_leaves()

    # introduce an index for all the nodes
    parents = {}
    N = 0
    #print("Traversing the tree")
    for idx, node in enumerate(tree.traverse("levelorder")):
        N += 1
        # set node index
        node.add_features(index=idx)

    #print("total number of nodes in the tree:", N)
    #print("number of leaves in the tree:", len(tree.get_leaves()))


    for n in tree.traverse("levelorder"):
        if not n.is_root():
            ancestor = n.up.index
            parents[n.index] = ancestor

    inverse_covariance = np.zeros((N * d, N * d))

    for i in parents:
        pi_ind = parents[i]
        inverse_covariance[i * d: (i + 1) * d, i * d: (i + 1) * d] += np.identity(d)
        inverse_covariance[pi_ind * d: (pi_ind + 1) * d, pi_ind * d: (pi_ind + 1) * d] += np.identity(d)
        inverse_covariance[pi_ind * d: (pi_ind + 1) * d, i * d: (i + 1) * d] += - np.identity(d)
        inverse_covariance[i * d: (i + 1) * d, pi_ind * d: (pi_ind + 1) * d] += - np.identity(d)

    inverse_covariance[0:d, 0:d] += np.identity(d)

    # invert precision matrix
    t1 = time.time()
    full_covariance = np.linalg.inv(inverse_covariance)

    #print("inversion of leave covariance took {} seconds".format(time.time() - t1))

    #  delete the columns and the row corresponding to non-terminal nodes (leaves covariance)

    to_delete = []
    for i, n in enumerate(tree.traverse("levelorder")):
        if not n.is_leaf():
            to_delete += [n.index * d, n.index * d + 1]
    len(to_delete)

    # delete rows
    x = np.delete(full_covariance, to_delete, 0)
    # delete columns
    leaves_covariance = np.delete(x, to_delete, 1)

    #print("There are {} leaves".format(len(leaves)))
    #print("The shape of the leaves precision matrix is {}".format(leaves_covariance.shape))

    return leaves_covariance, full_covariance


def marginalize_covariance(covariance, delete_list, d):
    to_delete = []
    for i in range(len(delete_list)):
        to_delete.append([])
    for i, to_delete_idx in enumerate(delete_list):
        for k in to_delete_idx:
            to_delete[i].append(k * d)
            to_delete[i].append(k * d + 1)
    if len(delete_list) == 1:
        x = np.delete(covariance, to_delete, 0)
        marg_covariance = np.delete(x, to_delete, 1)
    else:
        x = np.delete(covariance, to_delete[0], 0)
        marg_covariance = np.delete(x, to_delete[1], 1)
    return marg_covariance


