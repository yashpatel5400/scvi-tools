import time
import numpy as np
from ete3 import Tree


def precision_matrix(tree, d, branch_length):
    """
    :param tree_name: path of the ete3 tree file
    :param d: dimension of latent space
    :param: branch_length: constant branch length along the tree, or dict of branch lengths
    :return: the covariance matrix of the gaussian vector induced by the tree,
     after inversion and post processing of the constructed precision matrix
    """

    # load tree
    if type(tree) == str:
        suffix = tree.split('.')[-1]
        if suffix == "txt":
            with open(tree, "r") as myfile:
                tree_string = myfile.readlines()
                tree = Tree(tree_string[0], 1)
        else:
            tree = Tree(tree, 1)

    # introduce an index for all the nodes
    parents = {}
    N = 0
    for idx, node in enumerate(tree.traverse("levelorder")):
        N += 1
        # set node index
        node.add_features(index=idx)

    # ancestor indexing + branch length dict
    dist = {}
    for n in tree.traverse("levelorder"):
        if not n.is_root():
            ancestor = n.up.index
            parents[n.index] = ancestor
            if type(branch_length) == dict:
                dist[n.up.index] = n.up.dist

    # Intitalize precision matrix
    inverse_covariance = np.zeros((N * d, N * d))

    # the branch length is either constant along the tree, or
    if type(branch_length) != dict:
        t = 1 / branch_length
        for i in parents:
            pi_ind = parents[i]
            inverse_covariance[i * d: (i + 1) * d, i * d: (i + 1) * d] += np.identity(d) * t
            inverse_covariance[pi_ind * d: (pi_ind + 1) * d, pi_ind * d: (pi_ind + 1) * d] += np.identity(d) * t
            inverse_covariance[pi_ind * d: (pi_ind + 1) * d, i * d: (i + 1) * d] += - np.identity(d) * t
            inverse_covariance[i * d: (i + 1) * d, pi_ind * d: (pi_ind + 1) * d] += - np.identity(d) * t

        inverse_covariance[0:d, 0:d] += np.identity(d)
    else:
        for i in parents:
            pi_ind = parents[i]
            t = 1 / branch_length[str(pi_ind)]
            inverse_covariance[i * d: (i + 1) * d, i * d: (i + 1) * d] += np.identity(d) * t
            inverse_covariance[pi_ind * d: (pi_ind + 1) * d, pi_ind * d: (pi_ind + 1) * d] += np.identity(d) * t
            inverse_covariance[pi_ind * d: (pi_ind + 1) * d, i * d: (i + 1) * d] += - np.identity(d) * t
            inverse_covariance[i * d: (i + 1) * d, pi_ind * d: (pi_ind + 1) * d] += - np.identity(d) * t

        inverse_covariance[0:d, 0:d] += np.identity(d)


    # invert precision matrix
    full_covariance = np.linalg.inv(inverse_covariance)

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

    M = marginalize_covariance(full_covariance, to_delete, d)

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


