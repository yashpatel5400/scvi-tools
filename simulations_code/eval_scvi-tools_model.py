#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that evaluates model on simulated data.

Created on 2020/02/03
@author romain_lopez
"""

import os
import click
import numpy as np
from logzero import logger

from utils import get_mean_normal, find_location_index_cell_type, metrics_vector
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata

import scvi
from scvi.external import DestVI, SpatialStereoscope


PCA_path = "/home/ubuntu/simulation_LN/grtruth_PCA.npz"


@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input data directory')
@click.option('--model-subdir', type=click.STRING, default="out/", help='input model subdirectory')
@click.option('--model-string', type=click.STRING, default="description", help='input model description')
def main(input_dir, model_subdir, model_string):
    # Directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    # Load data
    grtruth_PCA = np.load(PCA_path)
    mean_, components_ = grtruth_PCA["mean_"], grtruth_PCA["components_"]

    C = components_.shape[0]
    D = components_.shape[1]

    # sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    # Create groundtruth
    logger.info("simulate cell-type specific gene expression for abundant cell types in abundant spots (used for imputation)")
    threshold_gt = 0.4
    spot_selection = np.where(st_adata.obsm["cell_type"].max(1) > threshold_gt)[0]
    s_location = st_adata.obsm["locations"][spot_selection]
    s_ct = st_adata.obsm["cell_type"][spot_selection, :].argmax(1)
    s_gamma = st_adata.obsm["gamma"][spot_selection]
    # get normal means
    s_groundtruth = get_mean_normal(s_ct[:, None], s_gamma[:, None], mean_, components_)[:, 0, :]
    s_groundtruth[s_groundtruth < 0] = 0
    s_groundtruth = np.expm1(s_groundtruth)
    s_groundtruth = s_groundtruth / np.sum(s_groundtruth, axis=1)[:, np.newaxis]

    if model_string == "DestVI" or "Stereoscope" in model_string:
        # first load the model
        if model_string == "DestVI":
            spatial_model = DestVI.load(input_dir+model_subdir, st_adata)
            nb_sub_ct=1
        else:
            spatial_model = SpatialStereoscope.load(input_dir+model_subdir, st_adata)
            index = int(model_string[-1])
            nb_sub_ct = st_adata.uns["target_list"][index]

        # second get the proportion estimates and get scores
        proportions = spatial_model.get_proportions(dataset=st_adata).values
        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]
        prop_score = metrics_vector(st_adata.obsm["cell_type"], agg_prop_estimates)

        # third impute at required locations
        all_res = []
        # for each cell type, query the model at certain locations and compare to groundtruth
        # create a global flush for comparaison across cell types
        imputed_expression = np.zeros_like(s_groundtruth)
        for ct in range(C):
            indices, _ = find_location_index_cell_type(st_adata.obsm["locations"], ct, 
                                                s_location, s_ct)
            n_location = indices.shape[0]
            ind_x = indices[:, np.newaxis].astype(np.long)
            x = st_adata.X[ind_x[:, 0]].A
            if nb_sub_ct == 1:
                y = ct * np.ones(shape=(n_location, 1), dtype=np.long)
                expression = spatial_model.get_scale_for_ct(x, ind_x, y)
            else:
                # hierarchical clusters in Stereoscope
                partial_cell_type = proportions[ind_x[:, 0], nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct] 
                partial_cell_type /= np.sum(partial_cell_type, axis=1)[:, np.newaxis] # shape (cells, nb_sub_ct)
                expression = np.zeros(shape=(ind_x.shape[0], x.shape[-1]))
                for t in range(nb_sub_ct):
                    y = t * np.ones(shape=(n_location, 1), dtype=np.long) + nb_sub_ct * ct
                    expression += partial_cell_type[:, [t]] * spatial_model.get_scale_for_ct(x, ind_x, y)

            normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # get local scores
            indices_gt = np.where(s_ct == ct)[0]
            # potentially filter genes for local scores only
            gene_list = np.unique(np.hstack([np.where(components_[ct, i] != 0)[0] for i in range(D)]))
            res = metrics_vector(s_groundtruth[indices_gt, :], normalized_expression, scaling=2e5, feature_shortlist=gene_list)
            all_res.append(pd.Series(res))
            # flush to global
            imputed_expression[indices_gt] = normalized_expression
        all_res.append(pd.Series(metrics_vector(s_groundtruth, imputed_expression, scaling=2e5)))

        df = pd.concat(all_res, axis=1)
        df = pd.concat([df, pd.Series(prop_score)], axis=1)
        df.columns = ["ct" + str(i) for i in range(5)]+["allct", "proportions"]    
        print(df)



if __name__ == '__main__':
    main()