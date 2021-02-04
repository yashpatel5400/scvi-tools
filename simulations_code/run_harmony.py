#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that runs Harmony on simulated data.

Created on 2020/02/03
@author romain_lopez
"""

import os
import click
import numpy as np
from logzero import logger

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata

import harmonypy as hm
from sklearn.decomposition import PCA


@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input gene expression directory')
@click.option("--output-suffix", type=click.STRING, default="harmony", help="output saved model")
def main(input_dir, output_suffix):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    #load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    logger.info("Running Harmony")

    # path management
    output_dir = input_dir + output_suffix + '/'

    if not os.path.isdir(output_dir):
        logger.info("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        logger.info(F"Found directory at:{output_dir}")
    
    dat1 = sc_adata.X.A
    dat2 = st_adata.X.A
    data_mat = PCA(n_components=10).fit_transform(np.log(1 + np.vstack([dat1, dat2])))
    meta_data = pd.DataFrame(data= dat1.shape[0] * ["b1"] + dat2.shape[0] * ["b2"], columns=["batch"])
    ho = hm.run_harmony(data_mat, meta_data, ["batch"])
    embedding1, embedding2 = ho.Z_corr.T[:dat1.shape[0]], ho.Z_corr.T[dat1.shape[0]:]
    np.savez_compressed(output_dir + 'embedding.npz', embedding_sc=embedding1, embedding_st=embedding2)

if __name__ == '__main__':
    main()