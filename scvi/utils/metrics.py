from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ks_pvalue(g, glm_samples, rep_samples):
    """
    :param g: number of genes
    :param glm_samples: simulated data
    :param rep_samples: cascVI-generated data
    :return -- > the pandas dataframe with p-values of K-S test for each sample
    """
    np.random.seed(42)

    data = []
    columns = ["Gene", "p-value", "H_0 accepted"]
    N = rep_samples.shape[0]

    for g_ in range(g):
        p_value = 0
        for n in range(N):
            x, y = glm_samples[:, n, g_].flatten(), rep_samples[n, g_, :].flatten()
            ks_test = stats.ks_2samp(x, y)
            p_value += ks_test[1]
        p_value /= N
        accepted = p_value >  0.05
        data.append([g_, p_value, accepted])
    df = pd.DataFrame(data,
                      columns=columns)
    return df

def accuracy_imputation(tree, groundtrtuh, imputed, gene):
    """
    :param tree: Cassiopeia ete3 tree
    :param groundtrtuh: ground truth gene expression value
    :param imputed: imputed gene expression value
    :param gene:
    :return:
    """
    accuracy = 0
    N = 0
    for n in tree.traverse('levelorder'):
        if not n.is_leaf() and groundtrtuh[n.name][gene] == imputed[n.name][gene]:
            accuracy += 1
        N += 1
    return (accuracy / N) * 100

def correlations(data, normalization=None, vis=True):
    """
    :param data: list: list of arrays with imputations (or ground truth) gene expression in this order: internal - imputed - baseline_avg, baseline_scvi - baseline_cascvi
    :param normalization: str: either "rank" or "quantile"
    :param: vis: bool: if True returns violin plots of the density of the correlation coefficients
    :return:
    """
    metrics = []
    columns = ["Method", "Spearman CC", "Pearson CC", "Kendall Tau"]

    # groundtruth and imputations
    internal_X, imputed_X, avg_X, scvi_X, scvi_X_2, cascvi_X, cascvi_X_2, cascvi_X_3  = data

    for i in range(internal_X.shape[1]):

        if normalization == "rank":
            data0 = stats.rankdata(internal_X[:, i])
            data1 = stats.rankdata(imputed_X[:, i])
            data2 = stats.rankdata(avg_X[:, i])
            data3 = stats.rankdata(scvi_X[:, i])
            data4 = stats.rankdata(scvi_X_2[:, i])
            data5 = stats.rankdata(cascvi_X[:, i])
            data6 = stats.rankdata(cascvi_X_2[:, i])
            data7 = stats.rankdata(cascvi_X_3[:, i])
        else:
            data0 = internal_X[:, i]
            data1 = imputed_X[:, i]
            data2 = avg_X[:, i]
            data3 = scvi_X[:, i]
            data4 = scvi_X_2[:, i]
            data5 = cascvi_X[:, i]
            data6 = cascvi_X_2[:, i]
            data7 = cascvi_X_3[:, i]

        metrics.append(["Average", stats.spearmanr(data2, data0)[0], stats.pearsonr(data2, data0)[0],
                        stats.kendalltau(data2, data0)[0]])
        metrics.append(["scVI Baseline 1", stats.spearmanr(data3, data0)[0], stats.pearsonr(data3, data0)[0],
                        stats.kendalltau(data3, data0)[0]])
        metrics.append(["scVI Baseline 2", stats.spearmanr(data4, data0)[0], stats.pearsonr(data4, data0)[0],
                        stats.kendalltau(data4, data0)[0]])
        metrics.append(["cascVI", stats.spearmanr(data1, data0)[0], stats.pearsonr(data1, data0)[0],
                        stats.kendalltau(data1, data0)[0]])
        metrics.append(["cascVI Baseline 1", stats.spearmanr(data5, data0)[0], stats.pearsonr(data5, data0)[0],
                        stats.kendalltau(data5, data0)[0]])
        metrics.append(["cascVI Baseline 2", stats.spearmanr(data6, data0)[0], stats.pearsonr(data6, data0)[0],
                        stats.kendalltau(data6, data0)[0]])
        metrics.append(["cascVI Baseline 3", stats.spearmanr(data7, data0)[0], stats.pearsonr(data7, data0)[0],
                        stats.kendalltau(data7, data0)[0]])

    df = pd.DataFrame(metrics, columns=columns)

    #plots
    if vis:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        sns.violinplot(ax=axes[0], x="Spearman CC", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[0].set_title("Spearman CC")

        sns.violinplot(ax=axes[1], x="Pearson CC", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[1].set_title("Pearson CC")

        sns.violinplot(ax=axes[2], x="Kendall Tau", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[2].set_title("Kendall Tau")

        plt.suptitle("Correlations", fontsize=16)

    return df








