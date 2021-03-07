import numpy as np
from torch.nn.functional import softplus

from scvi.data import setup_anndata, synthetic_iid
from scvi.external import SCTransform


def test_sctransform(save_path):
    adata = synthetic_iid(run_setup_anndata=False)
    adata.obs["log_total_count"] = np.log10(adata.X.sum(1) + 1)
    setup_anndata(adata, continuous_covariate_keys=["log_total_count"])
    model = SCTransform(adata)
    model.train(1, train_size=0.5)
    model.train(30, train_size=0.9)
    model.history["train_loss_epoch"].plot()

    weight = model.module.model.linear.weight.detach().cpu().numpy()
    gene_mean = adata.X.mean(0).A

    theta = softplus(model.module.model.theta_unsoft.detach().cpu()).numpy()
    # theta = np.exp(model.module.model.theta_log.detach().cpu().numpy())
    gene_mean = adata.X.mean(0).A
    post_params = dict(model.module.guide.named_parameters())
    weight = post_params["linear.weight_unconstrained"].detach().cpu().numpy()
    bias = post_params["linear.bias_unconstrained"].detach().cpu().numpy()

    import matplotlib.pyplot as plt

    plt.scatter(gene_mean, bias, s=0.5)
    plt.xscale("log")

    plt.scatter(gene_mean, weight.ravel(), s=0.2)
    plt.xscale("log")

    plt.scatter(gene_mean, np.log10(theta), s=0.5)
    plt.xscale("log")
