import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_tau_scores(tau1, tau0, cor_idx, n, title="ROC: QUE vs Naive Spectral"):
    """
    Plot ROC curves for tau1 (QUE) and tau0 (naive spectral).

    Args:
        tau1: 1D tensor/array of QUE scores, length n
        tau0: 1D tensor/array of spectral scores, length n
        cor_idx: 1D tensor/array of integer indices of outliers (len = n_outliers)
        n: total number of samples
        title: plot title
    """
    # Ensure numpy arrays
    if isinstance(tau1, torch.Tensor): tau1 = tau1.detach().cpu().numpy()
    if isinstance(tau0, torch.Tensor): tau0 = tau0.detach().cpu().numpy()
    if isinstance(cor_idx, torch.Tensor): cor_idx = cor_idx.detach().cpu().numpy()

    # Ground truth: 1 for outlier, 0 for inlier
    y_true = np.zeros(n, dtype=int)
    y_true[cor_idx] = 1

    # ROC for tau1 (QUE)
    fpr1, tpr1, _ = roc_curve(y_true, tau1)
    auc1 = auc(fpr1, tpr1)

    # ROC for tau0 (naive spectral)
    fpr0, tpr0, _ = roc_curve(y_true, tau0)
    auc0 = auc(fpr0, tpr0)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr1, tpr1, label=f"QUE (AUC = {auc1:.3f})")
    plt.plot(fpr0, tpr0, label=f"Naive Spectral (AUC = {auc0:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
