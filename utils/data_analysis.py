"""
Utilities functions for data and analysis
"""
import numpy as np
import scipy

import os
import sys
import time
from pathlib import Path
from colorama import Fore, Style

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def compute_binned_mean_sd(x, v, min_x=0.0, max_x=10, bins=100):
    """compute average and sd after bin the data
        x : values used to bin the data
        v : values to compute mean and sd
    """

    x = x.flatten()
    v = v.flatten()

    indices = np.where(np.logical_and(x >= min_x, x < max_x))

    # compute moving mean and sd
    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(np.log(x[indices]), v[indices], statistic='mean', bins=bins, range=np.log([min_x, max_x]))
    bin_sds, bin_edges, binnumber = scipy.stats.binned_statistic(np.log(x[indices]), v[indices], statistic='std', bins=bins, range=np.log([min_x, max_x]))

    y = bin_means
    nans, x = nan_helper(y)
    bin_means[nans]= np.interp(x(nans), x(~nans), y[~nans])

    y = bin_sds
    nans, x = nan_helper(y)
    bin_sds[nans]= np.interp(x(nans), x(~nans), y[~nans])

    bin_edges = np.exp(bin_edges)

    return bin_means, bin_sds, bin_edges, binnumber

# -------------------------------------------------------------------------------------------------

def compute_auc(bin_means, bin_edges):
    """Compute AUC for bin means

    Args:
        bin_means (N): N values of binned means
        bin_edges (N+1): bin edges

    Returns:
        auc: auc computed with trapz integral
    """
    auc = np.trapz(bin_means, 0.5 * (bin_edges[0:-1:]+bin_edges[1::]))
    return auc

# -------------------------------------------------------------------------------------------------
def plot_with_CI(x, v, min_x, max_x, bin_means, bin_sds, bin_edges, xlabel='snr', ylabel='ssim', ylim=[0, 1]):
    """Create the plot with data and CI

    Args:
        x (N): input values
        v (N): response values, y-axis
        min_x (float): min x to plot
        max_x (float): max x to plot
        bin_means (array): computed bin means
        bin_sds (array): computed bin sds
    """

    x = x.flatten()
    v = v.flatten()
    assert x.shape[0] == v.shape[0]

    indices = np.where(np.logical_and(x >= min_x, x < max_x))
    bin_c = 0.5 * (bin_edges[0:-1:]+bin_edges[1::])

    auc = np.trapz(bin_means, bin_c)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x[indices], v[indices], marker='.', c='b')
    ax.plot(bin_c, bin_means, 'r-')
    ax.fill_between(bin_c, (bin_means-1.65*bin_sds), (bin_means+1.65*bin_sds), color='r', alpha=.1)
    ax.set_xticks(np.arange(0.0, max_x+0.2, 0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True)
    ax.set_title(f"auc = {auc:.4f}")
    ax.set_xscale('log')
    return fig, auc

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":

    x = np.random.uniform(0.01, 12.0, 2048)
    y = np.sin(np.arange(2048)) + np.random.uniform(0.01, 2.0, 2048)

    bin_means, bin_sds, bin_edges, binnumber = compute_binned_mean_sd(x, y, min_x=0.01, max_x=10, bins=100)
    auc = compute_auc(bin_means, bin_edges)
    fig, auc = plot_with_CI(x, y, min_x=2.0, max_x=10.0, bin_means=bin_means, bin_sds=bin_sds, bin_edges=bin_edges, xlabel='snr', ylabel='ssim', ylim=[-3, 3])
    fig.savefig(os.path.join('/export/Lab-Xue/projects/mri/results', 'CI.png'), dpi=300)