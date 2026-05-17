from __future__ import annotations

import copy
import glob as _glob
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, norm
from src.metrics import redshift_metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

GALAXY_PALETTE = {
    "GOLD":        "#F3D79A",
    "CATMINT":     "#C7A7DD",
    "LUPINE":      "#9F7BC3",
    "BLUE_CUE":    "#7B80CD",
    "ASHTON_BLUE": "#5B69BB",
    "PREFECT":     "#4759A8",
    "FARAWAY_SKY": "#334B97",
    "VICTORIA":    "#2E346C",
}

GP = list(GALAXY_PALETTE.values())
GALAXY_CMAP = LinearSegmentedColormap.from_list("galaxy", GP)

SPLIT_COLORS = {
    "train": GALAXY_PALETTE["ASHTON_BLUE"],
    "val":   GALAXY_PALETTE["LUPINE"],
    "test":  GALAXY_PALETTE["GOLD"],
}

MODEL_COLOR_MAP = {
    "Random Forest":          "#F3D79A",
    "knn":                    "#C7A7DD",
    "cnn":                    "#E8A0C0",
    "AlexNet":                "#9F7BC3",
    "VGG16":                  "#7B80CD",
    "ViT":                    "#5B69BB",
    "ResNet101":              "#4759A8",
    "FusionCrossAttn":        "#334B97",
    "LateFusionEfficientNet": "#2E346C",
}

MODEL_PALETTE = list(MODEL_COLOR_MAP.values())

LSST_REQ = {
    "nmad": 0.02,
    "lsst_cat_outlier_pct": 10.0,
    "bias": 0.003,
}

plt.rcParams.update({
    "figure.facecolor": "#F7F5FC",
    "axes.facecolor": "#F7F5FC",
    "axes.edgecolor": "#CFC8E8",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.color": "#DDD9F3",
    "grid.linewidth": 0.7,
    "grid.linestyle": "--",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titlepad": 8,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#D7D0EC",
    "legend.borderpad": 0.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica Neue"],
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "patch.linewidth": 0.6,
})

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return the best available device: MPS (Apple) → CUDA → CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    no_improve: int,
    train_losses: list,
    val_losses: list,
    val_metrics: Optional[dict] = None,
    template: Optional[dict] = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        path          : File path (.pth).
        model         : PyTorch model.
        optimizer     : Current optimizer.
        scheduler     : Current LR scheduler.
        epoch         : Current epoch (0-indexed).
        best_val_loss : Best validation loss seen so far.
        no_improve    : Epochs without improvement (for early stopping).
        train_losses  : List of all training losses up to this epoch.
        val_losses    : List of all validation losses up to this epoch.
        val_metrics   : Optional dict of validation metrics (stored in best ckpt).
        template      : Optional dict of static metadata (model name, batch size, etc).
    """
    payload = copy.deepcopy(template) if template else {}
    payload.update({
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch"               : epoch,
        "best_val_loss"       : best_val_loss,
        "no_improve"          : no_improve,
        "train_losses"        : train_losses.copy(),
        "val_losses"          : val_losses.copy(),
    })
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a checkpoint saved by save_checkpoint().

    Returns the full payload dict so callers can read epoch,
    best_val_loss, train_losses, val_losses etc.
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint ← {path}  (epoch {ckpt.get('epoch', '?')+1})")
    return ckpt

def _safe_log_scale(ax, series: np.ndarray) -> bool:
    """
    Apply a log y-scale only when all finite values are strictly positive.

    Returns True if log scale was applied, False if linear was used instead.
    """
    finite = series[np.isfinite(series)]
    if len(finite) == 0:
        return False
    if finite.min() > 0:
        ax.set_yscale("log")
        return True
    return False

def _load_training_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single training-metrics CSV and normalise column names.

    Adds a 'model' column derived from the filename and ensures a uniform
    'gap' column regardless of whether the file uses 'train_val_gap' or 'gap'.
    """
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "train_val_gap" in df.columns and "gap" not in df.columns:
        df.rename(columns={"train_val_gap": "gap"}, inplace=True)

    if "epoch" in df.columns:
        epochs = df["epoch"].values.astype(float)
        diffs  = np.diff(epochs, prepend=epochs[0] - 1)
        resets = diffs <= 0
        if resets.sum() > 0:
            if "phase" not in df.columns:
                df["phase"] = np.cumsum(resets) + 1
            df["epoch"] = np.arange(1, len(df) + 1)

    stem = Path(path).stem
    for suffix in ("_training_metrics", "_metrics"):
        stem = stem.replace(suffix, "")
    df["model"] = stem.replace("_", " ").title()
    return df

def _load_all_training_csvs(
    models_dir: Union[str, Path] = "models",
) -> Dict[str, pd.DataFrame]:
    """
    Discover and load every *_metrics.csv under `models_dir`.

    Returns:
        dict mapping friendly model name → DataFrame.
        Empty CSVs are silently skipped.
    """
    models_dir = Path(models_dir)
    csvs = sorted(models_dir.glob("*_metrics*.csv"))
    result = {}
    for p in csvs:
        if "lsst_metrics" in p.name:
            continue
        df = _load_training_csv(p)
        if df.empty:
            continue
        name = df["model"].iloc[0]
        result[name] = df
    return result

def _get_colors(n: int) -> list:
    """Return n perceptually distinct colours from MODEL_PALETTE, cycling if needed."""
    return [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(n)]


def _model_colors(names: list) -> list:
    """Return per-model colours from MODEL_COLOR_MAP, falling back to MODEL_PALETTE."""
    return [MODEL_COLOR_MAP.get(name, MODEL_PALETTE[i % len(MODEL_PALETTE)])
            for i, name in enumerate(names)]


def scatter_density(ax, y_true, y_pred, name, cax=None):
    """
    Density scatter into ax with a tight colorbar.
    """
    xy = np.vstack([y_true, y_pred])
    density = gaussian_kde(xy)(xy)
    idx = density.argsort()
    sc = ax.scatter(y_true[idx], y_pred[idx],
                    c=density[idx], s=5, cmap="viridis",
                    alpha=0.7, linewidths=0, rasterized=True)

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = ax.get_figure().colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=20)

    x_lo = max(0.0, y_true.min() - 0.05)
    x_hi = 2.5
    y_lo = max(0.0, y_pred.min() - 0.05)
    y_hi = y_pred.max() + 0.05
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    z_line = np.linspace(x_lo, x_hi, 300)
    ax.plot(z_line, z_line, color="black", linewidth=0.9, zorder=4)
    ax.plot(z_line, z_line + 0.15 * (1 + z_line), color="black", linewidth=0.9,
            linestyle="--", zorder=4)
    ax.plot(z_line, z_line - 0.15 * (1 + z_line), color="black", linewidth=0.9,
            linestyle="--", zorder=4)
    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=20)
    ax.set_ylabel(r"$z_{\rm phot}$", fontsize=20)
    ax.set_title(name, fontsize=18, fontweight="normal", pad=8)

    dz = (y_pred - y_true) / (1 + y_true)
    b = float(np.median(dz))
    nmad = 1.4826 * np.median(np.abs(dz - b))
    eta = np.mean(np.abs(dz) > 0.15) * 100
    ann = (fr"$\sigma_{{\rm NMAD}}={nmad:.4f}$" + "\n"
            fr"$b={b:+.4f}$"                      + "\n"
            fr"$\eta={eta:.1f}\%$")
    ax.text(0.97, 0.04, ann,
            transform=ax.transAxes, fontsize=18,
            va="bottom", ha="right", color="white", linespacing=1.5)


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_path: Optional[str] = None,
    ax: Optional[object] = None,
) -> None:
    """Hexbin scatter of predicted vs. true redshift with a 1:1 reference line."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    lim_lo = max(0, min(y_true.min(), y_pred.min()) - 0.05)
    lim_hi = max(y_true.max(), y_pred.max()) + 0.05

    hb = ax.hexbin(y_true, y_pred, gridsize=90, cmap="plasma",
                   mincnt=1, norm=mcolors.PowerNorm(gamma=0.5),
                   extent=[lim_lo, lim_hi, lim_lo, lim_hi],
                   linewidths=0.1, rasterized=True)
    if standalone:
        cb = plt.colorbar(hb, ax=ax, shrink=0.80, pad=0.02)
        cb.set_label("Count", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="#66FF00", linewidth=1.5, linestyle="-",
            label="1:1", zorder=5)

    zs = np.array([lim_lo, lim_hi])
    for sign in [1, -1]:
        ax.plot(zs, zs + sign * 0.15 * (1 + zs),
                color="#FF6B6B", linewidth=1.1, linestyle="--",
                label=r"$|\Delta z/(1+z)| = 0.15$" if sign == 1 else "",
                zorder=5)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=12)
    ax.set_ylabel(r"$z_{\rm phot}$", fontsize=12)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="#AAAAAA")
    ax.set_aspect("equal", adjustable="box")

    dz = (y_pred - y_true) / (1.0 + y_true)
    _b = float(np.median(dz))
    _nmad = float(1.4826 * np.median(np.abs(dz - _b)))
    _eta = float(np.mean(np.abs(dz) > 0.15) * 100)
    _n = len(y_true)
    ann = (fr"$N={_n:,}$" + "\n"
           fr"$\sigma_{{\rm NMAD}}={_nmad:.4f}$" + "\n"
           fr"Bias$={_b:+.4f}$" + "\n"
           fr"$\eta={_eta:.1f}\%$")
    ax.annotate(ann, xy=(0.97, 0.03), xycoords="axes fraction",
                fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          alpha=0.80, ec="#AAAAAA"))

    if standalone:
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Scatter plot saved → {save_path}")
        plt.show()


def plot_residual_distribution(
    ax, y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> None:
    """Histogram of Δz/(1+z) residuals with a Gaussian fit and outlier threshold lines."""
    dz = (y_pred - y_true) / (1.0 + y_true)
    ax.hist(dz, bins=70, color=GALAXY_PALETTE["ASHTON_BLUE"],
            edgecolor="white", linewidth=0.4, density=True, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
    for sign in [1, -1]:
        ax.axvline(sign * 0.15, color="#E63946", linestyle="--", linewidth=1.1,
                   label=r"$|\Delta z/(1+z)| = 0.15$" if sign == 1 else "")
    mu, std = norm.fit(dz)
    x = np.linspace(-0.6, 0.6, 400)
    ax.plot(x, norm.pdf(x, mu, std),
            color="#F4A261", linewidth=2.0, linestyle="-",
            label=fr"Gaussian fit ($\sigma={std:.3f}$)")
    ax.set_xlabel(r"$\Delta z\,/\,(1 + z_{\rm spec})$")
    ax.set_ylabel("Probability density")
    ax.set_title(r"Residual $\Delta z$ Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, axis="y", alpha=0.35)


def plot_bias_vs_redshift(
    ax, y_true: np.ndarray, y_pred: np.ndarray, name: str, n_bins: int = 20
) -> None:
    """
    Median Δz bias and ±1σ envelope as a function of true redshift.
    Reveals systematic over/under-prediction at specific redshift ranges.
    """
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_centers, median_bias, std_bias = [], [], []

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 5:
            continue
        dz = (y_pred[mask] - y_true[mask]) / (1.0 + y_true[mask])
        bin_centers.append(np.median(y_true[mask]))
        median_bias.append(np.median(dz))
        std_bias.append(np.std(dz))

    bin_centers = np.array(bin_centers)
    median_bias = np.array(median_bias)
    std_bias = np.array(std_bias)

    ax.plot(bin_centers, median_bias, "o-", color=GALAXY_PALETTE["ASHTON_BLUE"])
    ax.fill_between(bin_centers,
                    median_bias - std_bias,
                    median_bias + std_bias,
                    alpha=0.25, color=GALAXY_PALETTE["ASHTON_BLUE"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline( LSST_REQ["bias"], color="red", linestyle="--", linewidth=1, label="LSST req")
    ax.axhline(-LSST_REQ["bias"], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"Median $\Delta z\,/\,(1 + z)$")
    ax.set_title("Bias vs Redshift")
    ax.legend(fontsize=8)


def plot_nmad_vs_redshift(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    n_bins: int = 15,
) -> None:
    """
    σ_NMAD per true-redshift bin — shows where the model's scatter
    degrades. 
    """
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    centers, nmads, counts = [], [], []

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 10:
            continue
        dz = (y_pred[mask] - y_true[mask]) / (1.0 + y_true[mask])
        nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        centers.append(np.median(y_true[mask]))
        nmads.append(nmad)
        counts.append(mask.sum())

    centers = np.array(centers)
    nmads = np.array(nmads)

    ax.plot(centers, nmads, "s-", color=GALAXY_PALETTE["BLUE_CUE"], linewidth=2, markersize=6)
    ax.axhline(LSST_REQ["nmad"], color="red", linestyle="--", linewidth=1.2,
               label="LSST req")
    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"$\sigma_{\rm NMAD}$")
    ax.set_title(r"$\sigma_{\rm NMAD}$ vs Redshift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_outlier_rate_vs_redshift(
    ax, y_true: np.ndarray, y_pred: np.ndarray, name: str, n_bins: int = 20
) -> None:
    """
    Catastrophic outlier rate (|Δz| > 0.15) as a function of true redshift.
    Shows where the model fails — typically at high-z or low-S/N galaxies.
    """
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_centers, outlier_rates = [], []

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 5:
            continue
        dz_bin = np.abs(y_pred[mask] - y_true[mask]) / (1.0 + y_true[mask])
        bin_centers.append(np.median(y_true[mask]))
        outlier_rates.append(np.mean(dz_bin > 0.15) * 100)

    ax.plot(bin_centers, outlier_rates, "o-", color=GALAXY_PALETTE["CATMINT"])
    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"Outlier rate $\eta$ (%)")
    ax.set_title("Outlier Rate vs Redshift")
    ax.grid(True, alpha=0.3)


def plot_binned_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    n_bins: int = 15,
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = (0.0, 2.5),
) -> None:
    """
    Three-panel figure (shared x-axis) showing σ_NMAD, outlier rate, and bias
    as a function of true redshift — the core binned diagnostics in one view.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if z_range is not None:
        z_min, z_max = z_range
        mask = (y_true >= z_min) & (y_true <= z_max)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    else:

     bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    centers = []
    nmads = []
    outlier_rates = []
    lsst_cat_rates = []
    biases = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 10:
            continue
        yt_b, yp_b = y_true[mask], y_pred[mask]
        dz = (yp_b - yt_b) / (1.0 + yt_b)
        nmad_b = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
        centers.append(float(np.median(yt_b)))
        nmads.append(nmad_b)
        outlier_rates.append(float(np.mean(np.abs(dz) > 0.15) * 100))
        lsst_cat_rates.append(float(np.mean(np.abs(yp_b - yt_b) > 1.0) * 100))
        biases.append(float(np.median(dz)))
        counts.append(int(mask.sum()))

    centers = np.array(centers)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(name,
                fontsize=14, fontweight="bold", y=1.01)

    span  = centers[-1] - centers[0] if len(centers) > 1 else 1.0
    bar_w = span / (len(centers) * 2.2)

    ax = axes[0]
    ax.bar(centers, nmads, width=bar_w * 2, color=GALAXY_PALETTE["ASHTON_BLUE"], alpha=0.75)
    ax.plot(centers, nmads, "o-", color=GALAXY_PALETTE["FARAWAY_SKY"],
            linewidth=1.5, markersize=5)
    ax.axhline(LSST_REQ["nmad"], color="red", linestyle="--", linewidth=1.4,
               label="LSST req")
    ax.set_ylabel(r"$\sigma_{\rm NMAD}$", fontsize=11)
    ax.set_title(r"$\sigma_{\rm NMAD}$", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(centers - bar_w / 2, outlier_rates, width=bar_w,
           color="#E63946", alpha=0.75, label=r"$\eta_{0.15}$")
    ax.bar(centers + bar_w / 2, lsst_cat_rates, width=bar_w,
           color="#F4A261", alpha=0.75, label=r"LSST $O_c$")
    ax.axhline(LSST_REQ["lsst_cat_outlier_pct"], color="red", linestyle="--", linewidth=1.4,
               label=r"LSST $O_c$ req")
    ax.set_ylabel("Outlier rate (%)", fontsize=11)
    ax.set_title(r"Outlier Rates", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    colors_b = [GALAXY_PALETTE["ASHTON_BLUE"] if b >= 0 else GALAXY_PALETTE["CATMINT"]
                for b in biases]
    ax.bar(centers, biases, width=bar_w * 2, color=colors_b, alpha=0.75)
    ax.plot(centers, biases, "s-", color=GALAXY_PALETTE["VICTORIA"],
            linewidth=1.5, markersize=5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline( LSST_REQ["bias"], color="red", linestyle="--", linewidth=1.4,
                label="LSST req")
    ax.axhline(-LSST_REQ["bias"], color="red", linestyle="--", linewidth=1.4)
    ax.set_ylabel(r"Bias $\langle\Delta z/(1+z)\rangle$", fontsize=11)
    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=11)
    ax.set_title("Bias", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Binned performance saved → {save_path}")
    plt.show()


def plot_cumulative_dz(
    ax, y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> None:
    """
    Cumulative fraction of galaxies within |Δz| ≤ threshold.
    """
    dz         = np.abs(y_pred - y_true) / (1.0 + y_true)
    thresholds = np.linspace(0, 0.5, 500)
    fractions  = [np.mean(dz <= t) for t in thresholds]

    ax.plot(thresholds, fractions, color=GALAXY_PALETTE["ASHTON_BLUE"], linewidth=2)
    for thresh, color in [(0.05, "red"), (0.15, "orange")]:
        frac = np.mean(dz <= thresh)
        ax.axvline(thresh, color=color, linestyle="--", linewidth=1)
        ax.axhline(frac,   color=color, linestyle=":",  linewidth=1,
                   label=fr"$<{thresh}$: {frac*100:.1f}\%")

    ax.set_xlabel(r"$|\Delta z|\,/\,(1 + z_{\rm spec})$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(r"Cumulative $|\Delta z|$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_nz_distribution(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    n_bins: int = 60,
) -> None:
    """
    Overlaid histograms of the true and predicted redshift distributions.
    """
    bins = np.linspace(
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
        n_bins + 1,
    )
    ax.hist(y_true, bins=bins, alpha=0.55, color=GALAXY_PALETTE["ASHTON_BLUE"], density=True,
            label=r"$z_{\rm spec}$ (true)")
    ax.hist(y_pred, bins=bins, alpha=0.55, color=GALAXY_PALETTE["LUPINE"], density=True,
            label=r"$z_{\rm phot}$ (pred)")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Normalised density")
    ax.set_title("N(z) Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_qq(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> None:
    """
    Quantile–quantile plot of normalised residuals Δz/(1+z).
    """
    dz = (y_pred - y_true) / (1.0 + y_true)
    dz_sorted = np.sort(dz)
    n = len(dz_sorted)
    theoretical = norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))

    ax.scatter(theoretical, dz_sorted, s=2, alpha=0.4, color=GALAXY_PALETTE["BLUE_CUE"],
               rasterized=True)
    lims = [theoretical.min(), theoretical.max()]
    q25, q75 = np.percentile(dz_sorted, [25, 75])
    t25, t75 = norm.ppf(0.25), norm.ppf(0.75)
    slope = (q75 - q25) / (t75 - t25)
    intercept = q25 - slope * t25
    ax.plot(lims, [slope * x + intercept for x in lims],
            "r--", linewidth=1.5, label="Reference line")

    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel(r"Sample quantiles $\Delta z / (1+z)$")
    ax.set_title("Q–Q Plot")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_pit_histogram(
    ax,
    pit_values: np.ndarray,
    name: str,
    n_bins: int = 20,
) -> None:
    """
    Probability Integral Transform (PIT) histogram for calibration.
    """
    ax.hist(pit_values, bins=n_bins, density=True, color=GALAXY_PALETTE["ASHTON_BLUE"],
            edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="Uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_learning_curves(
    df: Union[pd.DataFrame, str, Path],
    name: Optional[str] = None,
    log_scale: bool = True,
    phase1_end: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Train / validation loss curves from a training-metrics CSV.
    """
    if not isinstance(df, pd.DataFrame):
        df = _load_training_csv(df)
    if df.empty:
        print("Empty CSV — nothing to plot.")
        return
    if name is None:
        name = df["model"].iloc[0] if "model" in df.columns else "Model"

    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)

    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"].values, label="Train",
                linewidth=2.0, color=SPLIT_COLORS["train"])
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"].values, label="Validation",
                linewidth=2.0, linestyle="--", color=SPLIT_COLORS["val"])

    if phase1_end and "train_loss" in df.columns and phase1_end < len(df):
        ax.axvline(x=phase1_end, color="#2DC653", linestyle="--",
                   linewidth=1.5, label="Phase 2 start")

    all_loss = np.concatenate([
        df["train_loss"].values if "train_loss" in df.columns else np.array([]),
        df["val_loss"].values if "val_loss"   in df.columns else np.array([]),
    ])
    used_log = log_scale and _safe_log_scale(ax, all_loss)
    ax.set_ylabel("Loss (log scale)" if used_log else "Loss")

    ax.set_xlabel("Epoch")
    ax.set_title(f"Training Loss — {name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600)
    plt.show()


def plot_metrics_vs_epoch(
    df: Union[pd.DataFrame, str, Path],
    name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Multi-panel plot of validation metrics vs epoch from a training CSV.
    """
    if not isinstance(df, pd.DataFrame):
        df = _load_training_csv(df)
    if df.empty:
        print("Empty CSV — nothing to plot.")
        return
    if name is None:
        name = df["model"].iloc[0] if "model" in df.columns else "Model"

    if metrics is None:
        metrics = [
            "val_loss", "val_mae", "val_nmad",
            "val_outlier_pct", "within_0.05z", "val_r2",
        ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print(f"No plottable metrics found for {name}.")
        return

    n = len(metrics)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"{name} — Metric Evolution", fontsize=14, fontweight="bold")

    labels = {
        "val_loss":        "Validation Loss",
        "val_mae":         "MAE",
        "val_rmse":        "RMSE",
        "val_nmad":        r"$\sigma_{\rm NMAD}$",
        "val_outlier_pct": "Outlier %",
        "within_0.05z":    "Within 0.05z",
        "within_0.1z":     "Within 0.10z",
        "val_r2":          r"$R^2$",
        "val_bias":        "Bias",
        "rms_lsst":        "RMS (LSST)",
    }

    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)

    for i, metric in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.plot(epochs, df[metric].values, linewidth=2, color=GALAXY_PALETTE["BLUE_CUE"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(labels.get(metric, metric))
        ax.set_title(labels.get(metric, metric))
        ax.grid(True, alpha=0.3)

        if metric in ("val_loss", "val_mae", "val_rmse", "val_nmad",
                       "val_outlier_pct", "val_bias", "rms_lsst"):
            best_idx = df[metric].abs().idxmin() if metric == "val_bias" else df[metric].idxmin()
            ax.axvline(epochs[best_idx], color="red", linestyle=":", alpha=0.6,
                       label=f"ep {epochs[best_idx]}")
            ax.legend(fontsize=7)
        elif metric in ("within_0.05z", "within_0.1z", "val_r2"):
            best_idx = df[metric].idxmax()
            ax.axvline(epochs[best_idx], color="red", linestyle=":", alpha=0.6,
                       label=f"ep {epochs[best_idx]}")
            ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_lr_schedule(
    df: Union[pd.DataFrame, str, Path],
    name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the learning-rate schedule from a training CSV.
    """
    if not isinstance(df, pd.DataFrame):
        df = _load_training_csv(df)
    if df.empty:
        print("Empty CSV — nothing to plot.")
        return
    if name is None:
        name = df["model"].iloc[0] if "model" in df.columns else "Model"

    if "lr" not in df.columns:
        print(f"No 'lr' column found for {name}.")
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)
    ax.plot(epochs, df["lr"].values, linewidth=2, color=GALAXY_PALETTE["GOLD"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if "phase" in df.columns:
        phases = df["phase"].values
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                ax.axvline(epochs[i], color="#2DC653", linestyle="--",
                           linewidth=1.2, alpha=0.8,
                           label=f"Phase {phases[i]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.show()


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Six-panel single-model evaluation figure.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if z_range is not None:
        z_min, z_max = z_range
        mask = (y_true >= z_min) & (y_true <= z_max)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        width_ratios=[1, 0.05, 1, 0.05],
    )

    ax00  = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])
    ax01  = fig.add_subplot(gs[0, 2])
    ax10  = fig.add_subplot(gs[1, 0])
    ax11  = fig.add_subplot(gs[1, 2])
    ax20  = fig.add_subplot(gs[2, 0])
    ax21  = fig.add_subplot(gs[2, 2])

    scatter_density(ax00, y_true, y_pred, name, cax=cax00)
    plot_residual_distribution(ax01, y_true, y_pred, name)
    plot_bias_vs_redshift(ax10, y_true, y_pred, name)
    plot_nmad_vs_redshift(ax11, y_true, y_pred, name)
    plot_outlier_rate_vs_redshift(ax20, y_true, y_pred, name)
    plot_cumulative_dz(ax21, y_true, y_pred, name)

    fig.suptitle(name,
                 fontsize=15, fontweight="bold", y=1.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Full evaluation saved → {save_path}")
    plt.show()


def full_evaluation_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
    pit_values: Optional[np.ndarray] = None,
) -> None:
    """
    Extended 8-panel evaluation figure (adds N(z) and Q-Q or PIT).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if z_range is not None:
        z_min, z_max = z_range
        mask   = (y_true >= z_min) & (y_true <= z_max)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    fig = plt.figure(figsize=(14, 22))
    gs  = gridspec.GridSpec(
        4, 4,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        width_ratios=[1, 0.05, 1, 0.05],
    )

    ax00  = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])
    ax01  = fig.add_subplot(gs[0, 2])
    ax10  = fig.add_subplot(gs[1, 0])
    ax11  = fig.add_subplot(gs[1, 2])
    ax20  = fig.add_subplot(gs[2, 0])
    ax21  = fig.add_subplot(gs[2, 2])
    ax30  = fig.add_subplot(gs[3, 0])
    ax31  = fig.add_subplot(gs[3, 2])

    scatter_density(ax00, y_true, y_pred, name, cax=cax00)
    plot_residual_distribution(ax01, y_true, y_pred, name)
    plot_bias_vs_redshift(ax10, y_true, y_pred, name)
    plot_nmad_vs_redshift(ax11, y_true, y_pred, name)
    plot_outlier_rate_vs_redshift(ax20, y_true, y_pred, name)
    plot_cumulative_dz(ax21, y_true, y_pred, name)
    plot_nz_distribution(ax30, y_true, y_pred, name)

    if pit_values is not None:
        plot_pit_histogram(ax31, pit_values, name)
    else:
        plot_qq(ax31, y_true, y_pred, name)

    fig.suptitle(name,
                 fontsize=15, fontweight="bold", y=1.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Extended evaluation saved → {save_path}")
    plt.show()


def compare_scatter(predictions, save_path=None, z_range=None):
    """Grid of z_phot vs. z_spec hexbin panels, one per model, with per-panel metrics."""
    names = list(predictions.keys())

    model_labels = {
        "Random Forest": "RandomForest",
        "knn": "kNN",
        "cnn": "CNN",
        "FusionCrossAttn": "Fusion-XA",
        "LateFusionEfficientNet": "PhysLate-EffNet",
    }
    short_names = [model_labels.get(name, name[:12]) for name in names]

    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9 * ncols, 10 * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.15, wspace=0.25, top=0.96)

    last_row = nrows - 1

    for idx, (name, short_name) in enumerate(zip(names, short_names)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        y_true, y_pred = predictions[name]
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if z_range is not None:
            mask = (y_true >= z_range[0]) & (y_true <= z_range[1])
            y_true, y_pred = y_true[mask], y_pred[mask]

        scatter_density(ax, y_true, y_pred, short_name)

        if col != 0:
            ax.set_ylabel("")
        max_idx_in_col = len(names) - 1
        last_row_of_col = divmod(max_idx_in_col, ncols)[0] if col <= (max_idx_in_col % ncols) else last_row - 1
        if row != last_row_of_col and row != last_row:
            ax.set_xlabel("")

        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, color="#DDD9F3")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#CFC8E8")
        ax.spines["bottom"].set_color("#CFC8E8")
        ax.tick_params(labelsize=18)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.title.set_size(22)
        ax.title.set_fontweight("normal")

    for j in range(len(names), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(r"Model Comparison — $z_{\rm phot}$ vs $z_{\rm spec}$",
                 fontsize=20, fontweight="bold", y=0.99)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

def compare_residuals(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Overlaid residual Δz/(1+z) histograms for multiple models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _get_colors(len(predictions))

    for (name, (y_true, y_pred)), color in zip(predictions.items(), colors):
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if z_range is not None:
            z_min, z_max = z_range
            mask = (y_true >= z_min) & (y_true <= z_max)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        dz = (y_pred - y_true) / (1.0 + y_true)
        ax.hist(dz, bins=80, density=True, alpha=0.55,
                color=color, label=name, histtype="stepfilled")

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
    for sign in [1, -1]:
        ax.axvline(sign * 0.15, color="#E63946", linestyle="--", linewidth=1.1,
                   label=r"$|\Delta z/(1+z)| = 0.15$" if sign == 1 else "")

    ax.set_xlabel(r"$\Delta z\,/\,(1 + z_{\rm spec})$")
    ax.set_ylabel("Probability density")
    ax.set_title(r"Normalized Residuals $\Delta z\,/\,(1+z_{\rm spec})$")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_cumulative_dz(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Overlaid cumulative |Δz|/(1+z) curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _model_colors(list(predictions.keys()))
    thresholds = np.linspace(0, 0.5, 500)

    for (name, (y_true, y_pred)), color in zip(predictions.items(), colors):
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if z_range is not None:
            z_min, z_max = z_range
            mask = (y_true >= z_min) & (y_true <= z_max)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        dz = np.abs(y_pred - y_true) / (1.0 + y_true)
        fractions = [np.mean(dz <= t) for t in thresholds]
        ax.plot(thresholds, fractions, linewidth=2, color=color, label=name)

    for thresh, c in [(0.05, GALAXY_PALETTE["GOLD"]), (0.15, GALAXY_PALETTE["CATMINT"])]:
        ax.axvline(thresh, color=c, linestyle="--", linewidth=1.2, alpha=0.85)

    ax.set_xlabel(r"$|\Delta z|\,/\,(1 + z_{\rm spec})$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(r"Cumulative Fraction vs $|\Delta z|\,/\,(1+z_{\rm spec})$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_bias(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 20,
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Overlaid median bias vs redshift curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _get_colors(len(predictions))

    for (name, (y_true, y_pred)), color in zip(predictions.items(), colors):
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if z_range is not None:
            z_min, z_max = z_range
            mask = (y_true >= z_min) & (y_true <= z_max)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        centers, biases = [], []
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.sum() < 5:
                continue
            dz = (y_pred[mask] - y_true[mask]) / (1.0 + y_true[mask])
            centers.append(np.median(y_true[mask]))
            biases.append(np.median(dz))
        ax.plot(centers, biases, "o-", color=color, linewidth=1.8,
                markersize=4, label=name)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.axhline( LSST_REQ["bias"], color="red", linestyle="--", linewidth=1.2, label="LSST req")
    ax.axhline(-LSST_REQ["bias"], color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"Median $\Delta z\,/\,(1 + z)$")
    ax.set_title(r"Median Bias vs Redshift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_nmad_vs_redshift(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 15,
    save_path: Optional[str] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Overlaid σ_NMAD vs redshift curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = _model_colors(list(predictions.keys()))

    for (name, (y_true, y_pred)), color in zip(predictions.items(), colors):
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if z_range is not None:
            z_min, z_max = z_range
            mask = (y_true >= z_min) & (y_true <= z_max)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        centers, nmads = [], []
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.sum() < 10:
                continue
            dz = (y_pred[mask] - y_true[mask]) / (1.0 + y_true[mask])
            nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
            centers.append(np.median(y_true[mask]))
            nmads.append(nmad)
        ax.plot(centers, nmads, "s-", color=color, linewidth=1.8,
                markersize=4, label=name)

    ax.axhline(LSST_REQ["nmad"], color=GALAXY_PALETTE["GOLD"], linestyle="--",
               linewidth=1.5, label="LSST req")

    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"$\sigma_{\rm NMAD}$")
    ax.set_title(r"$\sigma_{\rm NMAD}$ vs Redshift")
    ax.legend(fontsize=8, facecolor="white", edgecolor="#CFC8E8")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def compare_bar(predictions, save_path=None, z_range=None):
    """Grouped bar chart comparing NMAD, bias, outlier rate, and MAE across models."""
    metrics = {}
    for name, (y_true, y_pred) in predictions.items():
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        if z_range:
            mask = (y_t >= z_range[0]) & (y_t <= z_range[1])
            y_t, y_p = y_t[mask], y_p[mask]
        m = redshift_metrics(y_p, y_t)
        m["bias"] = abs(m["bias"])
        metrics[name] = m

    models = list(metrics)
    model_labels = {
        "Random Forest": "RandomForest",
        "knn": "kNN",
        "cnn": "CNN",
        "FusionCrossAttn": "Fusion-XA",
        "LateFusionEfficientNet": "PhysLate-EffNet",
    }
    short_labels = [model_labels.get(m, m[:12]) for m in models]

    x = np.arange(len(models)) * 1.3

    _PANELS = [
        ("bias", r"$|$Bias$|$", LSST_REQ["bias"]),
        ("nmad", r"$\sigma_\mathrm{NMAD}$", LSST_REQ["nmad"]),
        ("lsst_cat_outlier_pct", "Catastrophic Outlier Rate (%)", LSST_REQ["lsst_cat_outlier_pct"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(60, 20))

    for i, (ax, (key, ylabel, req)) in enumerate(zip(axes, _PANELS)):
        vals = [metrics[m][key] for m in models]

        bar_color = "#7B7DC8"
        best_idx = np.argmin(vals)
        bars = ax.bar(x, vals, color=bar_color, alpha=0.8, width=0.9)
        bars[best_idx].set_color(GALAXY_PALETTE["GOLD"])

        ax.axhline(req, color="red", ls=":", lw=4, label=f"LSST req {req}", zorder=1)
        ax.legend(loc="upper right", fontsize=28)

        for j, (bar, v) in enumerate(zip(bars, vals)):
            bar_height = bar.get_height()

            if abs(bar_height - req) < 0.001:
                label_y = req * 1.08
            elif bar_height > req:
                label_y = bar_height + 0.0009
            else:
                label_y = bar_height + 0.0003

            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                   f"{v:.4f}", ha="center", va="bottom", fontsize=24)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=28)
        ax.set_ylabel(ylabel, fontsize=36)
        ax.tick_params(axis='y', labelsize=28)
        ax.set_xlim(-1, x[-1] + 0.8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Performance Comparison", fontsize=44, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_model_comparison_bar(metrics_dict, metrics_to_plot=None, save_path=None):
    """Bar chart for a custom selection of metrics across models, from a pre-computed dict."""
    if metrics_to_plot is None:
        metrics_to_plot = ["nmad", "bias", "lsst_cat_outlier_pct", "mae"]

    _METRIC_LABELS = {
        "nmad":               r"$\sigma_{\rm NMAD}$",
        "val_nmad":           r"$\sigma_{\rm NMAD}$",
        "bias":               r"Median Bias",
        "val_bias":           r"Median Bias",
        "lsst_cat_outlier_pct": r"$O_c$ (%)",
        "val_outlier_pct":    r"Outlier Rate $\eta$ (%)",
        "outlier_pct":        r"Outlier Rate $\eta$ (%)",
        "mae":                r"MAE",
        "val_mae":            r"MAE",
        "rmse":               r"RMSE",
        "val_rmse":           r"RMSE",
    }

    models = list(metrics_dict)
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.5*len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [metrics_dict[m].get(metric, np.nan) for m in models]

        bar_color = "#7B7DC8"
        best_idx = np.nanargmin(vals)
        bars = ax.bar(x, vals, color=bar_color, alpha=0.8, width=0.7)
        bars[best_idx].set_color(GALAXY_PALETTE["GOLD"])

        _lsst_label = None
        if metric in ("nmad", "val_nmad"):
            ax.axhline(LSST_REQ["nmad"], color="red", ls=":", lw=2,
                       label=f"LSST req {LSST_REQ['nmad']}")
            _lsst_label = True
        elif metric in ("lsst_cat_outlier_pct",):
            ax.axhline(LSST_REQ["lsst_cat_outlier_pct"], color="red", ls=":", lw=2,
                       label=f"LSST req {LSST_REQ['lsst_cat_outlier_pct']}")
            _lsst_label = True
        elif metric in ("bias", "val_bias"):
            req = LSST_REQ["bias"]
            ax.axhline(req, color="red", ls=":", lw=2, label=f"LSST req ±{req}")
            _lsst_label = True

        if _lsst_label:
            ax.legend(loc="upper right", fontsize=9)

        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0009,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Performance Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_learning_curves(
    models_dir: Union[str, Path] = "models",
    metric: str = "val_loss",
    log_scale: bool = True,
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare one metric curve across all models.

    Loss metrics (val_loss, train_loss, or any column ending in '_loss') are shown
    in separate per-model subplots with independent y-scales, because models use
    different loss functions (SmoothL1, custom RedshiftLoss, probabilistic NLL)
    whose absolute values are not directly comparable.
    """
    all_dfs = _load_all_training_csvs(models_dir)
    if not all_dfs:
        print(f"No training CSVs found in {models_dir}")
        return

    is_loss = (metric in {"val_loss", "train_loss", "gap"} or metric.endswith("_loss"))

    if is_loss and not normalize:
        models_with_metric = {n: df for n, df in all_dfs.items()
                              if "val_loss" in df.columns or "train_loss" in df.columns}
        n = len(models_with_metric)
        ncols  = min(3, n)
        nrows  = (n + ncols - 1) // ncols
        colors = _get_colors(n)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4.0 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            "Learning Curves — Train vs. Val Loss",
            fontsize=12, fontweight="bold",
        )

        for idx, ((name, df), color) in enumerate(zip(models_with_metric.items(), colors)):
            r, c   = divmod(idx, ncols)
            ax = axes[r][c]
            epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)

            all_vals: list[np.ndarray] = []
            if "train_loss" in df.columns:
                tv = df["train_loss"].values.astype(float)
                ax.plot(epochs, tv, linewidth=1.6, color=color,
                        linestyle="--", alpha=0.75, label="Train")
                all_vals.append(tv[np.isfinite(tv)])
            if "val_loss" in df.columns:
                vv = df["val_loss"].values.astype(float)
                ax.plot(epochs, vv, linewidth=1.8, color=color,
                        linestyle="-", label="Val")
                all_vals.append(vv[np.isfinite(vv)])

            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            combined = np.concatenate(all_vals) if all_vals else np.array([1.0])
            used_log = log_scale and _safe_log_scale(ax, combined)
            ax.set_ylabel("Loss" + (" (log)" if used_log else ""))
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].set_visible(False)

    else:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = _get_colors(len(all_dfs))

        for (name, df), color in zip(all_dfs.items(), colors):
            if metric not in df.columns:
                continue
            epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)
            vals = df[metric].values.astype(float)
            if normalize:
                v_min, v_max = vals.min(), vals.max()
                vals = (vals - v_min) / (v_max - v_min + 1e-9)
            ax.plot(epochs, vals, linewidth=1.8, color=color, label=name)

        if log_scale and not normalize:
            all_plotted: list[np.ndarray] = []
            for name, df in all_dfs.items():
                if metric in df.columns:
                    all_plotted.append(df[metric].values.astype(float))
            if all_plotted:
                _safe_log_scale(ax, np.concatenate(all_plotted))

        ylabel = (f"{metric} (normalised)" if normalize else metric).replace("_", " ")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Learning Curve Comparison — {metric.replace('_', ' ')}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_models_from_csv(
    models_dir: Union[str, Path] = "models",
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Build a model-comparison bar chart directly from training CSVs,
    using the best-epoch value for each metric.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["val_nmad", "val_outlier_pct", "val_bias", "val_mae"]

    all_dfs = _load_all_training_csvs(models_dir)
    if not all_dfs:
        print(f"No training CSVs found in {models_dir}")
        return

    lower_is_better = {"val_nmad", "val_outlier_pct", "val_bias", "val_mae",
                       "val_rmse", "val_loss", "lsst_cat_outlier_pct", "nmad",
                       "outlier_pct", "bias", "mae", "rmse"}

    best_metrics: Dict[str, dict] = {}
    for name, df in all_dfs.items():
        row = {}
        for m in metrics_to_plot:
            if m not in df.columns:
                continue
            vals = df[m].dropna()
            if vals.empty:
                continue
            row[m] = float(vals.min() if m in lower_is_better else vals.max())
        best_metrics[name] = row

    plot_model_comparison_bar(best_metrics, metrics_to_plot=metrics_to_plot,
                              save_path=save_path)


# ---------------------------------------------------------------------------
# Image display helpers (used by GalaxyLDM notebook and ldm.py)
# ---------------------------------------------------------------------------

_BAND_MEAN = torch.tensor(
    [0.07781, 0.15729, 0.23103, 0.30393, 0.36941], dtype=torch.float32
).view(-1, 1, 1)

_BAND_STD = torch.tensor(
    [0.82795, 1.44230, 1.78690, 2.59873, 3.14001], dtype=torch.float32
).view(-1, 1, 1)


def denorm(x: torch.Tensor,
           mean: torch.Tensor = _BAND_MEAN,
           std: torch.Tensor = _BAND_STD) -> torch.Tensor:
    """Reverse per-band normalisation. x: (C,H,W) or (B,C,H,W)."""
    m = mean.to(x.device)
    s = std.to(x.device)
    return x * s + m


def to_rgb(x: torch.Tensor, bands: tuple = (2, 1, 0)) -> np.ndarray:
    """Convert a normalised 5-band image tensor to a display RGB array.

    Args:
        x: (C, H, W) normalised tensor.
        bands: channel indices mapped to (R, G, B). Default uses i, r, g.

    Returns:
        (H, W, 3) float32 array clipped to [0, 1].
    """
    x = denorm(x.cpu())
    rgb = x[[bands[0], bands[1], bands[2]]].numpy()
    lo, hi = np.percentile(rgb, [0.5, 99.5])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-8), 0, 1)
    return rgb.transpose(1, 2, 0)


def show_galaxy_grid(images: list, titles: list, ncols: int = 8,
                     bg: str = "black", figsize=None):
    """Display a list of RGB images in a grid.

    Args:
        images: list of (H, W, 3) arrays.
        titles: per-image title strings.
        ncols: number of columns.
        bg: background colour ("black" or "white").
        figsize: optional (width, height) tuple.

    Returns:
        matplotlib Figure.
    """
    nrows = math.ceil(len(images) / ncols)
    if figsize is None:
        figsize = (ncols * 1.8, nrows * 1.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=bg, constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, origin="lower")
        ax.set_title(title, color="white" if bg == "black" else "black",
                     fontsize=7, pad=2)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    return fig
