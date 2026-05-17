import numpy as np
import torch
from typing import Union
import pandas as pd


def delta_z(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Normalised photometric redshift error.
    Δz = (z_pred - z_true) / (1 + z_true)
    """
    return (y_pred - y_true) / (1.0 + y_true)


def sigma_nmad(dz: np.ndarray) -> float:
    """
    σ_NMAD = 1.4826 × median( |Δz/(1+z) − b| )
    where b = median(Δz/(1+z)) is the bias.
    """
    b = np.median(dz)
    return float(1.4826 * np.median(np.abs(dz - b)))

def redshift_metrics(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    outlier_threshold: float = 0.15,
) -> dict:
    """
    Compute the full photo-z evaluation metrics.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    finite_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        import warnings
        warnings.warn(
            f"redshift_metrics: {n_bad} non-finite value(s) detected and "
            f"removed before computing metrics.",
            RuntimeWarning, stacklevel=2,
        )
        y_pred = y_pred[finite_mask]
        y_true = y_true[finite_mask]

    dz = delta_z(y_pred, y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nmad = sigma_nmad(dz)

    b = float(np.median(dz))

    return {
        "rmse" : float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae"  : float(np.mean(np.abs(y_pred - y_true))),
        "bias" : b,
        "r2"   : float(1 - ss_res / ss_tot),
        "nmad" : nmad,
        "outlier_pct" : float(np.mean(np.abs(dz) > outlier_threshold) * 100),
        "cat_outlier_pct" : float(np.mean(np.abs(dz) > 0.3) * 100),
        "three_sigma_outlier_pct" : float(np.mean(np.abs(dz - b) > 3 * nmad) * 100),
        "outlier_abs_pct" : float(np.mean(np.abs(y_pred - y_true) > 1.0) * 100),
        "lsst_cat_outlier_pct": float(np.mean(np.abs(y_pred - y_true) > 1.0) * 100),
        "within_0.1z"  : float(np.mean(np.abs(dz) < 0.10)),
        "within_0.05z" : float(np.mean(np.abs(dz) < 0.05)),
    }

def evaluate_by_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
) -> "pd.DataFrame":
    """
    Print and return per-redshift-bin performance breakdown.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    finite_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    rows = []
    print(f"\n{'Bin':<12} {'N':>7} {'RMSE':>8} {'σ_NMAD':>8} {'η (%)':>10} {'η_abs (%)':>11} {'Bias':>8}")
    print("-" * 68)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 10:
            continue
        yt, yp  = y_true[mask], y_pred[mask]
        dz = (yp - yt) / (1.0 + yt)
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        nmad = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
        outlier = float(np.mean(np.abs(dz) > 0.15) * 100)
        out_abs = float(np.mean(np.abs(yp - yt) > 1.0) * 100)
        bias = float(np.median(dz))
        rows.append({"Bin": f"{lo:.1f}–{hi:.1f}", "N": int(mask.sum()),
                     "RMSE": rmse, "σ_NMAD": nmad,
                     "η (%)": outlier, "η_abs (%)": out_abs, "Bias": bias})
        print(
            f"{lo:.1f}–{hi:.1f}{'':>5} {mask.sum():>7,} {rmse:>8.4f} "
            f"{nmad:>8.4f} {outlier:>9.2f}% {out_abs:>10.2f}% {bias:>8.4f}"
        )
    return pd.DataFrame(rows)