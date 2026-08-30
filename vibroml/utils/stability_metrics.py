"""Stability metrics for MLIP-driven phonon screening.

This module is the **single source of truth** for the dynamical stability
classification criterion and the continuous instability descriptor
(:math:`I_\\mathrm{curv}`) used in VibroML.

The criterion is a three-tier classification based solely on the
**soft-mode fraction** (``soft_frac``), defined as the fraction of
phonon modes with imaginary frequency on a uniform q-point mesh::

    Stable         soft_frac <  0.02   (2%)
    Quasi-Stable   0.02 <= soft_frac < 0.08   (8%)
    Unstable       soft_frac >= 0.08

The soft-mode fraction is **dimensionless** and therefore mass-independent,
making the criterion transferable across systems with different atomic masses.

The continuous descriptor :math:`I_\\mathrm{curv}` is the normalised mean
negative curvature of the potential energy surface::

    I_curv = (1/M) * sum_i max(0, -omega_i^2)     [THz^2]

where M is the total number of vibrational modes on the uniform mesh
(excluding the 3 acoustic modes at Gamma) and the sum runs over all
negative-frequency modes.  Equivalently::

    I_curv = Tr(D_-) / (3N)

where ``D_-`` is the negative-curvature part of the mass-weighted dynamical
matrix and N is the number of atoms in the primitive cell.

The characteristic instability frequency is::

    Omega_inst = sqrt(I_curv)     [THz]

Only the 3 acoustic modes at the Gamma point are excluded (by q-point/mode
index).  **No frequency threshold is applied** — the only boundary is
:math:`\\lambda = 0` (i.e. :math:`\\omega = 0`).
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

# ============================================================================
# THRESHOLDS — single source of truth
# ============================================================================
# These constants are imported by tests/test_stability_metrics.py which
# enforces that the LaTeX table captions use the same values (AC2).
# Do NOT change these without updating the manuscript table captions.

SOFT_FRAC_STABLE_MAX = 0.02    # < 2%  → Stable
SOFT_FRAC_UNSTABLE_MIN = 0.08  # >= 8% → Unstable

DEFAULT_Q_MESH: Tuple[int, int, int] = (6, 6, 6)


# ============================================================================
# Classification
# ============================================================================

def classify_stability(soft_frac: float, stable_max: float = None, unstable_min: float = None) -> str:
    """Classify a structure into Stable / Quasi-Stable / Unstable.

    Parameters
    ----------
    soft_frac : float
        Fraction of imaginary phonon modes on a uniform q-point mesh
        (3 Gamma acoustic modes excluded).
    stable_max : float, optional
        Upper bound for Stable classification. Defaults to
        ``SOFT_FRAC_STABLE_MAX`` (0.02).
    unstable_min : float, optional
        Lower bound for Unstable classification. Defaults to
        ``SOFT_FRAC_UNSTABLE_MIN`` (0.08).

    Returns
    -------
    str
        ``"Stable"``, ``"Quasi-Stable"``, or ``"Unstable"``.
    """
    s_max = stable_max if stable_max is not None else SOFT_FRAC_STABLE_MAX
    u_min = unstable_min if unstable_min is not None else SOFT_FRAC_UNSTABLE_MIN
    if soft_frac >= u_min:
        return "Unstable"
    elif soft_frac >= s_max:
        return "Quasi-Stable"
    else:
        return "Stable"


# ============================================================================
# Core metric computation
# ============================================================================

def compute_soft_fraction(
    freqs_THz: np.ndarray,
    gamma_indices: Optional[np.ndarray] = None,
) -> float:
    """Compute the soft-mode fraction.

    Parameters
    ----------
    freqs_THz : np.ndarray, shape (N_q, N_bands)
        Phonon frequencies in THz on a uniform q-point mesh.
    gamma_indices : np.ndarray, optional
        Indices of Gamma q-points in the mesh (rows to exclude the 3
        acoustic modes from).

    Returns
    -------
    float
        Fraction of modes with frequency < 0, excluding the 3 acoustic
        modes at Gamma.
    """
    exclude_mask = _build_exclusion_mask(freqs_THz, gamma_indices)
    kept = freqs_THz[~exclude_mask]
    if len(kept) == 0:
        return 0.0
    n_imag = int((kept < 0).sum())
    return float(n_imag) / len(kept)


def compute_i_curv(
    freqs_THz: np.ndarray,
    gamma_indices: Optional[np.ndarray] = None,
    p: float = 1.0,
) -> Dict:
    """Compute the instability metric I_p and derived quantities.

    For the default ``p=1`` this gives :math:`I_\\mathrm{curv}`, the
    normalised mean negative curvature::

        I_p = [ (1/M) * sum_i max(0, -omega_i^2)^p ]^(1/p)

    where M is the total mode count (Gamma acoustic excluded) and the
    sum runs over negative-frequency modes only.

    Parameters
    ----------
    freqs_THz : np.ndarray, shape (N_q, N_bands)
        Phonon frequencies in THz on a uniform q-point mesh.
    gamma_indices : np.ndarray, optional
        Indices of Gamma q-points (to exclude 3 acoustic modes).
    p : float, default 1.0
        Exponent in the generalised mean.  ``p=1`` gives the standard
        :math:`I_\\mathrm{curv}`.

    Returns
    -------
    dict with keys:
        ``i_p``         : float  — I_p in THz^(2)  (for p=1, this is I_curv)
        ``omega_p``     : float  — sqrt(I_p) in THz (characteristic frequency)
        ``i_curv``      : float  — alias for i_p when p=1, else None
        ``omega_inst``  : float  — alias for omega_p when p=1, else None
        ``omega_min``   : float  — most negative frequency (from ALL modes)
        ``soft_frac``   : float  — soft-mode fraction (Gamma acoustic excluded)
        ``n_modes``     : int    — number of modes after exclusion
        ``n_imag``      : int    — number of imaginary modes after exclusion
        ``n_excluded``  : int    — number of excluded modes
        ``p``           : float  — the exponent used
    """
    freqs_flat = freqs_THz.ravel()
    omega_min = float(min(freqs_flat.min(), 0.0))

    exclude_mask = _build_exclusion_mask(freqs_THz, gamma_indices)
    kept = freqs_THz[~exclude_mask]
    n_modes = len(kept)
    n_excluded = int(exclude_mask.sum())

    if n_modes == 0:
        return {
            "i_p": 0.0,
            "omega_p": 0.0,
            "i_curv": 0.0 if p == 1.0 else None,
            "omega_inst": 0.0 if p == 1.0 else None,
            "omega_min": omega_min,
            "soft_frac": 0.0,
            "n_modes": 0,
            "n_imag": 0,
            "n_excluded": n_excluded,
            "p": p,
        }

    n_imag = int((kept < 0).sum())
    soft_frac = float(n_imag) / n_modes

    # I_p computation
    neg_curvature = np.where(kept < 0, kept ** 2, 0.0)  # |omega|^2 for negative modes
    if p == 1.0:
        i_p = float(neg_curvature.mean())
    else:
        # Generalised: I_p = [ (1/M) * sum |omega|^(2p) ]^(1/p)
        neg_vals = np.abs(kept[kept < 0])
        if len(neg_vals) == 0:
            i_p = 0.0
        else:
            i_p = float(np.mean(neg_vals ** (2 * p)) ** (1.0 / p))

    omega_p = float(np.sqrt(i_p)) if i_p > 0 else 0.0

    return {
        "i_p": i_p,
        "omega_p": omega_p,
        "i_curv": i_p if p == 1.0 else None,
        "omega_inst": omega_p if p == 1.0 else None,
        "omega_min": omega_min,
        "soft_frac": soft_frac,
        "n_modes": n_modes,
        "n_imag": n_imag,
        "n_excluded": n_excluded,
        "p": p,
    }


def compute_all_metrics(
    freqs_THz: np.ndarray,
    gamma_indices: Optional[np.ndarray] = None,
    p_values: Optional[List[float]] = None,
) -> Dict:
    """Compute all stability metrics in one call.

    Returns a dict with:
        - soft_frac, omega_min, classification
        - i_curv (p=1), omega_inst
        - i_p / omega_p for each requested p value
        - n_modes, n_imag, n_excluded
    """
    if p_values is None:
        p_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]

    # Core metrics at p=1
    result = compute_i_curv(freqs_THz, gamma_indices, p=1.0)
    classification = classify_stability(result["soft_frac"])

    out = {
        "soft_frac": result["soft_frac"],
        "omega_min": result["omega_min"],
        "i_curv": result["i_curv"],
        "omega_inst": result["omega_inst"],
        "classification": classification,
        "n_modes": result["n_modes"],
        "n_imag": result["n_imag"],
        "n_excluded": result["n_excluded"],
    }

    # p-sweep
    for p in p_values:
        if p == 1.0:
            continue
        r = compute_i_curv(freqs_THz, gamma_indices, p=p)
        out[f"i_p_{p}"] = r["i_p"]
        out[f"omega_p_{p}"] = r["omega_p"]

    return out


# ============================================================================
# Internal helpers
# ============================================================================

def _build_exclusion_mask(
    freqs_THz: np.ndarray,
    gamma_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the mode exclusion mask.

    Only the 3 acoustic modes at Gamma are excluded (by q-point/mode index).
    No frequency threshold is applied.
    """
    exclude_mask = np.zeros_like(freqs_THz, dtype=bool)
    if gamma_indices is not None and len(gamma_indices) > 0:
        for gi in gamma_indices:
            exclude_mask[gi, :3] = True
    return exclude_mask
