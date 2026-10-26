r"""Stability metrics for MLIP-driven phonon screening.

This module provides the canonical implementation of the continuous
stability descriptors used in VibroML:

- **soft-mode fraction** (``soft_frac``): the fraction of phonon modes
  with imaginary frequency on a uniform q-point mesh.  Dimensionless
  and mass-independent.
- **mean negative curvature** (:math:`I_\\mathrm{curv}`): the normalised
  mean of :math:`\\omega^2` over imaginary modes, providing a severity
  scale in THz\ :sup:`2`.

These are reported as continuous descriptors; no categorical stability
labels or hard thresholds are imposed, because dynamical stability is a
continuous property whose actionable boundary depends on the downstream
application.

Only the 3 acoustic modes at the Gamma point are excluded (by q-point/mode
index).  **No frequency threshold is applied** — the only boundary is
:math:`\\lambda = 0` (i.e. :math:`\\omega = 0`).
"""

import math
import numpy as np
from typing import Optional, Dict, List, Tuple, Union

DEFAULT_Q_MESH: Tuple[int, int, int] = (6, 6, 6)

# Adaptive q-mesh parameters
# Target maximum q-point spacing of 0.30 1/Å along each reciprocal direction,
# with a minimum mesh of 6x6x6.  This was found as a good computational
# compromise: it gives ~8x8x8 for small cells (e.g. LiF, a~2.56 Å) while
# keeping 6x6x6 for larger cells, consistent with the convergence sweep
# performed on the four benchmark aristotypes (LiF, CsPbI3, HfO2, Bi2Sn2O7).
ADAPTIVE_Q_DENSITY: float = 0.30  # 1/Å
ADAPTIVE_Q_FLOOR: Tuple[int, int, int] = (6, 6, 6)


def compute_adaptive_q_mesh(
    cell: Union[np.ndarray, Tuple[float, ...], List[float]],
    target_dq: float = ADAPTIVE_Q_DENSITY,
    floor: Tuple[int, int, int] = ADAPTIVE_Q_FLOOR,
) -> Tuple[int, int, int]:
    """Compute a density-based q-point mesh with a minimum floor.

    Chooses Monkhorst-Pack mesh dimensions so that the q-point spacing
    along each reciprocal direction does not exceed ``target_dq``,
    while enforcing a minimum mesh size via ``floor``.

    For a cubic cell with lattice constant *a*, an ``N×N×N`` grid has
    spacing ``Δq ≈ 2π/(aN)``.  At ``target_dq = 0.30 1/Å``:

    - LiF  (a ≈ 2.56 Å) → ~8×8×8  (above the floor)
    - HfO2 (a ≈ 5.05 Å) → 6×6×6  (at the floor)
    - CsPbI3 (a ≈ 6.39 Å) → 6×6×6 (at the floor)
    - Bi2Sn2O7 (a ≈ 10.73 Å) → 6×6×6 (at the floor)

    Parameters
    ----------
    cell : array-like, shape (3,) or (3, 3)
        Real-space cell.  If 1-D, interpreted as orthogonal cell lengths
        [a, b, c] in Å.  If 2-D, interpreted as cell vectors (rows) in Å.
    target_dq : float, default 0.15
        Target maximum q-point spacing in 1/Å along each reciprocal
        lattice direction.
    floor : tuple of int, default (6, 6, 6)
        Minimum mesh dimensions.

    Returns
    -------
    tuple of int
        Monkhorst-Pack mesh dimensions (N1, N2, N3).
    """
    cell = np.asarray(cell, dtype=float)
    if cell.ndim == 1:
        # Orthogonal cell — cell lengths
        b_lengths = 2.0 * np.pi / cell
    else:
        # General cell — 3x3 matrix of cell vectors (rows)
        a1, a2, a3 = cell
        volume = abs(np.dot(a1, np.cross(a2, a3)))
        if volume < 1e-12:
            raise ValueError("Degenerate cell (zero volume)")
        b1 = np.cross(a2, a3)
        b2 = np.cross(a3, a1)
        b3 = np.cross(a1, a2)
        b_lengths = 2.0 * np.pi * np.array([
            np.linalg.norm(b1),
            np.linalg.norm(b2),
            np.linalg.norm(b3),
        ]) / volume

    mesh = []
    for b_len, f in zip(b_lengths, floor):
        n = max(f, int(math.ceil(b_len / target_dq)))
        mesh.append(n)
    return tuple(mesh)


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
        - soft_frac, omega_min
        - i_curv (p=1), omega_inst
        - i_p / omega_p for each requested p value
        - n_modes, n_imag, n_excluded
    """
    if p_values is None:
        p_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]

    # Core metrics at p=1
    result = compute_i_curv(freqs_THz, gamma_indices, p=1.0)

    out = {
        "soft_frac": result["soft_frac"],
        "omega_min": result["omega_min"],
        "i_curv": result["i_curv"],
        "omega_inst": result["omega_inst"],
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
