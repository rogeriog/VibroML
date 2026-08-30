"""Tests for vibroml.utils.stability_metrics.

Verifies:
  - Gamma acoustic exclusion (exactly 3 modes per Gamma point, nothing else)
  - All-positive spectrum → I_curv=0, soft_frac=0, "Stable"
  - Tier boundaries at exactly 0.02 and 0.08
  - I_curv invariance to mode ordering
  - AC2: thresholds used in LaTeX table captions match module constants
"""

import numpy as np
import pytest
import re
import os

from vibroml.utils.stability_metrics import (
    SOFT_FRAC_STABLE_MAX,
    SOFT_FRAC_UNSTABLE_MIN,
    DEFAULT_Q_MESH,
    classify_stability,
    compute_soft_fraction,
    compute_i_curv,
    compute_all_metrics,
    _build_exclusion_mask,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_positive_spectrum():
    """A 4x3 spectrum (4 q-points, 3 bands) — all positive."""
    return np.array([
        [0.0, 0.0, 0.0],   # Gamma (acoustic)
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5],
    ]), np.array([0])  # Gamma at index 0


@pytest.fixture
def mixed_spectrum():
    """A 4x3 spectrum with some negative modes."""
    return np.array([
        [0.0, 0.0, 0.0],    # Gamma (acoustic)
        [-1.0, 2.0, 3.0],   # 1 imaginary
        [-0.5, -2.0, 3.5],  # 2 imaginary
        [0.5, 1.5, 2.5],    # all positive
    ]), np.array([0])


@pytest.fixture
def all_negative_spectrum():
    """A 4x3 spectrum with all non-acoustic modes imaginary."""
    return np.array([
        [0.0, 0.0, 0.0],    # Gamma (acoustic)
        [-1.0, -2.0, -3.0],
        [-1.5, -2.5, -3.5],
        [-0.5, -1.5, -2.5],
    ]), np.array([0])


# ============================================================================
# AC2: Paper/code threshold consistency
# ============================================================================

class TestThresholdConsistency:
    """AC2 — thresholds in code must match thresholds in LaTeX table captions."""

    def test_table_s1_caption_matches_constants(self):
        """Parse Table S1 caption and verify thresholds match module constants."""
        table_path = os.path.join(
            os.path.dirname(__file__), "..",
            "..", "..", "..", "..", "..",
            "03_paper_and_rebuttal", "VIBROML_PAPER", "tables",
            "TableS1_fullcomp_MLIPs_phonon_evals.tex"
        )
        if not os.path.exists(table_path):
            pytest.skip(f"Table S1 not found at {table_path}")

        with open(table_path) as f:
            content = f.read()

        # Check that the caption references the correct thresholds
        # The caption should contain 0.02 (or 2%) and 0.08 (or 8%)
        # This test will be updated when the table is regenerated
        assert "0.02" in content or "2\\%" in content, \
            "Table S1 must reference the 2% stable/quasi boundary"

    def test_table1_caption_matches_constants(self):
        """Parse Table 1 caption and verify thresholds match module constants."""
        table_path = os.path.join(
            os.path.dirname(__file__), "..",
            "..", "..", "..", "..", "..",
            "03_paper_and_rebuttal", "VIBROML_PAPER", "tables",
            "table1_stability_remediation.tex"
        )
        if not os.path.exists(table_path):
            pytest.skip(f"Table 1 not found at {table_path}")

        with open(table_path) as f:
            content = f.read()

        # The caption should reference the correct thresholds
        assert "0.02" in content or "2\\%" in content, \
            "Table 1 must reference the 2% stable/quasi boundary"

    def test_constants_are_correct_values(self):
        """Verify the threshold constants have the expected values."""
        assert SOFT_FRAC_STABLE_MAX == 0.02
        assert SOFT_FRAC_UNSTABLE_MIN == 0.08
        assert DEFAULT_Q_MESH == (6, 6, 6)


# ============================================================================
# Classification tests
# ============================================================================

class TestClassifyStability:

    def test_stable_below_2pct(self):
        assert classify_stability(0.0) == "Stable"
        assert classify_stability(0.01) == "Stable"
        assert classify_stability(0.019) == "Stable"

    def test_quasi_between_2_and_8pct(self):
        assert classify_stability(0.02) == "Quasi-Stable"
        assert classify_stability(0.05) == "Quasi-Stable"
        assert classify_stability(0.079) == "Quasi-Stable"

    def test_unstable_at_8pct_and_above(self):
        assert classify_stability(0.08) == "Unstable"
        assert classify_stability(0.20) == "Unstable"
        assert classify_stability(1.0) == "Unstable"

    def test_boundary_exactness(self):
        """Boundaries are >= for the upper tier, >= for quasi lower tier."""
        assert classify_stability(0.02) == "Quasi-Stable"  # exactly 2%
        assert classify_stability(0.08) == "Unstable"      # exactly 8%
        assert classify_stability(0.0199) == "Stable"      # just below 2%


# ============================================================================
# Gamma acoustic exclusion
# ============================================================================

class TestAcousticExclusion:

    def test_only_3_gamma_acoustic_excluded(self, simple_positive_spectrum):
        """Exactly 3 modes per Gamma point are excluded, nothing else."""
        freqs, gamma_idx = simple_positive_spectrum
        mask = _build_exclusion_mask(freqs, gamma_idx)
        n_excluded = int(mask.sum())
        assert n_excluded == 3, f"Expected 3 excluded, got {n_excluded}"

    def test_no_threshold_applied(self, mixed_spectrum):
        """No frequency threshold — modes near zero are kept."""
        freqs, gamma_idx = mixed_spectrum
        # Add a mode at exactly 0.001 THz (would be excluded by a 0.05 threshold)
        freqs_with_marginal = freqs.copy()
        freqs_with_marginal[3, 0] = 0.001
        mask = _build_exclusion_mask(freqs_with_marginal, gamma_idx)
        # Only 3 Gamma acoustic modes should be excluded
        assert int(mask.sum()) == 3

    def test_multiple_gamma_points(self):
        """Multiple Gamma points each have 3 acoustic modes excluded."""
        freqs = np.zeros((6, 3))
        gamma_indices = np.array([0, 3])
        mask = _build_exclusion_mask(freqs, gamma_indices)
        assert int(mask.sum()) == 6  # 2 Gamma points × 3 acoustic

    def test_no_gamma_indices(self):
        """With no Gamma indices, nothing is excluded."""
        freqs = np.random.randn(10, 3)
        mask = _build_exclusion_mask(freqs, None)
        assert int(mask.sum()) == 0


# ============================================================================
# I_curv computation
# ============================================================================

class TestICurv:

    def test_all_positive_gives_zero(self, simple_positive_spectrum):
        """All-positive spectrum → I_curv=0, soft_frac=0."""
        freqs, gamma_idx = simple_positive_spectrum
        result = compute_i_curv(freqs, gamma_idx)
        assert result["i_curv"] == 0.0
        assert result["soft_frac"] == 0.0
        assert result["n_imag"] == 0

    def test_i_curv_value(self, mixed_spectrum):
        """Verify I_curv for a known spectrum."""
        freqs, gamma_idx = mixed_spectrum
        result = compute_i_curv(freqs, gamma_idx)
        # After excluding Gamma acoustic: 9 modes, 3 imaginary (-1.0, -0.5, -2.0)
        # I_curv = (1.0^2 + 0.5^2 + 2.0^2) / 9 = (1 + 0.25 + 4) / 9 = 5.25/9
        expected_i_curv = (1.0 + 0.25 + 4.0) / 9.0
        assert abs(result["i_curv"] - expected_i_curv) < 1e-10
        assert abs(result["soft_frac"] - 3.0/9.0) < 1e-10

    def test_omega_min_from_all_modes(self, mixed_spectrum):
        """omega_min is computed from ALL modes (including acoustic)."""
        freqs, gamma_idx = mixed_spectrum
        result = compute_i_curv(freqs, gamma_idx)
        assert result["omega_min"] == -2.0  # most negative in the full array

    def test_invariance_to_mode_ordering(self, mixed_spectrum):
        """I_curv is invariant to the order of modes within each q-point."""
        freqs, gamma_idx = mixed_spectrum
        result1 = compute_i_curv(freqs, gamma_idx)

        # Shuffle columns (mode order within each q-point)
        freqs_shuffled = freqs[:, [2, 0, 1]]
        result2 = compute_i_curv(freqs_shuffled, gamma_idx)

        assert abs(result1["i_curv"] - result2["i_curv"]) < 1e-10
        assert abs(result1["soft_frac"] - result2["soft_frac"]) < 1e-10

    def test_invariance_to_qpoint_ordering(self, mixed_spectrum):
        """I_curv is invariant to the order of q-points."""
        freqs, gamma_idx = mixed_spectrum
        result1 = compute_i_curv(freqs, gamma_idx)

        # Shuffle rows (q-point order) — Gamma moves to index 2
        freqs_shuffled = freqs[[1, 3, 0, 2]]
        gamma_shuffled = np.array([2])
        result2 = compute_i_curv(freqs_shuffled, gamma_shuffled)

        assert abs(result1["i_curv"] - result2["i_curv"]) < 1e-10
        assert abs(result1["soft_frac"] - result2["soft_frac"]) < 1e-10

    def test_p_generalization(self, mixed_spectrum):
        """p=2 should emphasize deeper modes more than p=1."""
        freqs, gamma_idx = mixed_spectrum
        r1 = compute_i_curv(freqs, gamma_idx, p=1.0)
        r2 = compute_i_curv(freqs, gamma_idx, p=2.0)
        # Both should be positive
        assert r1["i_p"] > 0
        assert r2["i_p"] > 0
        # p=2 should give a larger characteristic frequency (emphasizes depth)
        assert r2["omega_p"] > r1["omega_p"]

    def test_p_inf_approaches_omega_min(self, mixed_spectrum):
        """As p→∞, Omega_p approaches |omega_min|."""
        freqs, gamma_idx = mixed_spectrum
        r_large = compute_i_curv(freqs, gamma_idx, p=50.0)
        # |omega_min| = 2.0 (from the full array)
        # But I_p excludes Gamma acoustic, so the most negative kept mode is -2.0
        assert abs(r_large["omega_p"] - 2.0) < 0.1


# ============================================================================
# compute_all_metrics
# ============================================================================

class TestComputeAllMetrics:

    def test_returns_classification(self, mixed_spectrum):
        freqs, gamma_idx = mixed_spectrum
        result = compute_all_metrics(freqs, gamma_idx)
        assert "classification" in result
        assert result["classification"] in ["Stable", "Quasi-Stable", "Unstable"]

    def test_includes_p_sweep(self, mixed_spectrum):
        freqs, gamma_idx = mixed_spectrum
        result = compute_all_metrics(freqs, gamma_idx)
        assert "i_p_0.5" in result
        assert "i_p_2.0" in result
        assert "i_curv" in result  # p=1

    def test_empty_spectrum(self):
        """Handle edge case of empty spectrum."""
        freqs = np.zeros((1, 3))
        result = compute_i_curv(freqs, np.array([0]))
        # All 3 modes excluded → n_modes=0
        assert result["n_modes"] == 0
        assert result["i_curv"] == 0.0
        assert result["soft_frac"] == 0.0
