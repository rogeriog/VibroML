"""Tests for vibroml.utils.stability_metrics.

Verifies:
  - Gamma acoustic exclusion (exactly 3 modes per Gamma point, nothing else)
  - All-positive spectrum → I_curv=0, soft_frac=0
  - I_curv invariance to mode ordering and q-point ordering
  - p-generalization behavior
  - Empty-spectrum edge case
  - AC2: table captions do not contain the superseded -1.5 THz heuristic
    or categorical tier labels
"""

import numpy as np
import pytest
import re
import os

from vibroml.utils.stability_metrics import (
    DEFAULT_Q_MESH,
    ADAPTIVE_Q_DENSITY,
    ADAPTIVE_Q_FLOOR,
    compute_soft_fraction,
    compute_i_curv,
    compute_all_metrics,
    compute_adaptive_q_mesh,
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
# AC2: Paper/code consistency — tables must not contain tier labels or -1.5 THz
# ============================================================================

def _paper_table(name):
    """Absolute path to a LaTeX table in the paper tree.

    tests/ lives at  <root>/01_codebase/VibroML/tests/, and the paper at
    <root>/03_paper_and_rebuttal/, so we go up exactly THREE levels.
    (A previous version used six levels, which escaped the workspace and made
    these tests skip silently -- hiding a real paper/code mismatch.)
    """
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "03_paper_and_rebuttal", "VIBROML_PAPER",
                        "tables", name)


class TestTableConsistency:
    """AC2 — table captions must not contain superseded heuristics or tier labels."""

    def test_table_s1_no_tier_labels(self):
        """Table S1 caption must not contain tier labels or -1.5 THz."""
        table_path = _paper_table("TableS1_fullcomp_MLIPs_phonon_evals.tex")
        if not os.path.exists(table_path):
            pytest.skip(f"Table S1 not found at {table_path}")

        with open(table_path) as f:
            content = f.read()

        assert "-1.5" not in content, (
            "Table S1 still contains the superseded '-1.5 THz' heuristic.")
        assert "Quasi-Stable" not in content, (
            "Table S1 still contains 'Quasi-Stable' tier label.")
        # The \\stb, \\qst, \\ust macros should not appear
        assert r"\stb" not in content, (
            "Table S1 still uses the \\stb stability-label macro.")
        assert r"\qst" not in content, (
            "Table S1 still uses the \\qst stability-label macro.")
        assert r"\ust" not in content, (
            "Table S1 still uses the \\ust stability-label macro.")

    def test_table1_no_tier_labels(self):
        """Table 1 caption must not contain tier labels or -1.5 THz."""
        table_path = _paper_table("table1_stability_remediation.tex")
        if not os.path.exists(table_path):
            pytest.skip(f"Table 1 not found at {table_path}")

        with open(table_path) as f:
            content = f.read()

        assert "-1.5" not in content, (
            "Table 1 still contains the superseded '-1.5 THz' heuristic.")
        assert "Quasi-Stable" not in content, (
            "Table 1 still contains 'Quasi-Stable' tier label.")
        assert r"\stb" not in content, (
            "Table 1 still uses the \\stb stability-label macro.")
        assert r"\qst" not in content, (
            "Table 1 still uses the \\qst stability-label macro.")
        assert r"\ust" not in content, (
            "Table 1 still uses the \\ust stability-label macro.")

    def test_default_q_mesh(self):
        """Verify the default q-mesh constant."""
        assert DEFAULT_Q_MESH == (6, 6, 6)


# ============================================================================
# Adaptive q-mesh
# ============================================================================

class TestAdaptiveQMesh:
    """Tests for compute_adaptive_q_mesh."""

    def test_cubic_lif_above_floor(self):
        """LiF (a~2.56 Å) should get a mesh above the 6x6x6 floor."""
        mesh = compute_adaptive_q_mesh([2.560, 2.560, 2.560])
        assert mesh[0] > 6
        assert mesh == (mesh[0],) * 3  # cubic -> isotropic mesh

    def test_cubic_cspbi3_at_floor(self):
        """CsPbI3 (a~6.39 Å) should hit the 6x6x6 floor."""
        mesh = compute_adaptive_q_mesh([6.387, 6.387, 6.387])
        assert mesh == (6, 6, 6)

    def test_cubic_hfo2_at_floor(self):
        """HfO2 (a~5.05 Å) should hit the 6x6x6 floor."""
        mesh = compute_adaptive_q_mesh([5.050, 5.050, 5.050])
        assert mesh == (6, 6, 6)

    def test_cubic_bi2sn2o7_at_floor(self):
        """Bi2Sn2O7 (a~10.73 Å) should hit the 6x6x6 floor."""
        mesh = compute_adaptive_q_mesh([10.733, 10.733, 10.733])
        assert mesh == (6, 6, 6)

    def test_very_small_cell_exceeds_floor(self):
        """A very small cell (a=1.0 Å) should get a large mesh."""
        mesh = compute_adaptive_q_mesh([1.0, 1.0, 1.0])
        n_expected = int(np.ceil(2 * np.pi / (1.0 * ADAPTIVE_Q_DENSITY)))
        assert mesh == (n_expected, n_expected, n_expected)
        assert mesh[0] > 6

    def test_very_large_cell_at_floor(self):
        """A very large cell (a=50 Å) should hit the floor."""
        mesh = compute_adaptive_q_mesh([50.0, 50.0, 50.0])
        assert mesh == (6, 6, 6)

    def test_anisotropic_cell(self):
        """Anisotropic cell lengths should give direction-dependent mesh."""
        # a=2.56 (above floor), b=10.0 (floor), c=5.0 (floor)
        mesh = compute_adaptive_q_mesh([2.560, 10.0, 5.0])
        assert mesh[0] > 6
        assert mesh[1] == 6
        assert mesh[2] == 6

    def test_general_cell_matrix(self):
        """A 3x3 cell matrix (cubic) should match the 1-D form."""
        a = 2.560
        cell_3x3 = np.diag([a, a, a])
        mesh_3x3 = compute_adaptive_q_mesh(cell_3x3)
        mesh_1d = compute_adaptive_q_mesh([a, a, a])
        assert mesh_3x3 == mesh_1d

    def test_custom_target_dq(self):
        """A denser target should give a larger mesh."""
        mesh_default = compute_adaptive_q_mesh([2.560, 2.560, 2.560])
        mesh_dense = compute_adaptive_q_mesh([2.560, 2.560, 2.560], target_dq=0.05)
        assert mesh_dense[0] > mesh_default[0]

    def test_custom_floor(self):
        """A higher floor should override the density calculation."""
        mesh = compute_adaptive_q_mesh([10.0, 10.0, 10.0], floor=(8, 8, 8))
        assert mesh == (8, 8, 8)

    def test_adaptive_constants(self):
        """Verify the adaptive constants."""
        assert ADAPTIVE_Q_DENSITY == 0.30
        assert ADAPTIVE_Q_FLOOR == (6, 6, 6)

    def test_q_spacing_within_target(self):
        """For meshes above the floor, q-spacing should not exceed target."""
        a = 2.560
        mesh = compute_adaptive_q_mesh([a, a, a])
        dq = 2 * np.pi / (a * mesh[0])
        assert dq <= ADAPTIVE_Q_DENSITY + 1e-10


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

    def test_no_classification_key(self, mixed_spectrum):
        """compute_all_metrics must not return a 'classification' key."""
        freqs, gamma_idx = mixed_spectrum
        result = compute_all_metrics(freqs, gamma_idx)
        assert "classification" not in result

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
