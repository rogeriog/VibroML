#!/usr/bin/env python
"""
Minimal integration tests for VibroML's PET-MAD (upet) engine support.

Requires an environment with the upet package installed:
    pip install upet
  which brings in metatrain, metatomic-ase, warp-lang, and other dependencies.

Run with the upet environment active, e.g.:
    /path/to/upet_env/bin/python tests/test_pet_engine.py

Or as a pytest module (models are downloaded from HuggingFace on first run):
    pytest tests/test_pet_engine.py -v

Available model strings for --model_name when using --engine pet:
    pet-mad-xs  (smallest, fast – good for testing)
    pet-mad-s   (default)
    pet-omat-xs, pet-omat-s, pet-omat-m, pet-omat-l, pet-omat-xl
    pet-oam-l, pet-oam-xl
    pet-omatpes-l
    pet-spice-s, pet-spice-l  (molecules only)
"""

import sys
import os
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_upet_available():
    try:
        from upet.calculator import UPETCalculator  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Unit-level tests (no model download needed)
# ---------------------------------------------------------------------------

def test_vibroml_imports():
    """VibroML utilities must import cleanly regardless of upet availability."""
    from vibroml.utils.structure_utils import initialize_calculator  # noqa: F401
    from vibroml.utils.utils import HAVE_UPET, UPETCalculator  # noqa: F401


def test_have_upet_flag():
    """HAVE_UPET flag must be a bool and reflect whether upet is installed."""
    from vibroml.utils.utils import HAVE_UPET
    assert isinstance(HAVE_UPET, bool)
    expected = _check_upet_available()
    assert HAVE_UPET == expected, (
        f"HAVE_UPET={HAVE_UPET} but upet importable={expected}"
    )


# ---------------------------------------------------------------------------
# Integration tests (require upet + network for model download)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _check_upet_available(), reason="upet not installed")
def test_calculator_initialization_default():
    """initialize_calculator('pet') with default model_name returns a calculator."""
    from vibroml.utils.structure_utils import initialize_calculator
    calc = initialize_calculator(engine="pet")
    assert calc is not None, "initialize_calculator returned None for 'pet' engine"
    print(f"  Calculator type: {type(calc).__name__}")


@pytest.mark.skipif(not _check_upet_available(), reason="upet not installed")
@pytest.mark.parametrize("model_name", ["pet-mad-xs", "pet-mad-s"])
def test_calculator_initialization_model_variants(model_name):
    """initialize_calculator('pet') with explicit PET-MAD model variants."""
    from vibroml.utils.structure_utils import initialize_calculator
    calc = initialize_calculator(engine="pet", model_name=model_name)
    assert calc is not None, f"initialize_calculator returned None for model '{model_name}'"
    print(f"  {model_name}: {type(calc).__name__}")


@pytest.mark.skipif(not _check_upet_available(), reason="upet not installed")
def test_energy_forces_stress_lif():
    """PET-MAD-xs must return energy, forces, and stress for a LiF unit cell."""
    from vibroml.utils.structure_utils import initialize_calculator
    from ase.build import bulk

    atoms = bulk("LiF", "rocksalt", a=4.02)
    calc = initialize_calculator(engine="pet", model_name="pet-mad-xs")
    assert calc is not None
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    assert isinstance(energy, float), "Energy is not a float"
    assert forces.shape == (len(atoms), 3), f"Unexpected forces shape: {forces.shape}"
    assert stress.shape == (6,), f"Unexpected stress shape: {stress.shape}"
    print(f"  LiF energy = {energy:.4f} eV")
    print(f"  Max |force| = {abs(forces).max():.4f} eV/Å")


@pytest.mark.skipif(not _check_upet_available(), reason="upet not installed")
def test_energy_forces_stress_diamond_si():
    """PET-MAD-xs must return sensible results for diamond-cubic Si."""
    from vibroml.utils.structure_utils import initialize_calculator
    from ase.build import bulk
    import numpy as np

    atoms = bulk("Si", "diamond", a=5.43)
    calc = initialize_calculator(engine="pet", model_name="pet-mad-xs")
    assert calc is not None
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    assert isinstance(energy, float)
    assert forces.shape == (len(atoms), 3)
    assert stress.shape == (6,)
    # By symmetry, forces on Si in diamond-cubic should be near zero
    assert np.abs(forces).max() < 1.0, (
        f"Forces on ideal Si too large: {np.abs(forces).max():.4f} eV/Å"
    )
    print(f"  Si energy = {energy:.4f} eV  (per atom: {energy/len(atoms):.4f} eV)")


@pytest.mark.skipif(not _check_upet_available(), reason="upet not installed")
def test_checkpoint_path_loading():
    """initialize_calculator can load a local metatrain .ckpt checkpoint.

    Uses hf_hub_download to cache the .ckpt file (same path used internally
    by get_upet), so no extra network traffic after the first run.
    """
    from huggingface_hub import hf_hub_download
    from vibroml.utils.structure_utils import initialize_calculator
    from ase.build import bulk

    # Download (or use cached) metatrain checkpoint – the same file get_upet() uses
    ckpt = hf_hub_download(
        repo_id="lab-cosmo/upet",
        filename="pet-mad-xs-v1.5.0.ckpt",
        subfolder="models",
    )
    assert os.path.isfile(ckpt), "Downloaded .ckpt not found"

    calc = initialize_calculator(engine="pet", checkpoint_model_path=ckpt)
    assert calc is not None

    atoms = bulk("Cu", "fcc", a=3.61)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    print(f"  Cu energy from .ckpt checkpoint = {energy:.4f} eV")


# ---------------------------------------------------------------------------
# CLI smoke-test (optional, prints output when run directly)
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("VibroML + PET-MAD (upet) Integration Test")
    print("=" * 60)

    upet_available = _check_upet_available()
    print(f"upet available: {upet_available}")

    results = []

    # Always-run tests
    try:
        test_vibroml_imports()
        results.append(("VibroML imports", True))
    except Exception as e:
        results.append(("VibroML imports", False))
        print(f"  FAIL: {e}")

    try:
        test_have_upet_flag()
        results.append(("HAVE_UPET flag", True))
    except Exception as e:
        results.append(("HAVE_UPET flag", False))
        print(f"  FAIL: {e}")

    if not upet_available:
        print("\nupet not installed – skipping model-download tests.")
        print("Install upet and re-run:  pip install upet")
    else:
        for test_fn, label in [
            (test_calculator_initialization_default, "Calculator init (default)"),
            (test_energy_forces_stress_lif, "Energy/forces/stress – LiF"),
            (test_energy_forces_stress_diamond_si, "Energy/forces/stress – Si"),
        ]:
            try:
                test_fn()
                results.append((label, True))
                print(f"  PASS: {label}")
            except Exception as e:
                results.append((label, False))
                print(f"  FAIL: {label} – {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary:")
    for label, passed in results:
        print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {label}")
    print("=" * 60)

    all_passed = all(p for _, p in results)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
