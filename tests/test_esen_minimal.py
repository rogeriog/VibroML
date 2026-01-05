#!/usr/bin/env python
"""Minimal test script for VibroML's eSEN-30m-omat integration."""

import sys
import os

def test_imports():
    """Test if VibroML utilities can be imported."""
    print("Testing imports...")
    try:
        from vibroml.utils.structure_utils import initialize_calculator
        print("✓ VibroML utilities imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import VibroML utilities: {e}")
        return False

def test_calculator_initialization():
    """Test if eSEN calculator can be initialized through VibroML."""
    print("\nTesting eSEN calculator initialization...")
    try:
        from vibroml.utils.structure_utils import initialize_calculator

        # Initialize eSEN calculator through VibroML
        calc = initialize_calculator(engine="esen")

        if calc is None:
            print("✗ initialize_calculator returned None")
            return False

        print(f"✓ eSEN calculator initialized successfully")
        print(f"  Calculator type: {type(calc).__name__}")
        return True
    except Exception as e:
        print(f"✗ Error initializing eSEN calculator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_energy_calculation():
    """Test if eSEN calculator can calculate energy through VibroML."""
    print("\nTesting energy calculation with eSEN...")
    try:
        from vibroml.utils.structure_utils import initialize_calculator
        from ase.build import bulk

        # Create a simple test structure
        atoms = bulk('Cu', 'fcc', a=3.6)
        print(f"✓ Created test structure: {atoms.get_chemical_formula()}")

        # Initialize eSEN calculator through VibroML
        calc = initialize_calculator(engine="esen")

        if calc is None:
            print("✗ Failed to initialize calculator")
            return False

        # Try to calculate energy
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        print(f"✓ Energy calculation successful: {energy:.4f} eV")

        # Verify energy is reasonable (should be negative for stable structure)
        if energy < 0:
            print(f"✓ Energy value is reasonable (negative)")
            return True
        else:
            print(f"⚠ Warning: Energy is positive, may indicate issue")
            return True  # Still pass, as calculation worked

    except Exception as e:
        print(f"✗ Error testing energy calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VibroML + eSEN-30m-omat Integration Test")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("VibroML Imports", test_imports()))

    # Test 2: Calculator Initialization
    results.append(("Calculator Initialization", test_calculator_initialization()))

    # Test 3: Energy Calculation
    results.append(("Energy Calculation", test_energy_calculation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed - VibroML + eSEN integration working!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

