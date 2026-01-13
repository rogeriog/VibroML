#!/usr/bin/env python3
"""
Test script to verify fairchem-core upgrade impact on eSEN and UMA models.
"""
import sys
import os

print("=" * 70)
print("STEP 1: Check Current fairchem-core Version")
print("=" * 70)

try:
    import pkg_resources
    version = pkg_resources.get_distribution("fairchem-core").version
    print(f"✓ fairchem-core version: {version}")
except Exception as e:
    print(f"✗ Could not get fairchem-core version: {e}")
    sys.exit(1)

# Check if units module exists
print("\nChecking for units module...")
try:
    from fairchem.core import units
    print("✓ fairchem.core.units is available")
    has_units = True
except ImportError:
    print("✗ fairchem.core.units is NOT available")
    has_units = False

# Test eSEN model
print("\n" + "=" * 70)
print("STEP 2: Test eSEN Model")
print("=" * 70)

try:
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    from ase.build import bulk
    
    model_path = os.path.abspath("fairchem_models/esen_30m_omat.pt")
    print(f"Loading eSEN model from: {model_path}")
    
    calc = OCPCalculator(checkpoint_path=model_path, cpu=True)
    print("✓ eSEN model loaded successfully")
    
    # Quick energy test
    atoms = bulk('LiF', 'rocksalt', a=4.0)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    print(f"✓ eSEN energy calculation works: {energy:.6f} eV")
    esen_works = True
    
except Exception as e:
    print(f"✗ eSEN model test FAILED: {e}")
    esen_works = False

# Test UMA model
print("\n" + "=" * 70)
print("STEP 3: Test UMA Model")
print("=" * 70)

try:
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    from ase.build import bulk
    
    model_path = os.path.abspath("fairchem_models/uma-m-1p1.pt")
    print(f"Loading UMA model from: {model_path}")
    
    calc = OCPCalculator(checkpoint_path=model_path, cpu=True)
    print("✓ UMA model loaded successfully")
    
    # Quick energy test
    atoms = bulk('LiF', 'rocksalt', a=4.0)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    print(f"✓ UMA energy calculation works: {energy:.6f} eV")
    uma_works = True
    
except Exception as e:
    print(f"✗ UMA model test FAILED: {e}")
    uma_works = False

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"fairchem-core version: {version}")
print(f"Has units module: {has_units}")
print(f"eSEN model works: {esen_works}")
print(f"UMA model works: {uma_works}")

if esen_works and uma_works:
    print("\n✓ SUCCESS: Both models work with this fairchem-core version!")
    sys.exit(0)
elif esen_works and not uma_works:
    print("\n⚠ PARTIAL: eSEN works but UMA doesn't (need newer fairchem-core)")
    sys.exit(1)
elif not esen_works and uma_works:
    print("\n✗ PROBLEM: UMA works but eSEN broke (need to revert)")
    sys.exit(2)
else:
    print("\n✗ FAILURE: Neither model works!")
    sys.exit(3)

