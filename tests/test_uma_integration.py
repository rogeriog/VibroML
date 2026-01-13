#!/usr/bin/env python3
"""Test UMA environment integration with VibroML"""

import sys
import os

print("=" * 70)
print("UMA Environment Integration Test")
print("=" * 70)
print()

# Test 1: Check Python version
print("Test 1: Python version")
print(f"  Python: {sys.version}")
print()

# Test 2: Import VibroML
print("Test 2: Import VibroML")
try:
    import vibroml
    print("  ✓ VibroML imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import VibroML: {e}")
    sys.exit(1)
print()

# Test 3: Import fairchem.core.units
print("Test 3: Import fairchem.core.units")
try:
    from fairchem.core import units
    print("  ✓ fairchem.core.units imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import fairchem.core.units: {e}")
    sys.exit(1)
print()

# Test 4: Import OCPCalculator
print("Test 4: Import OCPCalculator")
try:
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    print("  ✓ OCPCalculator imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import OCPCalculator: {e}")
    sys.exit(1)
print()

# Test 5: Check UMA model file
print("Test 5: Check UMA model file")
uma_model = "/globalscratch/ucl/modl/rgouvea/VibroML/fairchem_models/uma-m-1p1.pt"
if os.path.exists(uma_model):
    size_gb = os.path.getsize(uma_model) / (1024**3)
    print(f"  ✓ UMA model found: {uma_model}")
    print(f"    Size: {size_gb:.2f} GB")
else:
    print(f"  ✗ UMA model not found: {uma_model}")
    sys.exit(1)
print()

# Test 6: Load UMA model
print("Test 6: Load UMA model with OCPCalculator")
try:
    calc = OCPCalculator(checkpoint_path=uma_model, cpu=True)
    print("  ✓ UMA model loaded successfully")
except Exception as e:
    print(f"  ✗ Failed to load UMA model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 7: Test energy calculation
print("Test 7: Test energy calculation with UMA model")
try:
    from ase.build import bulk
    atoms = bulk('LiF', 'rocksalt', a=4.0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"  ✓ Energy calculation successful")
    print(f"    LiF energy: {energy:.6f} eV")
except Exception as e:
    print(f"  ✗ Energy calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 8: Check VibroML calculator initialization
print("Test 8: Check VibroML calculator initialization")
try:
    from vibroml.utils.structure_utils import initialize_calculator
    calc_vibroml = initialize_calculator(engine="esen", checkpoint_path=uma_model)
    print("  ✓ VibroML calculator initialization successful")
except Exception as e:
    print(f"  ✗ VibroML calculator initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 70)
print("✓ All tests passed! UMA environment is ready for integration.")
print("=" * 70)

