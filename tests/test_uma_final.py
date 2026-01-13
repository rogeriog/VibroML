#!/usr/bin/env python3
"""
Final comprehensive test for UMA integration.
Run this in the uma_env to verify everything works.
"""

import sys
sys.path.insert(0, '/globalscratch/ucl/modl/rgouvea/VibroML')

from vibroml.utils.structure_utils import initialize_calculator
from vibroml.utils.relaxation_utils import relax_structure
from ase.build import bulk
import tempfile

def main():
    print("\n" + "="*70)
    print("FINAL UMA INTEGRATION TEST")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Calculator initialization
    print("\n[TEST 1] UMA Calculator Initialization")
    try:
        calc = initialize_calculator('uma')
        print("✓ PASS: UMA calculator initialized")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        all_passed = False
        return 1
    
    # Test 2: Energy calculation
    print("\n[TEST 2] Energy Calculation")
    try:
        atoms = bulk('LiF', 'rocksalt', a=4.0)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"✓ PASS: Energy = {energy:.6f} eV")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        all_passed = False
    
    # Test 3: Force calculation
    print("\n[TEST 3] Force Calculation")
    try:
        forces = atoms.get_forces()
        max_force = (forces**2).sum(axis=1).max()**0.5
        print(f"✓ PASS: Max force = {max_force:.6f} eV/Å")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        all_passed = False
    
    # Test 4: Stress calculation
    print("\n[TEST 4] Stress Calculation")
    try:
        stress = atoms.get_stress()
        print(f"✓ PASS: Stress calculated")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        all_passed = False
    
    # Test 5: Structure relaxation
    print("\n[TEST 5] Structure Relaxation")
    try:
        atoms = bulk('LiF', 'rocksalt', a=4.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            relaxed = relax_structure(
                atoms.copy(),
                calc,
                'uma',
                fmax=0.01,
                output_dir=tmpdir,
                original_cif_path='test.cif',
                save_trajectory=False,
                relaxation_patience=5
            )
            if relaxed is not None:
                print(f"✓ PASS: Relaxation successful ({len(relaxed)} atoms)")
            else:
                print("✗ FAIL: Relaxation returned None")
                all_passed = False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

