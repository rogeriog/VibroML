#!/usr/bin/env python3
"""
Comprehensive test for UMA engine integration with VibroML.
Tests calculator initialization, energy/force calculations, and relaxation.
"""

import sys
import os
sys.path.insert(0, '/globalscratch/ucl/modl/rgouvea/VibroML')

from vibroml.utils.structure_utils import initialize_calculator, load_structure
from vibroml.utils.relaxation_utils import relax_structure
from ase.build import bulk
import tempfile

def test_uma_calculator_init():
    """Test UMA calculator initialization"""
    print("\n" + "="*60)
    print("TEST 1: UMA Calculator Initialization")
    print("="*60)
    try:
        calc = initialize_calculator('uma')
        print("✓ UMA calculator initialized successfully")
        print(f"  Calculator type: {type(calc).__name__}")
        return calc
    except Exception as e:
        print(f"✗ Failed to initialize UMA calculator: {e}")
        return None

def test_uma_energy_calculation(calc):
    """Test energy and force calculations"""
    print("\n" + "="*60)
    print("TEST 2: Energy and Force Calculations")
    print("="*60)
    try:
        atoms = bulk('LiF', 'rocksalt', a=4.0)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        print(f"✓ Energy calculated: {energy:.6f} eV")
        
        forces = atoms.get_forces()
        max_force = (forces**2).sum(axis=1).max()**0.5
        print(f"✓ Forces calculated: max force = {max_force:.6f} eV/Å")
        
        stress = atoms.get_stress()
        print(f"✓ Stress calculated: {stress}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uma_relaxation(calc):
    """Test structure relaxation"""
    print("\n" + "="*60)
    print("TEST 3: Structure Relaxation")
    print("="*60)
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
                print(f"✓ Relaxation successful!")
                print(f"  Relaxed structure: {len(relaxed)} atoms")
                return True
            else:
                print("✗ Relaxation returned None")
                return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("UMA ENGINE INTEGRATION TEST SUITE")
    print("="*60)
    
    # Test 1: Calculator initialization
    calc = test_uma_calculator_init()
    if calc is None:
        print("\n✗ FAILED: Could not initialize UMA calculator")
        return False
    
    # Test 2: Energy/force calculations
    if not test_uma_energy_calculation(calc):
        print("\n✗ FAILED: Energy/force calculations")
        return False
    
    # Test 3: Relaxation
    if not test_uma_relaxation(calc):
        print("\n✗ FAILED: Structure relaxation")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

