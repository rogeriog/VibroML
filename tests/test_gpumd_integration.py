#!/usr/bin/env python3
"""
Test script to verify GPUMD integration into VibroML.

This script tests:
1. GPUMD binary detection
2. Calculator initialization
3. Basic energy and force calculations
4. Integration with VibroML structure utilities
"""

import os
import sys
import tempfile
import shutil

# Add VibroML to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibroml.utils.structure_utils import initialize_calculator, load_structure
from vibroml.utils.utils import HAVE_GPUMD, GPUMD_BINARY_PATH
from ase.build import bulk
import numpy as np

def test_gpumd_detection():
    """Test GPUMD binary detection."""
    print("\n" + "="*70)
    print("TEST 1: GPUMD Binary Detection")
    print("="*70)
    
    if HAVE_GPUMD:
        print(f"✓ GPUMD binary found at: {GPUMD_BINARY_PATH}")
        return True
    else:
        print("✗ GPUMD binary not found")
        return False

def test_calculator_initialization():
    """Test GPUMD calculator initialization."""
    print("\n" + "="*70)
    print("TEST 2: GPUMD Calculator Initialization")
    print("="*70)
    
    try:
        calculator = initialize_calculator("gpumd")
        if calculator is not None:
            print("✓ GPUMD calculator initialized successfully")
            return True, calculator
        else:
            print("✗ Failed to initialize GPUMD calculator")
            return False, None
    except Exception as e:
        print(f"✗ Error initializing GPUMD calculator: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_energy_force_calculation(calculator):
    """Test energy and force calculation with GPUMD."""
    print("\n" + "="*70)
    print("TEST 3: Energy and Force Calculation")
    print("="*70)
    
    try:
        # Create a test structure (Te with 2x2x2 supercell)
        atoms = bulk('Te', 'sc', a=3.0)
        atoms = atoms.repeat((2, 2, 2))
        
        print(f"Test structure: {atoms.get_chemical_formula()}")
        print(f"Number of atoms: {len(atoms)}")
        
        # Set calculator
        atoms.set_calculator(calculator)
        
        # Calculate energy
        energy = atoms.get_potential_energy()
        print(f"✓ Energy calculated: {energy:.6f} eV")
        
        # Calculate forces
        forces = atoms.get_forces()
        print(f"✓ Forces calculated: {len(forces)} force vectors")
        print(f"  Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.6e} eV/Å")
        print(f"  Mean force magnitude: {np.mean(np.linalg.norm(forces, axis=1)):.6e} eV/Å")
        
        return True
    except Exception as e:
        print(f"✗ Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_structures(calculator):
    """Test with multiple structures."""
    print("\n" + "="*70)
    print("TEST 4: Multiple Structure Tests")
    print("="*70)
    
    test_cases = [
        ("Te SC (2x2x2)", lambda: bulk('Te', 'sc', a=3.0).repeat((2, 2, 2))),
        ("Te SC (3x3x3)", lambda: bulk('Te', 'sc', a=3.0).repeat((3, 3, 3))),
    ]
    
    all_passed = True
    for name, structure_fn in test_cases:
        try:
            atoms = structure_fn()
            atoms.set_calculator(calculator)
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            print(f"✓ {name}: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
            print(f"  Energy: {energy:.6f} eV")
            print(f"  Max force: {np.max(np.linalg.norm(forces, axis=1)):.6e} eV/Å")
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GPUMD Integration Test Suite")
    print("="*70)
    
    results = {}
    
    # Test 1: Binary detection
    results['detection'] = test_gpumd_detection()
    if not results['detection']:
        print("\n✗ GPUMD binary not found. Cannot proceed with further tests.")
        return False
    
    # Test 2: Calculator initialization
    results['init'], calculator = test_calculator_initialization()
    if not results['init'] or calculator is None:
        print("\n✗ Failed to initialize calculator. Cannot proceed with further tests.")
        return False
    
    # Test 3: Energy and force calculation
    results['calc'] = test_energy_force_calculation(calculator)
    
    # Test 4: Multiple structures
    results['multi'] = test_multiple_structures(calculator)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    test_names = {
        'detection': 'GPUMD Binary Detection',
        'init': 'Calculator Initialization',
        'calc': 'Energy/Force Calculation',
        'multi': 'Multiple Structures'
    }
    
    all_passed = True
    for key, name in test_names.items():
        status = "✓ PASS" if results.get(key, False) else "✗ FAIL"
        print(f"{status}: {name}")
        if not results.get(key, False):
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nGPUMD is successfully integrated into VibroML!")
        return True
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

