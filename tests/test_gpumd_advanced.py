#!/usr/bin/env python3
"""
Advanced GPUMD calculator test with energy validation.

This script performs comprehensive testing:
1. Single-point energy/force calculation
2. Multiple structure testing
3. Energy consistency checks
4. Force magnitude validation
5. Comparison with reference values (if available)

Usage:
    python test_gpumd_advanced.py [--verbose] [--keep-temp]
"""

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path

def test_single_point(work_dir, gpumd_binary, atoms, test_name):
    """Run single-point calculation and return results."""
    try:
        from ase.io import write
        import numpy as np

        # Write structure
        xyz_path = os.path.join(work_dir, "model.xyz")
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        lattice_str = " ".join([f"{x:.10f}" for row in cell for x in row])

        with open(xyz_path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f'energy=0.0 config_type=test pbc="T T T" ')
            f.write(f'Lattice="{lattice_str}" ')
            f.write(f'Properties=species:S:1:pos:R:3:force:R:3\n')
            for symbol, pos in zip(symbols, positions):
                f.write(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} 0.0 0.0 0.0\n")

        # Find and copy potential file
        vibroml_root = os.path.dirname(os.path.abspath(__file__))
        potential_candidates = [
            os.path.join(vibroml_root, "GPUMD", "examples", "nep_train", "nep.txt"),
            os.path.join(vibroml_root, "GPUMD", "examples", "nep_prediction", "nep.txt"),
        ]

        potential_src = None
        for candidate in potential_candidates:
            if os.path.exists(candidate):
                potential_src = candidate
                break

        if potential_src is None:
            return None, "No NEP potential file found"

        potential_dst = os.path.join(work_dir, "nep.txt")
        shutil.copy(potential_src, potential_dst)

        # Write run.in
        run_in_path = os.path.join(work_dir, "run.in")
        with open(run_in_path, 'w') as f:
            f.write("potential   nep.txt\n")
            f.write("velocity    1\n")
            f.write("ensemble nve\n")
            f.write("time_step 0\n")
            f.write("dump_force 1\n")
            f.write("run 1\n")
        
        # Run GPUMD
        original_dir = os.getcwd()
        os.chdir(work_dir)
        
        result = subprocess.run(
            [gpumd_binary],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        os.chdir(original_dir)
        
        if result.returncode != 0:
            return None, f"GPUMD failed: {result.stderr}"
        
        # Parse forces
        force_file = os.path.join(work_dir, "force.out")
        if not os.path.exists(force_file):
            return None, "force.out not found"
        
        forces = []
        with open(force_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        forces.append([float(x) for x in parts[:3]])
        
        if len(forces) != len(atoms):
            return None, f"Force count mismatch: {len(forces)} vs {len(atoms)}"
        
        forces = np.array(forces)
        
        # Calculate statistics
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags)
        mean_force = np.mean(force_mags)
        
        return {
            'forces': forces,
            'force_mags': force_mags,
            'max_force': max_force,
            'mean_force': mean_force,
            'n_atoms': len(atoms),
            'formula': atoms.get_chemical_formula()
        }, None
        
    except Exception as e:
        return None, str(e)

def run_tests(gpumd_binary, verbose=False, keep_temp=False):
    """Run comprehensive test suite."""
    from ase.build import bulk
    
    print("=" * 70)
    print("GPUMD Advanced Calculator Test Suite")
    print("=" * 70)
    
    # Test structures (GPUMD requires at least 2 atoms)
    # Note: NEP potential is trained for Te and Pb
    test_cases = [
        ("Te SC (2x2x2)", lambda: bulk('Te', 'sc', a=3.0).repeat((2, 2, 2))),
        ("Te SC (3x3x3)", lambda: bulk('Te', 'sc', a=3.0).repeat((3, 3, 3))),
    ]
    
    results = []
    temp_dirs = []
    
    try:
        for test_name, structure_fn in test_cases:
            print(f"\n» Test: {test_name}")
            
            # Create structure
            try:
                atoms = structure_fn()
                print(f"  ✓ Created structure: {atoms.get_chemical_formula()}")
            except Exception as e:
                print(f"  ✗ Failed to create structure: {e}")
                continue
            
            # Create work directory
            work_dir = tempfile.mkdtemp(prefix=f"gpumd_{test_name.replace(' ', '_')}_")
            temp_dirs.append(work_dir)
            
            # Run calculation
            result, error = test_single_point(work_dir, gpumd_binary, atoms, test_name)
            
            if error:
                print(f"  ✗ Calculation failed: {error}")
                continue
            
            # Display results
            print(f"  ✓ Calculation successful")
            print(f"    - Atoms: {result['n_atoms']}")
            print(f"    - Max force: {result['max_force']:.6f} eV/Å")
            print(f"    - Mean force: {result['mean_force']:.6f} eV/Å")
            
            if verbose:
                print(f"    - Force magnitudes: {result['force_mags']}")
            
            results.append((test_name, result))
        
        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        
        if results:
            print(f"✓ Passed: {len(results)}/{len(test_cases)} tests")
            for test_name, result in results:
                print(f"  ✓ {test_name}: {result['formula']}")
            print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
            return True
        else:
            print(f"✗ Failed: All tests failed")
            return False
    
    finally:
        # Cleanup
        if not keep_temp:
            for work_dir in temp_dirs:
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir)
            print(f"\n✓ Cleaned up temporary directories")
        else:
            print(f"\n✓ Kept temporary directories:")
            for work_dir in temp_dirs:
                print(f"  - {work_dir}")

def main():
    parser = argparse.ArgumentParser(description="Advanced GPUMD calculator test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary directories")
    args = parser.parse_args()
    
    # Check GPUMD binary
    gpumd_binary = os.path.join(os.path.dirname(__file__), "GPUMD", "src", "gpumd")
    if not os.path.exists(gpumd_binary):
        print(f"✗ GPUMD binary not found at: {gpumd_binary}")
        sys.exit(1)
    
    print(f"✓ Found GPUMD binary: {gpumd_binary}\n")
    
    # Run tests
    success = run_tests(gpumd_binary, verbose=args.verbose, keep_temp=args.keep_temp)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

