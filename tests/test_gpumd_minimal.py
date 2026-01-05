#!/usr/bin/env python3
"""
Minimal test script for GPUMD calculator validation.

This script tests basic GPUMD functionality:
1. Loads a compiled GPUMD binary
2. Creates a test structure
3. Computes energies and forces
4. Validates results

Usage:
    # On GPU compute node with fosscuda/2020b loaded:
    python test_gpumd_minimal.py

Requirements:
    - GPUMD compiled in GPUMD/src/
    - ASE (Atomic Simulation Environment)
    - NumPy
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add VibroML to path for structure utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_structure():
    """Create a simple test structure (Te with supercell).

    Note: The NEP potential in GPUMD/examples/nep_train/nep.txt is trained for Te and Pb.
    """
    try:
        from ase.build import bulk
        from ase.io import write

        # Create Te structure with supercell
        # GPUMD requires at least 2 atoms
        # Using Te since the NEP potential is trained for Te and Pb
        # Use simple cubic structure for Te
        atoms = bulk('Te', 'sc', a=3.0)
        atoms = atoms.repeat((2, 2, 2))
        print(f"✓ Created test structure: {atoms.get_chemical_formula()}")
        print(f"  - Atoms: {len(atoms)}")
        print(f"  - Cell: {atoms.cell.cellpar()[:3]}")
        return atoms
    except Exception as e:
        print(f"✗ Failed to create test structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def write_gpumd_input(work_dir, atoms):
    """Write GPUMD input files (model.xyz and run.in) in GPUMD native format."""
    try:
        import numpy as np
        import shutil

        # Write structure in GPUMD extended XYZ format
        xyz_path = os.path.join(work_dir, "model.xyz")

        # Get structure info
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # Format lattice vector for GPUMD (row-major, 9 components)
        lattice_str = " ".join([f"{x:.10f}" for row in cell for x in row])

        with open(xyz_path, 'w') as f:
            # Header line with atom count
            f.write(f"{len(atoms)}\n")

            # Properties line (GPUMD format)
            f.write(f'energy=0.0 config_type=test pbc="T T T" ')
            f.write(f'Lattice="{lattice_str}" ')
            f.write(f'Properties=species:S:1:pos:R:3:force:R:3\n')

            # Atom lines (species, x, y, z, fx, fy, fz)
            for symbol, pos in zip(symbols, positions):
                f.write(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} ")
                f.write(f"0.0 0.0 0.0\n")  # Initial forces (zeros)

        # Find and copy the potential file
        vibroml_root = os.path.dirname(os.path.abspath(__file__))
        potential_candidates = [
            os.path.join(vibroml_root, "GPUMD", "examples", "nep_train", "nep.txt"),
            os.path.join(vibroml_root, "GPUMD", "examples", "nep_prediction", "nep.txt"),
            os.path.join(vibroml_root, "GPUMD", "potentials", "nep", "nep.txt"),
        ]

        potential_src = None
        for candidate in potential_candidates:
            if os.path.exists(candidate):
                potential_src = candidate
                break

        if potential_src is None:
            print(f"✗ Error: No NEP potential file found in:")
            for c in potential_candidates:
                print(f"    {c}")
            return None, None

        # Copy potential to work directory
        potential_dst = os.path.join(work_dir, "nep.txt")
        shutil.copy(potential_src, potential_dst)

        # Create GPUMD run.in file for single-point energy/force calculation
        run_in_path = os.path.join(work_dir, "run.in")

        with open(run_in_path, 'w') as f:
            f.write("# GPUMD single-point calculation\n")
            f.write("potential   nep.txt\n")
            f.write("velocity    1\n")
            f.write("ensemble nve\n")
            f.write("time_step 0\n")
            f.write("dump_force 1\n")
            f.write("run 1\n")

        print(f"✓ Created GPUMD input files in {work_dir}")
        print(f"  - Copied potential from: {potential_src}")
        return xyz_path, run_in_path
    except Exception as e:
        print(f"✗ Failed to write GPUMD input files: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_gpumd_calculation(work_dir, gpumd_binary):
    """Run GPUMD single-point calculation."""
    try:
        # Change to work directory
        original_dir = os.getcwd()
        os.chdir(work_dir)
        
        # Run GPUMD
        print(f"\n» Running GPUMD calculation...")
        result = subprocess.run(
            [gpumd_binary],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        os.chdir(original_dir)
        
        if result.returncode != 0:
            print(f"✗ GPUMD execution failed:")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            return False
        
        print(f"✓ GPUMD calculation completed successfully")
        print(f"  Output:\n{result.stdout}")
        return True
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        print(f"✗ GPUMD calculation timed out")
        return False
    except Exception as e:
        os.chdir(original_dir)
        print(f"✗ GPUMD execution error: {e}")
        return False

def parse_gpumd_output(work_dir):
    """Parse GPUMD output files (force.out)."""
    try:
        force_file = os.path.join(work_dir, "force.out")
        if not os.path.exists(force_file):
            print(f"✗ Force output file not found: {force_file}")
            return None
        
        # Read forces
        forces = []
        with open(force_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        forces.append([float(x) for x in parts[:3]])
        
        if forces:
            print(f"✓ Parsed {len(forces)} force vectors from output")
            print(f"  Sample force (atom 0): {forces[0]}")
            return forces
        else:
            print(f"✗ No forces found in output file")
            return None
    except Exception as e:
        print(f"✗ Failed to parse GPUMD output: {e}")
        return None

def main():
    """Main test routine."""
    print("=" * 70)
    print("GPUMD Calculator Minimal Test")
    print("=" * 70)
    
    # Check GPUMD binary
    gpumd_binary = os.path.join(os.path.dirname(__file__), "GPUMD", "src", "gpumd")
    if not os.path.exists(gpumd_binary):
        print(f"✗ GPUMD binary not found at: {gpumd_binary}")
        print(f"  Please compile GPUMD first: cd GPUMD/src && make")
        sys.exit(1)
    print(f"✓ Found GPUMD binary: {gpumd_binary}")
    
    # Create test structure
    print("\n» Creating test structure...")
    atoms = create_test_structure()
    if atoms is None:
        sys.exit(1)
    
    # Create temporary work directory
    work_dir = tempfile.mkdtemp(prefix="gpumd_test_")
    print(f"\n» Using work directory: {work_dir}")
    
    try:
        # Write GPUMD input files
        print("\n» Writing GPUMD input files...")
        xyz_path, run_in_path = write_gpumd_input(work_dir, atoms)
        if xyz_path is None:
            sys.exit(1)
        
        # Run GPUMD calculation
        success = run_gpumd_calculation(work_dir, gpumd_binary)
        if not success:
            sys.exit(1)
        
        # Parse output
        print("\n» Parsing GPUMD output...")
        forces = parse_gpumd_output(work_dir)
        if forces is None:
            sys.exit(1)
        
        # Validation
        print("\n» Validating results...")
        if len(forces) == len(atoms):
            print(f"✓ Force count matches atom count: {len(forces)}")
        else:
            print(f"✗ Force count mismatch: {len(forces)} vs {len(atoms)}")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓✓✓ GPUMD CALCULATOR TEST PASSED ✓✓✓")
        print("=" * 70)
        
    finally:
        # Cleanup
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"\n✓ Cleaned up work directory")

if __name__ == "__main__":
    main()

