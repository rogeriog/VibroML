#!/usr/bin/env python
"""
Minimal test script to verify VibroML works with eSEN-30m-omat calculator.
Tests phonon band structure calculation on a simple cubic structure.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

def test_vibroml_esen():
    """Test VibroML with eSEN calculator."""
    print("=" * 70)
    print("VibroML + eSEN-30m-omat Integration Test")
    print("=" * 70)
    
    # Step 1: Import VibroML utilities
    print("\n[1/5] Importing VibroML utilities...")
    try:
        from vibroml.utils.structure_utils import load_structure, initialize_calculator
        from vibroml.utils.phonon_utils import run_single_phonon_analysis
        print("✓ VibroML utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VibroML utilities: {e}")
        return False
    
    # Step 2: Load test structure
    print("\n[2/5] Loading test structure...")
    try:
        struct_path = "tests/test_structures/simple_cubic.cif"
        if not os.path.exists(struct_path):
            print(f"✗ Test structure not found at {struct_path}")
            return False
        
        struct, atoms = load_structure(struct_path)
        if atoms is None:
            print("✗ Failed to load structure")
            return False
        
        print(f"✓ Loaded structure: {atoms.get_chemical_formula()}")
        print(f"  Atoms: {len(atoms)}, Volume: {atoms.get_volume():.2f} Ų")
    except Exception as e:
        print(f"✗ Error loading structure: {e}")
        return False
    
    # Step 3: Initialize eSEN calculator
    print("\n[3/5] Initializing eSEN-30m-omat calculator...")
    try:
        # Use the automatic path resolution from initialize_calculator
        # by passing checkpoint_path=None
        print(f"  Loading checkpoint from default location...")
        calculator = initialize_calculator(
            engine="esen",
            checkpoint_path=None
        )
        if calculator is None:
            print("✗ Failed to initialize eSEN calculator")
            return False

        print(f"✓ eSEN calculator initialized")

        # Quick test: calculate energy
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        print(f"  Test energy: {energy:.4f} eV")
    except ImportError as e:
        print(f"✗ fairchem-core not available: {e}")
        print("  Make sure to activate the esen_env environment:")
        print("  conda activate /auto/globalscratch/users/r/g/rgouvea/esen_env")
        return False
    except SystemExit as e:
        print(f"✗ Initialization error: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to initialize calculator: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Run phonon calculation
    print("\n[4/5] Running phonon band structure calculation...")
    try:
        # Create temporary output directory
        output_dir = tempfile.mkdtemp(prefix="vibroml_esen_test_")
        print(f"  Output directory: {output_dir}")
        
        # Run phonon analysis with minimal settings for speed
        soft_modes, bsmin, elapsed_time, tracking_data = run_single_phonon_analysis(
            atoms=atoms,
            calculator=calculator,
            engine="esen",
            units="THz",
            supercell_dims=(2, 2, 2),  # Small supercell for speed
            delta=0.01,
            fmax=0.1,
            output_dir=output_dir,
            prefix="esen_test",
            phonon_path_npoints=50,  # Reduced for speed
            phonon_dos_grid=(20, 20, 20),  # Reduced for speed
            num_modes_to_return=2
        )
        
        print(f"✓ Phonon calculation completed in {elapsed_time:.2f} seconds")
        print(f"  Softest mode frequency: {bsmin:.4f} THz")
        print(f"  Number of soft modes found: {len(soft_modes)}")
        
        if soft_modes:
            for i, mode in enumerate(soft_modes[:2]):
                freq = mode.get('frequency', 'N/A')
                print(f"    Mode {i+1}: {freq} THz")
        
        # Cleanup
        shutil.rmtree(output_dir)
        
    except Exception as e:
        print(f"✗ Phonon calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify results
    print("\n[5/5] Verifying results...")
    try:
        if bsmin is None or bsmin > 100:  # Sanity check
            print(f"✗ Unreasonable frequency value: {bsmin}")
            return False
        
        if len(soft_modes) == 0:
            print("⚠ Warning: No soft modes found (may be normal for this structure)")
        
        print("✓ Results are reasonable")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print()
    success = test_vibroml_esen()
    print("\n" + "=" * 70)
    if success:
        print("✓ ALL TESTS PASSED - VibroML + eSEN integration is working!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("✗ TESTS FAILED - See errors above")
        print("=" * 70)
        sys.exit(1)

