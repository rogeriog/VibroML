#!/usr/bin/env python3
"""
Test script for the new band.yaml input feature.
This script tests the ability to load eigenmode data from an existing band.yaml file
and use it to generate displaced supercells.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# Add the vibroml package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibroml.utils.phonon_utils import load_eigenmode_from_band_yaml
from vibroml.utils.structure_utils import load_structure

def test_load_eigenmode_from_band_yaml():
    """Test loading eigenmode data from an existing band.yaml file."""
    
    # Use the existing band.yaml file
    band_yaml_path = "other/simple_cubic_band.yaml"
    
    if not os.path.exists(band_yaml_path):
        print(f"Error: Test band.yaml file not found at {band_yaml_path}")
        return False
    
    print(f"Testing eigenmode loading from: {band_yaml_path}")
    
    # Test loading the first mode at Gamma point (0,0,0)
    target_q_point = np.array([0.0, 0.0, 0.0])
    target_band_idx = 0
    
    print(f"Target q-point: {target_q_point}")
    print(f"Target band index: {target_band_idx}")
    
    # Load the eigenmode
    frequency, eigenvector, lattice, natom = load_eigenmode_from_band_yaml(
        band_yaml_path, target_q_point, target_band_idx
    )
    
    if frequency is None:
        print("Failed to load eigenmode data")
        return False
    
    print(f"Successfully loaded eigenmode:")
    print(f"  Frequency: {frequency:.4f} THz")
    print(f"  Number of atoms: {natom}")
    print(f"  Lattice shape: {lattice.shape}")
    print(f"  Eigenvector shape: {eigenvector.shape}")
    
    # Verify the data makes sense
    if natom != 2:
        print(f"Error: Expected 2 atoms for simple_cubic, got {natom}")
        return False
    
    if eigenvector.shape != (2, 3):
        print(f"Error: Expected eigenvector shape (2, 3), got {eigenvector.shape}")
        return False
    
    if lattice.shape != (3, 3):
        print(f"Error: Expected lattice shape (3, 3), got {lattice.shape}")
        return False
    
    print("✓ Eigenmode loading test passed!")
    return True

def test_command_line_interface():
    """Test the command-line interface with the new band.yaml feature."""
    
    # Use the existing structure file
    cif_path = "test_structures/simple_cubic.cif"
    band_yaml_path = "other/simple_cubic_band.yaml"
    
    if not os.path.exists(cif_path):
        print(f"Error: Test CIF file not found at {cif_path}")
        return False
    
    if not os.path.exists(band_yaml_path):
        print(f"Error: Test band.yaml file not found at {band_yaml_path}")
        return False
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing command-line interface with temporary output: {temp_dir}")
        
        # Construct the command
        cmd = [
            "python", "-m", "vibroml.main",
            "--cif", cif_path,
            "--band_yaml_path", band_yaml_path,
            "--q", "0.0,0.0,0.0",
            "--band_idx", "0",
            "--displacement", "0.1",
            "--engine", "mace",
            "--no-relax",
            "--units", "THz"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Change to temp directory for output
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Run the command
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=original_cwd)
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout[-1000:])  # Last 1000 characters
            if result.stderr:
                print("STDERR:")
                print(result.stderr[-1000:])  # Last 1000 characters
            
            # Check if the command succeeded
            if result.returncode == 0:
                print("✓ Command-line interface test passed!")
                
                # Check if expected output files were created
                expected_files = [
                    "simple_cubic_preloaded_eigenmode_summary.txt"
                ]
                
                for expected_file in expected_files:
                    if os.path.exists(expected_file):
                        print(f"✓ Found expected output file: {expected_file}")
                    else:
                        print(f"⚠ Expected output file not found: {expected_file}")
                
                return True
            else:
                print("✗ Command-line interface test failed!")
                return False
                
        finally:
            os.chdir(original_cwd)

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing new band.yaml input feature")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Load eigenmode from band.yaml
    print("\n" + "-" * 40)
    print("Test 1: Loading eigenmode from band.yaml")
    print("-" * 40)
    if test_load_eigenmode_from_band_yaml():
        tests_passed += 1
    
    # Test 2: Command-line interface
    print("\n" + "-" * 40)
    print("Test 2: Command-line interface")
    print("-" * 40)
    if test_command_line_interface():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())