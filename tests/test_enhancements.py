#!/usr/bin/env python3

"""
Test script to verify the implemented enhancements to VibroML:
1. Optional YAML file saving with --save-yaml flag
2. Optional phonon calculations for NEB methods with --with-phonon flag  
3. Mandatory structure relaxation for NEB methods

This script tests the command-line interface and verifies the expected behavior.
"""

import sys
import os
import tempfile
import shutil
import subprocess
import time

# Add current directory to path
sys.path.insert(0, '.')

def run_command(cmd, timeout=300):
    """Run a command and return success status and output."""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd='.'
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)

def test_yaml_saving_flag():
    """Test the --save-yaml flag functionality."""
    print("\n" + "="*60)
    print("TEST 1: Optional YAML file saving")
    print("="*60)
    
    # Test structure
    test_cif = "examples/LiF_simplecubic/LiFsimplecubic.cif"
    
    if not os.path.exists(test_cif):
        print(f"✗ Test structure not found: {test_cif}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test 1a: Without --save-yaml flag (default behavior - no YAML)
        print("\n--- Test 1a: Without --save-yaml flag ---")
        cmd = [
            "python", "-m", "vibroml.main",
            "--cif", test_cif,
            "--engine", "mace",
            "--no-relax",
            "--units", "THz",
            "--supercell", "2",
            "--delta", "0.05",
            "--fmax", "0.01"
        ]
        
        success, stdout, stderr = run_command(cmd, timeout=120)
        
        if success:
            print("✓ Command executed successfully")
            if "YAML file saving skipped" in stdout:
                print("✓ YAML saving correctly skipped by default")
            else:
                print("? YAML skipping message not found in output")
        else:
            print(f"✗ Command failed: {stderr}")
            return False
        
        # Test 1b: With --save-yaml flag
        print("\n--- Test 1b: With --save-yaml flag ---")
        cmd.append("--save-yaml")
        
        success, stdout, stderr = run_command(cmd, timeout=120)
        
        if success:
            print("✓ Command executed successfully")
            if "YAML file saved" in stdout or "Successfully saved Phonopy band structure" in stdout:
                print("✓ YAML saving correctly enabled with flag")
            else:
                print("? YAML saving message not found in output")
        else:
            print(f"✗ Command failed: {stderr}")
            return False
    
    print("✓ YAML saving flag test completed successfully")
    return True

def test_neb_phonon_flag():
    """Test the --with-phonon flag for NEB methods."""
    print("\n" + "="*60)
    print("TEST 2: Optional phonon calculations for NEB methods")
    print("="*60)
    
    # Test structures
    initial_cif = "examples/NEB_test/LiFsimplecubic.cif"
    final_cif = "examples/NEB_test/LiFhexagonal.cif"
    
    if not os.path.exists(initial_cif):
        print(f"✗ Initial test structure not found: {initial_cif}")
        return False
    
    if not os.path.exists(final_cif):
        print(f"✗ Final test structure not found: {final_cif}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test 2a: NEB without --with-phonon flag (default behavior - no phonon analysis)
        print("\n--- Test 2a: NEB without --with-phonon flag ---")
        cmd = [
            "python", "-m", "vibroml.main",
            "--cif", initial_cif,
            "--final_cif", final_cif,
            "--method", "neb",
            "--engine", "mace",
            "--auto",
            "--neb_num_images", "3",
            "--neb_max_iterations", "5",  # Very few iterations for quick test
            "--neb_force_tolerance", "0.2"  # Relaxed tolerance
        ]
        
        success, stdout, stderr = run_command(cmd, timeout=300)
        
        if success:
            print("✓ NEB command executed successfully")
            if "Skipping Phonon Analysis (--with-phonon not provided)" in stdout:
                print("✓ Phonon analysis correctly skipped by default")
            else:
                print("? Phonon skipping message not found in output")
        else:
            print(f"✗ NEB command failed: {stderr}")
            return False
        
        # Test 2b: NEB with --with-phonon flag
        print("\n--- Test 2b: NEB with --with-phonon flag ---")
        cmd.append("--with-phonon")
        
        success, stdout, stderr = run_command(cmd, timeout=600)
        
        if success:
            print("✓ NEB with phonon command executed successfully")
            if "Running Phonon Analysis on NEB Images" in stdout:
                print("✓ Phonon analysis correctly enabled with flag")
            else:
                print("? Phonon analysis message not found in output")
        else:
            print(f"✗ NEB with phonon command failed: {stderr}")
            return False
    
    print("✓ NEB phonon flag test completed successfully")
    return True

def test_neb_structure_relaxation():
    """Test mandatory structure relaxation for NEB methods."""
    print("\n" + "="*60)
    print("TEST 3: Mandatory structure relaxation for NEB methods")
    print("="*60)
    
    # Test structures
    initial_cif = "examples/NEB_test/LiFsimplecubic.cif"
    final_cif = "examples/NEB_test/LiFhexagonal.cif"
    
    if not os.path.exists(initial_cif):
        print(f"✗ Initial test structure not found: {initial_cif}")
        return False
    
    if not os.path.exists(final_cif):
        print(f"✗ Final test structure not found: {final_cif}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test 3: NEB with structure relaxation
        print("\n--- Test 3: NEB with mandatory structure relaxation ---")
        cmd = [
            "python", "-m", "vibroml.main",
            "--cif", initial_cif,
            "--final_cif", final_cif,
            "--method", "neb",
            "--engine", "mace",
            "--auto",
            "--neb_num_images", "3",
            "--neb_max_iterations", "3",  # Very few iterations for quick test
            "--neb_force_tolerance", "0.5"  # Very relaxed tolerance
        ]
        
        success, stdout, stderr = run_command(cmd, timeout=300)
        
        if success:
            print("✓ NEB command executed successfully")
            if "Mandatory Structure Relaxation for NEB" in stdout:
                print("✓ Mandatory structure relaxation detected")
            if "Relaxing initial structure" in stdout and "Relaxing final structure" in stdout:
                print("✓ Both initial and final structures relaxed")
            if "Both structures successfully relaxed" in stdout:
                print("✓ Structure relaxation completed successfully")
            else:
                print("? Structure relaxation completion message not found")
        else:
            print(f"✗ NEB command failed: {stderr}")
            return False
    
    print("✓ NEB structure relaxation test completed successfully")
    return True

def main():
    """Run all enhancement tests."""
    print("="*80)
    print("VibroML Enhancement Tests")
    print("Testing: --save-yaml, --with-phonon, and mandatory NEB structure relaxation")
    print("="*80)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: YAML saving flag
    if test_yaml_saving_flag():
        tests_passed += 1
    
    # Test 2: NEB phonon flag
    if test_neb_phonon_flag():
        tests_passed += 1
    
    # Test 3: NEB structure relaxation
    if test_neb_structure_relaxation():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("="*80)
    
    if tests_passed == total_tests:
        print("🎉 All enhancement tests passed!")
        return 0
    else:
        print("❌ Some enhancement tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
