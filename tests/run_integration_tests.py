#!/usr/bin/env python3
"""
Test runner for VibroML integration tests.

This script runs the comprehensive integration tests for VibroML,
including tests for anisotropic supercells, auto modes, displaced
supercell generation, and units functionality.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Test configuration
CONDA_ENV_PATH = "/globalscratch/ucl/modl/rgouvea/vibroml_env/"
TEST_DIR = Path(__file__).parent


def run_command(cmd, timeout=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def check_conda_environment():
    """Check if the conda environment is available."""
    print("Checking conda environment...")
    
    cmd = ["conda", "run", "-p", CONDA_ENV_PATH, "python", "-c", "import vibroml; print('VibroML available')"]
    result = run_command(cmd, timeout=30)
    
    if result is None or result.returncode != 0:
        print(f"ERROR: Conda environment not available or VibroML not installed")
        print(f"Expected environment path: {CONDA_ENV_PATH}")
        if result:
            print(f"Error output: {result.stderr}")
        return False
    
    print("✓ Conda environment and VibroML are available")
    return True


def run_unit_tests():
    """Run the unit tests for anisotropic supercell functionality."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    test_files = [
        "test_comprehensive_integration.py::TestAnisotropicSupercellConfiguration",
    ]
    
    for test_file in test_files:
        print(f"\nRunning unit tests in {test_file}...")
        
        cmd = ["python", "-m", "pytest", str(TEST_DIR / test_file), "-v", "-x"]
        result = run_command(cmd, timeout=120)
        
        if result is None or result.returncode != 0:
            print(f"❌ Unit tests failed in {test_file}")
            if result:
                print(f"Error output: {result.stderr}")
                print(f"Standard output: {result.stdout}")
            return False
        else:
            print(f"✓ Unit tests passed in {test_file}")
    
    return True


def run_integration_tests(test_type="all", quick=False):
    """Run the integration tests."""
    print("\n" + "="*60)
    print(f"RUNNING INTEGRATION TESTS ({test_type.upper()})")
    print("="*60)
    
    # Define test categories
    test_categories = {  
        "basic": [  
            "test_comprehensive_integration.py::TestVibroMLIntegration::test_basic_phonon_calculation",  
            "test_auto_traditional.py::TestAutoTraditionalMode::test_anisotropic_supercells",  
        ],  
        "auto": [  
            "test_auto_traditional.py::TestAutoTraditionalMode::test_basic_auto_mode",  
            "test_auto_ga.py::TestAutoGAMode::test_auto_ga_basic_run_and_workflow_structure",  
        ],  
        "displaced": [  
            "test_displaced_supercell.py::TestDisplacedSupercellGeneration::test_displaced_supercell_comprehensive",  
        ],  
        "band_yaml": [  
            "test_band_yaml.py::test_load_eigenmode_from_band_yaml",  
            "test_band_yaml.py::test_command_line_interface",  
        ],  
        "units": [  
            "test_units_functionality.py::TestUnitsConversion::test_units_comprehensive",  
        ],  
        "all": []  # Will be populated below  
    }
    
    # Populate "all" category
    for category_tests in test_categories.values():
        if category_tests:  # Skip the empty "all" list
            test_categories["all"].extend(category_tests)
    
    # Add quick mode tests
    if quick:
        test_categories["quick"] = [
            "test_comprehensive_integration.py::TestVibroMLIntegration::test_basic_phonon_calculation",
            "test_units_functionality.py::TestUnitsConversion::test_units_thz_basic",
        ]
    
    # Select tests to run
    if test_type not in test_categories:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {list(test_categories.keys())}")
        return False
    
    tests_to_run = test_categories[test_type]
    
    if not tests_to_run:
        print("No tests selected to run")
        return False
    
    print(f"Running {len(tests_to_run)} integration tests...")
    
    # Run each test
    passed = 0
    failed = 0
    
    for test in tests_to_run:
        print(f"\n{'='*40}")
        print(f"Running: {test}")
        print(f"{'='*40}")
        
        cmd = ["python", "-m", "pytest", str(TEST_DIR / test), "-v", "-s", "--tb=short"]
        
        # Set timeout based on test type
        timeout = 180 if quick else 600  # 3 minutes for quick, 10 minutes for full
        
        start_time = time.time()
        result = run_command(cmd, timeout=timeout)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result is None:
            print(f"❌ Test timed out after {timeout} seconds: {test}")
            failed += 1
        elif result.returncode == 0:
            print(f"✓ Test passed ({duration:.1f}s): {test}")
            passed += 1
        else:
            print(f"❌ Test failed ({duration:.1f}s): {test}")
            print(f"Error output: {result.stderr}")
            print(f"Standard output: {result.stdout}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    return failed == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run VibroML integration tests")
    parser.add_argument(
        "--type", 
        choices=["all", "basic", "auto", "displaced", "units", "quick"],
        default="quick",
        help="Type of tests to run (default: quick)"
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip conda environment check"
    )
    parser.add_argument(
        "--unit-tests-only",
        action="store_true",
        help="Run only unit tests (no integration tests)"
    )
    
    args = parser.parse_args()
    
    print("VibroML Integration Test Runner")
    print("="*60)
    
    # Check conda environment
    if not args.skip_env_check:
        if not check_conda_environment():
            print("\nERROR: Environment check failed. Use --skip-env-check to bypass.")
            return 1
    
    # Run unit tests
    if not run_unit_tests():
        print("\nERROR: Unit tests failed.")
        return 1
    
    # Run integration tests (unless unit-tests-only)
    if not args.unit_tests_only:
        if not run_integration_tests(args.type, quick=(args.type == "quick")):
            print("\nERROR: Integration tests failed.")
            return 1
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
