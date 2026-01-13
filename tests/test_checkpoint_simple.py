#!/usr/bin/env python3
"""
Simple checkpoint validation test for all optimization modes.
Tests basic checkpoint saving and loading functionality.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vibroml.checkpointing import CheckpointManager

def test_ga_mode():
    """Test GA mode checkpoint functionality."""
    print("\n" + "="*60)
    print("TESTING GA MODE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir, 'ga')
        
        # Test saving
        try:
            result_path = manager.save_checkpoint_ga(
                main_iteration=2,
                ga_generation=3,
                sample_index=5,
                all_iterations_results=[],
                ga_state={'population_size': 20}
            )
            print(f"✓ GA checkpoint saved: {Path(result_path).name}")
        except Exception as e:
            print(f"✗ GA save failed: {e}")
            return False
        
        # Test loading
        try:
            result = manager.load_latest_checkpoint()
            if result:
                data, state = result
                print(f"✓ GA checkpoint loaded successfully")
                print(f"  State: {state}")
                return True
            else:
                print("✗ GA load returned None")
                return False
        except Exception as e:
            print(f"✗ GA load failed: {e}")
            return False

def test_traditional_mode():
    """Test traditional mode checkpoint functionality."""
    print("\n" + "="*60)
    print("TESTING TRADITIONAL MODE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir, 'traditional')
        
        # Test saving
        try:
            result_path = manager.save_checkpoint_traditional(
                iteration=3,
                sample_index=8,
                all_iterations_results=[]
            )
            print(f"✓ Traditional checkpoint saved: {Path(result_path).name}")
        except Exception as e:
            print(f"✗ Traditional save failed: {e}")
            return False
        
        # Test loading
        try:
            result = manager.load_latest_checkpoint()
            if result:
                data, state = result
                print(f"✓ Traditional checkpoint loaded successfully")
                print(f"  State: {state}")
                return True
            else:
                print("✗ Traditional load returned None")
                return False
        except Exception as e:
            print(f"✗ Traditional load failed: {e}")
            return False

def test_traditional_all_mode():
    """Test traditional_all mode checkpoint functionality."""
    print("\n" + "="*60)
    print("TESTING TRADITIONAL_ALL MODE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir, 'traditional_all')
        
        # Test saving
        try:
            result_path = manager.save_checkpoint_traditional_all(
                iteration=2,
                pairing_index=1,
                config_index=0,
                sample_index=4,
                all_iterations_results=[]
            )
            print(f"✓ Traditional_all checkpoint saved: {Path(result_path).name}")
        except Exception as e:
            print(f"✗ Traditional_all save failed: {e}")
            return False
        
        # Test loading
        try:
            result = manager.load_latest_checkpoint()
            if result:
                data, state = result
                print(f"✓ Traditional_all checkpoint loaded successfully")
                print(f"  State: {state}")
                return True
            else:
                print("✗ Traditional_all load returned None")
                return False
        except Exception as e:
            print(f"✗ Traditional_all load failed: {e}")
            return False

def test_opt_random_mode():
    """Test opt_random mode checkpoint functionality."""
    print("\n" + "="*60)
    print("TESTING OPT_RANDOM MODE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir, 'opt_random')
        
        # Test saving
        try:
            result_path = manager.save_checkpoint_opt_random(
                iteration=4,
                sample_index=6,
                all_iterations_results=[],
                current_best_supercell=(3, 3, 3)
            )
            print(f"✓ Opt_random checkpoint saved: {Path(result_path).name}")
        except Exception as e:
            print(f"✗ Opt_random save failed: {e}")
            return False
        
        # Test loading
        try:
            result = manager.load_latest_checkpoint()
            if result:
                data, state = result
                print(f"✓ Opt_random checkpoint loaded successfully")
                print(f"  State: {state}")
                return True
            else:
                print("✗ Opt_random load returned None")
                return False
        except Exception as e:
            print(f"✗ Opt_random load failed: {e}")
            return False

def test_mode_validation():
    """Test mode validation."""
    print("\n" + "="*60)
    print("TESTING MODE VALIDATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create GA checkpoint
        ga_manager = CheckpointManager(temp_dir, 'ga')
        ga_manager.save_checkpoint_ga(
            main_iteration=1,
            ga_generation=1,
            sample_index=1,
            all_iterations_results=[],
            ga_state={}
        )
        
        # Try to load with traditional manager
        traditional_manager = CheckpointManager(temp_dir, 'traditional')
        result = traditional_manager.load_latest_checkpoint()
        
        if result is None:
            print("✓ Cross-mode loading correctly prevented")
            return True
        else:
            print("✗ Cross-mode loading not prevented")
            return False

def run_all_tests():
    """Run all checkpoint tests."""
    print("="*80)
    print("CHECKPOINT VALIDATION TESTS")
    print("="*80)
    
    tests = [
        ("GA Mode", test_ga_mode),
        ("Traditional Mode", test_traditional_mode),
        ("Traditional_all Mode", test_traditional_all_mode),
        ("Opt_random Mode", test_opt_random_mode),
        ("Mode Validation", test_mode_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL CHECKPOINT TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ {total-passed} TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)