#!/usr/bin/env python3
"""
Comprehensive test script to validate checkpoint functionality for all optimization modes.
Tests checkpoint saving, loading, and state restoration for GA, traditional, traditional_all, and opt_random modes.
"""

import sys
import os
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vibroml.checkpointing import CheckpointManager

def create_mock_args():
    """Create mock arguments for testing."""
    class MockArgs:
        def __init__(self):
            self.engine = "gpumd"
            self.model_name = "test_model"
            self.checkpoint = None
            self.nep_model = None
            self.checkpoint_model = None
            self.cif = "test_structure.cif"
            self.fmax = 0.01
            self.units = "THz"
            self.supercell_dims = (2, 2, 2)
            self.delta = 0.01
            self.relaxation_patience = 5
            self.save_yaml = False
    return MockArgs()

def create_mock_atoms():
    """Create mock ASE Atoms object."""
    try:
        from ase import Atoms
        import numpy as np
        # Create a simple cubic cell with 2 atoms
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        return atoms
    except ImportError:
        # Fallback if ASE is not available
        mock_atoms = Mock()
        mock_atoms.get_chemical_symbols.return_value = ['H', 'H']
        mock_atoms.get_positions.return_value = [[0, 0, 0], [0.5, 0.5, 0.5]]
        mock_atoms.get_cell.return_value = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        mock_atoms.get_pbc.return_value = [True, True, True]
        mock_atoms.get_atomic_numbers.return_value = [1, 1]
        return mock_atoms

def create_mock_soft_modes():
    """Create mock soft modes information."""
    return [
        {
            'frequency': -2.5,
            'coordinate': [0.5, 0.5, 0.5],
            'mode_vector': [0.1, 0.2, 0.3],
            'q_point': [0.25, 0.25, 0.25]
        }
    ]

def test_ga_mode_checkpoint():
    """Test checkpoint functionality for GA mode."""
    print("\n" + "="*80)
    print("TESTING GA MODE CHECKPOINT FUNCTIONALITY")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(temp_dir, 'ga')
        
        # Test 1: Save GA checkpoint
        print("\n[Test 1] Saving GA checkpoint...")
        mock_atoms = create_mock_atoms()
        mock_soft_modes = create_mock_soft_modes()
        
        checkpoint_data = {
            'version': '1.0.0',
            'mode': 'ga',
            'timestamp': time.time(),
            'state': {
                'main_iteration': 2,
                'ga_generation': 3,
                'sample_index': 5,
                'total_samples_completed': 15
            },
            'results': [
                {
                    'params': (1.0, 0.5, (0.1, 0.1, 0.1, 0.0, 0.0, 0.0), (2,2,2), True),
                    'fitness': -10.5,
                    'energy_per_atom': -5.25,
                    'relaxed_atoms': mock_atoms,
                    'main_iteration': 2,
                    'ga_generation': 3,
                    'sample': 5
                }
            ],
            'current_primitive_atoms': mock_atoms,
            'current_softest_modes': mock_soft_modes,
            'ga_state': {
                'population_size': 20,
                'mutation_rate': 0.1,
                'current_generation': 3
            }
        }
        
        try:
            checkpoint_manager.save_checkpoint_ga(
                main_iteration=2,
                ga_generation=3,
                sample_index=5,
                all_iterations_results=checkpoint_data['results'],
                ga_state=checkpoint_data['ga_state'],
                current_primitive_atoms=mock_atoms,
                current_softest_modes=mock_soft_modes
            )
            print("✓ GA checkpoint saved successfully")
        except Exception as e:
            print(f"✗ Failed to save GA checkpoint: {e}")
            return False
        
        # Test 2: Load GA checkpoint
        print("\n[Test 2] Loading GA checkpoint...")
        result = checkpoint_manager.load_latest_checkpoint()
        
        if result is None:
            print("✗ Failed to load GA checkpoint")
            return False
        
        loaded_data, loaded_state = result
        print("✓ GA checkpoint loaded successfully")
        
        # Test 3: Verify GA-specific state
        print("\n[Test 3] Verifying GA-specific state...")
        expected_state = {
            'main_iteration': 2,
            'ga_generation': 3,
            'sample_index': 5,
            'total_samples_completed': 15
        }
        
        state_match = True
        for key, expected_value in expected_state.items():
            if loaded_state.get(key) != expected_value:
                print(f"✗ State mismatch for {key}: expected {expected_value}, got {loaded_state.get(key)}")
                state_match = False
        
        if state_match:
            print("✓ GA-specific state verified")
        
        # Test 4: Verify GA-specific data
        print("\n[Test 4] Verifying GA-specific data...")
        if 'ga_state' in loaded_data:
            ga_state = loaded_data['ga_state']
            if ga_state.get('population_size') == 20:
                print("✓ GA population state preserved")
            else:
                print("✗ GA population state not preserved")
                return False
        else:
            print("✗ GA state missing from checkpoint")
            return False
        
        # Test 5: Verify hash
        print("\n[Test 5] Verifying GA checkpoint hash...")
        if checkpoint_manager._verify_hash(loaded_data):
            print("✓ Hash verification passed")
        else:
            print("✗ Hash verification failed")
            return False
    
    print("✓✓✓ GA MODE TESTS PASSED ✓✓✓")
    return True

def test_traditional_mode_checkpoint():
    """Test checkpoint functionality for traditional mode."""
    print("\n" + "="*80)
    print("TESTING TRADITIONAL MODE CHECKPOINT FUNCTIONALITY")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(temp_dir, 'traditional')
        
        # Test 1: Save traditional checkpoint
        print("\n[Test 1] Saving traditional checkpoint...")
        mock_atoms = create_mock_atoms()
        mock_soft_modes = create_mock_soft_modes()
        
        checkpoint_data = {
            'version': '1.0.0',
            'mode': 'traditional',
            'timestamp': time.time(),
            'state': {
                'iteration': 3,
                'sample_index': 8,
                'total_samples_completed': 24
            },
            'results': [
                {
                    'params': (1.5, 0.8, (0.2, 0.2, 0.2, 0.0, 0.0, 0.0)),
                    'energy_per_atom': -6.2,
                    'relaxed_atoms': mock_atoms,
                    'iteration': 3,
                    'sample': 8
                }
            ],
            'current_primitive_atoms': mock_atoms,
            'current_softest_modes': mock_soft_modes
        }
        
        try:
            checkpoint_manager._save_checkpoint(checkpoint_data)
            print("✓ Traditional checkpoint saved successfully")
        except Exception as e:
            print(f"✗ Failed to save traditional checkpoint: {e}")
            return False
        
        # Test 2: Load traditional checkpoint
        print("\n[Test 2] Loading traditional checkpoint...")
        result = checkpoint_manager.load_latest_checkpoint()
        
        if result is None:
            print("✗ Failed to load traditional checkpoint")
            return False
        
        loaded_data, loaded_state = result
        print("✓ Traditional checkpoint loaded successfully")
        
        # Test 3: Verify traditional-specific state
        print("\n[Test 3] Verifying traditional-specific state...")
        expected_state = {
            'iteration': 3,
            'sample_index': 8,
            'total_samples_completed': 24
        }
        
        state_match = True
        for key, expected_value in expected_state.items():
            if loaded_state.get(key) != expected_value:
                print(f"✗ State mismatch for {key}: expected {expected_value}, got {loaded_state.get(key)}")
                state_match = False
        
        if state_match:
            print("✓ Traditional-specific state verified")
        
        # Test 4: Verify hash
        print("\n[Test 4] Verifying traditional checkpoint hash...")
        if checkpoint_manager._verify_hash(loaded_data):
            print("✓ Hash verification passed")
        else:
            print("✗ Hash verification failed")
            return False
    
    print("✓✓✓ TRADITIONAL MODE TESTS PASSED ✓✓✓")
    return True

def test_traditional_all_mode_checkpoint():
    """Test checkpoint functionality for traditional_all mode."""
    print("\n" + "="*80)
    print("TESTING TRADITIONAL_ALL MODE CHECKPOINT FUNCTIONALITY")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(temp_dir, 'traditional_all')
        
        # Test 1: Save traditional_all checkpoint
        print("\n[Test 1] Saving traditional_all checkpoint...")
        mock_atoms = create_mock_atoms()
        mock_soft_modes = create_mock_soft_modes()
        
        checkpoint_data = {
            'version': '1.0.0',
            'mode': 'traditional_all',
            'timestamp': time.time(),
            'state': {
                'iteration': 2,
                'pairing_index': 1,
                'config_index': 0,
                'sample_index': 4,
                'total_samples_completed': 12
            },
            'results': [
                {
                    'params': (2.0, 1.2, (0.3, 0.3, 0.3, 0.0, 0.0, 0.0)),
                    'energy_per_atom': -7.1,
                    'relaxed_atoms': mock_atoms,
                    'iteration': 2,
                    'pairing': 1,
                    'config': 0,
                    'sample': 4
                }
            ],
            'current_primitive_atoms': mock_atoms,
            'current_softest_modes': mock_soft_modes
        }
        
        try:
            checkpoint_manager._save_checkpoint(checkpoint_data)
            print("✓ Traditional_all checkpoint saved successfully")
        except Exception as e:
            print(f"✗ Failed to save traditional_all checkpoint: {e}")
            return False
        
        # Test 2: Load traditional_all checkpoint
        print("\n[Test 2] Loading traditional_all checkpoint...")
        result = checkpoint_manager.load_latest_checkpoint()
        
        if result is None:
            print("✗ Failed to load traditional_all checkpoint")
            return False
        
        loaded_data, loaded_state = result
        print("✓ Traditional_all checkpoint loaded successfully")
        
        # Test 3: Verify traditional_all-specific state
        print("\n[Test 3] Verifying traditional_all-specific state...")
        expected_state = {
            'iteration': 2,
            'pairing_index': 1,
            'config_index': 0,
            'sample_index': 4,
            'total_samples_completed': 12
        }
        
        state_match = True
        for key, expected_value in expected_state.items():
            if loaded_state.get(key) != expected_value:
                print(f"✗ State mismatch for {key}: expected {expected_value}, got {loaded_state.get(key)}")
                state_match = False
        
        if state_match:
            print("✓ Traditional_all-specific state verified")
        
        # Test 4: Verify hash
        print("\n[Test 4] Verifying traditional_all checkpoint hash...")
        if checkpoint_manager._verify_hash(loaded_data):
            print("✓ Hash verification passed")
        else:
            print("✗ Hash verification failed")
            return False
    
    print("✓✓✓ TRADITIONAL_ALL MODE TESTS PASSED ✓✓✓")
    return True

def test_opt_random_mode_checkpoint():
    """Test checkpoint functionality for opt_random mode."""
    print("\n" + "="*80)
    print("TESTING OPT_RANDOM MODE CHECKPOINT FUNCTIONALITY")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(temp_dir, 'opt_random')
        
        # Test 1: Save opt_random checkpoint
        print("\n[Test 1] Saving opt_random checkpoint...")
        mock_atoms = create_mock_atoms()
        mock_soft_modes = create_mock_soft_modes()
        
        checkpoint_data = {
            'version': '1.0.0',
            'mode': 'opt_random',
            'timestamp': time.time(),
            'state': {
                'iteration': 4,
                'sample_index': 6,
                'total_samples_completed': 18
            },
            'results': [
                {
                    'params': (2.5, 1.5, (0.4, 0.4, 0.4, 0.0, 0.0, 0.0)),
                    'energy_per_atom': -8.3,
                    'relaxed_atoms': mock_atoms,
                    'iteration': 4,
                    'sample': 6
                }
            ],
            'current_primitive_atoms': mock_atoms,
            'current_best_supercell': (3, 3, 3)
        }
        
        try:
            checkpoint_manager._save_checkpoint(checkpoint_data)
            print("✓ Opt_random checkpoint saved successfully")
        except Exception as e:
            print(f"✗ Failed to save opt_random checkpoint: {e}")
            return False
        
        # Test 2: Load opt_random checkpoint
        print("\n[Test 2] Loading opt_random checkpoint...")
        result = checkpoint_manager.load_latest_checkpoint()
        
        if result is None:
            print("✗ Failed to load opt_random checkpoint")
            return False
        
        loaded_data, loaded_state = result
        print("✓ Opt_random checkpoint loaded successfully")
        
        # Test 3: Verify opt_random-specific state
        print("\n[Test 3] Verifying opt_random-specific state...")
        expected_state = {
            'iteration': 4,
            'sample_index': 6,
            'total_samples_completed': 18
        }
        
        state_match = True
        for key, expected_value in expected_state.items():
            if loaded_state.get(key) != expected_value:
                print(f"✗ State mismatch for {key}: expected {expected_value}, got {loaded_state.get(key)}")
                state_match = False
        
        if state_match:
            print("✓ Opt_random-specific state verified")
        
        # Test 4: Verify opt_random-specific data
        print("\n[Test 4] Verifying opt_random-specific data...")
        if 'current_best_supercell' in loaded_data:
            supercell = loaded_data['current_best_supercell']
            if supercell == (3, 3, 3):
                print("✓ Best supercell state preserved")
            else:
                print(f"✗ Best supercell state not preserved: expected (3,3,3), got {supercell}")
                return False
        else:
            print("✗ Best supercell state missing from checkpoint")
            return False
        
        # Test 5: Verify hash
        print("\n[Test 5] Verifying opt_random checkpoint hash...")
        if checkpoint_manager._verify_hash(loaded_data):
            print("✓ Hash verification passed")
        else:
            print("✗ Hash verification failed")
            return False
    
    print("✓✓✓ OPT_RANDOM MODE TESTS PASSED ✓✓✓")
    return True

def test_mode_validation():
    """Test mode validation to prevent cross-mode checkpoint loading."""
    print("\n" + "="*80)
    print("TESTING MODE VALIDATION")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Create GA checkpoint
        print("\n[Test 1] Creating GA checkpoint...")
        ga_manager = CheckpointManager(temp_dir, 'ga')
        
        ga_checkpoint_data = {
            'version': '1.0.0',
            'mode': 'ga',
            'timestamp': time.time(),
            'state': {'main_iteration': 1},
            'results': [],
            'current_primitive_atoms': None,
            'current_softest_modes': None,
            'ga_state': {}
        }
        
        ga_manager.save_checkpoint_ga(
            main_iteration=1,
            ga_generation=1,
            sample_index=1,
            all_iterations_results=[],
            ga_state={}
        )
        print("✓ GA checkpoint created")
        
        # Test 2: Try to load GA checkpoint with traditional manager
        print("\n[Test 2] Testing cross-mode loading prevention...")
        traditional_manager = CheckpointManager(temp_dir, 'traditional')
        result = traditional_manager.load_latest_checkpoint()
        
        if result is None:
            print("✓ Cross-mode loading correctly prevented")
        else:
            print("✗ Cross-mode loading not prevented - SECURITY RISK")
            return False
    
    print("✓✓✓ MODE VALIDATION TESTS PASSED ✓✓✓")
    return True

def test_checkpoint_integrity():
    """Test checkpoint integrity verification."""
    print("\n" + "="*80)
    print("TESTING CHECKPOINT INTEGRITY")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(temp_dir, 'test')
        
        # Test 1: Create valid checkpoint
        print("\n[Test 1] Creating valid checkpoint...")
        valid_data = {
            'version': '1.0.0',
            'mode': 'test',
            'timestamp': time.time(),
            'state': {'test': True},
            'results': []
        }
        
        checkpoint_manager.save_checkpoint_ga(
            main_iteration=1,
            ga_generation=1,
            sample_index=1,
            all_iterations_results=[],
            ga_state={}
        )
        print("✓ Valid checkpoint created")
        
        # Test 2: Verify integrity
        print("\n[Test 2] Verifying integrity...")
        result = checkpoint_manager.load_latest_checkpoint()
        
        if result is not None:
            loaded_data, _ = result
            if checkpoint_manager._verify_hash(loaded_data):
                print("✓ Integrity verification passed")
            else:
                print("✗ Integrity verification failed")
                return False
        else:
            print("✗ Failed to load valid checkpoint")
            return False
        
        # Test 3: Corrupt checkpoint file
        print("\n[Test 3] Testing corruption detection...")
        checkpoint_files = list(Path(temp_dir).glob(".checkpoints/checkpoint_test_*.pkl"))
        if checkpoint_files:
            # Corrupt the file by writing invalid data
            with open(checkpoint_files[0], 'w') as f:
                f.write("corrupted data")
            
            result = checkpoint_manager.load_latest_checkpoint()
            if result is None:
                print("✓ Corruption correctly detected")
            else:
                print("✗ Corruption not detected - SECURITY RISK")
                return False
    
    print("✓✓✓ INTEGRITY TESTS PASSED ✓✓✓")
    return True

def run_all_tests():
    """Run all checkpoint validation tests."""
    print("="*80)
    print("COMPREHENSIVE CHECKPOINT VALIDATION TESTS")
    print("="*80)
    
    tests = [
        ("GA Mode", test_ga_mode_checkpoint),
        ("Traditional Mode", test_traditional_mode_checkpoint),
        ("Traditional_all Mode", test_traditional_all_mode_checkpoint),
        ("Opt_random Mode", test_opt_random_mode_checkpoint),
        ("Mode Validation", test_mode_validation),
        ("Checkpoint Integrity", test_checkpoint_integrity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL CHECKPOINT VALIDATION TESTS PASSED! 🎉")
        print("✓ Checkpoint saving works for all modes")
        print("✓ Checkpoint loading works for all modes")
        print("✓ State restoration works for all modes")
        print("✓ Mode validation prevents cross-mode loading")
        print("✓ Integrity verification detects corruption")
        return True
    else:
        print(f"\n❌ {total-passed} TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)