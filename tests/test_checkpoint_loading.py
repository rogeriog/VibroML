#!/usr/bin/env python3
"""
Test script to verify checkpoint loading functionality.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vibroml.checkpointing import CheckpointManager

def test_checkpoint_loading():
    """Test loading reconstructed checkpoints."""
    
    print("=" * 80)
    print("Testing Checkpoint Loading")
    print("=" * 80)
    
    run_dir = "examples/LiF_simplecubic/LiFsimplecubic_UMA_GA_phonon_output_20251201-005549"
    checkpoint_manager = CheckpointManager(run_dir, "ga")
    
    # Test 1: Load latest checkpoint
    print("\n[Test 1] Loading latest checkpoint...")
    result = checkpoint_manager.load_latest_checkpoint()
    
    if result is None:
        print("✗ Failed to load checkpoint")
        return False
    
    checkpoint_data, state = result
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Mode: {checkpoint_data.get('mode')}")
    print(f"  State: {state}")
    print(f"  Total samples: {len(checkpoint_data.get('results', []))}")
    
    # Test 2: Verify hash
    print("\n[Test 2] Verifying checkpoint hash...")
    if checkpoint_manager._verify_hash(checkpoint_data):
        print("✓ Hash verification passed")
    else:
        print("✗ Hash verification failed")
        return False
    
    # Test 3: Check results structure
    print("\n[Test 3] Checking results structure...")
    results = checkpoint_data.get('results', [])
    if results:
        first_result = results[0]
        print(f"✓ Found {len(results)} results")
        print(f"  First result keys: {list(first_result.keys())}")
        
        # Check if atoms were deserialized
        if 'relaxed_atoms' in first_result and first_result['relaxed_atoms'] is not None:
            atoms = first_result['relaxed_atoms']
            print(f"  ✓ Atoms deserialized: {len(atoms)} atoms")
        else:
            print(f"  ⚠ No atoms in first result")
    else:
        print("✗ No results found in checkpoint")
        return False
    
    # Test 4: List all checkpoints
    print("\n[Test 4] Listing all checkpoints...")
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"✓ Found {len(checkpoints)} checkpoints in metadata")
    if checkpoints:
        print(f"  First checkpoint: {checkpoints[0]['filename']}")
        print(f"  Last checkpoint: {checkpoints[-1]['filename']}")
    
    # Test 5: Check metadata file
    print("\n[Test 5] Checking metadata file...")
    metadata_path = checkpoint_manager.metadata_path
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata file exists")
        print(f"  Version: {metadata.get('version')}")
        print(f"  Mode: {metadata.get('mode')}")
        print(f"  Checkpoints recorded: {len(metadata.get('checkpoints', []))}")
    else:
        print("✗ Metadata file not found")
        return False
    
    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)