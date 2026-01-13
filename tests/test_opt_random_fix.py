#!/usr/bin/env python3
"""
Quick test to verify opt_random checkpoint_path fix.
This test checks that the calculator is initialized with the correct NEP potential.
"""

import os
import sys
import tempfile
import shutil

# Add VibroML to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibroml.utils.structure_utils import initialize_calculator
from vibroml.utils.utils import GPUMD_BINARY_PATH

def test_opt_random_checkpoint_path():
    """Test that opt_random can initialize calculator with checkpoint_path."""
    
    print("=" * 70)
    print("Testing opt_random checkpoint_path fix")
    print("=" * 70)
    
    # Check GPUMD binary
    if GPUMD_BINARY_PATH is None:
        print("✗ GPUMD binary not found")
        return False
    print(f"✓ GPUMD binary found: {GPUMD_BINARY_PATH}")
    
    # Test NEP potential path
    nep_path = "/globalscratch/ucl/modl/rgouvea/VibroML/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"
    if not os.path.exists(nep_path):
        print(f"✗ NEP potential not found at {nep_path}")
        return False
    print(f"✓ NEP potential found: {nep_path}")
    
    # Initialize calculator with nep_model_path (simulating what opt_random should do)
    try:
        calculator = initialize_calculator("gpumd", model_name=None, nep_model_path=nep_path)
        if calculator is None:
            print("✗ Calculator initialization returned None")
            return False
        print("✓ Calculator initialized successfully")
        
        # Check that calculator has the correct potential_path
        if hasattr(calculator, 'potential_path'):
            if calculator.potential_path == nep_path:
                print(f"✓ Calculator has correct potential_path: {calculator.potential_path}")
            else:
                print(f"✗ Calculator potential_path mismatch:")
                print(f"  Expected: {nep_path}")
                print(f"  Got: {calculator.potential_path}")
                return False
        else:
            print("✗ Calculator does not have potential_path attribute")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing calculator: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_opt_random_checkpoint_path()
    print("\n" + "=" * 70)
    if success:
        print("✓✓✓ TEST PASSED ✓✓✓")
        sys.exit(0)
    else:
        print("✗✗✗ TEST FAILED ✗✗✗")
        sys.exit(1)

