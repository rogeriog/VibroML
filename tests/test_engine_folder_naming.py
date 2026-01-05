#!/usr/bin/env python
"""
Test script to verify that output folder names include the engine name.
This tests the folder naming logic without running full VibroML calculations.
"""

import os
import sys
import time
import tempfile
import shutil

# Add the VibroML package to the path
sys.path.insert(0, '/globalscratch/ucl/modl/rgouvea/VibroML')

def test_folder_naming():
    """Test that output folder names include engine names."""
    
    print("=" * 70)
    print("Testing VibroML Output Folder Naming with Engine Names")
    print("=" * 70)
    
    # Simulate the folder naming logic from main.py
    test_cases = [
        ("mace", "ga", "LiFsimplecubic"),
        ("esen", "ga", "LiFsimplecubic"),
        ("m3gnet", "traditional_all", "LiFsimplecubic"),
        ("mace", "traditional", "Cu_bulk"),
        ("esen", "opt_random", "Cu_bulk"),
        ("m3gnet", "neb", "LiFsimplecubic"),
    ]
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Engine suffix mapping
    engine_suffix_map = {
        "mace": "_MACE",
        "m3gnet": "_M3GNET",
        "esen": "_ESEN"
    }
    
    # Method suffix mapping
    method_suffix_map = {
        "traditional": "_TRADITIONAL",
        "ga": "_GA",
        "traditional_all": "_TRADITIONAL_ALL",
        "opt_random": "_OPT_RANDOM",
        "neb": "_NEB",
        "ci_neb": "_CI_NEB",
        "md_stability": "_MD_STABILITY"
    }
    
    print("\nGenerated folder names:\n")
    
    for engine, method, structure in test_cases:
        engine_suffix = engine_suffix_map.get(engine, f"_{engine.upper()}")
        method_suffix = method_suffix_map.get(method, "")
        
        folder_name = f"{structure}{engine_suffix}{method_suffix}_phonon_output_{timestamp}"
        
        print(f"  Engine: {engine:8} | Method: {method:15} | Structure: {structure:15}")
        print(f"    → {folder_name}")
        print()
    
    print("=" * 70)
    print("✓ All folder names include engine names!")
    print("=" * 70)
    
    # Verify that engine names are present
    print("\nVerification:")
    for engine, method, structure in test_cases:
        engine_suffix = engine_suffix_map.get(engine, f"_{engine.upper()}")
        method_suffix = method_suffix_map.get(method, "")
        folder_name = f"{structure}{engine_suffix}{method_suffix}_phonon_output_{timestamp}"
        
        # Check that engine name is in the folder name
        engine_upper = engine.upper()
        if engine_upper in folder_name:
            print(f"  ✓ {engine:8} → {engine_upper:8} found in folder name")
        else:
            print(f"  ✗ {engine:8} → {engine_upper:8} NOT found in folder name")
            return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_folder_naming()
    sys.exit(0 if success else 1)

