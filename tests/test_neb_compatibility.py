#!/usr/bin/env python3

"""
Test script to verify NEB structure compatibility handling.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_structure_compatibility():
    """Test the structure compatibility functions."""
    print("Testing NEB structure compatibility handling...")
    
    try:
        from vibroml.utils.structure_utils import load_structure
        from vibroml.utils.neb_utils import (
            check_chemical_compatibility, 
            make_structures_compatible,
            linear_interpolate_structures
        )
        
        # Load the test structures
        initial_cif = "examples/NEB_test/LiFsimplecubic.cif"
        final_cif = "examples/NEB_test/LiFhexagonal.cif"
        
        if not os.path.exists(initial_cif):
            print(f"✗ Initial structure not found: {initial_cif}")
            return False
            
        if not os.path.exists(final_cif):
            print(f"✗ Final structure not found: {final_cif}")
            return False
        
        print(f"Loading structures...")
        _, initial_atoms = load_structure(initial_cif)
        _, final_atoms = load_structure(final_cif)
        
        if initial_atoms is None or final_atoms is None:
            print("✗ Could not load structures")
            return False
        
        print(f"Initial structure: {len(initial_atoms)} atoms")
        print(f"Final structure: {len(final_atoms)} atoms")
        
        # Test chemical compatibility
        print("\n1. Testing chemical compatibility...")
        is_compatible = check_chemical_compatibility(initial_atoms, final_atoms)
        print(f"Chemical compatibility: {is_compatible}")
        
        if not is_compatible:
            print("✗ Structures are not chemically compatible")
            return False
        
        # Test structure compatibility (supercell creation)
        print("\n2. Testing structure compatibility (supercell creation)...")
        try:
            compatible_initial, compatible_final = make_structures_compatible(initial_atoms, final_atoms)
            print(f"✓ Successfully created compatible structures")
            print(f"Compatible initial: {len(compatible_initial)} atoms")
            print(f"Compatible final: {len(compatible_final)} atoms")
            
            if len(compatible_initial) != len(compatible_final):
                print("✗ Compatible structures have different atom counts")
                return False
                
        except Exception as e:
            print(f"✗ Failed to create compatible structures: {e}")
            return False
        
        # Test linear interpolation with compatibility handling
        print("\n3. Testing linear interpolation with automatic compatibility...")
        try:
            images = linear_interpolate_structures(initial_atoms, final_atoms, 3)
            print(f"✓ Successfully created {len(images)} images")
            
            # Verify all images have the same number of atoms
            atom_counts = [len(img) for img in images]
            if len(set(atom_counts)) != 1:
                print(f"✗ Images have different atom counts: {atom_counts}")
                return False
            
            print(f"All images have {atom_counts[0]} atoms")
            
        except Exception as e:
            print(f"✗ Linear interpolation failed: {e}")
            return False
        
        print("\n✓ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the compatibility test."""
    print("=" * 60)
    print("NEB Structure Compatibility Test")
    print("=" * 60)
    
    success = test_structure_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! NEB compatibility handling is working.")
    else:
        print("✗ Some tests failed. Check the output above.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
