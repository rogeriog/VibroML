#!/usr/bin/env python3
"""
Test eSEN engine backward compatibility after UMA integration.
"""

import sys
sys.path.insert(0, '/globalscratch/ucl/modl/rgouvea/VibroML')

from vibroml.utils.structure_utils import initialize_calculator
from ase.build import bulk

def test_esen_engine():
    """Test eSEN engine still works"""
    print("\n" + "="*60)
    print("ESEN ENGINE BACKWARD COMPATIBILITY TEST")
    print("="*60)
    
    try:
        print("\nInitializing eSEN calculator...")
        calc = initialize_calculator('esen')
        print('✓ eSEN calculator initialized successfully')
        
        print("\nCreating test structure...")
        atoms = bulk('LiF', 'rocksalt', a=4.0)
        atoms.calc = calc
        
        print("Calculating energy...")
        energy = atoms.get_potential_energy()
        print(f'✓ Energy calculated: {energy:.6f} eV')
        
        print("Calculating forces...")
        forces = atoms.get_forces()
        max_force = (forces**2).sum(axis=1).max()**0.5
        print(f'✓ Forces calculated: max force = {max_force:.6f} eV/Å')
        
        print("\n" + "="*60)
        print("✓ ESEN ENGINE WORKS CORRECTLY!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f'\n✗ FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_esen_engine()
    sys.exit(0 if success else 1)

