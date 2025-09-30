#!/usr/bin/env python3

"""
Test script to verify the complete NEB workflow with structure compatibility.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_neb_workflow():
    """Test the complete NEB workflow."""
    print("Testing complete NEB workflow with structure compatibility...")
    
    try:
        from vibroml.utils.structure_utils import load_structure, initialize_calculator
        from vibroml.utils.neb_utils import run_neb_optimization
        
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
        
        # Set up calculator
        print("Setting up MACE calculator...")
        calculator = initialize_calculator("mace")  # Use MACE calculator
        
        # Test NEB optimization with minimal settings
        print("Running NEB optimization...")
        output_dir = "test_neb_output"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            results = run_neb_optimization(
                initial_atoms=initial_atoms,
                final_atoms=final_atoms,
                calculator=calculator,
                num_images=3,
                spring_constant=1.0,
                max_iterations=3,     # Very few iterations for quick test
                force_tolerance=0.2,  # Relaxed tolerance for quick test
                output_dir=output_dir,
                prefix="test_neb",
                climbing_start_iteration=10  # Won't activate in 3 iterations
            )
            
            print(f"✓ NEB optimization completed successfully!")
            print(f"Results keys: {list(results.keys())}")
            
            if 'converged' in results:
                print(f"Converged: {results['converged']}")
            if 'final_max_force' in results:
                print(f"Final max force: {results['final_max_force']:.4f} eV/Å")
            if 'iterations' in results:
                print(f"Iterations completed: {results['iterations']}")
            
            return True
            
        except Exception as e:
            print(f"✗ NEB optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the NEB workflow test."""
    print("=" * 60)
    print("NEB Workflow Test with Structure Compatibility")
    print("=" * 60)
    
    success = test_neb_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ NEB workflow test passed!")
    else:
        print("✗ NEB workflow test failed.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
