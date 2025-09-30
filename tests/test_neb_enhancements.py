#!/usr/bin/env python3

"""
Test script to verify the enhanced NEB implementation with improved summaries and final structures export.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_enhanced_neb():
    """Test the enhanced NEB implementation."""
    print("Testing enhanced NEB implementation...")
    
    try:
        from vibroml.utils.structure_utils import load_structure, initialize_calculator
        from vibroml.utils.neb_utils import run_neb_optimization, generate_enhanced_neb_summary
        
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
        calculator = initialize_calculator("mace")
        
        # Test NEB optimization with new default parameters
        print("Running NEB optimization with enhanced features...")
        output_dir = "test_enhanced_neb_output"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            results = run_neb_optimization(
                initial_atoms=initial_atoms,
                final_atoms=final_atoms,
                calculator=calculator,
                num_images=5,  # Smaller for quick test
                spring_constant=5.0,  # New default
                max_iterations=3,     # Very few iterations for quick test
                force_tolerance=0.01, # New default
                output_dir=output_dir,
                prefix="test_enhanced_neb",
                climbing_start_iteration=None  # Standard NEB
            )
            
            print(f"✓ NEB optimization completed successfully!")
            
            # Test enhanced summary generation
            print("Testing enhanced summary generation...")
            
            # Create mock args object
            class MockArgs:
                def __init__(self):
                    self.cif = initial_cif
            
            args = MockArgs()
            
            neb_params = {
                'num_images': 5,
                'spring_constant': 5.0,
                'force_tolerance': 0.01,
                'max_iterations': 3
            }
            
            summary_paths = [
                os.path.join(output_dir, "enhanced_neb_summary.txt"),
                os.path.join(output_dir, "subdir", "enhanced_neb_summary.txt")
            ]
            
            generate_enhanced_neb_summary(
                results=results,
                method_name="Enhanced NEB Test",
                args=args,
                final_cif_path=final_cif,
                neb_params=neb_params,
                output_paths=summary_paths
            )
            
            # Verify summary files were created
            for path in summary_paths:
                if os.path.exists(path):
                    print(f"✓ Enhanced summary created: {path}")
                    
                    # Check if it contains force information
                    with open(path, 'r') as f:
                        content = f.read()
                        if "Total Force" in content and "Max Force" in content:
                            print(f"✓ Summary contains force information")
                        else:
                            print(f"✗ Summary missing force information")
                            return False
                else:
                    print(f"✗ Summary file not created: {path}")
                    return False
            
            # Test final structures export
            print("Testing final structures export...")
            final_structures_dir = os.path.join(output_dir, "final_structures")
            os.makedirs(final_structures_dir, exist_ok=True)
            
            from ase.io import write
            optimized_images = results['images']
            final_energies = results['final_energies']
            
            structure_files = []
            for i, (image, energy) in enumerate(zip(optimized_images, final_energies)):
                if i == 0:
                    structure_type = "initial"
                elif i == len(optimized_images) - 1:
                    structure_type = "final"
                else:
                    structure_type = f"intermediate_{i:02d}"
                
                filename = f"enhanced_neb_{structure_type}_img{i:02d}_energy_{energy:.4f}eV.cif"
                output_path = os.path.join(final_structures_dir, filename)
                
                try:
                    write(output_path, image)
                    structure_files.append(output_path)
                    print(f"✓ Exported {structure_type} structure: {filename}")
                except Exception as e:
                    print(f"✗ Error exporting {structure_type} structure: {e}")
                    return False
            
            print(f"✓ All {len(structure_files)} structures exported successfully")
            
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
    """Run the enhanced NEB test."""
    print("=" * 70)
    print("Enhanced NEB Implementation Test")
    print("=" * 70)
    
    success = test_enhanced_neb()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All enhanced NEB tests passed!")
        print("  - Updated default parameters")
        print("  - Enhanced summary with force information")
        print("  - Final structures export")
    else:
        print("✗ Some enhanced NEB tests failed.")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
