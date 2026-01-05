#!/usr/bin/env python3
import sys
import os

# Quick test - write results to file
output = []

try:
    output.append("Testing UMA environment...")
    
    # Test imports
    import vibroml
    output.append("✓ VibroML imported")
    
    from fairchem.core import units
    output.append("✓ fairchem.core.units imported")
    
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    output.append("✓ OCPCalculator imported")
    
    # Test model loading
    uma_model = "/globalscratch/ucl/modl/rgouvea/VibroML/fairchem_models/uma-m-1p1.pt"
    if os.path.exists(uma_model):
        output.append(f"✓ UMA model file exists ({os.path.getsize(uma_model)/(1024**3):.2f} GB)")
    else:
        output.append(f"✗ UMA model file not found")
        sys.exit(1)
    
    # Load calculator
    calc = OCPCalculator(checkpoint_path=uma_model, cpu=True)
    output.append("✓ UMA model loaded successfully")
    
    # Test energy calculation
    from ase.build import bulk
    atoms = bulk('LiF', 'rocksalt', a=4.0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    output.append(f"✓ Energy calculation successful: {energy:.6f} eV")
    
    # Test VibroML integration
    from vibroml.utils.structure_utils import initialize_calculator
    calc_vib = initialize_calculator(engine="esen", checkpoint_path=uma_model)
    output.append("✓ VibroML calculator initialization successful")
    
    output.append("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    
except Exception as e:
    output.append(f"✗ ERROR: {e}")
    import traceback
    output.append(traceback.format_exc())

# Write to file
with open("/tmp/uma_quick_test.txt", "w") as f:
    f.write("\n".join(output))

# Also print
print("\n".join(output))

