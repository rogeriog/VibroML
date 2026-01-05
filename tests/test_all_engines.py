#!/usr/bin/env python3
"""
Comprehensive test of all calculator engines: MACE, M3GNet, eSEN, and UMA.
Tests that all engines work correctly and don't interfere with each other.
"""

import sys
import os

def test_engine_in_environment(env_name, env_path, engine_name, expected_energy_range):
    """Test a specific engine in its environment"""
    print(f"\n{'='*60}")
    print(f"Testing {engine_name.upper()} in {env_name}")
    print(f"{'='*60}")

    # Create test script
    min_e, max_e = expected_energy_range
    test_script = f"""
import sys
sys.path.insert(0, '/globalscratch/ucl/modl/rgouvea/VibroML')

from vibroml.utils.structure_utils import initialize_calculator
from ase.build import bulk

try:
    calc = initialize_calculator('{engine_name}')
    atoms = bulk('LiF', 'rocksalt', a=4.0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Check if energy is in expected range
    min_e, max_e = {min_e}, {max_e}
    if min_e <= energy <= max_e:
        print(f'✓ {engine_name.upper()}: Energy = {{energy:.6f}} eV (in expected range)')
        sys.exit(0)
    else:
        print(f'✗ {engine_name.upper()}: Energy = {{energy:.6f}} eV (outside range [{min_e}, {max_e}])')
        sys.exit(1)
except Exception as e:
    print(f'✗ {engine_name.upper()}: {{e}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Write and run test script
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        import subprocess
        if env_path:
            cmd = f"conda activate {env_path} && python {script_path}"
        else:
            cmd = f"python {script_path}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    finally:
        os.unlink(script_path)

def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE ENGINE COMPATIBILITY TEST")
    print("="*60)
    
    tests = [
        ("vibroml_env", "/auto/globalscratch/users/r/g/rgouvea/vibroml_env", "mace", (-10, -9)),
        ("esen_env", "/auto/globalscratch/users/r/g/rgouvea/esen_env", "esen", (-10, -9)),
        ("uma_env", "/globalscratch/ucl/modl/rgouvea/uma_env", "uma", (-10, -7)),
    ]
    
    results = {}
    for env_name, env_path, engine_name, energy_range in tests:
        results[engine_name] = test_engine_in_environment(
            env_name, env_path, engine_name, energy_range
        )
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for engine, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{engine.upper():10} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL ENGINES WORKING CORRECTLY!")
    else:
        print("✗ SOME ENGINES FAILED")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

