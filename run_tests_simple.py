#!/usr/bin/env python3
"""Simple test runner for VibroML that doesn't require pytest."""

import sys
import os
import traceback

# Add the current directory to path so we can import vibroml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def mock_dependencies():
    """Mock external dependencies that might not be available."""
    import types
    
    # Mock torch
    if 'torch' not in sys.modules:
        torch_mock = types.ModuleType('torch')
        torch_mock.cuda = types.ModuleType('cuda')
        torch_mock.cuda.is_available = lambda: False
        torch_mock.cuda.get_device_name = lambda x: "Mock GPU"
        sys.modules['torch'] = torch_mock
    
    # Mock tensorflow
    if 'tensorflow' not in sys.modules:
        tf_mock = types.ModuleType('tensorflow')
        tf_mock.get_logger = lambda: types.SimpleNamespace(setLevel=lambda x: None)
        sys.modules['tensorflow'] = tf_mock
    
    # Mock m3gnet
    if 'm3gnet' not in sys.modules:
        m3gnet_mock = types.ModuleType('m3gnet')
        models_mock = types.ModuleType('models')
        models_mock.M3GNet = type('M3GNet', (), {'load': lambda: None})
        models_mock.M3GNetCalculator = type('M3GNetCalculator', (), {})
        models_mock.Potential = type('Potential', (), {})
        m3gnet_mock.models = models_mock
        sys.modules['m3gnet'] = m3gnet_mock
        sys.modules['m3gnet.models'] = models_mock
    
    # Mock mace
    if 'mace' not in sys.modules:
        mace_mock = types.ModuleType('mace')
        calc_mock = types.ModuleType('calculators')
        calc_mock.mace_mp = lambda: None
        calc_mock.MACECalculator = type('MACECalculator', (), {})
        mace_mock.calculators = calc_mock
        sys.modules['mace'] = mace_mock
        sys.modules['mace.calculators'] = calc_mock
    
    # Mock ase
    if 'ase' not in sys.modules:
        ase_mock = types.ModuleType('ase')
        io_mock = types.ModuleType('io')
        build_mock = types.ModuleType('build')
        
        io_mock.read = lambda x: None
        build_mock.bulk = lambda x: None
        build_mock.make_supercell = lambda x, y: None
        
        ase_mock.io = io_mock
        ase_mock.build = build_mock
        ase_mock.Atoms = type('Atoms', (), {})
        
        sys.modules['ase'] = ase_mock
        sys.modules['ase.io'] = io_mock
        sys.modules['ase.build'] = build_mock
    
    # Mock phonopy
    if 'phonopy' not in sys.modules:
        phonopy_mock = types.ModuleType('phonopy')
        phonopy_mock.Phonons = type('Phonons', (), {})
        sys.modules['phonopy'] = phonopy_mock
    
    # Mock ASE
    if 'ase' not in sys.modules:
        ase_mock = types.ModuleType('ase')
        build_mock = types.ModuleType('build')
        build_mock.bulk = lambda x: None
        build_mock.make_supercell = lambda atoms, matrix: atoms
        ase_mock.build = build_mock
        
        # Mock Atoms class
        ase_mock.Atoms = type('Atoms', (), {})
        
        # Mock io module
        io_mock = types.ModuleType('io')
        io_mock.read = lambda x: None
        ase_mock.io = io_mock
        
        sys.modules['ase'] = ase_mock
        sys.modules['ase.build'] = build_mock
        sys.modules['ase.io'] = io_mock

def test_parse_supercell_dimensions():
    """Test the new parse_supercell_dimensions function."""
    print("Testing parse_supercell_dimensions...")
    
    from vibroml.utils.utils import parse_supercell_dimensions
    
    test_cases = [
        ("3", (3, 3, 3)),
        ("2,3,4", (2, 3, 4)),
        ("1,1,2", (1, 1, 2)),
        (3, (3, 3, 3)),
        ([2, 3, 4], (2, 3, 4)),
        ((2, 3, 4), (2, 3, 4)),
        ("2, 3, 4", (2, 3, 4)),  # With spaces
        (" 2 , 3 , 4 ", (2, 3, 4)),  # With extra spaces
    ]
    
    for i, (input_val, expected) in enumerate(test_cases, 1):
        try:
            result = parse_supercell_dimensions(input_val)
            if result == expected:
                print(f"  ✓ Test {i}: {input_val} -> {result}")
            else:
                print(f"  ✗ Test {i}: {input_val} -> {result}, expected {expected}")
                return False
        except Exception as e:
            print(f"  ✗ Test {i}: {input_val} raised {type(e).__name__}: {e}")
            return False
    
    # Test error cases
    error_cases = [
        "1,2",       # Too few dimensions
        "1,2,3,4",   # Too many dimensions  
        "a,b,c",     # Non-numeric
        "0,1,1",     # Zero dimension
        "-1,2,3",    # Negative dimension
        "",          # Empty string
        "1,,3",      # Missing value
    ]
    
    for i, input_val in enumerate(error_cases, 1):
        try:
            result = parse_supercell_dimensions(input_val)
            print(f"  ✗ Error test {i}: {input_val} should have raised ValueError, got {result}")
            return False
        except ValueError:
            print(f"  ✓ Error test {i}: {input_val} correctly raised ValueError")
        except Exception as e:
            print(f"  ✗ Error test {i}: {input_val} raised {type(e).__name__}, expected ValueError: {e}")
            return False
    
    return True

def test_existing_functions():
    """Test that existing functions still work."""
    print("\nTesting existing functions...")
    
    try:
        from vibroml.utils.utils import load_default_settings
        settings = load_default_settings()
        if isinstance(settings, dict):
            print("  ✓ load_default_settings works")
        else:
            print("  ✗ load_default_settings returned non-dict")
            return False
    except Exception as e:
        print(f"  ✗ load_default_settings failed: {e}")
        return False
    
    try:
        from vibroml.utils.utils import HAVE_MACE
        if isinstance(HAVE_MACE, bool):
            print("  ✓ HAVE_MACE constant exists")
        else:
            print("  ✗ HAVE_MACE is not boolean")
            return False
    except Exception as e:
        print(f"  ✗ HAVE_MACE import failed: {e}")
        return False
    
    try:
        from vibroml.utils.config import EV_TO_THZ_FACTOR, THZ_TO_CM_FACTOR
        if isinstance(EV_TO_THZ_FACTOR, float) and isinstance(THZ_TO_CM_FACTOR, float):
            print("  ✓ Conversion factors exist")
        else:
            print("  ✗ Conversion factors are not floats")
            return False
    except Exception as e:
        print(f"  ✗ Conversion factors import failed: {e}")
        return False
    
    try:
        from vibroml.utils.structure_utils import estimate_commensurate_supercell_size
        result = estimate_commensurate_supercell_size([0.0, 0.0, 0.0])
        if result == (1, 1, 1):
            print("  ✓ estimate_commensurate_supercell_size works")
        else:
            print(f"  ✗ estimate_commensurate_supercell_size returned {result}, expected (1,1,1)")
            return False
    except Exception as e:
        print(f"  ✗ estimate_commensurate_supercell_size failed: {e}")
        return False
    
    return True

def test_argument_parsing():
    """Test argument parsing."""
    print("\nTesting argument parsing...")
    
    try:
        # Mock the settings loading
        import unittest.mock
        mock_settings = {
            "default_supercell_n": 3,
            "screen_supercell_ns": [2, 3, 4],
            "default_delta": 0.01,
            "default_fmax": 0.01,
            "default_engine": "mace",
            "default_model_name": "medium",
            "default_units": "THz",
            "phonon_path_npoints": 100,
            "phonon_dos_grid": [20, 20, 20],
            "default_traj_kT": 1.0,
            "negative_phonon_threshold_thz": -0.1,
            "screen_deltas": [0.05, 0.03, 0.01],
            "screen_fmax_values": [0.001, 0.0005, 0.0001],
            "soft_mode_max_iterations": 3,
            "soft_mode_displacement_scales": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            "mode2_ratio_scales": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
            "soft_mode_num_top_structures_to_analyze": 3,
            "cell_scale_factors": [-0.05, 0.0, 0.05, 0.10],
            "num_modes_to_return": 2,
            "ga_population_size": 50,
            "ga_mutation_rate": 0.1,
            "num_new_points_per_iteration": 30,
            "default_method": "ga"
        }
        
        with unittest.mock.patch('vibroml.utils.utils.load_default_settings', return_value=mock_settings):
            from vibroml.utils.utils import get_arg_parser_and_settings, parse_supercell_dimensions
            
            parser, settings = get_arg_parser_and_settings()
            
            # Test new supercell argument
            args = parser.parse_args(['--cif', 'test.cif', '--supercell', '2,3,4'])
            if args.supercell == '2,3,4':
                print("  ✓ New --supercell argument works")
                
                parsed = parse_supercell_dimensions(args.supercell)
                if parsed == (2, 3, 4):
                    print("  ✓ Supercell parsing works")
                else:
                    print(f"  ✗ Supercell parsing failed: got {parsed}")
                    return False
            else:
                print(f"  ✗ Supercell argument failed: got {args.supercell}")
                return False
            
            # Test backward compatibility
            args = parser.parse_args(['--cif', 'test.cif', '--supercell_n', '3'])
            if args.supercell_n == 3 and args.supercell is None:
                print("  ✓ Backward compatibility works")
            else:
                print(f"  ✗ Backward compatibility failed")
                return False
            
    except Exception as e:
        print(f"  ✗ Argument parsing test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_new_functions():
    """Test our new functions."""
    print("\nTesting new functions...")
    
    try:
        from vibroml.utils.structure_utils import generate_supercell_variants
        variants = generate_supercell_variants((2, 2, 2), max_variants=3)
        if isinstance(variants, list) and len(variants) > 0 and (2, 2, 2) in variants:
            print("  ✓ generate_supercell_variants works")
        else:
            print(f"  ✗ generate_supercell_variants failed: got {variants}")
            return False
    except Exception as e:
        print(f"  ✗ generate_supercell_variants failed: {e}")
        return False
    
    try:
        from vibroml.utils.structure_utils import estimate_commensurate_supercell_size_custom
        result = estimate_commensurate_supercell_size_custom([0.5, 0.0, 0.0], (1, 1, 1))
        if result == (2, 1, 1):
            print("  ✓ estimate_commensurate_supercell_size_custom works")
        else:
            print(f"  ✗ estimate_commensurate_supercell_size_custom failed: got {result}")
            return False
    except Exception as e:
        print(f"  ✗ estimate_commensurate_supercell_size_custom failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("VibroML Test Suite")
    print("=" * 60)
    
    # Mock dependencies first
    mock_dependencies()
    
    success = True
    
    try:
        success &= test_parse_supercell_dimensions()
    except Exception as e:
        print(f"✗ parse_supercell_dimensions test failed: {e}")
        traceback.print_exc()
        success = False
    
    try:
        success &= test_existing_functions()
    except Exception as e:
        print(f"✗ existing functions test failed: {e}")
        traceback.print_exc()
        success = False
    
    try:
        success &= test_argument_parsing()
    except Exception as e:
        print(f"✗ argument parsing test failed: {e}")
        traceback.print_exc()
        success = False
    
    try:
        success &= test_new_functions()
    except Exception as e:
        print(f"✗ new functions test failed: {e}")
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nYour VibroML modifications are working correctly:")
        print("  - Supercell parsing function works")
        print("  - Argument parsing supports new --supercell format")
        print("  - Backward compatibility maintained")
        print("  - New helper functions work")
        print("  - Existing functions still work")
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nPlease check the error messages above.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
