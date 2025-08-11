"""Test existing VibroML functions that actually exist in the codebase."""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.build import bulk


class TestExistingUtils:
    """Test existing utility functions."""
    
    def test_parse_supercell_dimensions(self):
        """Test the new parse_supercell_dimensions function we added."""
        from vibroml.utils.utils import parse_supercell_dimensions
        
        # Test various input formats
        assert parse_supercell_dimensions("3") == (3, 3, 3)
        assert parse_supercell_dimensions("2,3,4") == (2, 3, 4)
        assert parse_supercell_dimensions(3) == (3, 3, 3)
        assert parse_supercell_dimensions([2, 3, 4]) == (2, 3, 4)
        assert parse_supercell_dimensions((2, 3, 4)) == (2, 3, 4)
        
        # Test error cases
        with pytest.raises(ValueError):
            parse_supercell_dimensions("1,2")  # Too few dimensions
        with pytest.raises(ValueError):
            parse_supercell_dimensions("0,1,1")  # Zero dimension
        with pytest.raises(ValueError):
            parse_supercell_dimensions("a,b,c")  # Non-numeric
    
    def test_load_default_settings(self):
        """Test loading default settings."""
        from vibroml.utils.utils import load_default_settings
        
        # This should work with the existing default_settings.json
        settings = load_default_settings()
        assert isinstance(settings, dict)
        
        # Should contain expected keys from actual file
        expected_keys = ["default_supercell_n", "screen_supercell_ns"]
        for key in expected_keys:
            assert key in settings
    
    def test_have_mace_constant(self):
        """Test HAVE_MACE constant exists."""
        from vibroml.utils.utils import HAVE_MACE
        assert isinstance(HAVE_MACE, bool)
    
    def test_clean_phonon_cache(self):
        """Test phonon cache cleaning."""
        from vibroml.utils.utils import clean_phonon_cache
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake phonon cache files
            cache_file = os.path.join(temp_dir, "phonopy.yaml")
            with open(cache_file, 'w') as f:
                f.write("test")
            
            # Function should run without error
            clean_phonon_cache(temp_dir)


class TestExistingStructureUtils:
    """Test existing structure utility functions."""
    
    def test_load_structure_function_exists(self):
        """Test that load_structure function exists and is callable."""
        from vibroml.utils.structure_utils import load_structure
        assert callable(load_structure)
    
    def test_initialize_calculator_function_exists(self):
        """Test that initialize_calculator function exists."""
        from vibroml.utils.structure_utils import initialize_calculator
        assert callable(initialize_calculator)
    
    def test_estimate_commensurate_supercell_size(self):
        """Test the existing commensurate supercell estimation."""
        from vibroml.utils.structure_utils import estimate_commensurate_supercell_size
        
        # Test Gamma point
        result = estimate_commensurate_supercell_size([0.0, 0.0, 0.0])
        assert result == (1, 1, 1)
        
        # Test simple fractions
        result = estimate_commensurate_supercell_size([0.5, 0.0, 0.0])
        assert result == (2, 1, 1)


class TestExistingPhononUtils:
    """Test existing phonon utility functions."""
    
    @patch('vibroml.utils.phonon_utils.Phonons')
    def test_run_phonon_calculation_exists(self, mock_phonons_class):
        """Test that run_phonon_calculation function works."""
        from vibroml.utils.phonon_utils import run_phonon_calculation
        
        mock_ph = Mock()
        mock_phonons_class.return_value = mock_ph
        
        atoms = bulk('Si')
        calculator = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phonon_calculation(atoms, calculator, 3, 0.01, temp_dir)
            assert result == mock_ph
    
    @patch('vibroml.utils.phonon_utils.Phonons')
    def test_run_phonon_calculation_with_custom_supercell(self, mock_phonons_class):
        """Test our new custom supercell function."""
        from vibroml.utils.phonon_utils import run_phonon_calculation_with_custom_supercell
        
        mock_ph = Mock()
        mock_phonons_class.return_value = mock_ph
        
        atoms = bulk('Si')
        calculator = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phonon_calculation_with_custom_supercell(
                atoms, calculator, (2, 3, 4), 0.01, temp_dir
            )
            
            assert result == mock_ph
            # Verify correct supercell was used
            call_args = mock_phonons_class.call_args
            assert call_args[1]['supercell'] == (2, 3, 4)
    
    def test_run_single_phonon_analysis_exists(self):
        """Test that run_single_phonon_analysis function exists."""
        from vibroml.utils.phonon_utils import run_single_phonon_analysis
        assert callable(run_single_phonon_analysis)


class TestExistingConfig:
    """Test existing configuration constants."""
    
    def test_conversion_factors(self):
        """Test conversion factors exist."""
        from vibroml.utils.config import EV_TO_THZ_FACTOR, THZ_TO_CM_FACTOR
        
        assert isinstance(EV_TO_THZ_FACTOR, float)
        assert isinstance(THZ_TO_CM_FACTOR, float)
        assert EV_TO_THZ_FACTOR > 0
        assert THZ_TO_CM_FACTOR > 0


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    @patch('vibroml.utils.utils.load_default_settings')
    def test_get_arg_parser_and_settings(self, mock_load_settings):
        """Test argument parser creation."""
        mock_load_settings.return_value = {
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
        
        from vibroml.utils.utils import get_arg_parser_and_settings
        
        parser, settings = get_arg_parser_and_settings()
        assert parser is not None
        assert isinstance(settings, dict)
        
        # Test parsing new supercell argument
        args = parser.parse_args(['--cif', 'test.cif', '--supercell', '2,3,4'])
        assert args.supercell == '2,3,4'
        
        # Test backward compatibility
        args = parser.parse_args(['--cif', 'test.cif', '--supercell_n', '3'])
        assert args.supercell_n == 3


class TestSupercellIntegration:
    """Test integration of supercell modifications."""
    
    def test_supercell_parsing_integration(self):
        """Test that supercell parsing works end-to-end."""
        from vibroml.utils.utils import parse_supercell_dimensions
        
        test_cases = [
            ("3", (3, 3, 3)),
            ("2,3,4", (2, 3, 4)),
            ("1,1,2", (1, 1, 2)),
            (3, (3, 3, 3)),
            ([2, 3, 4], (2, 3, 4)),
            ((2, 3, 4), (2, 3, 4)),
        ]
        
        for input_val, expected in test_cases:
            result = parse_supercell_dimensions(input_val)
            assert result == expected, f"Failed for input {input_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
