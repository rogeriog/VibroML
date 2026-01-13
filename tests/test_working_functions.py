"""Tests for functions that actually exist and work in VibroML."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock


class TestActualSupercellFunctions:
    """Test the supercell functions we actually implemented."""
    
    def test_parse_supercell_dimensions(self):
        """Test the parse_supercell_dimensions function we added."""
        from vibroml.utils.utils import parse_supercell_dimensions
        
        # Test basic functionality
        assert parse_supercell_dimensions("3") == (3, 3, 3)
        assert parse_supercell_dimensions("2,3,4") == (2, 3, 4)
        assert parse_supercell_dimensions(3) == (3, 3, 3)
        assert parse_supercell_dimensions([2, 3, 4]) == (2, 3, 4)
        assert parse_supercell_dimensions((2, 3, 4)) == (2, 3, 4)
        
        # Test with spaces
        assert parse_supercell_dimensions("2, 3, 4") == (2, 3, 4)
        assert parse_supercell_dimensions(" 2 , 3 , 4 ") == (2, 3, 4)
        
        # Test error cases
        with pytest.raises(ValueError):
            parse_supercell_dimensions("1,2")  # Too few
        with pytest.raises(ValueError):
            parse_supercell_dimensions("1,2,3,4")  # Too many
        with pytest.raises(ValueError):
            parse_supercell_dimensions("a,b,c")  # Non-numeric
        with pytest.raises(ValueError):
            parse_supercell_dimensions("0,1,1")  # Zero
        with pytest.raises(ValueError):
            parse_supercell_dimensions("-1,2,3")  # Negative
    
    def test_generate_supercell_variants(self):
        """Test the generate_supercell_variants function we added."""
        from vibroml.utils.structure_utils import generate_supercell_variants
        
        # Test basic functionality
        variants = generate_supercell_variants((2, 2, 2), max_variants=3)
        
        assert isinstance(variants, list)
        assert len(variants) > 0
        assert len(variants) <= 3
        assert (2, 2, 2) in variants  # Original should be included
        
        # All variants should be valid tuples
        for variant in variants:
            assert isinstance(variant, tuple)
            assert len(variant) == 3
            assert all(isinstance(x, int) and x > 0 for x in variant)
    
    def test_estimate_commensurate_supercell_size_custom(self):
        """Test the custom commensurate supercell function we added."""
        from vibroml.utils.structure_utils import estimate_commensurate_supercell_size_custom
        
        # Test Gamma point
        result = estimate_commensurate_supercell_size_custom([0.0, 0.0, 0.0], (1, 1, 1))
        assert result == (1, 1, 1)
        
        # Test simple fractions
        result = estimate_commensurate_supercell_size_custom([0.5, 0.0, 0.0], (1, 1, 1))
        assert result == (2, 1, 1)
        
        # Test with base supercell
        result = estimate_commensurate_supercell_size_custom([0.5, 0.0, 0.0], (2, 2, 2))
        assert result == (4, 2, 2)


class TestExistingFunctions:
    """Test existing functions that we know work."""
    
    def test_load_default_settings(self):
        """Test that load_default_settings works."""
        from vibroml.utils.utils import load_default_settings
        
        settings = load_default_settings()
        assert isinstance(settings, dict)
        
        # Check for expected keys from the actual default_settings.json
        expected_keys = ["default_supercell_n", "screen_supercell_ns"]
        for key in expected_keys:
            assert key in settings
    
    def test_estimate_commensurate_supercell_size(self):
        """Test the original function."""
        from vibroml.utils.structure_utils import estimate_commensurate_supercell_size
        
        # Test Gamma point
        result = estimate_commensurate_supercell_size([0.0, 0.0, 0.0])
        assert result == (1, 1, 1)
        
        # Test simple fractions
        result = estimate_commensurate_supercell_size([0.5, 0.0, 0.0])
        assert result == (2, 1, 1)


class TestPhononFunctions:
    """Test phonon functions that actually exist."""
    
    @patch('vibroml.utils.phonon_utils.Phonons')
    def test_run_phonon_calculation_with_custom_supercell(self, mock_phonons_class):
        """Test our new custom supercell phonon function."""
        from vibroml.utils.phonon_utils import run_phonon_calculation_with_custom_supercell
        
        mock_ph = Mock()
        mock_phonons_class.return_value = mock_ph
        
        # Create mock atoms and calculator
        atoms = Mock()
        atoms.set_calculator = Mock()
        calculator = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phonon_calculation_with_custom_supercell(
                atoms, calculator, (2, 3, 4), 0.01, temp_dir
            )
            
            assert result == mock_ph
            
            # Check that Phonons was called with correct supercell
            call_args = mock_phonons_class.call_args
            assert call_args[1]['supercell'] == (2, 3, 4)
            
            mock_ph.run.assert_called_once()
            mock_ph.read.assert_called_once_with(acoustic=True)
            mock_ph.clean.assert_called_once()
    
    @patch('vibroml.utils.phonon_utils.Phonons')
    def test_run_phonon_calculation_with_tuple_input(self, mock_phonons_class):
        """Test that run_phonon_calculation handles tuple input."""
        from vibroml.utils.phonon_utils import run_phonon_calculation
        
        mock_ph = Mock()
        mock_phonons_class.return_value = mock_ph
        
        atoms = Mock()
        atoms.set_calculator = Mock()
        calculator = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with tuple input (should work with our modifications)
            result = run_phonon_calculation(atoms, calculator, (2, 3, 4), 0.01, temp_dir)
            
            assert result == mock_ph
            
            # Should use the tuple directly as supercell
            call_args = mock_phonons_class.call_args
            assert call_args[1]['supercell'] == (2, 3, 4)


class TestGeneticAlgorithmActual:
    """Test the actual GeneticAlgorithm class with correct parameters."""
    
    def test_genetic_algorithm_initialization(self):
        """Test GA initialization with actual parameters."""
        from vibroml.utils.genetic_algorithm import GeneticAlgorithm
        
        # Use actual parameters based on the real constructor
        ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.1,
            displacement_scale_bounds=(0.1, 2.0),
            ratio_mode2_to_mode1_bounds=(-1.0, 1.0),
            cell_scale_bounds=(-0.1, 0.1),
            cell_angle_bounds=(-5.0, 5.0),
            supercell_variants=[(2, 2, 2), (3, 3, 3)]
        )
        
        assert ga.population_size == 10
        assert ga.mutation_rate == 0.1
        assert ga.supercell_variants == [(2, 2, 2), (3, 3, 3)]
    
    def test_genetic_algorithm_generate_individual(self):
        """Test individual generation with actual GA class."""
        from vibroml.utils.genetic_algorithm import GeneticAlgorithm

        ga = GeneticAlgorithm(
            population_size=5,
            mutation_rate=0.1,
            displacement_scale_bounds=(0.1, 2.0),
            ratio_mode2_to_mode1_bounds=(-1.0, 1.0),
            cell_scale_bounds=(-0.1, 0.1),
            cell_angle_bounds=(-5.0, 5.0),
            supercell_variants=[(2, 2, 2)]
        )

        # Test the actual method name - now returns (individual_params, mutation_data)
        individual_result = ga._generate_random_individual()

        # Check the structure matches what the actual function returns
        assert isinstance(individual_result, tuple)
        assert len(individual_result) == 2  # (individual_params, mutation_data)

        individual_params, mutation_data = individual_result
        assert isinstance(individual_params, tuple)
        assert len(individual_params) == 5  # Based on actual implementation
        assert isinstance(mutation_data, dict)
        assert 'mode_replaced' in mutation_data
        assert 'selected_mode' in mutation_data


class TestConfigurationValues:
    """Test configuration constants."""
    
    def test_conversion_factors(self):
        """Test that conversion factors exist and are valid."""
        from vibroml.utils.config import EV_TO_THZ_FACTOR, THZ_TO_CM_FACTOR
        
        assert isinstance(EV_TO_THZ_FACTOR, float)
        assert isinstance(THZ_TO_CM_FACTOR, float)
        assert EV_TO_THZ_FACTOR > 0
        assert THZ_TO_CM_FACTOR > 0
    
    def test_have_mace_constant(self):
        """Test HAVE_MACE constant."""
        from vibroml.utils.utils import HAVE_MACE
        assert isinstance(HAVE_MACE, bool)


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    @patch('vibroml.utils.utils.load_default_settings')
    def test_argument_parser_with_supercell(self, mock_load_settings):
        """Test that argument parser works with supercell arguments."""
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
        
        from vibroml.utils.utils import get_arg_parser_and_settings, parse_supercell_dimensions
        
        parser, settings = get_arg_parser_and_settings()
        
        # Test new supercell argument
        args = parser.parse_args(['--cif', 'test.cif', '--supercell', '2,3,4'])
        assert args.supercell == '2,3,4'
        
        # Test parsing
        parsed = parse_supercell_dimensions(args.supercell)
        assert parsed == (2, 3, 4)
        
        # Test backward compatibility
        args = parser.parse_args(['--cif', 'test.cif', '--supercell_n', '3'])
        assert args.supercell_n == 3
        assert args.supercell is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
