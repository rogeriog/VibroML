#!/usr/bin/env python3
"""
Integration tests for MD stability method in VibroML.

This test validates the MD stability implementation by:
1. Testing argument parsing for MD-specific parameters
2. Testing MD utilities functions with mock data
3. Testing integration with the main workflow
4. Validating output file generation
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path to import vibroml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vibroml.utils.utils import get_arg_parser_and_settings
from vibroml.utils.md_utils import (
    parse_supercell_size, 
    calculate_rmsd, 
    calculate_rdf, 
    determine_stability
)


class TestMDArgumentParsing:
    """Test MD-specific argument parsing."""
    
    def test_md_method_in_choices(self):
        """Test that md_stability is available as a method choice."""
        parser, settings = get_arg_parser_and_settings()
        
        # Find the method argument
        method_action = None
        for action in parser._actions:
            if action.dest == 'method':
                method_action = action
                break
        
        assert method_action is not None, "Method argument not found"
        assert 'md_stability' in method_action.choices, "md_stability not in method choices"
    
    def test_md_specific_arguments(self):
        """Test that MD-specific arguments are properly parsed."""
        parser, settings = get_arg_parser_and_settings()
        
        # Test parsing with MD-specific arguments
        test_args = [
            '--cif', 'test.cif',
            '--method', 'md_stability',
            '--temp', '400',
            '--pressure', '1.0',
            '--time', '20.0',
            '--supercell-size', '3x3x3',
            '--equilibration-fraction', '0.3',
            '--output-prefix', 'test_prefix'
        ]
        
        args = parser.parse_args(test_args)
        
        assert args.method == 'md_stability'
        assert args.temp == 400.0
        assert args.pressure == 1.0
        assert args.time == 20.0
        assert args.supercell_size == '3x3x3'
        assert args.equilibration_fraction == 0.3
        assert args.output_prefix == 'test_prefix'
    
    def test_md_default_values(self):
        """Test that MD arguments have proper default values."""
        parser, settings = get_arg_parser_and_settings()
        
        # Test with minimal arguments
        test_args = ['--cif', 'test.cif', '--method', 'md_stability']
        args = parser.parse_args(test_args)
        
        # Check defaults from settings
        assert args.temp == settings['md_temperature']
        assert args.pressure == settings['md_pressure']
        assert args.time == settings['md_time']
        assert args.supercell_size == settings['md_supercell_size']
        assert args.equilibration_fraction == settings['md_equilibration_fraction']

        # Check new MLIP-optimized stability threshold defaults
        assert args.volume_threshold == 4.0  # Generous 4% threshold
        assert args.rmsd_threshold == 1.0     # Generous 1.0 Å threshold
        assert args.rdf_threshold == 0.5      # Lower 0.5 threshold

    def test_md_custom_thresholds(self):
        """Test that custom stability thresholds can be set."""
        parser, settings = get_arg_parser_and_settings()

        # Test with custom threshold arguments
        test_args = [
            '--cif', 'test.cif', '--method', 'md_stability',
            '--volume-threshold', '3.0',
            '--rmsd-threshold', '0.8',
            '--rdf-threshold', '0.7'
        ]
        args = parser.parse_args(test_args)

        # Check custom thresholds
        assert args.volume_threshold == 3.0
        assert args.rmsd_threshold == 0.8
        assert args.rdf_threshold == 0.7


class TestMDUtilities:
    """Test MD utility functions."""
    
    def test_parse_supercell_size(self):
        """Test supercell size parsing."""
        # Valid formats
        assert parse_supercell_size("2x2x2") == (2, 2, 2)
        assert parse_supercell_size("3x4x5") == (3, 4, 5)
        assert parse_supercell_size("1X1X1") == (1, 1, 1)  # Case insensitive
        
        # Invalid formats should raise ValueError
        with pytest.raises(ValueError):
            parse_supercell_size("2x2")  # Too few dimensions
        
        with pytest.raises(ValueError):
            parse_supercell_size("2x2x2x2")  # Too many dimensions
        
        with pytest.raises(ValueError):
            parse_supercell_size("0x2x2")  # Zero dimension
        
        with pytest.raises(ValueError):
            parse_supercell_size("invalid")  # Non-numeric
    
    def test_calculate_rmsd(self):
        """Test RMSD calculation."""
        # Create simple test data
        positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        positions2 = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
        cell1 = np.eye(3) * 5.0  # 5x5x5 cubic cell
        cell2 = np.eye(3) * 5.0
        
        rmsd = calculate_rmsd(positions1, positions2, cell1, cell2)
        
        # Expected RMSD for 0.1 Å displacement of both atoms
        expected_rmsd = 0.1
        assert abs(rmsd - expected_rmsd) < 1e-10
    
    def test_calculate_rdf_basic(self):
        """Test basic RDF calculation."""
        from ase import Atoms
        
        # Create a simple 2-atom system
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
        
        r, g_r = calculate_rdf(atoms, r_max=3.0, n_bins=30)
        
        assert len(r) == 30
        assert len(g_r) == 30
        assert np.all(r >= 0)
        assert np.all(r <= 3.0)
    
    def test_determine_stability(self):
        """Test stability determination logic with new MLIP-optimized thresholds."""
        # Create mock analysis results - should be stable with new generous thresholds
        stable_results = {
            'rmsd_mean': 0.8,  # Below new 1.0 Å threshold
            'rmsd_trend': 0.005,  # Low trend
            'volume_max_change': 3.5,  # Below new 4.0% threshold
            'rdf_correlation': 0.6  # Above new 0.5 threshold
        }

        # Test stable case with new defaults
        stability = determine_stability(stable_results)
        assert stability['verdict'] == 'STABLE'
        assert stability['confidence'] in ['HIGH', 'MODERATE']

        # Test unstable case - exceeds new thresholds
        unstable_results = {
            'rmsd_mean': 1.5,  # Above new 1.0 Å threshold
            'rmsd_trend': 0.02,  # High trend
            'volume_max_change': 5.0,  # Above new 4.0% threshold
            'rdf_correlation': 0.3  # Below new 0.5 threshold
        }

        stability = determine_stability(unstable_results)
        assert stability['verdict'] == 'UNSTABLE'

        # Test custom thresholds
        stability_custom = determine_stability(
            stable_results,
            rmsd_threshold=0.5,  # Stricter than default
            volume_threshold=2.0,  # Stricter than default
            rdf_threshold=0.8  # Stricter than default
        )
        # Should be unstable with stricter thresholds
        assert stability_custom['verdict'] == 'UNSTABLE'


class TestMDIntegration:
    """Test MD stability integration with main workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_cif = os.path.join(self.test_dir, 'test.cif')
        
        # Create a minimal CIF file for testing
        cif_content = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1 0.0 0.0 0.0
H2 0.5 0.5 0.5
"""
        with open(self.test_cif, 'w') as f:
            f.write(cif_content)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('vibroml.utils.structure_utils.initialize_calculator')
    @patch('vibroml.utils.md_utils.setup_nvt_ensemble')
    @patch('vibroml.utils.md_utils.setup_npt_ensemble')
    @patch('vibroml.utils.md_utils.analyze_trajectory_stability')
    @patch('vibroml.utils.md_utils.analyze_energy_trajectory')
    @patch('vibroml.utils.md_utils.determine_stability')
    def test_md_workflow_integration(self, mock_stability, mock_energy, mock_trajectory,
                                   mock_npt, mock_nvt, mock_calculator):
        """Test integration of MD workflow with mocked components."""
        from vibroml.auto_optimize import run_md_stability_analysis
        from vibroml.utils.structure_utils import load_structure
        
        # Mock calculator
        mock_calc = Mock()
        mock_calculator.return_value = mock_calc

        # Mock NVT dynamics
        mock_nvt_dynamics = Mock()
        mock_nvt_dynamics.nsteps = 0
        mock_nvt_dynamics.atoms.get_temperature.return_value = 300.0
        mock_nvt.return_value = mock_nvt_dynamics

        # Mock NPT dynamics
        mock_npt_dynamics = Mock()
        mock_npt_dynamics.atoms.get_temperature.return_value = 300.0
        mock_npt_dynamics.atoms.get_volume.return_value = 125.0
        mock_npt.return_value = mock_npt_dynamics
        
        # Mock trajectory analysis results
        mock_trajectory.return_value = {
            'rmsd_mean': 0.3,
            'rmsd_std': 0.1,
            'rmsd_trend': 0.001,
            'volume_mean_change': 1.0,
            'volume_std_change': 0.5,
            'volume_max_change': 2.0,
            'rdf_correlation': 0.95,
            'production_frames': 1000,
            'production_time_fs': np.arange(1000) * 1.0 * 8,  # Mock time array
            'volume_reference_frames': 63,  # 500 fs / (1.0 fs * 8 interval) ≈ 63
            'volume_reference_time_fs': 500.0,
            'timestep_fs': 1.0,
            'production_interval': 8,
            'average_volume': 125.0,
            'rmsd_values': np.random.rand(1000) * 0.1 + 0.25,  # Mock RMSD values
            'volume_changes': np.random.rand(1000) * 2.0 - 1.0,  # Mock volume changes
            'initial_rdf': (np.linspace(0, 8, 100), np.random.rand(100)),
            'final_rdf': (np.linspace(0, 8, 100), np.random.rand(100))
        }
        
        mock_energy.return_value = {
            'has_energy_data': True,
            'energy_mean': -1.5,
            'energy_std': 0.01,
            'energy_drift': 1e-6,
            'energy_values': np.random.rand(1000) * 0.01 - 1.5  # Mock energy values
        }
        
        mock_stability.return_value = {
            'verdict': 'STABLE',
            'confidence': 'HIGH',
            'criteria': {'rmsd_stable': True, 'volume_stable': True, 'rdf_stable': True},
            'reasoning': ['All criteria passed']
        }
        
        # Create mock arguments
        args = Mock()
        args.cif = self.test_cif
        args.engine = 'mace'
        args.model_name = 'medium-omat-0'
        args.temp = 300.0
        args.pressure = 0.0
        args.time = 1.0  # Short simulation for testing
        args.supercell_size = '2x2x2'
        args.equilibration_fraction = 0.2
        args.save_yaml = False
        
        # Load test structure
        struct, atoms = load_structure(self.test_cif)
        assert atoms is not None, "Failed to load test structure"
        
        # Run MD stability analysis
        with patch('vibroml.auto_optimize.write'), \
             patch('vibroml.auto_optimize.read'), \
             patch('ase.io.trajectory.Trajectory'), \
             patch('vibroml.utils.md_utils.StressMDLogger'), \
             patch('vibroml.utils.md_utils.save_trajectory_analysis_plots') as mock_plots:
            
            mock_plots.return_value = ['plot1.png', 'plot2.png']
            
            result = run_md_stability_analysis(args, self.test_dir, atoms)
        
        # Verify the result
        assert result is not None
        assert result['verdict'] == 'STABLE'
        assert result['confidence'] == 'HIGH'

        # Verify mocks were called
        mock_calculator.assert_called_once()
        mock_nvt.assert_called_once()  # NVT setup should be called
        assert mock_npt.call_count == 2  # NPT setup should be called twice (equilibration + production)
        mock_trajectory.assert_called_once()
        mock_energy.assert_called_once()
        mock_stability.assert_called_once()
    
    def test_md_method_validation(self):
        """Test MD method validation in main workflow."""
        from vibroml.main import main
        
        # Test invalid temperature
        with patch('sys.argv', ['vibroml', '--cif', self.test_cif, '--method', 'md_stability', '--temp', '-100']):
            with pytest.raises(SystemExit):
                main()
        
        # Test invalid pressure
        with patch('sys.argv', ['vibroml', '--cif', self.test_cif, '--method', 'md_stability', '--pressure', '-1']):
            with pytest.raises(SystemExit):
                main()
        
        # Test invalid equilibration fraction
        with patch('sys.argv', ['vibroml', '--cif', self.test_cif, '--method', 'md_stability', '--equilibration-fraction', '0.8']):
            with pytest.raises(SystemExit):
                main()
        
        # Test invalid supercell size
        with patch('sys.argv', ['vibroml', '--cif', self.test_cif, '--method', 'md_stability', '--supercell-size', 'invalid']):
            with pytest.raises(SystemExit):
                main()


if __name__ == '__main__':
    pytest.main([__file__])
