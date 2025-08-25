"""Test the new optical mode selection functionality."""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.build import bulk


class TestOpticalModeSelection:
    """Test the new optical mode selection behavior when no soft modes are found."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple cubic structure for testing
        self.atoms = bulk('Si', 'diamond', a=5.43)
        
        # Mock phonon object
        self.mock_ph = Mock()
        self.mock_ph.atoms = self.atoms
        
        # Mock path object with special points
        self.mock_path = Mock()
        self.mock_path.special_points = {
            'Gamma': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.0]),
            'L': np.array([0.5, 0.5, 0.5])
        }
        self.mock_path.kpts = np.array([
            [0.0, 0.0, 0.0],  # Gamma
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # X
            [0.5, 0.25, 0.25],
            [0.5, 0.5, 0.5]   # L
        ])
        
        # Mock band structure energies (all positive - no soft modes)
        # Shape: (n_kpoints, n_bands)
        self.bs_energies_no_soft = np.array([
            [1.0, 2.0, 3.0, 15.0, 16.0, 17.0],  # Gamma
            [1.5, 2.5, 3.5, 14.5, 15.5, 16.5],  # intermediate
            [2.0, 3.0, 4.0, 14.0, 15.0, 16.0],  # X
            [2.5, 3.5, 4.5, 13.5, 14.5, 15.5],  # intermediate
            [3.0, 4.0, 5.0, 13.0, 14.0, 15.0]   # L
        ])
        
        # Mock band structure energies with soft modes
        self.bs_energies_with_soft = np.array([
            [-2.0, -1.0, 3.0, 15.0, 16.0, 17.0],  # Gamma - has soft modes
            [1.5, 2.5, 3.5, 14.5, 15.5, 16.5],    # intermediate
            [2.0, 3.0, 4.0, 14.0, 15.0, 16.0],    # X
            [2.5, 3.5, 4.5, 13.5, 14.5, 15.5],    # intermediate
            [3.0, 4.0, 5.0, 13.0, 14.0, 15.0]     # L
        ])
        
        self.special_k_point_distances = [0.0, 1.0, 2.0]
        self.special_k_point_labels = ['Gamma', 'X', 'L']
        
    @patch('vibroml.utils.phonon_utils.get_eigenvector_for_q_and_band_index')
    @patch('vibroml.utils.phonon_utils._process_and_save_mode_data')
    def test_optical_mode_selection_when_no_soft_modes(self, mock_process_save, mock_get_eigenvector):
        """Test that optical modes are selected when no soft modes exist below threshold."""
        from vibroml.utils.phonon_utils import analyze_special_points_and_modes
        
        # Mock eigenvector function to return dummy displacements
        mock_get_eigenvector.return_value = np.random.rand(8, 3)  # 8 atoms, 3 directions
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = analyze_special_points_and_modes(
                ph=self.mock_ph,
                path=self.mock_path,
                bs_energies=self.bs_energies_no_soft,
                special_k_point_distances=self.special_k_point_distances,
                special_k_point_labels=self.special_k_point_labels,
                units="THz",
                output_dir=temp_dir,
                prefix="test",
                traj_kT=1.0,
                num_modes_to_return=2,
                negative_phonon_threshold=-0.1  # Threshold provided
            )
            
            # Should return 2 modes (highest frequency modes)
            assert len(result) == 2
            
            # Check that the returned modes are the highest frequency ones
            # From our mock data, highest should be 17.0 THz at Gamma, then 16.5 at intermediate point
            # But we only consider special points, so it should be 17.0 at Gamma and 16.0 at X
            assert result[0]['frequency'] == 17.0  # Highest at Gamma
            assert result[1]['frequency'] == 16.0  # Highest at X
            
            # Check that eigenvector retrieval was called
            assert mock_get_eigenvector.call_count == 2
            
            # Check that processing function was called with optical mode prefix
            assert mock_process_save.call_count == 2
            call_args = mock_process_save.call_args_list
            assert "optical_mode_1" in call_args[0][1]['prefix']
            assert "optical_mode_2" in call_args[1][1]['prefix']
    
    @patch('vibroml.utils.phonon_utils.get_eigenvector_for_q_and_band_index')
    @patch('vibroml.utils.phonon_utils._process_and_save_mode_data')
    def test_soft_mode_selection_when_soft_modes_exist(self, mock_process_save, mock_get_eigenvector):
        """Test that soft modes are selected normally when they exist below threshold."""
        from vibroml.utils.phonon_utils import analyze_special_points_and_modes
        
        # Mock eigenvector function to return dummy displacements
        mock_get_eigenvector.return_value = np.random.rand(8, 3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = analyze_special_points_and_modes(
                ph=self.mock_ph,
                path=self.mock_path,
                bs_energies=self.bs_energies_with_soft,
                special_k_point_distances=self.special_k_point_distances,
                special_k_point_labels=self.special_k_point_labels,
                units="THz",
                output_dir=temp_dir,
                prefix="test",
                traj_kT=1.0,
                num_modes_to_return=2,
                negative_phonon_threshold=-0.1  # Threshold provided
            )
            
            # Should return 2 soft modes
            assert len(result) == 2
            
            # Check that the returned modes are the softest (most negative)
            assert result[0]['frequency'] == -2.0  # Most negative at Gamma
            assert result[1]['frequency'] == -1.0  # Second most negative at Gamma
            
            # Check that processing function was called with softest mode prefix
            call_args = mock_process_save.call_args_list
            assert "softest_mode_1" in call_args[0][1]['prefix']
            assert "softest_mode_2" in call_args[1][1]['prefix']
    
    @patch('vibroml.utils.phonon_utils.get_eigenvector_for_q_and_band_index')
    def test_no_threshold_provided_returns_empty(self, mock_get_eigenvector):
        """Test that when no threshold is provided and no soft modes exist, empty list is returned."""
        from vibroml.utils.phonon_utils import analyze_special_points_and_modes
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = analyze_special_points_and_modes(
                ph=self.mock_ph,
                path=self.mock_path,
                bs_energies=self.bs_energies_no_soft,
                special_k_point_distances=self.special_k_point_distances,
                special_k_point_labels=self.special_k_point_labels,
                units="THz",
                output_dir=temp_dir,
                prefix="test",
                traj_kT=1.0,
                num_modes_to_return=2,
                negative_phonon_threshold=None  # No threshold provided
            )
            
            # Should return empty list
            assert len(result) == 0
            
            # Eigenvector function should not be called
            assert mock_get_eigenvector.call_count == 0
    
    def test_run_single_phonon_analysis_accepts_threshold_parameter(self):
        """Test that run_single_phonon_analysis accepts the new threshold parameter."""
        from vibroml.utils.phonon_utils import run_single_phonon_analysis
        import inspect
        
        # Check that the function signature includes the new parameter
        sig = inspect.signature(run_single_phonon_analysis)
        assert 'negative_phonon_threshold' in sig.parameters
        
        # Check that the parameter has the correct default value
        param = sig.parameters['negative_phonon_threshold']
        assert param.default is None
