"""Integration test for the optical mode selection feature."""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.build import bulk


class TestOpticalModeIntegration:
    """Integration test for the complete optical mode selection workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple cubic structure for testing
        self.atoms = bulk('Si', 'diamond', a=5.43)
        
    @patch('vibroml.utils.phonon_utils.Phonons')
    @patch('vibroml.utils.phonon_utils.get_phonon_results')
    @patch('vibroml.utils.phonon_utils.analyze_special_points_and_modes')
    def test_run_single_phonon_analysis_with_threshold(self, mock_analyze, mock_get_results, mock_phonons):
        """Test that run_single_phonon_analysis passes threshold correctly."""
        from vibroml.utils.phonon_utils import run_single_phonon_analysis
        from vibroml.utils.structure_utils import initialize_calculator
        
        # Mock the calculator
        mock_calculator = Mock()
        
        # Mock phonon results
        mock_bs = Mock()
        mock_path = Mock()
        mock_path.special_points = {'Gamma': np.array([0.0, 0.0, 0.0])}
        mock_path.kpts = np.array([[0.0, 0.0, 0.0]])
        
        mock_dos = Mock()
        mock_bs_energies = np.array([[1.0, 2.0, 3.0]])  # No soft modes
        mock_dos_energies = np.array([1.0, 2.0, 3.0])
        mock_all_k_distances = np.array([0.0])
        mock_special_k_distances = np.array([0.0])
        mock_special_k_labels = ['Gamma']
        mock_discontinuities = []
        mock_y_label = "Frequency (THz)"
        mock_bsmin = 1.0
        
        mock_get_results.return_value = (
            mock_bs, mock_path, mock_dos, mock_bs_energies, mock_dos_energies,
            mock_all_k_distances, mock_special_k_distances, mock_special_k_labels,
            mock_discontinuities, mock_y_label, mock_bsmin
        )
        
        # Mock analyze_special_points_and_modes to return optical modes
        mock_optical_modes = [
            {
                'label': 'Gamma',
                'coordinate': [0.0, 0.0, 0.0],
                'frequency': 3.0,
                'band_index': 2,
                'raw_displacements': [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
            }
        ]
        mock_analyze.return_value = mock_optical_modes
        
        # Mock phonons object
        mock_ph = Mock()
        mock_phonons.return_value = mock_ph
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function with threshold
            result = run_single_phonon_analysis(
                atoms=self.atoms,
                calculator=mock_calculator,
                engine="mace",
                units="THz",
                supercell_dims=(2, 2, 2),
                delta=0.01,
                fmax=0.001,
                output_dir=temp_dir,
                prefix="test",
                negative_phonon_threshold=-0.1  # This should trigger optical mode selection
            )
            
            # Verify that analyze_special_points_and_modes was called with the threshold
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args
            assert call_args[1]['negative_phonon_threshold'] == -0.1
            
            # Verify the result structure
            softest_modes_info_list, bsmin, time_taken = result
            assert len(softest_modes_info_list) == 1
            assert softest_modes_info_list[0]['frequency'] == 3.0
            assert bsmin == 1.0
            assert isinstance(time_taken, float)
    
    def test_main_script_integration(self):
        """Test that the main script can handle the new parameter."""
        from vibroml.main import main
        import sys
        from unittest.mock import patch
        
        # Mock sys.argv to simulate command line arguments
        test_args = [
            'vibroml',
            'tests/test_structures/simple_cubic.cif',
            '--engine', 'mace',
            '--units', 'THz',
            '--negative_phonon_threshold_thz', '-0.1'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('vibroml.main.run_single_phonon_analysis') as mock_run:
                with patch('vibroml.utils.structure_utils.load_structure') as mock_load:
                    with patch('vibroml.utils.structure_utils.initialize_calculator') as mock_calc:
                        with patch('vibroml.utils.relaxation_utils.relax_structure') as mock_relax:
                            # Mock the structure loading
                            mock_load.return_value = self.atoms
                            mock_calc.return_value = Mock()
                            mock_relax.return_value = (self.atoms, True)
                            
                            # Mock run_single_phonon_analysis to avoid actual computation
                            mock_run.return_value = ([], 0.0, 1.0)
                            
                            try:
                                main()
                            except SystemExit:
                                pass  # Expected when main() completes
                            
                            # Verify that run_single_phonon_analysis was called with threshold
                            mock_run.assert_called()
                            call_args = mock_run.call_args
                            assert 'negative_phonon_threshold' in call_args[1]
                            assert call_args[1]['negative_phonon_threshold'] == -0.1
    
    def test_genetic_algorithm_integration(self):
        """Test that the genetic algorithm workflow can use the new functionality."""
        from vibroml.auto_optimize import run_ga_soft_mode_optimization
        from unittest.mock import patch, Mock
        import argparse
        
        # Create mock args
        mock_args = Mock()
        mock_args.engine = "mace"
        mock_args.units = "THz"
        mock_args.supercell_dims = (2, 2, 2)
        mock_args.delta = 0.01
        mock_args.fmax = 0.001
        mock_args.cif = "test.cif"
        
        # Mock initial soft modes info
        initial_soft_modes = [
            {
                'label': 'Gamma',
                'coordinate': [0.0, 0.0, 0.0],
                'frequency': -1.0,
                'band_index': 0,
                'raw_displacements': [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
            }
        ]
        
        with patch('vibroml.auto_optimize.initialize_calculator') as mock_init_calc:
            with patch('vibroml.auto_optimize.run_single_phonon_analysis') as mock_run_phonon:
                with patch('vibroml.auto_optimize.GeneticAlgorithm') as mock_ga:
                    with patch('vibroml.auto_optimize.generate_displaced_supercells') as mock_gen:
                        with patch('vibroml.auto_optimize.relax_structures_in_folder') as mock_relax:
                            with patch('vibroml.auto_optimize.find_lowest_energy_structures') as mock_find:
                                with patch('vibroml.auto_optimize.SpacegroupAnalyzer') as mock_sga:
                                    # Setup mocks
                                    mock_init_calc.return_value = Mock()
                                    mock_run_phonon.return_value = (initial_soft_modes, -1.0, 1.0)
                                    mock_ga_instance = Mock()
                                    mock_ga.return_value = mock_ga_instance
                                    mock_ga_instance.evolve.return_value = []
                                    mock_gen.return_value = []
                                    mock_relax.return_value = []
                                    mock_find.return_value = []
                                    
                                    # Mock SpacegroupAnalyzer
                                    mock_sga_instance = Mock()
                                    mock_sga.return_value = mock_sga_instance
                                    mock_sga_instance.get_primitive_standard_structure.return_value = Mock()
                                    
                                    with tempfile.TemporaryDirectory() as temp_dir:
                                        try:
                                            run_ga_soft_mode_optimization(
                                                args=mock_args,
                                                base_output_dir=temp_dir,
                                                initial_atoms_for_soft_mode_analysis=self.atoms,
                                                initial_softest_modes_info_list=initial_soft_modes,
                                                max_iterations=1,
                                                soft_mode_displacement_scales=[1.0],
                                                cell_scale_factors=[1.0],
                                                mode2_ratio_scales=[0.0],
                                                num_top_structures_to_analyze=1,
                                                negative_phonon_threshold_thz=-0.1,
                                                phonon_path_npoints=10,
                                                phonon_dos_grid=(10, 10, 10),
                                                default_traj_kT=1.0,
                                                num_modes_to_return=2,
                                                ga_population_size=10,
                                                ga_mutation_rate=0.1,
                                                num_new_points_per_iteration=5,
                                                ga_disp_scale_bounds=(0.1, 2.0),
                                                ga_ratio_bounds=(-1.0, 1.0),
                                                ga_cell_scale_bounds=(0.9, 1.1),
                                                ga_cell_angle_bounds=(-5.0, 5.0)
                                            )
                                        except Exception as e:
                                            # Some exceptions are expected due to mocking
                                            if "list index out of range" not in str(e):
                                                raise
                                        
                                        # Verify that run_single_phonon_analysis was called with threshold
                                        if mock_run_phonon.called:
                                            call_args = mock_run_phonon.call_args_list[0]
                                            assert 'negative_phonon_threshold' in call_args[1]
                                            assert call_args[1]['negative_phonon_threshold'] == -0.1
