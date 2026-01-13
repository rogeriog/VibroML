#!/usr/bin/env python3

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, Mock
import numpy as np

# Add the parent directory to the path so we can import vibroml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vibroml.utils.neb_utils import (
    linear_interpolate_structures, 
    calculate_tangent, 
    calculate_spring_force,
    calculate_neb_forces,
    check_neb_convergence,
    update_image_positions,
    find_highest_energy_image,
    run_neb_optimization
)
from vibroml.utils.structure_utils import load_structure
from vibroml.auto_optimize import run_neb_soft_mode_optimization, run_ci_neb_soft_mode_optimization


class TestNEBUtils:
    """Test the core NEB utility functions."""
    
    def test_linear_interpolate_structures(self):
        """Test linear interpolation between two structures."""
        # Load test structures
        initial_cif = "test_structures/simple_cubic.cif"
        final_cif = "test_structures/hexagonal.cif"
        
        if not os.path.exists(initial_cif) or not os.path.exists(final_cif):
            pytest.skip("Test structures not available")
        
        _, initial_atoms = load_structure(initial_cif)
        _, final_atoms = load_structure(final_cif)
        
        if initial_atoms is None or final_atoms is None:
            pytest.skip("Could not load test structures")
        
        # Test interpolation with 3 intermediate images
        num_images = 3
        images = linear_interpolate_structures(initial_atoms, final_atoms, num_images)
        
        # Should have initial + intermediate + final = 5 total images
        assert len(images) == num_images + 2
        
        # First and last should match input structures
        assert len(images[0]) == len(initial_atoms)
        assert len(images[-1]) == len(final_atoms)
        
        # Check that intermediate images have reasonable positions
        initial_pos = initial_atoms.get_positions()
        final_pos = final_atoms.get_positions()
        
        for i in range(1, num_images + 1):
            img_pos = images[i].get_positions()
            # Each intermediate image should be between initial and final
            alpha = i / (num_images + 1)
            expected_pos = (1 - alpha) * initial_pos + alpha * final_pos
            np.testing.assert_allclose(img_pos, expected_pos, rtol=1e-10)
    
    def test_calculate_tangent(self):
        """Test tangent calculation for NEB."""
        # Create simple test structures
        from ase import Atoms
        
        # Simple 1D chain of atoms for easy testing
        prev_atoms = Atoms('H', positions=[[0, 0, 0]])
        curr_atoms = Atoms('H', positions=[[1, 0, 0]])
        next_atoms = Atoms('H', positions=[[2, 0, 0]])
        
        tangent = calculate_tangent(prev_atoms, curr_atoms, next_atoms)
        
        # For a straight line, tangent should be along x-direction
        expected_tangent = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-10)
    
    def test_calculate_spring_force(self):
        """Test spring force calculation."""
        from ase import Atoms
        
        # Create test structures with unequal spacing
        prev_atoms = Atoms('H', positions=[[0, 0, 0]])
        curr_atoms = Atoms('H', positions=[[1, 0, 0]])  # Distance 1 from prev
        next_atoms = Atoms('H', positions=[[3, 0, 0]])  # Distance 2 from curr
        
        tangent = np.array([1.0, 0.0, 0.0])  # Along x-direction
        spring_constant = 1.0
        
        spring_force = calculate_spring_force(prev_atoms, curr_atoms, next_atoms, tangent, spring_constant)
        
        # Spring force should try to equalize distances
        # Distance to next (2) > distance to prev (1), so force should be positive (toward next)
        expected_force = np.array([1.0, 0.0, 0.0])  # k * (2 - 1) * tangent
        np.testing.assert_allclose(spring_force, expected_force, rtol=1e-10)
    
    def test_check_neb_convergence(self):
        """Test NEB convergence checking."""
        # Test with converged forces
        small_forces = [np.array([0.01, 0.01, 0.01]), np.array([0.02, 0.01, 0.01])]
        converged, max_force = check_neb_convergence(small_forces, force_tolerance=0.05)
        assert converged
        assert max_force < 0.05
        
        # Test with non-converged forces
        large_forces = [np.array([0.1, 0.1, 0.1]), np.array([0.05, 0.05, 0.05])]
        converged, max_force = check_neb_convergence(large_forces, force_tolerance=0.05)
        assert not converged
        assert max_force > 0.05
    
    def test_find_highest_energy_image(self):
        """Test finding the highest energy intermediate image."""
        # Test energies: initial=1.0, intermediate=[2.0, 3.0, 1.5], final=1.2
        energies = [1.0, 2.0, 3.0, 1.5, 1.2]
        highest_idx = find_highest_energy_image(energies)
        
        # Should return index 2 (energy 3.0) among intermediate images
        assert highest_idx == 2
    
    def test_update_image_positions(self):
        """Test position updates for NEB images."""
        from ase import Atoms
        
        # Create test images
        images = [
            Atoms('H', positions=[[0, 0, 0]]),  # Fixed initial
            Atoms('H', positions=[[1, 0, 0]]),  # Intermediate
            Atoms('H', positions=[[2, 0, 0]])   # Fixed final
        ]
        
        # Force pointing in +y direction
        neb_forces = [np.array([0.0, 1.0, 0.0])]
        step_size = 0.1
        
        initial_pos = images[1].get_positions().copy()
        update_image_positions(images, neb_forces, step_size)
        
        # Only intermediate image should move
        expected_pos = initial_pos + step_size * np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(images[1].get_positions(), expected_pos, rtol=1e-10)


class TestNEBIntegration:
    """Integration tests for NEB methods."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('vibroml.utils.structure_utils.initialize_calculator')
    def test_neb_optimization_basic(self, mock_init_calc, temp_dir):
        """Test basic NEB optimization workflow."""
        # Skip if test structures not available
        initial_cif = "examples/LiF_simplecubic/LiFsimplecubic.cif"
        final_cif = "examples/LiF_simplecubic/GA_LiFsimplecubic_phonon_output_20250831-152432/final_structures/top_1_iter4_sample40_LiFsimplecubic_energy_m4p8264_conventional_freqp19p5107THz.cif"
        
        if not os.path.exists(initial_cif) or not os.path.exists(final_cif):
            pytest.skip("Test structures not available")
        
        _, initial_atoms = load_structure(initial_cif)
        _, final_atoms = load_structure(final_cif)
        
        if initial_atoms is None or final_atoms is None:
            pytest.skip("Could not load test structures")
        
        # Mock calculator
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -10.0
        mock_calculator.get_forces.return_value = np.zeros((len(initial_atoms), 3))
        mock_init_calc.return_value = mock_calculator
        
        # Run NEB optimization with minimal parameters
        results = run_neb_optimization(
            initial_atoms=initial_atoms,
            final_atoms=final_atoms,
            calculator=mock_calculator,
            num_images=3,
            spring_constant=1.0,
            max_iterations=5,  # Very few iterations for testing
            force_tolerance=0.1,  # Loose tolerance
            output_dir=temp_dir,
            prefix="test_neb"
        )
        
        # Check that results contain expected keys
        assert 'converged' in results
        assert 'final_max_force' in results
        assert 'iterations' in results
        assert 'optimization_time' in results
        assert 'images' in results
        assert 'convergence_history' in results
        
        # Check that images were created
        assert len(results['images']) == 5  # 3 intermediate + 2 endpoints
        
        # Check that output files were created
        assert os.path.exists(os.path.join(temp_dir, "images_iter_0000"))
    
    @patch('vibroml.utils.structure_utils.initialize_calculator')
    @patch('vibroml.utils.phonon_utils.run_single_phonon_analysis')
    def test_neb_soft_mode_optimization(self, mock_phonon_analysis, mock_init_calc, temp_dir):
        """Test the full NEB soft mode optimization workflow."""
        # Skip if test structures not available
        initial_cif = "examples/LiF_simplecubic/LiFsimplecubic.cif"
        final_cif = "examples/LiF_simplecubic/GA_LiFsimplecubic_phonon_output_20250831-152432/final_structures/top_1_iter4_sample40_LiFsimplecubic_energy_m4p8264_conventional_freqp19p5107THz.cif"
        
        if not os.path.exists(initial_cif) or not os.path.exists(final_cif):
            pytest.skip("Test structures not available")
        
        _, initial_atoms = load_structure(initial_cif)
        
        if initial_atoms is None:
            pytest.skip("Could not load initial structure")
        
        # Mock calculator
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -10.0
        mock_calculator.get_forces.return_value = np.zeros((len(initial_atoms), 3))
        mock_init_calc.return_value = mock_calculator
        
        # Mock phonon analysis
        mock_phonon_analysis.return_value = ([], -1.0, 10.0, {})
        
        # Mock args object
        mock_args = Mock()
        mock_args.cif = initial_cif
        mock_args.final_cif = final_cif
        mock_args.engine = "mace"
        mock_args.model_name = "medium"
        mock_args.units = "THz"
        mock_args.supercell_dims = (2, 2, 2)
        mock_args.delta = 0.01
        mock_args.fmax = 0.001
        
        # Run NEB soft mode optimization
        run_neb_soft_mode_optimization(
            args=mock_args,
            base_output_dir=temp_dir,
            initial_atoms_for_soft_mode_analysis=initial_atoms,
            initial_softest_modes_info_list=[],
            max_iterations=10,
            soft_mode_displacement_scales=[0.5, 1.0],
            cell_scale_factors=[0.0, 0.05],
            mode2_ratio_scales=[0.5, 1.0],
            num_top_structures_to_analyze=3,
            negative_phonon_threshold_thz=-0.5,
            phonon_path_npoints=50,
            phonon_dos_grid=(20, 20, 20),
            default_traj_kT=1.0,
            num_modes_to_return=2,
            neb_num_images=3,
            neb_spring_constant=1.0,
            neb_max_iterations=5,
            neb_force_tolerance=0.1,
            final_cif_path=final_cif
        )
        
        # Check that NEB output directory was created
        neb_dir = os.path.join(temp_dir, "neb_optimization")
        assert os.path.exists(neb_dir)
        
        # Check that summary file was created
        summary_file = os.path.join(neb_dir, "neb_summary.txt")
        assert os.path.exists(summary_file)


if __name__ == "__main__":
    pytest.main([__file__])
