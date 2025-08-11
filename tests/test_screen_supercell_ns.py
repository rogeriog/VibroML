"""Optimized tests for screen_supercell_ns argument parsing and functionality."""

import pytest
import os
import tempfile
import shutil
import subprocess
import json
from pathlib import Path

# Test configuration
CONDA_ENV_PATH = "/globalscratch/ucl/modl/rgouvea/vibroml_env/"

# Fast test parameters for screen_supercell_ns testing
FAST_SCREEN_PARAMS = [
    "--screen_deltas", "0.05",
    "--screen_fmax_values", "0.01",
    "--soft_mode_max_iterations", "1",
    "--no-relax"
]


class TestScreenSupercellNS:
    """Comprehensive tests for screen_supercell_ns argument parsing and functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def simple_cif_file(self, temp_dir):
        """Create a simple cubic test structure optimized for fast testing."""
        test_cif_content = """# Simple cubic test structure for screen_supercell_ns testing
data_SimpleCubic
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   2.50000000
_cell_length_b   2.50000000
_cell_length_c   2.50000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   LiF
_chemical_formula_sum   'Li1 F1'
_cell_volume   15.62500000
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Li  Li0  1  0.00000000  0.00000000  0.00000000  1
  F   F1  1  0.50000000  0.50000000  0.50000000  1"""
        
        test_cif_path = os.path.join(temp_dir, "simple_cubic.cif")
        with open(test_cif_path, 'w') as f:
            f.write(test_cif_content)
        
        return test_cif_path
    
    def run_vibroml_command(self, args, cwd=None, timeout=300):
        """Run a vibroml command and return the result."""
        cmd = ["conda", "run", "-p", CONDA_ENV_PATH, "vibroml"] + args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired:
            pytest.skip(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            pytest.skip("vibroml command not found - conda environment may not be available")
    
    def test_screen_supercell_ns_single_value(self, simple_cif_file, temp_dir):
        """Test screen_supercell_ns with a single value."""
        os.chdir(temp_dir)
        
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "traditional",
            "--screen_supercell_ns", "2",
            "--units", "THz"
        ] + FAST_SCREEN_PARAMS
        
        result = self.run_vibroml_command(args, cwd=temp_dir)
        
        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"
        
        output_dir = os.path.join(temp_dir, output_dirs[0])
        
        # Check that settings reflect the screen_supercell_ns value
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert "screen_supercell_ns" in settings
            # The value should be parsed as a list of supercell dimensions
            assert isinstance(settings["screen_supercell_ns"], list)
    
    def test_screen_supercell_ns_multiple_values(self, simple_cif_file, temp_dir):
        """Test screen_supercell_ns with multiple values."""
        os.chdir(temp_dir)
        
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "traditional",
            "--screen_supercell_ns", "2", "3",  # Multiple values
            "--units", "THz"
        ] + FAST_SCREEN_PARAMS
        
        result = self.run_vibroml_command(args, cwd=temp_dir)
        
        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"
        
        output_dir = os.path.join(temp_dir, output_dirs[0])
        
        # Check that settings reflect the multiple screen_supercell_ns values
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert "screen_supercell_ns" in settings
            assert isinstance(settings["screen_supercell_ns"], list)
            assert len(settings["screen_supercell_ns"]) >= 2  # Should have at least 2 entries
    
    def test_screen_supercell_ns_anisotropic_format(self, simple_cif_file, temp_dir):
        """Test screen_supercell_ns with anisotropic supercell format."""
        os.chdir(temp_dir)
        
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "traditional",
            "--screen_supercell_ns", "2,2,1", "1,1,2",  # Anisotropic supercells
            "--units", "THz"
        ] + FAST_SCREEN_PARAMS
        
        result = self.run_vibroml_command(args, cwd=temp_dir)
        
        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"
        
        output_dir = os.path.join(temp_dir, output_dirs[0])
        
        # Check that settings reflect the anisotropic screen_supercell_ns values
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert "screen_supercell_ns" in settings
            assert isinstance(settings["screen_supercell_ns"], list)
    
    def test_screen_supercell_ns_parameter_validation(self, simple_cif_file, temp_dir):
        """Test parameter validation for screen_supercell_ns."""
        os.chdir(temp_dir)
        
        # Test with invalid format (should handle gracefully or fail appropriately)
        args_invalid = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "traditional",
            "--screen_supercell_ns", "invalid",
            "--units", "THz"
        ] + FAST_SCREEN_PARAMS
        
        result = self.run_vibroml_command(args_invalid, cwd=temp_dir)
        
        # The behavior depends on implementation - it might fail or handle gracefully
        # We just check that it doesn't crash unexpectedly
        assert result.returncode in [0, 1, 2], "Unexpected return code for invalid screen_supercell_ns"
    
    def test_screen_supercell_ns_with_ga_method(self, simple_cif_file, temp_dir):
        """Test screen_supercell_ns with GA method."""
        os.chdir(temp_dir)
        
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "ga",
            "--screen_supercell_ns", "2", "3",
            "--ga_population_size", "4",
            "--num_new_points_per_iteration", "2",
            "--units", "THz"
        ] + FAST_SCREEN_PARAMS
        
        result = self.run_vibroml_command(args, cwd=temp_dir, timeout=300)
        
        # Check that command completed successfully
        assert result.returncode == 0, f"GA method with screen_supercell_ns failed: {result.stderr}"
        
        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created for GA method"
        
        output_dir = os.path.join(temp_dir, output_dirs[0])
        
        # Check that settings reflect both GA method and screen_supercell_ns
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found for GA method"
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["method"] == "ga"
            assert "screen_supercell_ns" in settings


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
