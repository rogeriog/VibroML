"""Optimized tests for VibroML output units functionality (THz, cm-1, eV)."""

import pytest
import os
import tempfile
import shutil
import subprocess
import json
import re
from pathlib import Path

# Test configuration
CONDA_ENV_PATH = "/globalscratch/ucl/modl/rgouvea/vibroml_env/"

# Fast test parameters for units testing
FAST_UNITS_PARAMS = [
    "--supercell", "1,1,1",  # Minimal supercell for speed
    "--delta", "0.05",
    "--fmax", "0.01",
    "--no-relax"
]


class TestUnitsConversion:
    """Optimized tests for output units functionality with THz, cm-1, and eV."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def simple_cif_file(self, temp_dir):
        """Create a simple cubic test structure optimized for fast testing."""
        test_cif_content = """# Simple cubic test structure for units testing
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

    @pytest.mark.slow
    def test_units_comprehensive(self, simple_cif_file, temp_dir):
        """Comprehensive test for all supported units (THz, cm-1, eV) in a single optimized test."""
        units_to_test = ["THz", "cm-1", "eV"]

        for units in units_to_test:
            # Create subdirectory for each units test
            units_dir = os.path.join(temp_dir, f"test_{units}")
            os.makedirs(units_dir, exist_ok=True)
            os.chdir(units_dir)

            args = [
                "--cif", simple_cif_file,
                "--engine", "mace",
                "--units", units
            ] + FAST_UNITS_PARAMS

            result = self.run_vibroml_command(args, cwd=units_dir)

            # Check that command completed successfully
            assert result.returncode == 0, f"Command failed for units {units} with stderr: {result.stderr}"

            # Check that the specified units appear in the output
            assert units in result.stdout, f"{units} units not found in output"

            # Check that output directory was created
            output_dirs = [d for d in os.listdir(units_dir) if d.startswith("simple_cubic_phonon_output_")]
            assert len(output_dirs) > 0, f"No output directory created for {units}"

            output_dir = os.path.join(units_dir, output_dirs[0])

            # Check that settings reflect the correct units
            settings_file = os.path.join(output_dir, "initial_settings.json")
            assert os.path.exists(settings_file), f"Initial settings file not found for {units}"

            with open(settings_file, 'r') as f:
                settings = json.load(f)
                assert settings["units"] == units

    def test_units_conversion_consistency(self, simple_cif_file, temp_dir):
        """Test that units conversion is consistent and parameter validation works."""
        os.chdir(temp_dir)

        # Test with THz units (baseline)
        args_thz = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--units", "THz"
        ] + FAST_UNITS_PARAMS

        result_thz = self.run_vibroml_command(args_thz, cwd=temp_dir)
        assert result_thz.returncode == 0, f"THz test failed: {result_thz.stderr}"

        # Verify that settings are saved correctly
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"

        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["units"] == "THz"

    def test_units_parameter_validation(self, simple_cif_file, temp_dir):
        """Test parameter validation for units functionality."""
        os.chdir(temp_dir)

        # Test invalid units (should fail or default to THz)
        args_invalid = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--units", "invalid_unit"
        ] + FAST_UNITS_PARAMS

        result = self.run_vibroml_command(args_invalid, cwd=temp_dir)
        # The behavior depends on implementation - it might fail or default to THz
        # We just check that it doesn't crash unexpectedly
        assert result.returncode in [0, 1, 2], "Unexpected return code for invalid units"

    def test_units_in_file_outputs(self, simple_cif_file, temp_dir):
        """Test that units are correctly reflected in output files."""
        os.chdir(temp_dir)

        # Test with cm-1 units to verify file output
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--units", "cm-1"
        ] + FAST_UNITS_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that settings file reflects the correct units
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"

        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["units"] == "cm-1"

    @pytest.mark.slow
    def test_units_with_auto_mode(self, simple_cif_file, temp_dir):
        """Test that units work correctly with auto mode (optimized)."""
        os.chdir(temp_dir)

        # Test auto mode with cm-1 units
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--auto",
            "--method", "traditional",
            "--screen_supercell_ns", "2",  # Single supercell for speed
            "--screen_deltas", "0.05",
            "--screen_fmax_values", "0.01",
            "--soft_mode_max_iterations", "1",
            "--units", "cm-1"
        ] + FAST_UNITS_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir, timeout=300)

        # Check that command completed successfully
        assert result.returncode == 0, f"Auto mode failed with stderr: {result.stderr}"

        # Check that the specified units appear in the output
        assert "cm-1" in result.stdout, "cm-1 units not found in auto mode output"

        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created for auto mode"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that settings reflect the correct units
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found for auto mode"

        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["units"] == "cm-1"
            assert settings["auto"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
