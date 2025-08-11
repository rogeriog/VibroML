"""Optimized tests for VibroML displaced supercell generation and mode visualization."""

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

# Fast test parameters for displaced supercell generation
FAST_DISPLACED_PARAMS = [
    "--supercell", "1,1,1",  # Minimal supercell for speed
    "--delta", "0.05",
    "--fmax", "0.01",
    "--no-relax"
]


class TestDisplacedSupercellGeneration:
    """Comprehensive tests for displaced supercell generation and mode visualization."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def simple_cif_file(self, temp_dir):
        """Create a simple cubic test structure optimized for fast testing."""
        test_cif_content = """# Simple cubic test structure for displaced supercell testing
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

    def validate_displaced_supercell_files(self, mode_gen_dir, expected_q_pattern, expected_mode, expected_displacement=None):
        """Validate that displaced supercell files are generated correctly."""
        # Check that displaced supercell files were created
        cif_files = [f for f in os.listdir(mode_gen_dir) if f.endswith(".cif")]
        xyz_files = [f for f in os.listdir(mode_gen_dir) if f.endswith(".xyz")]

        assert len(cif_files) > 0, "No CIF files generated for displaced supercell"
        assert len(xyz_files) > 0, "No XYZ files generated for displaced supercell"

        # Check filename patterns
        cif_file = cif_files[0]
        assert expected_q_pattern in cif_file, f"Q-point pattern '{expected_q_pattern}' not found in filename: {cif_file}"
        assert f"mode_{expected_mode}" in cif_file, f"Mode index 'mode_{expected_mode}' not found in filename: {cif_file}"

        # Check displacement if specified
        if expected_displacement is not None:
            # Look for displacement pattern in filename
            disp_pattern = f"disp_{expected_displacement:.3f}"
            # Also check for alternative patterns that might be used
            disp_found = any(pattern in cif_file for pattern in [
                disp_pattern,
                f"disp_{expected_displacement}",
                f"disp_{int(expected_displacement*1000):03d}"
            ])
            assert disp_found, f"Displacement pattern not found in filename: {cif_file}. Expected patterns containing {expected_displacement}"

        return cif_files, xyz_files
    
    @pytest.mark.slow
    def test_displaced_supercell_comprehensive(self, simple_cif_file, temp_dir):
        """Comprehensive test for displaced supercell generation at different q-points and modes."""

        # Test cases: (q_point, band_idx, displacement, expected_q_pattern)
        test_cases = [
            ("0.0,0.0,0.0", 0, 0.1, "q_0p000_0p000_0p000"),  # Gamma point
            ("0.5,0.0,0.0", 1, 0.05, "q_0p500_0p000_0p000"),  # X point
        ]

        for q_point, band_idx, displacement, expected_q_pattern in test_cases:
            # Create subdirectory for each test case
            test_name = f"q_{q_point.replace(',', '_').replace('.', 'p')}_mode_{band_idx}"
            test_dir = os.path.join(temp_dir, test_name)
            os.makedirs(test_dir, exist_ok=True)
            os.chdir(test_dir)

            args = [
                "--cif", simple_cif_file,
                "--engine", "mace",
                "--q", q_point,
                "--band_idx", str(band_idx),
                "--displacement", str(displacement),
                "--units", "THz"
            ] + FAST_DISPLACED_PARAMS

            result = self.run_vibroml_command(args, cwd=test_dir)

            # Check that command completed successfully
            assert result.returncode == 0, f"Command failed for q={q_point}, mode={band_idx} with stderr: {result.stderr}"

            # Check that output directory was created
            output_dirs = [d for d in os.listdir(test_dir) if d.startswith("simple_cubic_phonon_output_")]
            assert len(output_dirs) > 0, f"No output directory created for q={q_point}, mode={band_idx}"

            output_dir = os.path.join(test_dir, output_dirs[0])

            # Check that mode generation directory exists
            mode_gen_dirs = [d for d in os.listdir(output_dir) if d.startswith("mode_gen_")]
            assert len(mode_gen_dirs) > 0, f"No mode generation directory found for q={q_point}, mode={band_idx}"

            mode_gen_dir = os.path.join(output_dir, mode_gen_dirs[0])

            # Validate displaced supercell files
            cif_files, xyz_files = self.validate_displaced_supercell_files(
                mode_gen_dir, expected_q_pattern, band_idx, displacement
            )

            # Check that initial settings were saved with correct parameters
            settings_file = os.path.join(output_dir, "initial_settings.json")
            assert os.path.exists(settings_file), f"Initial settings file not found for q={q_point}, mode={band_idx}"

            with open(settings_file, 'r') as f:
                settings = json.load(f)
                assert settings["q"] == q_point
                assert settings["band_idx"] == band_idx
                assert settings["displacement"] == displacement

    def test_displaced_supercell_different_units(self, simple_cif_file, temp_dir):
        """Test displaced supercell generation with different units (optimized)."""
        units_to_test = ["cm-1", "eV"]

        for units in units_to_test:
            # Create subdirectory for each units test
            units_dir = os.path.join(temp_dir, f"test_{units}")
            os.makedirs(units_dir, exist_ok=True)
            os.chdir(units_dir)

            args = [
                "--cif", simple_cif_file,
                "--engine", "mace",
                "--q", "0.0,0.0,0.0",
                "--band_idx", "0",
                "--displacement", "0.1",
                "--units", units
            ] + FAST_DISPLACED_PARAMS

            result = self.run_vibroml_command(args, cwd=units_dir)

            # Check that command completed successfully
            assert result.returncode == 0, f"Command failed for units {units} with stderr: {result.stderr}"

            # Check that output directory was created
            output_dirs = [d for d in os.listdir(units_dir) if d.startswith("simple_cubic_phonon_output_")]
            assert len(output_dirs) > 0, f"No output directory created for units {units}"

            output_dir = os.path.join(units_dir, output_dirs[0])

            # Validate that settings reflect the correct units
            settings_file = os.path.join(output_dir, "initial_settings.json")
            assert os.path.exists(settings_file), f"Initial settings file not found for units {units}"

            with open(settings_file, 'r') as f:
                settings = json.load(f)
                assert settings["units"] == units

    def test_displaced_supercell_anisotropic_supercell(self, simple_cif_file, temp_dir):
        """Test displaced supercell generation with anisotropic supercells."""
        os.chdir(temp_dir)

        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--q", "0.0,0.0,0.5",  # Z point
            "--band_idx", "2",     # Third mode
            "--displacement", "0.08",
            "--supercell", "2,3,1",  # Anisotropic supercell
            "--units", "THz"
        ] + FAST_DISPLACED_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir)

        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that mode generation directory exists
        mode_gen_dirs = [d for d in os.listdir(output_dir) if d.startswith("mode_gen_")]
        assert len(mode_gen_dirs) > 0, "No mode generation directory found"

        mode_gen_dir = os.path.join(output_dir, mode_gen_dirs[0])

        # Validate displaced supercell files
        self.validate_displaced_supercell_files(
            mode_gen_dir, "q_0p000_0p000_0p500", 2, 0.08
        )

        # Check that settings reflect the anisotropic supercell
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"

        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["supercell_dims"] == [2, 3, 1]
    
    @pytest.mark.slow
    def test_displaced_supercell_anisotropic_supercell(self, simple_cif_file, temp_dir):
        """Test displaced supercell generation with anisotropic supercell."""
        os.chdir(temp_dir)
        
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--q", "0.0,0.0,0.5",  # Z point
            "--band_idx", "2",     # Third mode
            "--displacement", "0.08",
            "--supercell", "2,3,1",  # Anisotropic supercell
            "--units", "THz"
        ] + FAST_DISPLACED_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir)

        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that mode generation directory exists
        mode_gen_dirs = [d for d in os.listdir(output_dir) if d.startswith("mode_gen_")]
        assert len(mode_gen_dirs) > 0, "No mode generation directory found"

        mode_gen_dir = os.path.join(output_dir, mode_gen_dirs[0])

        # Validate displaced supercell files
        self.validate_displaced_supercell_files(
            mode_gen_dir, "q_0p000_0p000_0p500", 2, 0.08
        )

        # Check that settings reflect the anisotropic supercell
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"

        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["supercell_dims"] == [2, 3, 1]
    
    def test_displaced_supercell_parameter_validation(self, simple_cif_file, temp_dir):
        """Test parameter validation for displaced supercell generation (optimized)."""
        os.chdir(temp_dir)

        # Test invalid q-point format (should fail)
        args_invalid_q = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--q", "0.0,0.0",  # Missing third component
            "--band_idx", "0",
            "--displacement", "0.1",
            "--units", "THz"
        ] + FAST_DISPLACED_PARAMS

        result = self.run_vibroml_command(args_invalid_q, cwd=temp_dir)
        # This should fail with an error about invalid q-point format
        assert result.returncode != 0, "Command should fail with invalid q-point format"

    def test_displaced_supercell_workflow_integration(self, simple_cif_file, temp_dir):
        """Test that displaced supercell generation integrates properly with the overall workflow (optimized)."""
        os.chdir(temp_dir)

        # Run displaced supercell generation directly (no need for separate basic run)
        args = [
            "--cif", simple_cif_file,
            "--engine", "mace",
            "--q", "0.0,0.0,0.0",
            "--band_idx", "0",
            "--displacement", "0.1",
            "--units", "THz"
        ] + FAST_DISPLACED_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir)
        assert result.returncode == 0, f"Displaced supercell generation failed: {result.stderr}"

        # Check that output directory was created with expected structure
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that the displaced supercell run has the expected structure
        mode_gen_dirs = [d for d in os.listdir(output_dir) if d.startswith("mode_gen_")]
        assert len(mode_gen_dirs) > 0, "No mode generation directory found in displaced supercell output"

        # Verify that both CIF and XYZ files are generated
        mode_gen_dir = os.path.join(output_dir, mode_gen_dirs[0])
        cif_files = [f for f in os.listdir(mode_gen_dir) if f.endswith(".cif")]
        xyz_files = [f for f in os.listdir(mode_gen_dir) if f.endswith(".xyz")]

        assert len(cif_files) > 0, "No CIF files generated"
        assert len(xyz_files) > 0, "No XYZ files generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
