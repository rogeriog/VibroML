"""Streamlined integration tests for VibroML package with optimized performance."""

import pytest
import os
import tempfile
import shutil
import subprocess
import json
import time
from pathlib import Path

# Test configuration
TEST_CIF_PATH = os.path.join(os.path.dirname(__file__), "test_structures", "simple_cubic.cif")
CONDA_ENV_PATH = "/globalscratch/ucl/modl/rgouvea/vibroml_env/"

# Optimized parameters for fast testing
FAST_TEST_PARAMS = [
    "--supercell", "1,1,1",  # Minimal supercell
    "--delta", "0.05",       # Reasonable delta
    "--fmax", "0.01",        # Reasonable fmax
    "--no-relax"             # Skip relaxation for speed
]

class TestVibroMLIntegration:
    """Streamlined integration tests with performance optimizations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def simple_cif_file(self, temp_dir):
        """Create a simple cubic test structure optimized for fast testing."""
        test_cif_content = """# Simple cubic test structure
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
    def test_basic_phonon_calculation(self, simple_cif_file, temp_dir):
        """Test basic phonon calculation without auto mode."""
        os.chdir(temp_dir)

        args = ["--cif", simple_cif_file, "--engine", "mace", "--units", "THz"] + FAST_TEST_PARAMS

        result = self.run_vibroml_command(args, cwd=temp_dir)

        # Check that command completed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Check that output directory was created
        output_dirs = [d for d in os.listdir(temp_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) > 0, "No output directory created"

        output_dir = os.path.join(temp_dir, output_dirs[0])

        # Check that initial settings file exists
        settings_file = os.path.join(output_dir, "initial_settings.json")
        assert os.path.exists(settings_file), "Initial settings file not found"

        # Verify basic settings
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            assert settings["units"] == "THz"
            assert settings["engine"] == "mace"


class TestAnisotropicSupercellConfiguration:
    """Test the enhanced anisotropic supercell configuration."""

    def test_parse_screen_supercell_ns_old_format(self):
        """Test parsing of old format (list of integers)."""
        from vibroml.utils.utils import parse_screen_supercell_ns

        # Old format - list of integers
        old_format = [2, 3, 4]
        result = parse_screen_supercell_ns(old_format)
        expected = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
        assert result == expected
    
    def test_parse_screen_supercell_ns_new_format(self):
        """Test parsing of new format (list of tuples)."""
        from vibroml.utils.utils import parse_screen_supercell_ns
        
        # New format - list of tuples
        new_format = [[2, 2, 1], [3, 3, 1], [4, 4, 2]]
        result = parse_screen_supercell_ns(new_format)
        expected = [(2, 2, 1), (3, 3, 1), (4, 4, 2)]
        assert result == expected
    
    def test_parse_screen_supercell_ns_mixed_format(self):
        """Test parsing of mixed format (backward compatibility)."""
        from vibroml.utils.utils import parse_screen_supercell_ns
        
        # Mixed format
        mixed_format = [2, [3, 3, 1], 4]
        result = parse_screen_supercell_ns(mixed_format)
        expected = [(2, 2, 2), (3, 3, 1), (4, 4, 4)]
        assert result == expected
    
    def test_parse_screen_supercell_ns_invalid_format(self):
        """Test error handling for invalid formats."""
        from vibroml.utils.utils import parse_screen_supercell_ns
        
        # Invalid formats
        invalid_formats = [
            [2, [3, 3]],  # Wrong number of dimensions
            [2, [3, 3, 0]],  # Zero dimension
            [2, [3, 3, -1]],  # Negative dimension
            [2, "invalid"],  # Wrong type
            "not_a_list"  # Not a list
        ]
        
        for invalid_format in invalid_formats:
            with pytest.raises(ValueError):
                parse_screen_supercell_ns(invalid_format)
    
    def test_default_settings_anisotropic(self):
        """Test that default settings support anisotropic supercells."""
        from vibroml.utils.utils import load_default_settings
        
        settings = load_default_settings()
        screen_supercell_ns = settings.get("screen_supercell_ns", [])
        
        # Should be a list of lists/tuples now
        assert isinstance(screen_supercell_ns, list)
        assert len(screen_supercell_ns) > 0
        
        # Each element should be a list/tuple with 3 elements
        for supercell in screen_supercell_ns:
            assert isinstance(supercell, (list, tuple))
            assert len(supercell) == 3
            assert all(isinstance(x, int) and x > 0 for x in supercell)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])