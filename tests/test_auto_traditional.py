import pytest
import os
import shutil
import subprocess
import json
import time

# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CIF file, which is inside a subdirectory of the tests folder.
CIF_FILE_PATH = os.path.join(SCRIPT_DIR, "test_structures", "simple_cubic.cif")


# Define the path to the Conda environment
CONDA_ENV_PATH = "/globalscratch/ucl/modl/rgouvea/vibroml_env/"
VIBROML_COMMAND = os.path.join(CONDA_ENV_PATH, "bin", "vibroml")

# Define a base directory for test outputs
TEST_OUTPUT_BASE_DIR = "vibroml_test_outputs"

# --- EXPANDED Simplified parameters for fast testing ---
# This now includes overrides for the extensive grid search parameters.
SIMPLIFIED_PARAMS = [
    # Screening parameters - all lists will have length 1
    "--screen_supercell_ns", "2",
    "--screen_deltas", "0.03",
    "--screen_fmax_values", "0.01",

    # Grid search and soft mode parameters (set to minimal values)
    "--soft_mode_max_iterations", "1",
    "--soft_mode_displacement_scales", "1.0",
    "--soft_mode_num_top_structures_to_analyze", "1",
    "--cell_scale_factors", "0.0",
    "--mode2_ratio_scales", "0.0",
    "--num_modes_to_return", "1",
    
    # GA parameters (overridden for safety, though not used in traditional mode)
    "--ga_population_size", "2",
    "--num_new_points_per_iteration", "1",
    "--ga_mutation_rate", "0.1"
]

class TestAutoTraditionalMode:

    @classmethod
    def setup_class(cls):
        """
        Set up the test environment before any tests in this class run.
        """
        print(f"\n--- Setting up test class: {cls.__name__} ---")
        if os.path.exists(TEST_OUTPUT_BASE_DIR):
            shutil.rmtree(TEST_OUTPUT_BASE_DIR)
        os.makedirs(TEST_OUTPUT_BASE_DIR, exist_ok=True)
        print(f"  Created base test output directory: {TEST_OUTPUT_BASE_DIR}")

        print(f"  Verifying CIF file exists at: {CIF_FILE_PATH}")
        if not os.path.isfile(CIF_FILE_PATH):
            pytest.fail(f"CIF file not found at expected path: {CIF_FILE_PATH}.")
        print(f"  CIF file found: {CIF_FILE_PATH}")


    @classmethod
    def teardown_class(cls):
        """
        Clean up the test environment after all tests in this class have run.
        """
        print(f"\n--- Tearing down test class: {cls.__name__} ---")
        if os.path.exists(TEST_OUTPUT_BASE_DIR):
            shutil.rmtree(TEST_OUTPUT_BASE_DIR)
        print("--- Test class teardown complete ---")

    def setup_method(self, method):
        """
        Set up for each test method.
        """
        self.test_name = method.__name__
        self.current_test_dir = os.path.join(TEST_OUTPUT_BASE_DIR, self.test_name)
        print(f"\n--- Starting test method: {self.test_name} ---")
        os.makedirs(self.current_test_dir, exist_ok=True)
        
        self.test_cif_path = os.path.join(self.current_test_dir, os.path.basename(CIF_FILE_PATH))
        shutil.copy(CIF_FILE_PATH, self.test_cif_path)


    def teardown_method(self, method):
        """
        Clean up after each test method.
        """
        print(f"--- Finished test method: {self.test_name} ---")

    def _run_vibroml_command(self, args, cwd):
        """Helper to run vibroml command and capture output."""
        command = [VIBROML_COMMAND] + args
        print(f"  Executing command: {' '.join(command)} in {cwd}")
        start_time = time.time()
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                env={"PATH": os.environ["PATH"], "CONDA_PREFIX": CONDA_ENV_PATH}
            )
            end_time = time.time()
            print(f"  Command executed successfully in {end_time - start_time:.2f} seconds.")
            print(f"  STDOUT:\n{process.stdout}")
            if process.stderr:
                print(f"  STDERR:\n{process.stderr}")
            return process.stdout
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            print(f"  Command failed after {end_time - start_time:.2f} seconds.")
            print(f"  Command failed with exit code {e.returncode}")
            print(f"  STDOUT:\n{e.stdout}")
            print(f"  STDERR:\n{e.stderr}")
            pytest.fail(f"VibroML command failed: {e}")

    def _validate_output_files(self, base_dir, expected_files):
        """Validates that expected files exist."""
        for f in expected_files:
            path = os.path.join(base_dir, f)
            assert os.path.isfile(path), f"Expected file '{path}' not found."

    def _load_json_file(self, file_path):
        """Loads and returns content of a JSON file."""
        assert os.path.isfile(file_path), f"JSON file not found: {file_path}"
        with open(file_path, 'r') as f:
            return json.load(f)

    def test_basic_auto_mode(self):
        """
        Test the basic auto mode functionality with simplified, fast parameters.
        """
        print("#### Running test_basic_auto_mode ####")
        
        cif_filename = os.path.basename(self.test_cif_path)
        command_args = [
            "--auto", "--method", "traditional", "--cif", cif_filename, 
            "--fmax", "0.01", "--supercell", "1,1,1"
        ] + SIMPLIFIED_PARAMS
        
        self._run_vibroml_command(command_args, cwd=self.current_test_dir)

        output_dirs = [d for d in os.listdir(self.current_test_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) == 1, "Expected exactly one output directory."
        output_dir = os.path.join(self.current_test_dir, output_dirs[0])

        self._validate_output_files(output_dir, ["initial_settings.json"])
        
        settings_data = self._load_json_file(os.path.join(output_dir, "initial_settings.json"))
        
        # --- VALIDATION: Check that ALL our overrides were applied ---
        assert settings_data["supercell_dims"] == [1, 1, 1]
        assert settings_data["fmax"] == 0.01
        
        # --- THIS IS THE CORRECTED LINE ---
        # Your script converts the input '2' to [[2, 2, 2]] before saving.
        assert settings_data["screen_supercell_ns"] == [[2, 2, 2]]
        
        assert settings_data["screen_deltas"] == [0.03]
        assert settings_data["screen_fmax_values"] == [0.01]
        
        # Check the new grid search overrides
        assert settings_data["soft_mode_max_iterations"] == 1
        assert settings_data["soft_mode_displacement_scales"] == [1.0]
        assert settings_data["soft_mode_num_top_structures_to_analyze"] == 1
        assert settings_data["cell_scale_factors"] == [0.0]
        assert settings_data["mode2_ratio_scales"] == [0.0]
        assert settings_data["num_modes_to_return"] == 1
        
        print("#### test_basic_auto_mode PASSED ####")

    def test_anisotropic_supercells(self):
        """
        Test auto mode with anisotropic supercells and fast parameters.
        """
        print("#### Running test_anisotropic_supercells ####")
        
        cif_filename = os.path.basename(self.test_cif_path)
        command_args = [
            "--auto", "--method", "traditional", "--cif", cif_filename, 
            "--fmax", "0.01", "--supercell", "1,2,1"
        ] + SIMPLIFIED_PARAMS

        self._run_vibroml_command(command_args, cwd=self.current_test_dir)

        output_dirs = [d for d in os.listdir(self.current_test_dir) if d.startswith("simple_cubic_phonon_output_")]
        assert len(output_dirs) == 1, "Expected exactly one output directory."
        output_dir = os.path.join(self.current_test_dir, output_dirs[0])
        
        settings_data = self._load_json_file(os.path.join(output_dir, "initial_settings.json"))
        assert settings_data["supercell_dims"] == [1, 2, 1]
        print("#### test_anisotropic_supercells PASSED ####")

    