### Tests Directory

This directory contains the comprehensive test suite for the VibroML package, designed to ensure the stability, correctness, and performance of its functionalities.

#### Structure

The tests are organized into several files, each focusing on specific aspects of the VibroML package:

*   `run_integration_tests.py`: The main script for orchestrating and running the integration and unit tests.
*   `test_auto_ga.py`: Tests specifically for the Genetic Algorithm (GA) auto mode.
*   `test_auto_traditional.py`: Tests specifically for the traditional auto mode.
*   `test_comprehensive_integration.py`: Contains a basic phonon calculation smoke test and unit tests for anisotropic supercell configuration parsing.
*   `test_displaced_supercell.py`: Tests for the generation of displaced supercells.
*   `test_existing_functions.py`: General sanity checks and tests for core utility and existing functions.
*   `test_screen_supercell_ns.py`: Dedicated tests for the `screen_supercell_ns` argument, including anisotropic values.
*   `test_units_functionality.py`: Comprehensive tests for various output unit conversions (THz, cm-1, eV).
*   `test_working_functions.py`: Detailed tests for core utility functions and the `GeneticAlgorithm` class.

#### How to Run Tests

All tests should be run using the `run_integration_tests.py` script, which manages the test execution, environment checks, and reporting.

1.  **Navigate to the `tests/` directory:**
    ```bash
    cd tests/
    ```

2.  **Execute the test runner with desired options:**

    *   **Run default (quick) integration tests and unit tests:**
        ```bash
        python run_integration_tests.py
        ```

    *   **Run all integration tests and unit tests:**
        ```bash
        python run_integration_tests.py --type all
        ```

    *   **Run specific categories of integration tests:**
        ```bash
        python run_integration_tests.py --type basic
        python run_integration_tests.py --type auto
        python run_integration_tests.py --type displaced
        python run_integration_tests.py --type units
        ```

    *   **Run only unit tests (skipping integration tests):**
        ```bash
        python run_integration_tests.py --unit-tests-only
        ```

    *   **Skip the initial conda environment check (use with caution):**
        ```bash
        python run_integration_tests.py --skip-env-check
        ```

#### Requirements

*   A working Conda environment with `vibroml` and `pytest` installed.
*   The `CONDA_ENV_PATH` specified in `run_integration_tests.py` must point to your active VibroML Conda environment.