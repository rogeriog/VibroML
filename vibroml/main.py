# script.py (Revised)
import argparse
import os
import time
import json
import sys

# Import functions/classes from our new component files within the 'utils' package
from utils.utils import clean_phonon_cache
# Import the optimization functions from auto_optimize
from auto_optimize import run_parameter_sweep_optimization, run_soft_mode_optimization, run_single_phonon_analysis, NEGATIVE_PHONON_THRESHOLD_THZ # Import the unified single phonon analysis and threshold
from utils.structure_utils import load_structure, initialize_calculator
from utils.relaxation_utils import relax_structure


def main():
    # --- Step 0: Argument Parsing ---
    print("--- Step 0: Parsing Command Line Arguments ---")
    parser = argparse.ArgumentParser(description="Calculate phonon band structure and DOS for crystal structures, with optional relaxation and soft mode analysis.")
    parser.add_argument("--cif", type=str, required=True, help="Path to the CIF file.")
    parser.add_argument("--no-relax", action="store_true", help="Skip relaxation of the structure before calculation.")
    parser.add_argument("--engine", type=str, default="mace", help="Calculation engine (default: mace).")
    parser.add_argument("--units", type=str, default="THz", choices=["THz", "cm-1", "eV"],
                        help="Units for frequency (default: THz). Choose from THz, cm-1, eV.")
    parser.add_argument("--supercell_n", type=int, default=3,
                        help="Size of the supercell (N, N, N) for phonon calculation (default: 3).")
    parser.add_argument("--delta", type=float, default=0.03,
                        help="Displacement distance for finite difference phonon calculation (default: 0.03).")
    parser.add_argument("--fmax", type=float, default=0.001,
                        help="Maximum force tolerance for structure relaxation (default: 0.001 eV/Å).")
    parser.add_argument("--auto", action="store_true", help="Automatically test multiple settings (parameter sweep) to minimize negative imaginary phonons. If a soft mode persists, it will trigger the iterative soft mode workflow.")
    parser.add_argument("--run-soft-mode-after-single", action="store_true", help="If a soft mode is detected in a single phonon calculation, automatically run the iterative soft mode displacement and relaxation workflow.")


    args = parser.parse_args()

    # --- Step 0.1: Setup Output Directory and Clean Cache ---
    print("\n--- Step 0.1: Setting up Output Directory and Cleaning Cache ---")
    cif_filename_base = os.path.splitext(os.path.basename(args.cif))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder_name = f"{cif_filename_base}_phonon_output_{timestamp}"
    output_dir = os.path.join(os.getcwd(), output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Clean up any old phonon cache files
    clean_phonon_cache()

    # Save initial settings
    with open(os.path.join(output_dir, "initial_settings.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print("Initial settings saved.")

    # --- Step 0.2: Run Calculation based on Mode ---
    if args.auto:
        print("\n--- Step 0.2: Running in Parameter Sweep Auto Optimization Mode ---")
        # run_parameter_sweep_optimization now handles triggering soft mode optimization internally
        run_parameter_sweep_optimization(args, output_dir)
    else:
        print("\n--- Step 0.2: Running Single Calculation Mode ---")
        # In single mode, we still need to load and potentially relax the structure first
        print("\n--- Loading and potentially relaxing initial structure for single run ---")
        struct, initial_atoms = load_structure(args.cif)
        if struct is None or initial_atoms is None:
            print("Failed to load initial structure. Exiting.")
            sys.exit(1)

        calculator = initialize_calculator(args.engine)
        if calculator is None:
            print("Failed to initialize calculator. Exiting.")
            sys.exit(1)

        current_atoms = initial_atoms.copy()
        if not args.no_relax:
            initial_relax_dir = os.path.join(output_dir, "initial_relaxation_for_single_run")
            os.makedirs(initial_relax_dir, exist_ok=True)
            current_atoms = relax_structure(initial_atoms.copy(), calculator, args.engine, args.fmax, initial_relax_dir, args.cif)
            if current_atoms is None:
                print("Initial relaxation failed. Exiting single run.")
                sys.exit(1)
        else:
            print("Skipping initial structure relaxation for single run.")

        # Run a single phonon analysis with specified parameters on the (potentially relaxed) structure
        current_run_output_dir = os.path.join(output_dir, f"SingleRun_N{args.supercell_n}_D{args.delta}_F{args.fmax}")
        os.makedirs(current_run_output_dir, exist_ok=True)
        print(f"Single run output directory created: {current_run_output_dir}")

        run_settings = vars(args).copy()
        with open(os.path.join(current_run_output_dir, "run_settings.json"), 'w') as f:
            json.dump(run_settings, f, indent=4)
        print("Run settings saved.")

        # Execute the single phonon analysis step
        softest_mode_info, most_negative_freq, time_taken = run_single_phonon_analysis(
            current_atoms, calculator, args.engine, args.units, args.supercell_n, args.delta, args.fmax, current_run_output_dir, prefix=cif_filename_base
        )

        # If --run-soft-mode-after-single is enabled and a soft mode is detected
        threshold_in_current_units = NEGATIVE_PHONON_THRESHOLD_THZ
        if args.units == "cm-1":
            threshold_in_current_units *= 33.35641
        elif args.units == "eV":
            threshold_in_current_units *= 4.135667696e-3

        if args.run_soft_mode_after_single and softest_mode_info is not None and most_negative_freq < threshold_in_current_units:
            print(f"\nSoft mode detected ({most_negative_freq:.4f} {args.units}) in single run. Initiating iterative soft mode optimization...")
            run_soft_mode_optimization(
                args,
                output_dir,
                current_atoms, # The structure that had the soft mode
                softest_mode_info # The softest mode info from that structure
            )
        elif args.run_soft_mode_after_single:
            print(f"\nNo significant soft mode detected ({most_negative_freq:.4f} {args.units}) in single run. Skipping iterative soft mode optimization.")


    print("\n--- Phonon Calculation Script Finished ---")


if __name__ == "__main__":
    main()