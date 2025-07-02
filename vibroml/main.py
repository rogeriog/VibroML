# vibroml/main.py
import argparse
import os
import time
import json
import sys

# Import functions/classes from our new component files within the 'utils' package
from .utils.utils import clean_phonon_cache, load_default_settings
# Import the optimization functions from auto_optimize
from .auto_optimize import run_parameter_sweep_optimization, run_soft_mode_optimization, run_single_phonon_analysis
from .utils.structure_utils import load_structure, initialize_calculator
from .utils.relaxation_utils import relax_structure


def main():
    # --- Step 0.0: Load Default Settings ---
    # These will serve as the *default* values for the argparse arguments.
    # If the JSON file is not found or values are missing, hardcoded fallbacks are used.
    print("--- Step 0.0: Loading Default Settings ---")
    default_settings = load_default_settings()
    if not default_settings:
        print("Warning: Could not load default settings. Using hardcoded fallbacks.")
        # Define hardcoded fallbacks if JSON loading fails
        default_engine = "mace"
        default_model_name = "medium-omat-0"
        default_fmax = 0.001
        default_delta = 0.03
        default_supercell_n = 3
        screen_supercell_ns = [2, 3, 4]
        screen_deltas = [0.05, 0.03, 0.01]
        screen_fmax_values = [0.001, 0.0005, 0.0001]
        phonon_path_npoints = 100
        phonon_dos_grid = [40, 40, 40]
        default_units = "THz"
        default_traj_kT = 1.0
        negative_phonon_threshold_thz = -0.1
        soft_mode_max_iterations = 3
        soft_mode_displacement_scales = [10.0, 15.8, 25.1, 39.8, 63.1, 100.0]
        soft_mode_num_top_structures_to_analyze = 3
        cell_scale_factors = [-0.05, 0.0, 0.05, 0.10]
        num_modes_to_return = 2
        ga_population_size = 50
        ga_mutation_rate = 0.1
        num_new_points_per_iteration = 30
    else:
        # Use values from loaded JSON
        default_engine = default_settings.get("default_engine", "mace")
        default_model_name = default_settings.get("default_model_name", "medium-omat-0")
        default_fmax = default_settings.get("default_fmax", 0.001)
        default_delta = default_settings.get("default_delta", 0.03)
        default_supercell_n = default_settings.get("default_supercell_n", 3)
        screen_supercell_ns = default_settings.get("screen_supercell_ns", [2, 3, 4])
        screen_deltas = default_settings.get("screen_deltas", [0.05, 0.03, 0.01])
        screen_fmax_values = default_settings.get("screen_fmax_values", [0.001, 0.0005, 0.0001])
        phonon_path_npoints = default_settings.get("phonon_path_npoints", 100)
        phonon_dos_grid = default_settings.get("phonon_dos_grid", [40, 40, 40])
        default_units = default_settings.get("default_units", "THz")
        default_traj_kT = default_settings.get("default_traj_kT", 1.0)
        negative_phonon_threshold_thz = default_settings.get("negative_phonon_threshold_thz", -0.1)
        soft_mode_max_iterations = default_settings.get("soft_mode_max_iterations", 3)
        soft_mode_displacement_scales = default_settings.get("soft_mode_displacement_scales", [10.0, 15.8, 25.1, 39.8, 63.1, 100.0])
        soft_mode_num_top_structures_to_analyze = default_settings.get("soft_mode_num_top_structures_to_analyze", 3)
        cell_scale_factors = default_settings.get("cell_scale_factors", [-0.05, 0.0, 0.05, 0.10])
        num_modes_to_return = default_settings.get("num_modes_to_return", 2)
        ga_population_size = default_settings.get("ga_population_size", 50)
        ga_mutation_rate = default_settings.get("ga_mutation_rate", 0.1)
        num_new_points_per_iteration = default_settings.get("num_new_points_per_iteration", 30)


    # --- Step 0.1: Argument Parsing ---
    print("--- Step 0.1: Parsing Command Line Arguments ---")
    parser = argparse.ArgumentParser(description="Calculate phonon band structure and DOS for crystal structures, with optional relaxation and soft mode analysis.")
    parser.add_argument("--cif", type=str, required=True, help="Path to the CIF file.")
    parser.add_argument("--no-relax", action="store_true", help="Skip relaxation of the structure before calculation.")
    parser.add_argument("--engine", type=str, default=default_engine, help=f"Calculation engine (default: {default_engine}).")
    parser.add_argument("--units", type=str, default=default_units, choices=["THz", "cm-1", "eV"],
                        help=f"Units for frequency (default: {default_units}). Choose from THz, cm-1, eV.")
    parser.add_argument("--model_name", type=str, default=default_model_name,
                        help=f"Model name for the calculator (default: {default_model_name}).")
    parser.add_argument("--supercell_n", type=int, default=default_supercell_n,
                        help=f"Size of the supercell (N, N, N) for phonon calculation (default: {default_supercell_n}).")
    parser.add_argument("--delta", type=float, default=default_delta,
                        help=f"Displacement distance for finite difference phonon calculation (default: {default_delta}).")
    parser.add_argument("--fmax", type=float, default=default_fmax,
                        help=f"Maximum force tolerance for structure relaxation (default: {default_fmax} eV/Å).")
    parser.add_argument("--auto", action="store_true", help="Automatically test multiple settings (parameter sweep) to minimize negative imaginary phonons. If a soft mode persists, it will trigger the iterative soft mode workflow.")
    parser.add_argument("--run-soft-mode-after-single", action="store_true", help="If a soft mode is detected in a single phonon calculation, automatically run the iterative soft mode displacement and relaxation workflow.")
    parser.add_argument("--displace-primitive", action="store_true", help="Also perform soft mode displacements and relaxation directly on the primitive cell, in a separate directory.")

    # --- New Arguments for Parameter Sweep and Soft Mode Optimization ---
    parser.add_argument("--screen_supercell_ns", type=int, nargs='+', default=screen_supercell_ns,
                        help=f"List of supercell N values for parameter sweep (default: {screen_supercell_ns}).")
    parser.add_argument("--screen_deltas", type=float, nargs='+', default=screen_deltas,
                        help=f"List of delta values for parameter sweep (default: {screen_deltas}).")
    parser.add_argument("--screen_fmax_values", type=float, nargs='+', default=screen_fmax_values,
                        help=f"List of fmax values for parameter sweep (default: {screen_fmax_values}).")
    parser.add_argument("--phonon_path_npoints", type=int, default=phonon_path_npoints,
                        help=f"Number of points along the phonon path (default: {phonon_path_npoints}).")
    parser.add_argument("--phonon_dos_grid", type=int, nargs=3, default=phonon_dos_grid,
                        help=f"Grid for DOS calculation (e.g., 40 40 40) (default: {phonon_dos_grid}).")
    parser.add_argument("--traj_kT", type=float, default=default_traj_kT,
                        help=f"Temperature for trajectory generation (default: {default_traj_kT} eV).")
    parser.add_argument("--negative_phonon_threshold_thz", type=float, default=negative_phonon_threshold_thz,
                        help=f"Threshold for negative phonon frequency in THz to trigger soft mode optimization (default: {negative_phonon_threshold_thz} THz).")
    parser.add_argument("--soft_mode_max_iterations", type=int, default=soft_mode_max_iterations,
                        help=f"Maximum iterations for the soft mode optimization (default: {soft_mode_max_iterations}).")
    parser.add_argument("--soft_mode_displacement_scales", type=float, nargs='+', default=soft_mode_displacement_scales,
                        help=f"List of displacement scales for soft mode generation (default: {soft_mode_displacement_scales}).")
    parser.add_argument("--soft_mode_num_top_structures_to_analyze", type=int, default=soft_mode_num_top_structures_to_analyze,
                        help=f"Number of top structures to analyze in final soft mode optimization step (default: {soft_mode_num_top_structures_to_analyze}).")
    parser.add_argument("--cell_scale_factors", type=float, nargs='+', default=cell_scale_factors,
                        help=f"List of cell scale factors for soft mode optimization (default: {cell_scale_factors}).")
    parser.add_argument("--num_modes_to_return", type=int, default=num_modes_to_return,
                        help=f"Number of softest modes to return for analysis (default: {num_modes_to_return}).")
    parser.add_argument("--ga_population_size", type=int, default=ga_population_size,
                        help=f"Population size for the Genetic Algorithm (default: {ga_population_size}).")
    parser.add_argument("--ga_mutation_rate", type=float, default=ga_mutation_rate,
                        help=f"Mutation rate for the Genetic Algorithm (default: {ga_mutation_rate}).")
    parser.add_argument("--num_new_points_per_iteration", type=int, default=num_new_points_per_iteration,
                        help=f"Number of new structures to generate per GA iteration (default: {num_new_points_per_iteration}).")


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # --- Step 0.2: Setup Output Directory and Clean Cache ---
    print("\n--- Step 0.2: Setting up Output Directory and Cleaning Cache ---")
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

    # --- Step 0.3: Run Calculation based on Mode ---
    if args.auto:
        print("\n--- Step 0.3: Running in Parameter Sweep Auto Optimization Mode ---")
        # Pass args directly, as all parameters are now attributes of args
        run_parameter_sweep_optimization(
            args,
            output_dir,
            args.screen_supercell_ns, # Now from args
            args.screen_deltas,       # Now from args
            args.screen_fmax_values,  # Now from args
            args.negative_phonon_threshold_thz, # Now from args
            args.soft_mode_max_iterations,      # Now from args
            args.soft_mode_displacement_scales, # Now from args
            args.soft_mode_num_top_structures_to_analyze, # Now from args
            args.phonon_path_npoints, # Now from args
            args.phonon_dos_grid,     # Now from args
            args.traj_kT,             # Now from args
            args.cell_scale_factors,  # Now from args
            args.num_modes_to_return, # Now from args
            args.ga_population_size,  # Now from args
            args.ga_mutation_rate,    # Now from args
            args.num_new_points_per_iteration, # Now from args
        )
    else:
        print("\n--- Step 0.3: Running Single Calculation Mode ---")
        # In single mode, we still need to load and potentially relax the structure first
        print("\n--- Loading and potentially relaxing initial structure for single run ---")
        struct, initial_atoms = load_structure(args.cif)
        if struct is None or initial_atoms is None:
            print("Failed to load initial structure. Exiting.")
            sys.exit(1)

        calculator = initialize_calculator(args.engine, model_name=args.model_name)
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
        softest_modes_info, most_negative_freq, time_taken = run_single_phonon_analysis(
            current_atoms, calculator, args.engine, args.units, args.supercell_n, args.delta, args.fmax, current_run_output_dir, prefix=cif_filename_base,
            phonon_path_npoints=args.phonon_path_npoints, # Now from args
            phonon_dos_grid=args.phonon_dos_grid,     # Now from args
            traj_kT=args.traj_kT,                     # Now from args
            num_modes_to_return=args.num_modes_to_return, # Now from args
        )

        # If --run-soft-mode-after-single is enabled and a soft mode is detected
        threshold_in_current_units = args.negative_phonon_threshold_thz # Now from args
        if args.units == "cm-1":
            threshold_in_current_units *= 33.35641
        elif args.units == "eV":
            threshold_in_current_units *= 4.135667696e-3

        # Check if softest_modes_info is not empty and the first mode is negative
        if args.run_soft_mode_after_single and softest_modes_info and softest_modes_info[0]['frequency'] < threshold_in_current_units:
            print(f"\nSoft mode detected ({softest_modes_info[0]['frequency']:.4f} {args.units}) in single run. Initiating iterative soft mode optimization...")
            run_soft_mode_optimization(
                args,
                output_dir,
                current_atoms, # The structure that had the soft mode
                softest_modes_info, # The softest mode info list from that structure
                max_iterations=args.soft_mode_max_iterations, # Now from args
                soft_mode_displacement_scales=args.soft_mode_displacement_scales, # Now from args
                num_top_structures_to_analyze=args.soft_mode_num_top_structures_to_analyze, # Now from args
                negative_phonon_threshold_thz=args.negative_phonon_threshold_thz, # Now from args
                phonon_path_npoints=args.phonon_path_npoints, # Now from args
                phonon_dos_grid=args.phonon_dos_grid,     # Now from args
                default_traj_kT=args.traj_kT,             # Now from args
                cell_scale_factors=args.cell_scale_factors, # Now from args
                num_modes_to_return=args.num_modes_to_return, # Now from args
                ga_population_size=args.ga_population_size, # Now from args
                ga_mutation_rate=args.ga_mutation_rate,   # Now from args
                num_new_points_per_iteration=args.num_new_points_per_iteration, # Now from args
            )
        elif args.run_soft_mode_after_single:
            print(f"\nNo significant soft mode detected ({most_negative_freq:.4f} {args.units}) in single run. Skipping iterative soft mode optimization.")

        # Primitive cell displacement logic (now uses the same run_soft_mode_optimization)
        # Check if softest_modes_info is not empty and the first mode is negative
        if args.displace_primitive and softest_modes_info and softest_modes_info[0]['frequency'] < threshold_in_current_units:
            print(f"\nSoft mode detected ({softest_modes_info[0]['frequency']:.4f} {args.units}) in single run. Initiating iterative soft mode optimization directly on primitive cell...")
            primitive_output_dir = os.path.join(output_dir, "primitive_cell_soft_mode_optimization")
            os.makedirs(primitive_output_dir, exist_ok=True)
            run_soft_mode_optimization(
                args,
                primitive_output_dir, # New output directory for primitive cell
                current_atoms, # The primitive cell structure (already primitive if from run_single_phonon_analysis)
                softest_modes_info, # The softest mode info list from that structure
                max_iterations=args.soft_mode_max_iterations, # Now from args
                soft_mode_displacement_scales=args.soft_mode_displacement_scales, # Now from args
                num_top_structures_to_analyze=args.soft_mode_num_top_structures_to_analyze, # Now from args
                negative_phonon_threshold_thz=args.negative_phonon_threshold_thz, # Now from args
                phonon_path_npoints=args.phonon_path_npoints, # Now from args
                phonon_dos_grid=args.phonon_dos_grid,     # Now from args
                default_traj_kT=args.traj_kT,             # Now from args
                cell_scale_factors=args.cell_scale_factors, # Now from args
                num_modes_to_return=args.num_modes_to_return, # Now from args
                ga_population_size=args.ga_population_size, # Now from args
                ga_mutation_rate=args.ga_mutation_rate,   # Now from args
                num_new_points_per_iteration=args.num_new_points_per_iteration, # Now from args
            )
        elif args.displace_primitive:
            print(f"\nNo significant soft mode detected ({most_negative_freq:.4f} {args.units}) in single run. Skipping primitive cell iterative soft mode optimization.")


    print("\n--- Phonon Calculation Script Finished ---")