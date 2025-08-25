import os
import time
import json
import sys
import numpy as np
# Ensure all necessary functions are imported from utils.utils
from .utils.utils import clean_phonon_cache, get_arg_parser_and_settings, parse_supercell_dimensions, parse_cli_screen_supercell_ns, parse_screen_supercell_ns, load_default_settings
# Import the optimization functions from auto_optimize
from .auto_optimize import run_phonon_calculation_sweep_optimization, run_automatic_soft_mode_optimization
from .utils.structure_utils import load_structure, initialize_calculator
from .utils.relaxation_utils import relax_structure
from .utils.phonon_utils import run_single_phonon_analysis, load_eigenmode_from_band_yaml

def main():
    ###############################################
    print("--- Initilization: Loading Default Settings and Initializing Parser ---")
    ###############################################
    parser, _ = get_arg_parser_and_settings() 

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Manually handle defaults for list-based arguments to avoid argparse issues.
    # This ensures that user-provided flags REPLACE the defaults, not append to them.
    default_settings = load_default_settings()

    # Handle screening parameters: load from defaults ONLY if not provided by user
    if args.screen_supercell_ns is None:
        default_supercells = default_settings.get("screen_supercell_ns", [[2,2,2], [3,3,3], [4,4,4]])
        args.screen_supercell_ns = parse_screen_supercell_ns(default_supercells)
    else:
        # This path is taken by our test script, which provides a command-line value
        args.screen_supercell_ns = parse_cli_screen_supercell_ns(args.screen_supercell_ns)

    if args.screen_deltas is None:
        args.screen_deltas = default_settings.get("screen_deltas", [0.05, 0.03, 0.01])

    if args.screen_fmax_values is None:
        args.screen_fmax_values = default_settings.get("screen_fmax_values", [0.001, 0.0005, 0.0001])

    # Handle other list-based arguments for the soft mode optimization
    if args.soft_mode_displacement_scales is None:
        args.soft_mode_displacement_scales = default_settings.get("soft_mode_displacement_scales", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

    if args.mode2_ratio_scales is None:
        args.mode2_ratio_scales = default_settings.get("mode2_ratio_scales", [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])

    if args.cell_scale_factors is None:
        args.cell_scale_factors = default_settings.get("cell_scale_factors", [-0.05, 0.0, 0.05, 0.10])
    # --- END OF NEW LOGIC ---

    # Handle supercell argument - prefer new --supercell over old --supercell_n
    if args.supercell is not None:
        supercell_dims = parse_supercell_dimensions(args.supercell)
        print(f"Using custom supercell dimensions: {supercell_dims}")
    else:
        supercell_dims = parse_supercell_dimensions(args.supercell_n)
        print(f"Using default supercell dimensions: {supercell_dims}")
    
    # Store the parsed supercell dimensions for use throughout the script
    args.supercell_dims = supercell_dims
    args.ga_disp_scale_bounds = [float(x) for x in args.ga_disp_scale_bounds.split(',')]  
    args.ga_ratio_bounds = [float(x) for x in args.ga_ratio_bounds.split(',')]  
    args.ga_cell_scale_bounds = [float(x) for x in args.ga_cell_scale_bounds.split(',')]  
    args.ga_cell_angle_bounds = [float(x) for x in args.ga_cell_angle_bounds.split(',')]
    
    print("\n--- Initilization: Setting up Output Directory and Cleaning Cache ---")
    cif_filename_base = os.path.splitext(os.path.basename(args.cif))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder_name = f"{cif_filename_base}_phonon_output_{timestamp}"
    output_dir = os.path.join(os.getcwd(), output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Clean up any old phonon cache files
    clean_phonon_cache()

    # Save initial settings (filter based on method)
    settings_to_save = vars(args).copy()

    # Remove method-specific parameters that don't apply to the selected method
    if args.method == "traditional":
        # Remove GA-specific parameters for traditional method
        ga_specific_params = [
            "ga_population_size",
            "ga_mutation_rate",
            "num_new_points_per_iteration",
            "ga_disp_scale_bounds",
            "ga_ratio_bounds",
            "ga_cell_scale_bounds",
            "ga_cell_angle_bounds"
        ]
        for param in ga_specific_params:
            settings_to_save.pop(param, None)

    with open(os.path.join(output_dir, "initial_settings.json"), 'w') as f:
        json.dump(settings_to_save, f, indent=4)
    print(f"Initial settings saved to {os.path.join(output_dir, 'initial_settings.json')}")
    
    ###############################################
    print("\n--- Step 1: Loading and potentially relaxing initial structure for single run ---")
    ###############################################
    
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
    
    ###############################################
    running_mode = "Automatic Soft Mode Optimization" if args.auto else "Phonon Calculation Only"
    print(f"\n--- Step 2: Running Mode is {running_mode} ---")
    ###############################################
    
    if args.auto:
        print("\n--- Step 3: Running Phonon Calculation Sweep Optimization ---")
        best_negative_frequency, best_settings, best_softest_modes_info, best_relaxed_atoms_from_sweep = run_phonon_calculation_sweep_optimization(  
            args,  
            output_dir,  
            current_atoms.copy(), # Pass the relaxed initial atoms  
            calculator,  
            cif_filename_base,  
            args.screen_supercell_ns,  
            args.screen_deltas,  
            args.screen_fmax_values,  
            args.negative_phonon_threshold_thz,  
            args.phonon_path_npoints,  
            args.phonon_dos_grid,  
            args.traj_kT,  
            args.num_modes_to_return,  
        )
        print(f"Method for soft mode optimization: {args.method}")

        print(f"\n--- Step 4: Running Automatic Soft Mode Optimization with method: {args.method} ---")  
        run_automatic_soft_mode_optimization(  
            args,  
            output_dir,  
            best_negative_frequency,  
            best_settings,  
            best_softest_modes_info,  
            best_relaxed_atoms_from_sweep,  
            args.negative_phonon_threshold_thz, # Pass threshold again for clarity  
            args.soft_mode_max_iterations,  
            args.soft_mode_displacement_scales,
            args.mode2_ratio_scales,  
            args.soft_mode_num_top_structures_to_analyze,  
            args.phonon_path_npoints,  
            args.phonon_dos_grid,  
            args.traj_kT,  
            args.cell_scale_factors,  
            args.num_modes_to_return,  
            args.ga_population_size,  
            args.ga_mutation_rate,  
            args.num_new_points_per_iteration,  
            args.ga_disp_scale_bounds,  
            args.ga_ratio_bounds,  
            args.ga_cell_scale_bounds,  
            args.ga_cell_angle_bounds
        )
    else:
        # Initialize q_point as None by default
        q_point = None
        preloaded_eigenmode_data = None

        # Check if user wants to use eigenmode from existing band.yaml file
        if args.band_yaml_path and args.q and args.band_idx is not None:
            print(f"\n--- Loading eigenmode from existing band.yaml file ---")
            print(f"Band.yaml file: {args.band_yaml_path}")
            print(f"Target q-point: {args.q}")
            print(f"Target band index: {args.band_idx}")
            print(f"Displacement magnitude: {args.displacement}")

            # Parse the q-point argument
            try:
                q_point = [float(x.strip()) for x in args.q.split(',')]
                if len(q_point) != 3:
                    raise ValueError("Q-point must have 3 components (e.g., '0.5,0,0').")
                q_point = np.array(q_point)
            except Exception as e:
                print(f"Error parsing q-point '{args.q}': {e}. Please provide as 'x,y,z'. Exiting.")
                sys.exit(1)

            # Load eigenmode data from band.yaml file
            frequency, eigenvector, lattice, natom = load_eigenmode_from_band_yaml(
                args.band_yaml_path, q_point, args.band_idx
            )

            if frequency is None:
                print("Failed to load eigenmode from band.yaml file. Exiting.")
                sys.exit(1)

            preloaded_eigenmode_data = {
                'frequency': frequency,
                'eigenvector': eigenvector,
                'lattice': lattice,
                'natom': natom,
                'q_point': q_point,
                'band_idx': args.band_idx
            }

        elif args.q and args.band_idx is not None:
            print(f"\n--- Data for eigenmode at band {args.band_idx} at q-point {args.q} will be retrieved ---")
            print(f"The following displacement will be applied for commensurate supercell along specified eigenmode: {args.displacement}")

            # Parse the q-point argument
            try:
                q_point = [float(x.strip()) for x in args.q.split(',')]
                if len(q_point) != 3:
                    raise ValueError("Q-point must have 3 components (e.g., '0.5,0,0').")
                q_point = np.array(q_point)
            except Exception as e:
                print(f"Error parsing q-point '{args.q}': {e}. Please provide as 'x,y,z'. Exiting.")
                sys.exit(1)

        elif args.band_yaml_path:
            print("Error: --band_yaml_path requires both --q and --band_idx to be specified.")
            sys.exit(1)
            
        # Call the consolidated run_single_phonon_analysis function
        softest_modes_info_list, bsmin, time_taken = run_single_phonon_analysis(
            current_atoms.copy(),
            calculator,
            args.engine,
            args.units,
            args.supercell_dims,  # Use parsed supercell dimensions
            args.delta,
            args.fmax,
            output_dir, # Use the specific directory for single phonon run
            prefix=cif_filename_base,
            phonon_path_npoints=args.phonon_path_npoints,
            phonon_dos_grid=args.phonon_dos_grid,
            traj_kT=args.traj_kT,
            num_modes_to_return=args.num_modes_to_return,
            q_point_for_specific_mode=q_point,
            band_idx_for_specific_mode=args.band_idx,
            displacement_magnitude=args.displacement,
            preloaded_eigenmode_data=preloaded_eigenmode_data,
            negative_phonon_threshold=args.negative_phonon_threshold_thz
        )

        
    print("\n--- Phonon Calculation Script Finished ---")