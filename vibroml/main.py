import os
import time
import json
import sys
import numpy as np
# Ensure all necessary functions are imported from utils.utils
from .utils.utils import clean_phonon_cache, get_arg_parser_and_settings, parse_supercell_dimensions, parse_cli_screen_supercell_ns, parse_screen_supercell_ns, load_default_settings
# Import the optimization functions from auto_optimize
from .auto_optimize import run_phonon_calculation_sweep_optimization, run_automatic_soft_mode_optimization, run_neb_soft_mode_optimization, run_ci_neb_soft_mode_optimization, run_md_stability_analysis
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

    # Validate decomposition threshold
    if args.decomposition_threshold <= 0:
        print(f"Error: --decomposition_threshold must be a positive number, got {args.decomposition_threshold}")
        sys.exit(1)
    print(f"Using decomposition threshold: {args.decomposition_threshold:.3f} eV (structures with E_relax < E_ref - {args.decomposition_threshold:.3f} will be flagged as DECOMPOSED)")

    # Validate NEB-specific requirements
    if args.method in ["neb", "ci_neb"]:
        if not args.final_cif:
            print(f"Error: --final_cif is required when using method '{args.method}'")
            sys.exit(1)
        if not os.path.exists(args.final_cif):
            print(f"Error: Final CIF file not found: {args.final_cif}")
            sys.exit(1)
        print(f"NEB method '{args.method}' will use initial structure: {args.cif}")
        print(f"NEB method '{args.method}' will use final structure: {args.final_cif}")
        print(f"Number of intermediate images: {args.neb_num_images}")
        print(f"Spring constant: {args.neb_spring_constant} eV/Å²")
        print(f"Force tolerance: {args.neb_force_tolerance} eV/Å")
        if args.method == "ci_neb":
            print(f"Climbing image will start at iteration: {args.neb_climbing_start_iteration}")

    # Validate MD stability-specific requirements
    if args.method == "md_stability":
        if args.temp <= 0:
            print(f"Error: Temperature must be positive, got {args.temp} K")
            sys.exit(1)
        if args.pressure < 0:
            print(f"Error: Pressure must be non-negative, got {args.pressure} GPa")
            sys.exit(1)
        if args.time <= 0:
            print(f"Error: Simulation time must be positive, got {args.time} ps")
            sys.exit(1)
        if not (0.1 <= args.equilibration_fraction <= 0.5):
            print(f"Error: Equilibration fraction must be between 0.1 and 0.5, got {args.equilibration_fraction}")
            sys.exit(1)

        # Validate supercell size format
        try:
            supercell_parts = args.supercell_size.lower().split('x')
            if len(supercell_parts) != 3:
                raise ValueError("Must have exactly 3 dimensions")
            supercell_dims = [int(x) for x in supercell_parts]
            if any(dim < 1 for dim in supercell_dims):
                raise ValueError("All dimensions must be positive")
        except (ValueError, AttributeError) as e:
            print(f"Error: Invalid supercell size format '{args.supercell_size}'. Expected format: 'NxNxN' (e.g., '2x2x2')")
            sys.exit(1)

        print(f"MD stability method will use:")
        print(f"  Temperature: {args.temp} K")
        print(f"  Pressure: {args.pressure} GPa")
        print(f"  Total simulation time: {args.time} ps")
        print(f"  Supercell size: {args.supercell_size}")
        print(f"  Equilibration fraction: {args.equilibration_fraction}")
        print(f"  Stability assessment thresholds (generous for MLIP fluctuations):")
        print(f"    Volume fluctuation: ±{args.volume_threshold}%")
        print(f"    RMSD threshold: {args.rmsd_threshold} Å")
        print(f"    RDF correlation: >{args.rdf_threshold}")

    print("\n--- Initilization: Setting up Output Directory and Cleaning Cache ---")
    cif_filename_base = os.path.splitext(os.path.basename(args.cif))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Add method suffix to output folder name
    method_suffix_map = {
        "traditional": "_TRADITIONAL",
        "ga": "_GA",
        "traditional_all": "_TRADITIONAL_ALL",
        "opt_random": "_OPT_RANDOM",
        "neb": "_NEB",
        "ci_neb": "_CI_NEB",
        "md_stability": "_MD_STABILITY"
    }
    method_suffix = method_suffix_map.get(args.method, "")

    # Add custom prefix if provided
    prefix_part = ""
    if hasattr(args, 'output_prefix') and args.output_prefix:
        prefix_part = f"{args.output_prefix}_"

    output_folder_name = f"{prefix_part}{cif_filename_base}{method_suffix}_phonon_output_{timestamp}"
    output_dir = os.path.join(os.getcwd(), output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Clean up any old phonon cache files
    clean_phonon_cache()

    # Save initial settings (filter based on method)
    settings_to_save = vars(args).copy()

    # Remove method-specific parameters that don't apply to the selected method
    if args.method == "traditional":
        # Remove GA-specific and NEB-specific parameters for traditional method
        non_traditional_params = [
            "ga_population_size", "ga_mutation_rate", "num_new_points_per_iteration",
            "ga_disp_scale_bounds", "ga_ratio_bounds", "ga_cell_scale_bounds", "ga_cell_angle_bounds",
            "final_cif", "neb_num_images", "neb_spring_constant", "neb_max_iterations",
            "neb_force_tolerance", "neb_climbing_start_iteration"
        ]
        for param in non_traditional_params:
            settings_to_save.pop(param, None)
    elif args.method in ["ga", "traditional_all", "opt_random"]:
        # Remove NEB-specific and MD-specific parameters for non-NEB/non-MD methods
        neb_specific_params = [
            "final_cif", "neb_num_images", "neb_spring_constant", "neb_max_iterations",
            "neb_force_tolerance", "neb_climbing_start_iteration"
        ]
        md_specific_params = [
            "temp", "pressure", "time", "supercell_size", "equilibration_fraction"
        ]
        for param in neb_specific_params + md_specific_params:
            settings_to_save.pop(param, None)
    elif args.method == "md_stability":
        # Remove NEB-specific parameters for MD method
        neb_specific_params = [
            "final_cif", "neb_num_images", "neb_spring_constant", "neb_max_iterations",
            "neb_force_tolerance", "neb_climbing_start_iteration"
        ]
        for param in neb_specific_params:
            settings_to_save.pop(param, None)
    elif args.method in ["neb", "ci_neb"]:
        # Remove GA-specific and soft-mode-specific parameters for NEB methods
        non_neb_params = [
            "ga_population_size", "ga_mutation_rate", "num_new_points_per_iteration",
            "ga_disp_scale_bounds", "ga_ratio_bounds", "ga_cell_scale_bounds", "ga_cell_angle_bounds",
            "soft_mode_displacement_scales", "mode2_ratio_scales", "cell_scale_factors"
        ]
        for param in non_neb_params:
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
        current_atoms = relax_structure(initial_atoms.copy(), calculator, args.engine, args.fmax, initial_relax_dir, args.cif, relaxation_patience=getattr(args, 'relaxation_patience', 5))
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
            args.ga_generations,
            args.num_new_points_per_iteration,
            args.ga_disp_scale_bounds,
            args.ga_ratio_bounds,
            args.ga_cell_scale_bounds,
            args.ga_cell_angle_bounds
        )
    else:
        # Handle NEB methods even without --auto flag
        if args.method in ["neb", "ci_neb"]:
            print(f"\n--- Running {args.method.upper()} Method (without --auto) ---")
            print("Note: NEB methods require --auto flag for full workflow. Running direct NEB optimization only.")

            # For NEB methods without --auto, we still need to run the NEB workflow
            # but we'll use the current_atoms as the initial structure

            # Create a minimal softest_modes_info_list (empty since we're not doing phonon analysis first)
            initial_softest_modes_info_list = []

            if args.method == "neb":
                run_neb_soft_mode_optimization(
                    args, output_dir, current_atoms, initial_softest_modes_info_list,
                    max_iterations=1,  # Not used in NEB workflow
                    soft_mode_displacement_scales=[],  # Not used in NEB workflow
                    cell_scale_factors=[],  # Not used in NEB workflow
                    mode2_ratio_scales=[],  # Not used in NEB workflow
                    num_top_structures_to_analyze=1,  # Not used in NEB workflow
                    negative_phonon_threshold_thz=args.negative_phonon_threshold_thz,
                    phonon_path_npoints=args.phonon_path_npoints,
                    phonon_dos_grid=args.phonon_dos_grid,
                    default_traj_kT=args.traj_kT,
                    num_modes_to_return=args.num_modes_to_return,
                    neb_num_images=args.neb_num_images,
                    neb_spring_constant=args.neb_spring_constant,
                    neb_max_iterations=args.neb_max_iterations,
                    neb_force_tolerance=args.neb_force_tolerance,
                    final_cif_path=args.final_cif
                )
            elif args.method == "ci_neb":
                run_ci_neb_soft_mode_optimization(
                    args, output_dir, current_atoms, initial_softest_modes_info_list,
                    max_iterations=1,  # Not used in NEB workflow
                    soft_mode_displacement_scales=[],  # Not used in NEB workflow
                    cell_scale_factors=[],  # Not used in NEB workflow
                    mode2_ratio_scales=[],  # Not used in NEB workflow
                    num_top_structures_to_analyze=1,  # Not used in NEB workflow
                    negative_phonon_threshold_thz=args.negative_phonon_threshold_thz,
                    phonon_path_npoints=args.phonon_path_npoints,
                    phonon_dos_grid=args.phonon_dos_grid,
                    default_traj_kT=args.traj_kT,
                    num_modes_to_return=args.num_modes_to_return,
                    neb_num_images=args.neb_num_images,
                    neb_spring_constant=args.neb_spring_constant,
                    neb_max_iterations=args.neb_max_iterations,
                    neb_force_tolerance=args.neb_force_tolerance,
                    neb_climbing_start_iteration=args.neb_climbing_start_iteration,
                    final_cif_path=args.final_cif
                )

            print("\n--- NEB Method Completed ---")
            return

        # Handle MD stability method
        elif args.method == "md_stability":
            print(f"\n--- Running MD Stability Analysis ---")
            print("Note: MD stability analysis runs independently of phonon calculations.")

            # Run MD stability analysis
            stability_verdict = run_md_stability_analysis(args, output_dir, current_atoms)

            if stability_verdict:
                print(f"\n--- MD Stability Analysis Completed ---")
                print(f"Final verdict: {stability_verdict['verdict']} (confidence: {stability_verdict['confidence']})")
            else:
                print(f"\n--- MD Stability Analysis Failed ---")

            return

        # Initialize q_point as None by default for non-NEB/non-MD methods
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
        softest_modes_info_list, bsmin, time_taken, tracked_k_points_data = run_single_phonon_analysis(
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
            negative_phonon_threshold=args.negative_phonon_threshold_thz,
            save_yaml=args.save_yaml
        )

        
    print("\n--- Phonon Calculation Script Finished ---")