import os
import sys
import json
import time

from .utils.structure_utils import initialize_calculator, generate_displaced_supercells, estimate_commensurate_supercell_size, generate_random_displaced_structures, load_structure
from .utils.relaxation_utils import relax_structure, relax_structures_in_folder, find_lowest_energy_structures
from .utils.phonon_utils import run_single_phonon_analysis
from .utils.neb_utils import run_neb_optimization, generate_enhanced_neb_summary
from .utils.md_utils import (prepare_md_supercell, setup_nvt_ensemble, setup_npt_ensemble,
                            temperature_ramp_callback, monitor_equilibration_convergence,
                            analyze_trajectory_stability, analyze_energy_trajectory,
                            determine_stability, save_trajectory_analysis_plots)
from .checkpointing import CheckpointManager, should_skip_sample

from .utils.genetic_algorithm import GeneticAlgorithm

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write, read
import numpy as np


def _save_final_structure(result, output_dir, index, structure_type, original_prefix):
    """
    Helper function to save final structures with consistent naming.
    
    Args:
        result (dict): Structure result dictionary
        output_dir (str): Output directory path
        index (int): Structure index
        structure_type (str): Either "top" or "unique"
        original_prefix (str): Original structure prefix
    """
    top_structure_relaxed_atoms = result['relaxed_atoms']
    
    # Format energy for filename (replace . with p, - with m)
    energy_per_atom = result['energy_per_atom']
    energy_str = f"{abs(energy_per_atom):.4f}".replace('.', 'p')
    if energy_per_atom < 0:
        energy_str = 'm' + energy_str
    else:
        energy_str = 'p' + energy_str
    
    # FIX: Handle different iteration keys and sanitize the filename
    iteration_val = result.get('iteration', result.get('main_iteration', 'NA'))
    sample_val = result.get('sample', 'NA')
    iteration_str = str(iteration_val).replace('/', '_')
    sample_str = str(sample_val).replace('/', '_')
    
    # Create base filename
    base_filename = f"{structure_type}_{index}_iter{iteration_str}_sample{sample_str}_{original_prefix}_energy_{energy_str}"
    
    # Use main_iteration for GA, iteration for traditional
    iter_for_print = result.get('main_iteration', result.get('iteration', 'N/A'))
    sample_for_print = result.get('sample', 'N/A')
    print(f"  {index}. Iter {iter_for_print}, Sample {sample_for_print}: Energy: {result['energy_per_atom']:.6f} eV/atom")

    try:
        # Save main structure files
        final_cif_path = os.path.join(output_dir, f"{base_filename}.cif")
        final_xyz_path = os.path.join(output_dir, f"{base_filename}.xyz")
        
        write(final_cif_path, top_structure_relaxed_atoms)
        print(f"    Saved structure CIF to: {final_cif_path}")
        write(final_xyz_path, top_structure_relaxed_atoms)
        print(f"    Saved structure XYZ to: {final_xyz_path}")
        
        # Convert ASE Atoms to Pymatgen Structure for primitive and conventional cell analysis  
        pmg_structure = AseAtomsAdaptor.get_structure(top_structure_relaxed_atoms)  
        sga = SpacegroupAnalyzer(pmg_structure)  

        # Get and save the primitive standard structure  
        primitive_pmg_structure = sga.get_primitive_standard_structure()  
        primitive_atoms = AseAtomsAdaptor.get_atoms(primitive_pmg_structure)  
        primitive_path = os.path.join(output_dir, f"{base_filename}_primitive.cif")  
        write(primitive_path, primitive_atoms)  
        print(f"    Saved primitive cell to: {primitive_path}")  

        # Get and save the conventional standard structure  
        conventional_pmg_structure = sga.get_conventional_standard_structure()  
        conventional_atoms = AseAtomsAdaptor.get_atoms(conventional_pmg_structure)  
        conventional_path = os.path.join(output_dir, f"{base_filename}_conventional.cif")  
        write(conventional_path, conventional_atoms)  
        print(f"    Saved conventional cell to: {conventional_path}")
        
    except Exception as e:
        print(f"    Error saving files for {structure_type} structure {index}: {e}")
        import traceback
        traceback.print_exc()

# Assuming these are defined elsewhere or will be imported into this file
# from your_module import load_structure, initialize_calculator, relax_structure, run_single_phonon_analysis, run_ga_soft_mode_optimization, run_traditional_soft_mode_optimization

def run_phonon_calculation_sweep_optimization(args, output_dir, initial_atoms, calculator, original_prefix, supercell_ns, deltas, fmax_values, negative_phonon_threshold_thz,
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return):
    """
    Runs the parameter sweep to find optimal settings for phonon calculations.
    Returns the best negative frequency, settings, softest modes info, and the relaxed atoms.
    """
    print("Running parameter sweep to find optimal phonon calculation settings...")
    best_negative_frequency = -float('inf')
    best_settings = {}
    best_softest_modes_info = []
    best_relaxed_atoms = initial_atoms.copy() # Start with the initially relaxed atoms

    if not (len(supercell_ns) == len(deltas) == len(fmax_values)):
        print("Error: For --auto mode with sequential optimization, supercell_ns, deltas, and fmax_values lists must have the same length.")
        sys.exit(1)

    results = []
    previous_best_negative_frequency = - float('inf')

    threshold_in_current_units = negative_phonon_threshold_thz
    if args.units == "cm-1":
        threshold_in_current_units *= 33.35641
    elif args.units == "eV":
        threshold_in_current_units *= 4.135667696e-3

    for i in range(len(supercell_ns)):
        sc_dims = supercell_ns[i]  # Now this is a tuple like (2,2,2)
        d = deltas[i]
        fm = fmax_values[i]

        # Create a string representation for the supercell dimensions
        sc_dims_str = f"{sc_dims[0]}x{sc_dims[1]}x{sc_dims[2]}"
        print(f"\n--- Testing Supercell Dims: {sc_dims}, Delta: {d}, Fmax: {fm} ---")
        current_output_dir = os.path.join(output_dir, f"N{sc_dims_str}_D{d}_F{fm}")
        os.makedirs(current_output_dir, exist_ok=True)
        run_settings = vars(args).copy()
        run_settings['supercell_dims'] = sc_dims  # Store as tuple
        run_settings['supercell_n'] = sc_dims[0] if sc_dims[0] == sc_dims[1] == sc_dims[2] else f"{sc_dims[0]},{sc_dims[1]},{sc_dims[2]}"  # For backward compatibility
        run_settings['delta'] = d
        run_settings['fmax'] = fm
        with open(os.path.join(current_output_dir, "run_settings.json"), 'w') as f:
            json.dump(run_settings, f, indent=4)
        print(f"\nAttempting to relax structure for Fmax={fm}...")
        # Pass a copy of initial_atoms to relax_structure to avoid modifying it directly
        relaxed_atoms_for_current_params = relax_structure(
            initial_atoms.copy(), calculator, args.engine, fm, current_output_dir, args.cif,
            relaxation_patience=getattr(args, 'relaxation_patience', 5),
            volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
        )
        if relaxed_atoms_for_current_params is None:
            print(f"Skipping phonon analysis for N={sc_dims_str}, D={d}, F={fm} due to failed relaxation.")
            results.append({
                "supercell_dims": sc_dims,
                "supercell_n": sc_dims_str,  # For backward compatibility
                "delta": d,
                "fmax": fm,
                "negative_frequency_at_special_point": None,
                "time_taken": 0,
                "relaxation_status": "failed"
            })
            continue # Skip to the next iteration if relaxation fails  
        else:
            print(f"Relaxation successful for N={sc_dims_str}, D={d}, F={fm}. Proceeding with phonon analysis.")
            results.append({
                "supercell_dims": sc_dims,
                "supercell_n": sc_dims_str,  # For backward compatibility
                "delta": d,
                "fmax": fm,
                "relaxation_status": "successful"
            })
        # Expect a list of softest modes now
        softest_modes_info_current, neg_freq_at_special_point, time_taken, _ = run_single_phonon_analysis(
            relaxed_atoms_for_current_params.copy(), calculator, args.engine, args.units, sc_dims, d, fm, current_output_dir, prefix=original_prefix,
            phonon_path_npoints=phonon_path_npoints,
            phonon_dos_grid=phonon_dos_grid,
            traj_kT=default_traj_kT,
            num_modes_to_return=num_modes_to_return,
            negative_phonon_threshold=negative_phonon_threshold_thz,
            save_yaml=args.save_yaml
        )

        if neg_freq_at_special_point is not None:
            results[-1].update({  
                "negative_frequency_at_special_point": neg_freq_at_special_point,  
                "time_taken": time_taken  
            })
            if neg_freq_at_special_point > best_negative_frequency:
                best_negative_frequency = neg_freq_at_special_point
                best_settings = {"supercell_dims": sc_dims, "supercell_n": sc_dims_str, "delta": d, "fmax": fm}
                best_softest_modes_info = softest_modes_info_current
                best_relaxed_atoms = relaxed_atoms_for_current_params.copy()
            improvement = best_negative_frequency - previous_best_negative_frequency
            if improvement < abs(threshold_in_current_units * 0.5):
                print(f"Improvement in negative frequency ({improvement:.4f} {args.units}) is less than {abs(threshold_in_current_units * 0.5):.4f} {args.units}. Stopping parameter sweep.")
                break
            previous_best_negative_frequency = best_negative_frequency

    print("\n--- Parameter sweep auto-optimization complete ---")
    print(f"Most negative frequency at a special point found: {best_negative_frequency:.4f} {args.units}")
    print(f"Optimal settings: Supercell Dims = {best_settings.get('supercell_dims')}, Delta = {best_settings.get('delta')}, Fmax = {best_settings.get('fmax')}")

    with open(os.path.join(output_dir, "auto_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    return best_negative_frequency, best_settings, best_softest_modes_info, best_relaxed_atoms

def run_automatic_soft_mode_optimization(args, output_dir, best_negative_frequency, best_settings, best_softest_modes_info, best_relaxed_atoms, negative_phonon_threshold_thz,
                                     soft_mode_max_iterations, soft_mode_displacement_scales, mode2_ratio_scales, soft_mode_num_top_structures_to_analyze,
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, cell_scale_factors, num_modes_to_return,
                                     ga_population_size, ga_mutation_rate, ga_generations, num_new_points_per_iteration, ga_disp_scale_bounds, ga_ratio_bounds, ga_cell_scale_bounds, ga_cell_angle_bounds
    ):
    """
    Routes to the appropriate soft mode optimization method (GA or Traditional).
    """
    threshold_in_current_units = negative_phonon_threshold_thz
    if args.units == "cm-1":
        threshold_in_current_units *= 33.35641
    elif args.units == "eV":
        threshold_in_current_units *= 4.135667696e-3

    if best_negative_frequency < threshold_in_current_units and best_softest_modes_info and best_relaxed_atoms is not None:
        print(f"\nSoft mode detected ({best_negative_frequency:.4f} {args.units}) after parameter sweep. Initiating iterative soft mode optimization with method: {args.method}...")
        if args.method == "ga":
            run_ga_soft_mode_optimization(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                ga_population_size=ga_population_size,
                ga_mutation_rate=ga_mutation_rate,
                ga_generations=ga_generations,
                num_new_points_per_iteration=num_new_points_per_iteration,
                ga_disp_scale_bounds=args.ga_disp_scale_bounds,
                ga_ratio_bounds=args.ga_ratio_bounds,
                ga_cell_scale_bounds=args.ga_cell_scale_bounds,
                ga_cell_angle_bounds=args.ga_cell_angle_bounds
            )
        elif args.method == "traditional":
            run_traditional_soft_mode_optimization(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
            )
        elif args.method == "traditional_all":
            run_traditional_all_soft_mode_optimization(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
            )
        elif args.method == "opt_random":
            # Load random-specific parameters from settings
            from .utils.utils import load_default_settings
            default_settings = load_default_settings()
            random_displacement_bounds = default_settings.get("random_displacement_bounds", [0.1, 2.0])
            random_cell_perturbation = default_settings.get("random_cell_perturbation", True)
            random_seed = default_settings.get("random_seed", None)

            run_random_structure_search(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                num_new_points_per_iteration=num_new_points_per_iteration,
                random_displacement_bounds=random_displacement_bounds,
                random_cell_perturbation=random_cell_perturbation,
                random_seed=random_seed,
                ga_cell_scale_bounds=ga_cell_scale_bounds,
                ga_cell_angle_bounds=ga_cell_angle_bounds,
            )
        elif args.method == "neb":
            run_neb_soft_mode_optimization(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                neb_num_images=args.neb_num_images,
                neb_spring_constant=args.neb_spring_constant,
                neb_max_iterations=args.neb_max_iterations,
                neb_force_tolerance=args.neb_force_tolerance,
                final_cif_path=args.final_cif
            )
        elif args.method == "ci_neb":
            run_ci_neb_soft_mode_optimization(
                args,
                output_dir,
                best_relaxed_atoms,
                best_softest_modes_info,
                max_iterations=soft_mode_max_iterations,
                soft_mode_displacement_scales=soft_mode_displacement_scales,
                cell_scale_factors=cell_scale_factors,
                mode2_ratio_scales=mode2_ratio_scales,
                num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
                negative_phonon_threshold_thz=negative_phonon_threshold_thz,
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                default_traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                neb_num_images=args.neb_num_images,
                neb_spring_constant=args.neb_spring_constant,
                neb_max_iterations=args.neb_max_iterations,
                neb_force_tolerance=args.neb_force_tolerance,
                neb_climbing_start_iteration=args.neb_climbing_start_iteration,
                final_cif_path=args.final_cif
            )
        else:
            print(f"Error: Unknown soft mode optimization method: {args.method}")
    else:
        print("\nNo significant soft mode detected after parameter sweep, or structure is stable enough. Skipping iterative soft mode optimization.")

def convert_results_for_ga(results):  
    """Convert results format to GA-expected format"""  
    ga_format_results = []  
    for result in results:  
        if result.get('energy_per_atom') is not None:  
            ga_format_results.append({  
                'params': result['params'],  
                'fitness': result['energy_per_atom']  # GA expects 'fitness' key  
            })  
    return ga_format_results

def run_ga_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_modes_info_list, max_iterations,
                               soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, ga_population_size, ga_mutation_rate, ga_generations, num_new_points_per_iteration,
                               ga_disp_scale_bounds, ga_ratio_bounds, ga_cell_scale_bounds, ga_cell_angle_bounds
                               ):
    """
    Runs an iterative workflow to find low-energy structures using a Genetic Algorithm,
    and then performs a final phonon analysis on the best candidates found across all iterations.
    """
    print("\n--- Running Soft Mode Iterative Optimization (Genetic Algorithm) ---")

    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_modes_info_list = initial_softest_modes_info_list
    
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)
    if calculator is None: return

    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]
    checkpoint_manager = CheckpointManager(base_output_dir, 'ga')
    
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    checkpoint_state = None
    resume_from_checkpoint = False
    
    if checkpoint_data:
        checkpoint_full, checkpoint_state = checkpoint_data
        resume_from_checkpoint = True
        print("\n" + "="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        
        # Restore data
        if checkpoint_full.get('current_softest_modes'):
             current_softest_modes_info_list = checkpoint_full['current_softest_modes']
        
        # --- EXPLICIT CONVERSION TO NUMPY ---
        # Force conversion here in case CheckpointManager failed or used Lists
        if current_softest_modes_info_list:
            for mode in current_softest_modes_info_list:
                if 'raw_displacements' in mode:
                    mode['raw_displacements'] = np.array(mode['raw_displacements'], dtype=complex)
        
        if checkpoint_full.get('current_primitive_atoms'):
             current_primitive_atoms = checkpoint_full['current_primitive_atoms']

    # --- CHECK IF DATA IS VALID ---
    needs_recalc = False
    if not current_softest_modes_info_list:
        needs_recalc = True
        print("DEBUG: Recalc needed - List is empty")
    elif len(current_softest_modes_info_list) > 0:
        first_mode = current_softest_modes_info_list[0]
        if first_mode.get('label') == 'RESUME_PLACEHOLDER':
            needs_recalc = True
            print("DEBUG: Recalc needed - Placeholder detected")
        elif 'raw_displacements' not in first_mode:
            needs_recalc = True
            print("DEBUG: Recalc needed - missing raw_displacements key")
        else:
            disp = first_mode['raw_displacements']
            # Strict dimension checks
            if isinstance(disp, list) and len(disp) == 0:
                 needs_recalc = True
                 print("DEBUG: Recalc needed - Displacement is empty list")
            elif isinstance(disp, np.ndarray):
                 if disp.size == 0:
                     needs_recalc = True
                     print("DEBUG: Recalc needed - Displacement array is size 0")
                 elif disp.ndim < 2:
                     # This caused the AxisError. If it's 1D, we must recalc.
                     needs_recalc = True
                     print(f"DEBUG: Recalc needed - Invalid dimensions: {disp.shape} (Must be at least 2D)")

    if needs_recalc:
        print("⚠ RESUME NOTICE: Soft mode data is missing, invalid, or placeholder.")
        print("  Running immediate phonon analysis to recover eigenvectors before starting GA...")
        
        recovery_dir = os.path.join(base_output_dir, "resume_recovery_phonon_check")
        os.makedirs(recovery_dir, exist_ok=True)
        
        recovered_modes, _, _, recovered_tracking = run_single_phonon_analysis(
            current_primitive_atoms.copy(), calculator, args.engine, args.units,
            args.supercell_dims, args.delta, args.fmax, recovery_dir, 
            prefix="resume_recovery",
            phonon_path_npoints=phonon_path_npoints, phonon_dos_grid=phonon_dos_grid,
            traj_kT=default_traj_kT, num_modes_to_return=num_modes_to_return,
            negative_phonon_threshold=negative_phonon_threshold_thz,
            save_yaml=args.save_yaml
        )
        
        if recovered_modes:
            current_softest_modes_info_list = recovered_modes
            print(f"  ✓ Recovered {len(recovered_modes)} soft modes.")
        else:
            print("  ⚠ Warning: No soft modes found during recovery check.")

    threshold_in_current_units = negative_phonon_threshold_thz
    if args.units == "cm-1":
        threshold_in_current_units *= 33.35641
    elif args.units == "eV":
        threshold_in_current_units *= 4.135667696e-3

    # Store the reference energy of the primitive cell  
    # Ensure the calculator is set for initial_atoms_for_soft_mode_analysis  
    initial_atoms_for_soft_mode_analysis.set_calculator(calculator)  
    try:  
        E_ref = initial_atoms_for_soft_mode_analysis.get_potential_energy() / len(initial_atoms_for_soft_mode_analysis)  
        print(f"Reference energy (E_ref) of the primitive cell: {E_ref:.6f} eV/atom")  
    except Exception as e:  
        print(f"Warning: Could not get reference energy for primitive cell: {e}. Decomposition filter will be skipped.")  
        E_ref = None # Disable filter if E_ref cannot be obtained  
  
    # Define decomposition threshold from command line argument
    decomposition_threshold = -args.decomposition_threshold  # Convert to negative for comparison (E_relax < E_ref - threshold)
                
    # Determine supercell variants based on the q-point of the softest mode
    # If no soft mode info, default to (2,2,2)
    if 'coordinate' in current_softest_modes_info_list[0]:
        q_point_for_supercell = current_softest_modes_info_list[0]['coordinate']
        commensurate_supercell = estimate_commensurate_supercell_size(q_point_for_supercell)
        print(f"Using commensurate supercell size {commensurate_supercell} based on softest mode q-point {q_point_for_supercell}.")
    else:
        commensurate_supercell = (2,2,2) # Default to primitive if no q-point info
        print("No q-point information for softest mode. Defaulting to (2,2,2) supercell.")  
    
    # Always include these specific supercells  
    required_supercells = [(1,1,1), (2,1,1), (1,1,2), (2,2,1), (2,1,2), (2,2,2)]  
    supercell_variants = [commensurate_supercell] + [sc for sc in required_supercells if sc != commensurate_supercell]
                
    # Stores results for all iterations, including GA parameters, fitness (energy), and relaxed atoms object.
    if resume_from_checkpoint:
        all_iterations_results = checkpoint_full.get('results', [])
        # Restore current primitive atoms if available
        if checkpoint_full.get('current_primitive_atoms'):
            current_primitive_atoms = checkpoint_full['current_primitive_atoms']
        # Restore current soft modes if available
        if checkpoint_full.get('current_softest_modes'):
            current_softest_modes_info_list = checkpoint_full['current_softest_modes']
        print(f"Restored {len(all_iterations_results)} previous results from checkpoint")
    else:
        all_iterations_results = []

    # Define GA parameter bounds. These ranges should be configurable, perhaps from default_settings.json.
    disp_scale_bounds = (0.0, 10.0) # Example: 0 to 10 Angstrom displacement magnitude
    ratio_mode2_to_mode1_bounds = (-1.5, 1.5) # Example: 0 to 1 (0 means no mode 2, 1 means equal magnitude)
    cell_scale_bounds = (-0.50, 0.50) # Example: -50% to +50% change in a,b,c
    cell_angle_bounds = (-45.0, 45.0) # Example: -45 to +45 degrees change in alpha,beta,gamma

    # Initialize Genetic Algorithm with empty tracked data (will be updated during first guidance check)
    initial_tracked_data = {
        'soft_modes': [],
        'highest_freq_modes': [],
        'lowest_freq_modes': []
    }

    ga = GeneticAlgorithm(
        population_size=ga_population_size,
        mutation_rate=ga_mutation_rate,
        displacement_scale_bounds=ga_disp_scale_bounds, # Use new bounds
        ratio_mode2_to_mode1_bounds=ga_ratio_bounds, # Use new bounds
        cell_scale_bounds=ga_cell_scale_bounds, # Use new bounds
        cell_angle_bounds=ga_cell_angle_bounds, # Use new bounds
        supercell_variants=supercell_variants,
        num_offspring=num_new_points_per_iteration,
        tracked_k_points_data=initial_tracked_data
    )

    # --- 2. Main Iterative Loop (Genetic Algorithm Driven) ---
    # Determine starting iteration based on checkpoint
    start_iteration = 1
    if resume_from_checkpoint:
        start_iteration = checkpoint_state.get('main_iteration', 1)
        print(f"Resuming from main iteration {start_iteration}")
    
    for main_iteration_idx in range(start_iteration, max_iterations + 1):
        print(f"\n### Starting Main Iteration {main_iteration_idx} ({ga_generations} GA generations + phonon check) ###")

        # Run GA generations for this main iteration
        # Determine starting generation based on checkpoint
        start_generation = 1
        if resume_from_checkpoint and main_iteration_idx == checkpoint_state.get('main_iteration', 1):
            start_generation = checkpoint_state.get('ga_generation', 1)
            print(f"Resuming from GA generation {start_generation}")
        
        for ga_generation in range(start_generation, ga_generations + 1):
            print(f"\n--- GA Generation {ga_generation} of Main Iteration {main_iteration_idx} ---")

            # Initialize mutation summary to ensure it's always available
            mutation_summary = {
                'total_individuals': 0,
                'mode_replacements': 0,
                'replacement_rate': 0.0,
                'selected_modes': []
            }
            
            # CRITICAL IMPROVEMENT 1: Restore offspring parameters from checkpoint if resuming mid-generation
            new_offspring_params = None
            new_mutation_data = None
            
            if resume_from_checkpoint and checkpoint_full.get('current_offspring_params'):
                if main_iteration_idx == checkpoint_state.get('main_iteration', 1) and ga_generation == checkpoint_state.get('ga_generation', 1):
                    print("✓ Restoring offspring parameters from checkpoint to ensure GA consistency...")
                    new_offspring_params = checkpoint_full['current_offspring_params']
                    # Generate dummy mutation data for restored offspring (logging purposes)
                    new_mutation_data = [{'mode_replaced': False, 'selected_mode': None}] * len(new_offspring_params)
                    print(f"  Restored {len(new_offspring_params)} offspring parameters from checkpoint")
                    # Clear the checkpoint data so next generation doesn't use it
                    checkpoint_full['current_offspring_params'] = None
        
            # Prepare initial population for the first iteration, or evolve for subsequent
            if ga_generation == 1:
                print("First main iteration, first generation - creating initial population.")

                initial_ga_individuals = []
                for disp_scale in soft_mode_displacement_scales:
                    for cell_scale in cell_scale_factors:
                        for ratio_mode2 in mode2_ratio_scales:
                            for supercell_variant in [supercell_variants[0]]: # We keep the commensurate only for initial screening
                                initial_cell_transform_vec = (cell_scale, cell_scale, cell_scale, 0.0, 0.0, 0.0)
                                initial_ga_individuals.append({
                                    'params': (disp_scale, ratio_mode2, initial_cell_transform_vec, supercell_variant, True),  # phase factor true
                                    'fitness': None
                                })
                # Adds additional random samples to reach population_size
                # The GA object now handles this internally when filling up the population
                ga.initialize_population(initial_individuals=initial_ga_individuals)
                if new_offspring_params is None:  # Only if not restored from checkpoint
                    new_offspring_params, new_mutation_data = ga.get_population_with_mutation_data()

                # Get mutation summary for initialization (includes mode replacements during random generation)
                mutation_summary = ga.get_mutation_summary()
                if mutation_summary['total_individuals'] == 0:
                    # Fallback for initialization case
                    mutation_summary = {
                        'total_individuals': len(new_offspring_params),
                        'mode_replacements': sum(1 for ind in ga.population if ind.get('mutation_data', {}).get('mode_replaced', False)),
                        'replacement_rate': 0.0,
                        'selected_modes': []
                    }
                    if mutation_summary['total_individuals'] > 0:
                        mutation_summary['replacement_rate'] = mutation_summary['mode_replacements'] / mutation_summary['total_individuals']
            else:  
                # Subsequent generations: normal GA evolution (unless restored from checkpoint)
                if new_offspring_params is None:  # Only if not restored from checkpoint
                    current_generation_results = [r for r in all_iterations_results if r.get('main_iteration') == main_iteration_idx and r.get('ga_generation') == ga_generation - 1]  
                    if not current_generation_results:  
                        print(f"No results from previous generation to evolve from. Using all available results.")  
                        current_generation_results = all_iterations_results[-ga_population_size:] if len(all_iterations_results) >= ga_population_size else all_iterations_results  
                    
                    ga_format_current = convert_results_for_ga(current_generation_results)
                    if ga_format_current:
                        new_offspring_params, new_mutation_data = ga.evolve(ga_format_current)
                        # Get mutation summary for this generation
                        mutation_summary = ga.get_mutation_summary()
                    else:
                        print("No valid results to evolve from. Generating random population.")
                        ga.initialize_population()
                        new_offspring_params, new_mutation_data = ga.get_population_with_mutation_data()
                        # No mutation summary for random initialization
                        mutation_summary = {
                            'total_individuals': len(new_offspring_params),
                            'mode_replacements': 0,
                            'replacement_rate': 0.0,
                            'selected_modes': []
                        }
                else:
                    # Offspring were restored from checkpoint, use them as-is
                    mutation_summary = {
                        'total_individuals': len(new_offspring_params),
                        'mode_replacements': 0,
                        'replacement_rate': 0.0,
                        'selected_modes': []
                    }
                        
            if not new_offspring_params:
                print(f"GA did not generate any new offspring for main iteration {main_iteration_idx}, generation {ga_generation}. Stopping.")
                break

            iteration_results = [] # To store results for this specific GA iteration
            decomposed_count = 0  # Track structures flagged as decomposed in this generation

            # Generate, Relax, and Evaluate each new individual
            for i, (individual_params, individual_mutation_data) in enumerate(zip(new_offspring_params, new_mutation_data)):
                sample_index = i + 1
                
                # Check if we should skip this sample (already completed in checkpoint)
                if resume_from_checkpoint:
                    current_state_check = {
                        'main_iteration': main_iteration_idx,
                        'ga_generation': ga_generation,
                        'sample_index': sample_index
                    }
                    if should_skip_sample(checkpoint_state, current_state_check, 'ga'):
                        print(f"\n  ⏭ Skipping sample {sample_index} (already completed in checkpoint)")
                        continue
                
                scale_mode1, ratio_mode2_to_mode1, cell_transformation_vector, supercell_variant, use_phase_factor = individual_params

                sample_output_dir = os.path.join(base_output_dir, f"main_iter_{main_iteration_idx}_gen_{ga_generation}", f"sample_{sample_index}")
                os.makedirs(sample_output_dir, exist_ok=True)

                print(f"\n  Generating and relaxing structure for GA sample {sample_index} (Main Iter {main_iteration_idx}, Gen {ga_generation}):")

                # Retry mechanism for unphysical parameters
                max_retries = 10  # Maximum number of retry attempts
                retry_count = 0
                generated_cif_paths = []

                while not generated_cif_paths and retry_count < max_retries:
                    if retry_count > 0:
                        print(f"    Retry {retry_count}/{max_retries}: Generating new parameters due to unphysical cell parameters")
                        # Generate new parameters for this individual
                        new_individual_params, new_individual_mutation_data = ga._generate_random_individual()
                        scale_mode1, ratio_mode2_to_mode1, cell_transformation_vector, supercell_variant, use_phase_factor = new_individual_params
                        individual_mutation_data = new_individual_mutation_data

                    # Log the supercell being used for this individual
                    print(f"    Mode1 Scale: {scale_mode1:.3f}, Mode2 Ratio: {ratio_mode2_to_mode1:.3f}, Cell Transform: {cell_transformation_vector}, Supercell: {supercell_variant}, Phase Factor: {use_phase_factor}")

                    # Generate displaced supercells using the specific variant for this individual
                    generated_cif_paths = generate_displaced_supercells(
                        current_primitive_atoms.copy(),
                        current_softest_modes_info_list,
                        scale_mode1,
                        ratio_mode2_to_mode1,
                        [supercell_variant], # Pass the single variant as a list
                        sample_output_dir,
                        main_iteration_idx,
                        original_prefix,
                        cell_transformation_vector,
                        use_phase_factor,
                        individual_mutation_data,  # Pass mutation data for filename and logging
                        max_atoms_per_supercell=getattr(args, 'max_atoms_per_supercell', None)
                    )

                    retry_count += 1

                if not generated_cif_paths:
                    print(f"    Failed to generate valid structures after {max_retries} retries for sample {i+1}. Skipping relaxation.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None,
                        'relaxed_atoms': None,
                        'original_file': None,
                        'main_iteration': main_iteration_idx,
                        'ga_generation': ga_generation,
                        'sample': i + 1
                    })
                    continue
                else:
                    # Update the individual_params with the successful parameters if retries were needed
                    if retry_count > 1:
                        individual_params = (scale_mode1, ratio_mode2_to_mode1, cell_transformation_vector, supercell_variant, use_phase_factor)
                        print(f"    Successfully generated structures with retry parameters after {retry_count-1} attempts")

                # Relax the generated structures
                unique_supercell_folders = list(set(os.path.dirname(fpath) for fpath in generated_cif_paths))

                sample_relaxation_results = []
                for folder in unique_supercell_folders:
                    folder_relaxation_results = relax_structures_in_folder(
                        folder, calculator, args.engine, args.fmax,
                        relaxation_patience=getattr(args, 'relaxation_patience', 5),
                        volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
                    )
                    sample_relaxation_results.extend(folder_relaxation_results)

                if not sample_relaxation_results:
                    print(f"    No structures successfully relaxed for sample {i+1}.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None, 
                        'relaxed_atoms': None,
                        'original_file': None,
                        'num_atoms': 'N/A',
                        'international_symbol': 'N/A',
                        'crystal_system': 'N/A',
                        'main_iteration': main_iteration_idx,  
                        'ga_generation': ga_generation,  
                        'sample': i + 1  
                    })
                    continue

                # Find the lowest energy structure among the relaxed ones for this sample
                lowest_energy_sample = find_lowest_energy_structures(sample_relaxation_results, num_to_select=1)
                if lowest_energy_sample:
                    best_result_for_sample = lowest_energy_sample[0]

                    E_relax = best_result_for_sample['energy_per_atom']
                    if E_ref is not None and E_relax < (E_ref + decomposition_threshold): # Note: decomposition_threshold is negative
                        decomposed_count += 1
                        print(f"    Sample {i+1} (Energy: {E_relax:.6f} eV/atom) flagged as DECOMPOSED (E_relax < E_ref {E_ref:.6f} + {decomposition_threshold:.3f} eV). Excluding from ranking and further GA steps.")
                    else:
                        iteration_results.append({
                            'params': individual_params,
                            'energy_per_atom': best_result_for_sample['energy_per_atom'],
                            'relaxed_atoms': best_result_for_sample['relaxed_atoms'],
                            'original_file': best_result_for_sample['original_file'],
                            'num_atoms': best_result_for_sample.get('num_atoms', 'N/A'),
                            'international_symbol': best_result_for_sample.get('international_symbol', 'N/A'),
                            'crystal_system': best_result_for_sample.get('crystal_system', 'N/A'),
                            'main_iteration': main_iteration_idx,  
                            'ga_generation': ga_generation,  
                            'sample': sample_index
                        })
                        print(f"    Sample {sample_index} relaxed. Lowest energy: {best_result_for_sample['energy_per_atom']:.6f} eV/atom")
                        
                        # Save checkpoint after each successful sample
                        try:
                            checkpoint_manager.save_checkpoint_ga(
                                main_iteration=main_iteration_idx,
                                ga_generation=ga_generation,
                                sample_index=sample_index,
                                all_iterations_results=all_iterations_results + iteration_results,
                                ga_state={'population': ga.population, 'generation': ga_generation},
                                current_primitive_atoms=current_primitive_atoms,
                                current_softest_modes=current_softest_modes_info_list,
                                tracked_k_points_data=ga.tracked_k_points_data if hasattr(ga, 'tracked_k_points_data') else None,
                                current_offspring_params=new_offspring_params
                            )
                            checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
                        except Exception as e:
                            print(f"    ⚠ Warning: Failed to save checkpoint: {e}")
                else:
                    print(f"    Could not find lowest energy for sample {i+1}.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None, # Indicate failure
                        'relaxed_atoms': None,
                        'original_file': None,
                        'main_iteration': main_iteration_idx,  
                        'ga_generation': ga_generation,  
                        'sample': sample_index
                    })

            # Add this iteration's results to the master list for GA evolution in next iteration
            # Filter out failed relaxations (energy_per_atom is None) before adding to all_iterations_results
            valid_iteration_results = [r for r in iteration_results if r['energy_per_atom'] is not None]
            all_iterations_results.extend(valid_iteration_results)

            # Enhanced warning system for decomposition issues
            total_samples = len(new_offspring_params)
            if decomposed_count > 0:
                print(f"\n{'='*80}")
                print(f"DECOMPOSITION WARNING - Main Iteration {main_iteration_idx}, Generation {ga_generation}")
                print(f"{'='*80}")
                print(f"Structures flagged as DECOMPOSED: {decomposed_count}/{total_samples} ({decomposed_count/total_samples*100:.1f}%)")
                print(f"Reference energy (E_ref): {E_ref:.6f} eV/atom")
                print(f"Decomposition threshold: {args.decomposition_threshold:.3f} eV")
                print(f"Threshold condition: E_relax < E_ref - {args.decomposition_threshold:.3f} = {E_ref + decomposition_threshold:.6f} eV/atom")

                if decomposed_count == total_samples:
                    print(f"\n*** CRITICAL: ALL STRUCTURES FLAGGED AS DECOMPOSED ***")
                    print(f"This prevents GA progression as no valid structures remain for evolution.")
                    print(f"Consider adjusting the decomposition threshold using:")
                    print(f"  --decomposition_threshold {args.decomposition_threshold + 0.5:.1f}  (more permissive)")
                    print(f"  --decomposition_threshold {args.decomposition_threshold + 1.0:.1f}  (much more permissive)")
                elif decomposed_count > total_samples * 0.8:
                    print(f"\n*** WARNING: High decomposition rate ({decomposed_count/total_samples*100:.1f}%) ***")
                    print(f"This may severely limit GA evolution effectiveness.")
                    print(f"Consider adjusting --decomposition_threshold if this pattern persists.")
                print(f"{'='*80}\n")

            if not valid_iteration_results:
                print(f"No valid structures found in main iteration {main_iteration_idx}, generation {ga_generation}. Continuing anyway (exploratory mode).")

            # Create iteration-specific relaxation summary
            generation_dir = os.path.join(base_output_dir, f"main_iter_{main_iteration_idx}_gen_{ga_generation}")
            os.makedirs(generation_dir, exist_ok=True)
            iteration_summary_filepath = os.path.join(generation_dir, "relaxation_summary_generation.txt")

            with open(iteration_summary_filepath, 'w') as f:
                f.write(f"--- Relaxation Summary for Main Iteration {main_iteration_idx}, GA Generation {ga_generation} ---\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Add GA mutation tracking summary
                if mutation_summary is not None and 'total_individuals' in mutation_summary:
                    f.write(f"--- GA Mutation Tracking Summary ---\n")
                    f.write(f"Total Individuals: {mutation_summary['total_individuals']}\n")
                    f.write(f"Mode Replacements: {mutation_summary['mode_replacements']}\n")
                    f.write(f"Replacement Rate: {mutation_summary['replacement_rate']:.2%}\n")
                    if mutation_summary['selected_modes']:
                        f.write(f"Selected Modes for Replacement:\n")
                        for i, mode in enumerate(mutation_summary['selected_modes'][:5]):  # Show first 5
                            f.write(f"  {i+1}. {mode['label']} (freq: {mode['frequency']:.4f}, band: {mode['band_index']})\n")
                        if len(mutation_summary['selected_modes']) > 5:
                            f.write(f"  ... and {len(mutation_summary['selected_modes']) - 5} more\n")
                    else:
                        f.write(f"No modes were selected for replacement in this generation.\n")
                    f.write(f"\n")

                # Add decomposition tracking summary
                f.write(f"--- Decomposition Analysis Summary ---\n")
                f.write(f"Reference Energy (E_ref): {E_ref:.6f} eV/atom\n")
                f.write(f"Decomposition Threshold: {args.decomposition_threshold:.3f} eV\n")
                f.write(f"Threshold Condition: E_relax < E_ref - {args.decomposition_threshold:.3f} = {E_ref + decomposition_threshold:.6f} eV/atom\n")
                f.write(f"Total Samples: {total_samples}\n")
                f.write(f"Structures Flagged as DECOMPOSED: {decomposed_count} ({decomposed_count/total_samples*100:.1f}%)\n")
                f.write(f"Valid Structures for GA Evolution: {len(valid_iteration_results)} ({len(valid_iteration_results)/total_samples*100:.1f}%)\n")
                if decomposed_count == total_samples:
                    f.write(f"*** CRITICAL: ALL STRUCTURES FLAGGED AS DECOMPOSED - GA PROGRESSION BLOCKED ***\n")
                elif decomposed_count > total_samples * 0.8:
                    f.write(f"*** WARNING: High decomposition rate may limit GA effectiveness ***\n")
                f.write(f"\n")

                f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'Main Iter':<10} {'GA Gen':<8} {'Sample':<8} {'GA Params':<65}\n")
                f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*10:<10} {'-'*8:<8} {'-'*8:<8} {'-'*65:<65}\n")

                # Sort valid_iteration_results by energy_per_atom for this summary
                if valid_iteration_results:
                    sorted_iter_results = sorted(valid_iteration_results, key=lambda x: x['energy_per_atom'])
                else:
                    sorted_iter_results = []

                for result in sorted_iter_results:
                    num_atoms = result.get('num_atoms', 'N/A')
                    international_symbol = result.get('international_symbol', 'N/A')
                    crystal_system = result.get('crystal_system', 'N/A')
                    energy = result.get('energy_per_atom', 'FAIL')
                    params = result.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A'))
                    # Format cell_transformation_vector to 3 decimal places
                    if isinstance(params, tuple) and len(params) == 5:  
                        cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])  
                        sc_str = "x".join(map(str, params[3])) # e.g., (2,2,1) -> "2x2x1"  
                        ph_fac = params[4] if len(params) > 4 else 1.0  # Default to 1.0 if not provided
                        params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str}), SC:{sc_str}, PhFactor:{ph_fac}"  
                    else:  
                        params_str = str(params)
                    energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)  
                    f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('main_iteration', 'N/A')):<10} {str(result.get('ga_generation', 'N/A')):<8} {str(result.get('sample', 'N/A')):<8} {params_str:<80}\n")
            
            print(f"Generation results saved to {iteration_summary_filepath}")
            
        # End of GA generation loop  
          
        # After GA generations, perform phonon check and update primitive cell
        print(f"\n--- Completed {ga_generations} GA generations for Main Iteration {main_iteration_idx} ---")
        print("--- Performing phonon check and primitive cell update ---")  
      
        # --- Prepare for Next Iteration: Find the soft mode of the overall best structure found so far ---
        # This is crucial for guiding the next GA generation with relevant phonon information.
        # Sort all valid results collected so far to find the absolute best structure
        valid_overall_results_so_far = [
            r for r in all_iterations_results
            if isinstance(r, dict) and 'energy_per_atom' in r and r['energy_per_atom'] is not None
        ]
        
        if not valid_overall_results_so_far:
            print("No valid structures found across all iterations to guide next step. Continuing anyway (exploratory mode).")
            # Continue with current primitive atoms and soft modes
            continue

        sorted_overall_best_so_far = sorted(valid_overall_results_so_far, key=lambda x: x['energy_per_atom'])
        best_candidate_for_next_iter = sorted_overall_best_so_far[0]

        try:
            relaxed_supercell = best_candidate_for_next_iter['relaxed_atoms']
            pmg_structure = AseAtomsAdaptor.get_structure(relaxed_supercell)
            
            sga = SpacegroupAnalyzer(pmg_structure)  
            primitive_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(sga.get_primitive_standard_structure())
            
            # Run a minimal phonon check just to get the next guiding mode(s)
            check_dir = os.path.join(base_output_dir, f"main_iter_{main_iteration_idx}_guidance_phonon_check")
            os.makedirs(check_dir, exist_ok=True)
            print(f"\nPerforming guidance phonon check on best structure from Main Iteration {main_iteration_idx} to find next soft modes...")
            
            primitive_cif_path = os.path.join(check_dir, f"primitive_for_guidance_main_iter_{main_iteration_idx}.cif")  
            write(primitive_cif_path, primitive_atoms_for_next_iter)  
            print(f"Saved primitive structure for guidance phonon check to: {primitive_cif_path}")
            
            conventional_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(sga.get_conventional_standard_structure())  
            conventional_cif_path = os.path.join(check_dir, f"conventional_for_guidance_main_iter_{main_iteration_idx}.cif")  
            write(conventional_cif_path, conventional_atoms_for_next_iter)  
            print(f"Saved conventional structure for guidance phonon check to: {conventional_cif_path}")

            next_softest_modes_info_list, _, _, tracked_k_points_data = run_single_phonon_analysis(
                primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, check_dir, # FIX: Use the parsed dimensions
                prefix=f"guidance_check_main_iter_{main_iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            # Update the guiding atoms and soft mode list for the next loop
            current_primitive_atoms = primitive_atoms_for_next_iter.copy()
            current_softest_modes_info_list = next_softest_modes_info_list

            # Update GA with new tracked k-points data for mode replacement
            ga.tracked_k_points_data = tracked_k_points_data
            print(f"Updated GA with tracked k-points data: {len(tracked_k_points_data['soft_modes'])} soft modes, "
                  f"{len(tracked_k_points_data['highest_freq_modes'])} highest freq modes, "
                  f"{len(tracked_k_points_data['lowest_freq_modes'])} lowest freq modes")
            
            # Update supercell variants based on the new guiding mode's q-point
            if 'coordinate' in next_softest_modes_info_list[0]:
                q_point_for_supercell = next_softest_modes_info_list[0]['coordinate']
                
                # FIX: Add a standard 2x2x2 (or similar) as a backup option
                commensurate = estimate_commensurate_supercell_size(q_point_for_supercell)
                backup_sc = (2, 2, 2) # Or use args.supercell_dims
                
                # Create list with both. The filter will now reject 'commensurate' if it's huge 
                # and pick 'backup_sc' instead because it fits the atom limit.
                supercell_variants = [commensurate, backup_sc]
                
                print(f"Updated commensurate supercell size to {supercell_variants[0]} based on new softest mode q-point {q_point_for_supercell}.")
            else:
                supercell_variants = [(2,2,2)] # Default if no q-point info
                print("No q-point information for new softest mode. Defaulting to (2,2,2) supercell for next iteration.")

            # REMOVED: Don't stop if no soft modes found - this is exploratory
            if not current_softest_modes_info_list:  
                print(f"No soft modes found in best structure from main iteration {main_iteration_idx}. Continuing anyway (exploratory mode).")
                # Use the original soft modes to continue
                current_softest_modes_info_list = initial_softest_modes_info_list
                
        except Exception as e:
            print(f"Could not prepare for next iteration due to an error during phonon analysis: {e}")
            print("Continuing with current primitive atoms and soft modes (exploratory mode).")
            import traceback
            traceback.print_exc()


    # --- 3. Final Phonon Analysis on Overall Best Structures ---
    print("\n\n--- All GA iterations complete. ---")
    print("--- Analyzing overall best structures for final phonon properties. ---")

    if not all_iterations_results:
        print("No structures were successfully relaxed to perform final analysis on.")
        return

    # Filter the master list to ensure all entries are valid dictionaries with the required key before sorting.
    valid_overall_results = [
        r for r in all_iterations_results
        if isinstance(r, dict) and 'energy_per_atom' in r and r['energy_per_atom'] is not None
    ]

    if not valid_overall_results:
        print("No valid structures with energy information were found across all iterations.")
        return

    # Sort the VALID results by energy and select the absolute best
    sorted_overall_best = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])
    
    # Filter out structures with energy per atom higher than E_ref  
    if E_ref is not None:  
        filtered_overall_best = [  
            r for r in sorted_overall_best if r['energy_per_atom'] <= E_ref  
        ]  
        print(f"Filtered out {len(sorted_overall_best) - len(filtered_overall_best)} structures with energy higher than initial reference energy ({E_ref:.6f} eV/atom).")  
        sorted_overall_best = filtered_overall_best  
        if not sorted_overall_best:  
            print("No structures remaining after filtering by reference energy. Skipping final phonon analysis.")  
            return
    
    final_top_structures = sorted_overall_best[:num_top_structures_to_analyze]

    print("\nAll valid iteration results (sorted by energy):")
    for idx, result in enumerate(sorted_overall_best, 1):
        original_file = os.path.basename(result.get('original_file', 'unknown'))
        energy = result['energy_per_atom']
        # Also print the GA parameters for context
        params = result.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A'))
        cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
        print(f"  {idx}. Iter {result.get('iteration', 'N/A')}, Sample {result.get('sample', 'N/A')}: {original_file} (Disp1: {params[0]:.3f}, Ratio21: {params[1]:.3f}, Cell:({cell_transform_str}, {cell_transform_str})): {energy:.6f} eV/atom")


    # Get all unique energies from ALL structures with tolerance
    all_structures = sorted_overall_best
    unique_structures = []
    energy_tolerance = 5e-4  # 0.5 meV/atom tolerance

    for result in all_structures:
        energy = result['energy_per_atom']
        is_unique = True
        
        # Check if this energy is unique within tolerance
        for existing_result in unique_structures:
            if abs(energy - existing_result['energy_per_atom']) < energy_tolerance:
                is_unique = False
                break
        
        if is_unique:
            unique_structures.append(result)

    # Now separate: first N are "top", rest are "unique"
    final_top_structures = unique_structures[:num_top_structures_to_analyze]
    unique_remaining_structures = unique_structures[num_top_structures_to_analyze:]
    print(f"\nFound {len(unique_structures)} structures with unique energies.")  
    print(f"Selected the top {len(final_top_structures)} structures and {len(unique_remaining_structures)} additional unique structures for final analysis:")

    final_structure_dir = os.path.join(base_output_dir, f"final_structures")
    os.makedirs(final_structure_dir, exist_ok=True)

    # Process top structures
    for i, result in enumerate(final_top_structures):
        _save_final_structure(result, final_structure_dir, i+1, "top", original_prefix)

    # Process unique remaining structures  
    for i, result in enumerate(unique_remaining_structures):
        _save_final_structure(result, final_structure_dir, i+1, "unique", original_prefix)


    # Run phonon analysis on top structures  
    for i, result in enumerate(final_top_structures):  
        top_structure_relaxed_supercell = result['relaxed_atoms']  
        energy_per_atom = result['energy_per_atom']  
        energy_str = f"{abs(energy_per_atom):.4f}".replace('.', 'p')  
        if energy_per_atom < 0:  
            energy_str = 'm' + energy_str  
        else:  
            energy_str = 'p' + energy_str  
        
        final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_top_{i+1}_energy_{energy_str}")  
        print(f"\n--- Running Final Phonon Analysis on Top Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")  
        try:  
            pmg_structure = AseAtomsAdaptor.get_structure(top_structure_relaxed_supercell)  
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())  
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structure_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
        except Exception as e:  
            print(f"  Error during final phonon analysis for top structure {i+1}: {e}")  
            import traceback  
            traceback.print_exc()  
    
    # Run phonon analysis on unique structures (only if there are any)  
    if unique_remaining_structures:  
        print(f"\n--- Running Final Phonon Analysis on {len(unique_remaining_structures)} Additional Unique Structures ---")  
        for i, result in enumerate(unique_remaining_structures):  
            unique_structure_relaxed_supercell = result['relaxed_atoms']  
            energy_per_atom = result['energy_per_atom']  
            energy_str = f"{abs(energy_per_atom):.4f}".replace('.', 'p')  
            if energy_per_atom < 0:  
                energy_str = 'm' + energy_str  
            else:  
                energy_str = 'p' + energy_str  
            
            final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_unique_{i+1}_energy_{energy_str}")  
            print(f"\n--- Running Final Phonon Analysis on Unique Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")  
            try:  
                pmg_structure = AseAtomsAdaptor.get_structure(unique_structure_relaxed_supercell)  
                primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())  
                _, _, _, _ = run_single_phonon_analysis(
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                    args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",
                    phonon_path_npoints=phonon_path_npoints,
                    phonon_dos_grid=phonon_dos_grid,
                    traj_kT=default_traj_kT,
                    num_modes_to_return=num_modes_to_return,
                    final_structures_dir=final_structure_dir,
                    negative_phonon_threshold=negative_phonon_threshold_thz,
                    save_yaml=args.save_yaml
                )
            except Exception as e:  
                print(f"  Error during final phonon analysis for unique structure {i+1}: {e}")  
                import traceback  
                traceback.print_exc()  
    else:  
        print(f"\nNo additional unique structures found beyond the top {len(final_top_structures)} structures.")

    print("\n--- Soft Mode Iterative Optimization Complete ---")

    # Create comprehensive GA summary similar to traditional_all mode
    print("\n--- Creating Comprehensive GA Summary ---")
    overall_summary_filepath = os.path.join(base_output_dir, "overall_ga_summary.txt")

    # Identify unique structures based on energy tolerance
    print(f"\nIdentifying unique structures from all {len(all_iterations_results)} calculated structures...")
    unique_structures = []
    energy_tolerance = 5e-4  # 0.5 meV/atom tolerance

    # Sort all structures by energy
    sorted_overall_results = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])

    for structure in sorted_overall_results:
        is_unique = True
        for unique_struct in unique_structures:
            if abs(structure['energy_per_atom'] - unique_struct['energy_per_atom']) < energy_tolerance:
                is_unique = False
                break
        if is_unique:
            unique_structures.append(structure)

    print(f"Found {len(unique_structures)} unique structures (energy tolerance: {energy_tolerance:.1e} eV/atom)")

    # Get final top structures for analysis
    final_top_structures = sorted_overall_results[:num_top_structures_to_analyze]

    with open(overall_summary_filepath, 'w') as f:
        f.write("--- Genetic Algorithm Soft Mode Optimization Summary ---\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total iterations completed: {main_iteration_idx}\n")
        f.write(f"Total successful structures: {len(all_iterations_results)}\n")
        f.write(f"Total unique structures identified: {len(unique_structures)}\n")
        f.write(f"Final structures analyzed: {len(final_top_structures)}\n\n")

        # Optimization Settings
        f.write("Optimization Settings:\n")
        f.write(f"  Engine: {args.engine}\n")
        f.write(f"  Units: THz\n")
        f.write(f"  Supercell dimensions: {args.supercell}\n")
        f.write(f"  Delta: {args.delta}\n")
        f.write(f"  Fmax: {args.fmax}\n")
        f.write(f"  Displacement scales: {soft_mode_displacement_scales}\n")
        f.write(f"  Mode2 ratios: {mode2_ratio_scales}\n")
        f.write(f"  Cell scale factors: {cell_scale_factors}\n")
        f.write(f"  Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")
        f.write(f"  Max iterations: {max_iterations}\n")
        f.write(f"  Top structures to analyze: {num_top_structures_to_analyze}\n")
        f.write(f"  GA population size: 50\n")
        f.write(f"  GA generations per iteration: 3\n\n")

        # All Successful Structures Table
        f.write("All Successful Structures (sorted by energy):\n")
        f.write("=" * 280 + "\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<20} {'Iter':<5} {'Gen':<5} {'Sample':<8} {'GA Parameters':<120} {'Soft Mode Details':<80}\n")
        f.write("=" * 280 + "\n")

        for result in sorted_overall_results:
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result.get('energy_per_atom', 'FAIL')
            params = result.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A', 'N/A'))

            # Format GA parameters
            if isinstance(params, tuple) and len(params) == 5:
                cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
                sc_str = "x".join(map(str, params[3]))
                ph_fac = params[4] if len(params) > 4 else 1.0
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str}), SC:{sc_str}, PhFactor:{ph_fac}"
            else:
                params_str = str(params)

            # Soft mode details (placeholder - would need actual mode info)
            soft_mode_details = "P:0.00THz@[0, 0, 0], S:0.00THz@[0, 0, 0]"

            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<20} {str(result.get('main_iteration', 'N/A')):<5} {str(result.get('ga_generation', 'N/A')):<5} {str(result.get('sample', 'N/A')):<8} {params_str:<120} {soft_mode_details:<80}\n")

        f.write("=" * 280 + "\n\n")

        # Unique Structures Summary
        f.write(f"Unique Structures Summary (energy tolerance: {energy_tolerance:.1e} eV/atom):\n")
        f.write("-" * 150 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'Energy Difference (meV/atom)':<30} {'GA Parameters':<80}\n")
        f.write("-" * 150 + "\n")

        for i, unique_struct in enumerate(unique_structures):
            energy_diff = (unique_struct['energy_per_atom'] - unique_structures[0]['energy_per_atom']) * 1000  # Convert to meV

            # Format parameters for unique structure
            params = unique_struct.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A', 'N/A'))
            if isinstance(params, tuple) and len(params) >= 3:
                cell_vec = params[2]
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_vec[0]:.3f}, {cell_vec[1]:.3f}, {cell_vec[2]:.3f}...)"
            else:
                params_str = str(params)

            f.write(f"{i+1:<6} {unique_struct['energy_per_atom']:.6f}{'':>14} {energy_diff:.2f}{'':>26} {params_str:<80}\n")

        f.write("-" * 150 + "\n\n")

        # Final Top Structures Selected for Analysis
        f.write(f"Final Top {len(final_top_structures)} Structures Selected for Analysis:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'GA Parameters':<80}\n")
        f.write("-" * 120 + "\n")

        for i, struct in enumerate(final_top_structures):
            params = struct.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A', 'N/A'))
            if isinstance(params, tuple) and len(params) >= 3:
                cell_vec = params[2]
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_vec[0]:.3f}, {cell_vec[1]:.3f}, {cell_vec[2]:.3f}...)"
            else:
                params_str = str(params)

            f.write(f"{i+1:<6} {struct['energy_per_atom']:.6f}{'':>14} {params_str:<80}\n")

        f.write("-" * 120 + "\n\n")

        # Method Description
        f.write("Method: Genetic Algorithm with Soft Mode Optimization\n")
        f.write("  - Evolutionary optimization of displacement parameters\n")
        f.write("  - Population-based search with mutation and crossover\n")
        f.write("  - Multi-iteration refinement with phonon guidance\n")
        f.write("  - Automatic parameter space exploration\n")
        f.write(f"Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")

    print(f"Comprehensive GA summary saved to: {overall_summary_filepath}")

    # Also create the simple overall_relaxation_summary.txt for backward compatibility
    simple_summary_filepath = os.path.join(base_output_dir, "overall_relaxation_summary.txt")
    with open(simple_summary_filepath, 'w') as f:
        f.write(f"--- Overall Relaxation Summary (All Iterations) ---\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'Iter':<6} {'Sample':<8} {'GA Params':<80}\n")
        f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*6:<6} {'-'*8:<8} {'-'*80:<80} \n")

        for result in sorted_overall_results:
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result.get('energy_per_atom', 'FAIL')
            params = result.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A', 'N/A')) # Updated to 5 elements
            if isinstance(params, tuple) and len(params) == 5:
                cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
                sc_str = "x".join(map(str, params[3])) # e.g., (2,2,1) -> "2x2x1"
                ph_fac = params[4] if len(params) > 4 else 1.0  # Default to 1.0 if not provided
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str}), SC:{sc_str}, PhFactor:{ph_fac}"
            else:
                params_str = str(params)
            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('main_iteration', 'N/A')):<10} {str(result.get('ga_generation', 'N/A')):<8} {str(result.get('sample', 'N/A')):<8} {params_str:<80}\n")

    print(f"Simple relaxation summary saved to: {simple_summary_filepath}")

    # Print the comprehensive summary to screen
    with open(overall_summary_filepath, 'r') as f:
        print("\n" + f.read())


def run_traditional_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, softest_modes_info_list, max_iterations,
                               soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return):
    """
    Runs an iterative workflow to find low-energy structures using a traditional grid search
    for displacement scales and cell transformations, and then performs a final phonon analysis
    on the best candidates found across all iterations.
    """
    print("\n--- Running Soft Mode Iterative Optimization (Traditional Grid Search) ---")

    # --- 1. Initial Setup ---
    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)
    if calculator is None:
        print("Failed to initialize calculator. Exiting.")
        return
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(base_output_dir, 'traditional')
    
    # Try to load existing checkpoint
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    checkpoint_state = None
    resume_from_checkpoint = False
    
    if checkpoint_data:
        checkpoint_full, checkpoint_state = checkpoint_data
        resume_from_checkpoint = True
        print("\n" + "="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        print(f"Iteration: {checkpoint_state.get('iteration', 1)}")
        print(f"Sample Index: {checkpoint_state.get('sample_index', 0)}")
        print(f"Completed Samples: {checkpoint_state.get('total_samples_completed', 0)}")
        print("="*80 + "\n")
    
    threshold_in_current_units = negative_phonon_threshold_thz
    if args.units == "cm-1":
        threshold_in_current_units *= 33.35641
    elif args.units == "eV":
        threshold_in_current_units *= 4.135667696e-3

    # Determine supercell variants based on the q-point of the softest mode
    # If no soft mode info, default to (2,2,2)
    if 'coordinate' in softest_modes_info_list[0]:
        q_point_for_supercell = softest_modes_info_list[0]['coordinate']
        supercell_variants = [estimate_commensurate_supercell_size(q_point_for_supercell)]
        print(f"Using commensurate supercell size {supercell_variants[0]} based on softest mode q-point {q_point_for_supercell}.")
    else:
        supercell_variants = [(2,2,2)] # Default to primitive if no q-point info
        print("No q-point information for softest mode. Defaulting to (2,2,2) supercell.")

    # Restore state from checkpoint if available
    if resume_from_checkpoint:
        all_iterations_results = checkpoint_full.get('results', [])
        if checkpoint_full.get('current_primitive_atoms'):
            current_primitive_atoms = checkpoint_full['current_primitive_atoms']
        if checkpoint_full.get('current_softest_modes'):
            softest_modes_info_list = checkpoint_full['current_softest_modes']
        print(f"Restored {len(all_iterations_results)} previous results from checkpoint")
    else:
        all_iterations_results = [] # To store results for all iterations

    # --- 2. Main Iterative Loop (Traditional Grid Search) ---
    start_iteration = 1
    if resume_from_checkpoint:
        start_iteration = checkpoint_state.get('iteration', 1)
        print(f"Resuming from iteration {start_iteration}")
    
    for iteration_idx in range(start_iteration, max_iterations + 1):
        print(f"\n### Starting Traditional Iteration {iteration_idx} ###")
        
        if len(softest_modes_info_list) == 0:
            print(f"No softest mode information to guide iteration {iteration_idx}. Stopping.")
            break

        iteration_results = [] # To store results for this specific iteration

        # Iterate through predefined displacement scales, cell scale factors, AND mode2_ratio_scales  
        sample_counter = 0  
        for disp_scale in soft_mode_displacement_scales:  
            for cell_scale in cell_scale_factors:  
                for ratio_mode2_to_mode1 in mode2_ratio_scales: # New loop for the second mode ratio  
                    sample_counter += 1  

                    if resume_from_checkpoint:
                        current_state_check = {
                            'iteration': iteration_idx,
                            'sample_index': sample_counter
                        }
                        if should_skip_sample(checkpoint_state, current_state_check, 'traditional'):
                            print(f"\n  ⏭ Skipping sample {sample_counter} (already completed in checkpoint)")
                            continue
                    # Convert simple cell_scale to a 6-element vector (only a,b,c scaled equally, angles unchanged)  
                    cell_transformation_vector = (cell_scale, cell_scale, cell_scale, 0.0, 0.0, 0.0)  
  
                    sample_output_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}", f"sample_{sample_counter}")  
                    os.makedirs(sample_output_dir, exist_ok=True)  
  
                    print(f"\n  Generating and relaxing structure for Traditional sample {sample_counter} (Iter {iteration_idx}):")  
                    print(f"    Mode1 Scale: {disp_scale:.3f}, Mode2 Ratio: {ratio_mode2_to_mode1:.3f}, Cell Transform: {cell_transformation_vector}")  
  
                    # Generate displaced supercells
                    generated_cif_paths = generate_displaced_supercells(
                        current_primitive_atoms.copy(),
                        softest_modes_info_list,
                        disp_scale,
                        ratio_mode2_to_mode1, # Now using the looped ratio
                        supercell_variants, # Use the q-point commensurate supercell
                        sample_output_dir,
                        iteration_idx,
                        original_prefix,
                        cell_transformation_vector,
                        use_phase_factor=True,
                        mutation_data=None,
                        max_atoms_per_supercell=getattr(args, 'max_atoms_per_supercell', None)
                    )
  
                    if not generated_cif_paths:  
                        print(f"    No CIFs generated for sample {sample_counter}. Skipping relaxation.")  
                        iteration_results.append({  
                            'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),  
                            'energy_per_atom': None,  
                            'relaxed_atoms': None,  
                            'original_file': None,  
                            'iteration': iteration_idx,  
                            'sample': sample_counter  
                        })  
                        continue  
  
                    unique_supercell_folders = list(set(os.path.dirname(fpath) for fpath in generated_cif_paths))
                    sample_relaxation_results = []
                    for folder in unique_supercell_folders:
                        folder_relaxation_results = relax_structures_in_folder(
                            folder, calculator, args.engine, args.fmax,
                            relaxation_patience=getattr(args, 'relaxation_patience', 5),
                            volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
                        )
                        sample_relaxation_results.extend(folder_relaxation_results)
  
                    if not sample_relaxation_results:  
                        print(f"    No structures successfully relaxed for sample {sample_counter}.")  
                        iteration_results.append({  
                            'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),  
                            'energy_per_atom': None,  
                            'relaxed_atoms': None,  
                            'original_file': None,  
                            'iteration': iteration_idx,  
                            'sample': sample_counter  
                        })  
                        continue  
  
                    lowest_energy_sample = find_lowest_energy_structures(sample_relaxation_results, num_to_select=1)  
                    if lowest_energy_sample:  
                        best_result_for_sample = lowest_energy_sample[0]  
                        iteration_results.append({  
                            'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),  
                            'energy_per_atom': best_result_for_sample['energy_per_atom'],  
                            'relaxed_atoms': best_result_for_sample['relaxed_atoms'],  
                            'original_file': best_result_for_sample['original_file'],  
                            'num_atoms': best_result_for_sample.get('num_atoms', 'N/A'),  
                            'international_symbol': best_result_for_sample.get('international_symbol', 'N/A'),  
                            'crystal_system': best_result_for_sample.get('crystal_system', 'N/A'),  
                            'iteration': iteration_idx,  
                            'sample': sample_counter  
                        })  
                        print(f"    Sample {sample_counter} relaxed. Lowest energy: {best_result_for_sample['energy_per_atom']:.6f} eV/atom")
                        
                        # Save checkpoint after each successful sample
                        try:
                            checkpoint_manager.save_checkpoint_traditional(
                                iteration=iteration_idx,
                                sample_index=sample_counter,
                                all_iterations_results=all_iterations_results + iteration_results,
                                current_primitive_atoms=current_primitive_atoms,
                                current_softest_modes=softest_modes_info_list
                            )
                            checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
                        except Exception as e:
                            print(f"    ⚠ Warning: Failed to save checkpoint: {e}")
                    else:  
                        print(f"    Could not find lowest energy for sample {sample_counter}.")  
                        iteration_results.append({  
                            'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),  
                            'energy_per_atom': None,  
                            'relaxed_atoms': None,  
                            'original_file': None,  
                            'iteration': iteration_idx,  
                            'sample': sample_counter  
                        })

        valid_iteration_results = [r for r in iteration_results if r['energy_per_atom'] is not None]
        all_iterations_results.extend(valid_iteration_results)

        if not valid_iteration_results:
            print(f"No valid structures found in iteration {iteration_idx}. Stopping Traditional optimization.")
            break

        iteration_summary_filepath = os.path.join(base_output_dir, f"iter_{iteration_idx}", "relaxation_summary_iter.txt")
        with open(iteration_summary_filepath, 'w') as f:
            f.write(f"--- Relaxation Summary for Iteration {iteration_idx} (Traditional) ---\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'Iter':<6} {'Sample':<8} {'Params':<50}\n")
            f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*6:<6} {'-'*8:<8} {'-'*50:<50}\n")

            sorted_iter_results = sorted(valid_iteration_results, key=lambda x: x['energy_per_atom'])

            for result in sorted_iter_results:
                num_atoms = result.get('num_atoms', 'N/A')
                international_symbol = result.get('international_symbol', 'N/A')
                crystal_system = result.get('crystal_system', 'N/A')
                energy = result.get('energy_per_atom', 'FAIL')
                params = result.get('params', ('N/A', 'N/A', 'N/A'))
                cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str})" if isinstance(params, tuple) else str(params)

                energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
                f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('iteration', 'N/A')):<6} {str(result.get('sample', 'N/A')):<8} {params_str:<50} \n")
        print(f"Iteration {iteration_idx} relaxation summary saved to: {iteration_summary_filepath}")

        # --- Prepare for Next Iteration: Find the soft mode of the overall best structure found so far ---
        valid_overall_results_so_far = [
            r for r in all_iterations_results
            if isinstance(r, dict) and 'energy_per_atom' in r and r['energy_per_atom'] is not None
        ]
        if not valid_overall_results_so_far:
            print("No valid structures found across all iterations to guide next step. Stopping.")
            break

        sorted_overall_best_so_far = sorted(valid_overall_results_so_far, key=lambda x: x['energy_per_atom'])
        best_candidate_for_next_iter = sorted_overall_best_so_far[0]

        try:
            relaxed_supercell = best_candidate_for_next_iter['relaxed_atoms']
            pmg_structure = AseAtomsAdaptor.get_structure(relaxed_supercell)
            
            sga = SpacegroupAnalyzer(pmg_structure)  
            primitive_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(sga.get_primitive_standard_structure())

            check_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}_guidance_phonon_check")
            os.makedirs(check_dir, exist_ok=True)
            print(f"\nPerforming guidance phonon check on best structure from Iteration {iteration_idx} to find next soft modes...")
            
            primitive_cif_path = os.path.join(check_dir, f"primitive_for_guidance_iter_{iteration_idx}.cif")  
            write(primitive_cif_path, primitive_atoms_for_next_iter)  
            print(f"Saved primitive structure for guidance phonon check to: {primitive_cif_path}")

            conventional_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(sga.get_conventional_standard_structure())  
            conventional_cif_path = os.path.join(check_dir, f"conventional_for_guidance_{iteration_idx}.cif")  
            write(conventional_cif_path, conventional_atoms_for_next_iter)  
            print(f"Saved conventional structure for guidance phonon check to: {conventional_cif_path}")

            next_softest_modes_info_list, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, check_dir,
                prefix=f"guidance_check_iter_{iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            current_primitive_atoms = primitive_atoms_for_next_iter.copy()
            softest_modes_info_list = next_softest_modes_info_list

            # Update supercell variants based on the new guiding mode's q-point
            if softest_modes_info_list and 'coordinate' in softest_modes_info_list[0]:
                q_point_for_supercell = softest_modes_info_list[0]['coordinate']
                supercell_variants = [estimate_commensurate_supercell_size(q_point_for_supercell)]
                print(f"Updated commensurate supercell size to {supercell_variants[0]} based on new softest mode q-point {q_point_for_supercell}.")
            else:
                supercell_variants = [(2,2,2)] # Default to primitive if no q-point info
                print("No q-point information for new softest mode. Defaulting to (2,2,2) supercell for next iteration.")

            if not softest_modes_info_list:
                print(f"No soft modes found in best structure from iteration {iteration_idx}. Ending Traditional optimization.")
                break
        except Exception as e:
            print(f"Could not prepare for next iteration due to an error during phonon analysis: {e}")
            import traceback
            traceback.print_exc()
            break

    # --- 3. Final Phonon Analysis on Overall Best Structures ---
    print("\n\n--- All Traditional iterations complete. ---")
    print("--- Analyzing overall best structures for final phonon properties. ---")

    if not all_iterations_results:
        print("No structures were successfully relaxed to perform final analysis on.")
        return

    valid_overall_results = [
        r for r in all_iterations_results
        if isinstance(r, dict) and 'energy_per_atom' in r and r['energy_per_atom'] is not None
    ]

    if not valid_overall_results:
        print("No valid structures with energy information were found across all iterations.")
        return

    sorted_overall_best = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])
    final_top_structures = sorted_overall_best[:num_top_structures_to_analyze]

    print("\nAll valid iteration results (sorted by energy):")
    for idx, result in enumerate(sorted_overall_best, 1):
        original_file = os.path.basename(result.get('original_file', 'unknown'))
        energy = result['energy_per_atom']
        params = result.get('params', ('N/A', 'N/A', 'N/A'))
        cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
        print(f"  {idx}. Iter {result.get('iteration', 'N/A')}, Sample {result.get('sample', 'N/A')}: {original_file} (Disp1: {params[0]:.3f}, R21: {params[1]:.3f}, Cell:({cell_transform_str})): {energy:.6f} eV/atom")


    print(f"\nSelected the top {len(final_top_structures)} structures from all iterations for final analysis:")
    # Get all unique energies from ALL structures with tolerance  
    all_structures = sorted_overall_best  
    unique_structures = []  
    energy_tolerance = 5e-4  # 0.0005 eV/atom tolerance  
    
    for result in all_structures:  
        energy = result['energy_per_atom']  
        is_unique = True  
        
        # Check if this energy is unique within tolerance  
        for existing_result in unique_structures:  
            if abs(energy - existing_result['energy_per_atom']) < energy_tolerance:  
                is_unique = False  
                break  
        
        if is_unique:  
            unique_structures.append(result)  
    
    # Now separate: first N are "top", rest are "unique"  
    final_top_structures = unique_structures[:num_top_structures_to_analyze]  
    unique_remaining_structures = unique_structures[num_top_structures_to_analyze:]
    final_structure_dir = os.path.join(base_output_dir, f"final_structures")
    os.makedirs(final_structure_dir, exist_ok=True)

    # Process top structures
    for i, result in enumerate(final_top_structures):
        _save_final_structure(result, final_structure_dir, i+1, "top", original_prefix)

    # Process unique remaining structures  
    for i, result in enumerate(unique_remaining_structures):
        _save_final_structure(result, final_structure_dir, i+1, "unique", original_prefix)


    # Run phonon analysis on top structures  
    for i, result in enumerate(final_top_structures):  
        top_structure_relaxed_supercell = result['relaxed_atoms']  
        energy_per_atom = result['energy_per_atom']  
        energy_str = f"{abs(energy_per_atom):.4f}".replace('.', 'p')  
        if energy_per_atom < 0:  
            energy_str = 'm' + energy_str  
        else:  
            energy_str = 'p' + energy_str  
        
        final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_top_{i+1}_energy_{energy_str}")  
        print(f"\n--- Running Final Phonon Analysis on Top Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")  
        try:  
            pmg_structure = AseAtomsAdaptor.get_structure(top_structure_relaxed_supercell)  
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())  
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structure_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
        except Exception as e:  
            print(f"  Error during final phonon analysis for top structure {i+1}: {e}")  
            import traceback  
            traceback.print_exc()  
    
    # Run phonon analysis on unique structures (only if there are any)  
    if unique_remaining_structures:  
        print(f"\n--- Running Final Phonon Analysis on {len(unique_remaining_structures)} Additional Unique Structures ---")  
        for i, result in enumerate(unique_remaining_structures):  
            unique_structure_relaxed_supercell = result['relaxed_atoms']  
            energy_per_atom = result['energy_per_atom']  
            energy_str = f"{abs(energy_per_atom):.4f}".replace('.', 'p')  
            if energy_per_atom < 0:  
                energy_str = 'm' + energy_str  
            else:  
                energy_str = 'p' + energy_str  
            
            final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_unique_{i+1}_energy_{energy_str}")  
            print(f"\n--- Running Final Phonon Analysis on Unique Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")  
            try:  
                pmg_structure = AseAtomsAdaptor.get_structure(unique_structure_relaxed_supercell)  
                primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())  
                _, _, _, _ = run_single_phonon_analysis(
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                    args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",
                    phonon_path_npoints=phonon_path_npoints,
                    phonon_dos_grid=phonon_dos_grid,
                    traj_kT=default_traj_kT,
                    num_modes_to_return=num_modes_to_return,
                    final_structures_dir=final_structure_dir,
                    negative_phonon_threshold=negative_phonon_threshold_thz,
                    save_yaml=args.save_yaml
                )
            except Exception as e:  
                print(f"  Error during final phonon analysis for unique structure {i+1}: {e}")  
                import traceback  
                traceback.print_exc()  
    else:  
        print(f"\nNo additional unique structures found beyond the top {len(final_top_structures)} structures.")

    print("\n--- Soft Mode Iterative Optimization Complete ---")
    overall_summary_filepath = os.path.join(base_output_dir, "overall_relaxation_summary.txt")

    print("\n--- Creating Overall Relaxation Summary (All Iterations) ---")
    with open(overall_summary_filepath, 'w') as f:
        f.write(f"--- Overall Relaxation Summary (All Iterations) ---\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'Iter':<6} {'Sample':<8} {'Params':<50}\n")
        f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*6:<6} {'-'*8:<8} {'-'*50:<50} \n")

        sorted_overall_results = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])

        for result in sorted_overall_results:
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result.get('energy_per_atom', 'FAIL')
            params = result.get('params', ('N/A', 'N/A', 'N/A'))
            cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
            params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str})" if isinstance(params, tuple) else str(params)

            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('iteration', 'N/A')):<6} {str(result.get('sample', 'N/A')):<8} {params_str:<50}\n")
    print(f"Overall relaxation summary saved to: {overall_summary_filepath}")

    with open(overall_summary_filepath, 'r') as f:
        print("\n" + f.read())


def identify_all_soft_modes_from_phonon_analysis(softest_modes_info_list, tracked_k_points_data, negative_phonon_threshold_thz):
    """
    Identifies all soft modes from phonon analysis results.

    Args:
        softest_modes_info_list (list): List of softest modes from run_single_phonon_analysis
        tracked_k_points_data (dict): Comprehensive tracking data containing soft_modes
        negative_phonon_threshold_thz (float): Threshold for soft mode detection

    Returns:
        list: All soft modes below the threshold, sorted by frequency (most negative first)
    """
    all_soft_modes = []

    # Use tracked_k_points_data if available (more comprehensive)
    if tracked_k_points_data and 'soft_modes' in tracked_k_points_data:
        all_soft_modes = tracked_k_points_data['soft_modes'].copy()
    else:
        # Fallback to softest_modes_info_list
        for mode_info in softest_modes_info_list:
            if mode_info.get('frequency', 0) < negative_phonon_threshold_thz:
                all_soft_modes.append(mode_info)

    # Sort by frequency (most negative first)
    all_soft_modes.sort(key=lambda x: x.get('frequency', 0))

    print(f"Identified {len(all_soft_modes)} soft modes below threshold {negative_phonon_threshold_thz:.4f} THz")
    for i, mode in enumerate(all_soft_modes):
        print(f"  Mode {i+1}: {mode.get('label', 'unknown')} at {mode.get('frequency', 0):.4f} THz")

    return all_soft_modes


def generate_soft_mode_pairings(all_soft_modes):
    """
    Generates unique pairings of the softest mode with each other soft mode.
    Avoids duplicate pairings by checking mode properties.

    Args:
        all_soft_modes (list): List of all soft modes, sorted by frequency (most negative first)

    Returns:
        list: List of tuples (softest_mode, other_mode) for each unique pairing
    """
    if len(all_soft_modes) < 2:
        print("Less than 2 soft modes found. No pairings possible.")
        return []

    softest_mode = all_soft_modes[0]  # Most negative frequency
    other_modes = all_soft_modes[1:]  # All other soft modes

    pairings = []
    seen_pairings = set()  # Track unique pairings to avoid duplicates

    for other_mode in other_modes:
        # Create a unique identifier for this pairing based on mode properties
        pairing_key = (
            tuple(softest_mode.get('coordinate', [])),
            softest_mode.get('band_index', -1),
            softest_mode.get('label', ''),
            tuple(other_mode.get('coordinate', [])),
            other_mode.get('band_index', -1),
            other_mode.get('label', '')
        )

        # Only add if we haven't seen this pairing before
        if pairing_key not in seen_pairings:
            pairings.append((softest_mode, other_mode))
            seen_pairings.add(pairing_key)
        else:
            print(f"  Skipping duplicate pairing: {softest_mode.get('label', 'unknown')} + {other_mode.get('label', 'unknown')}")

    print(f"Generated {len(pairings)} unique pairings with softest mode ({softest_mode.get('label', 'unknown')} at {softest_mode.get('frequency', 0):.4f} THz)")
    for i, (mode1, mode2) in enumerate(pairings):
        print(f"  Pairing {i+1}: {mode1.get('label', 'unknown')} + {mode2.get('label', 'unknown')}")

    return pairings


def run_traditional_all_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, softest_modes_info_list, max_iterations,
                               soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return):
    """
    Runs an iterative workflow to find low-energy structures using the traditional_all method.

    For each iteration:
    1. Identifies the softest (most negative) phonon mode across all special k-points
    2. Combines the softest mode with ALL other soft modes one at a time
    3. Creates separate subfolders for each combination
    4. Runs phonon analysis on the lowest energy structure from each subfolder
    5. Continues until no more soft modes remain

    Args:
        args: Command line arguments
        base_output_dir (str): Base output directory
        initial_atoms_for_soft_mode_analysis (ase.Atoms): Initial structure
        softest_modes_info_list (list): Initial soft modes information
        max_iterations (int): Maximum number of iterations
        soft_mode_displacement_scales (list): Displacement scale factors
        cell_scale_factors (list): Cell scaling factors
        mode2_ratio_scales (list): Ratios for second mode
        num_top_structures_to_analyze (int): Number of top structures to analyze
        negative_phonon_threshold_thz (float): Threshold for soft mode detection
        phonon_path_npoints (int): Number of points for phonon path
        phonon_dos_grid (tuple): Grid for DOS calculation
        default_traj_kT (float): Temperature for trajectory generation
        num_modes_to_return (int): Number of modes to return
    """
    print("\n--- Running Soft Mode Iterative Optimization (Traditional All Method) ---")

    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)
    if calculator is None:
        print("Failed to initialize calculator. Exiting.")
        return

    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]
    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_modes_info_list = softest_modes_info_list.copy()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(base_output_dir, 'traditional_all')
    
    # Try to load existing checkpoint
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    checkpoint_state = None
    resume_from_checkpoint = False
    
    if checkpoint_data:
        checkpoint_full, checkpoint_state = checkpoint_data
        resume_from_checkpoint = True
        print("\n" + "="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        print(f"Iteration: {checkpoint_state.get('iteration', 1)}")
        print(f"Pairing Index: {checkpoint_state.get('pairing_index', 0)}")
        print(f"Config Index: {checkpoint_state.get('config_index', 0)}")
        print(f"Sample Index: {checkpoint_state.get('sample_index', 0)}")
        print("="*80 + "\n")
    
    # Restore state from checkpoint if available
    if resume_from_checkpoint:
        all_iterations_results = checkpoint_full.get('results', [])
        if checkpoint_full.get('current_primitive_atoms'):
            current_primitive_atoms = checkpoint_full['current_primitive_atoms']
        if checkpoint_full.get('current_softest_modes'):
            current_softest_modes_info_list = checkpoint_full['current_softest_modes']
        print(f"Restored {len(all_iterations_results)} previous results from checkpoint")
    else:
        all_iterations_results = []

    for iteration_idx in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"TRADITIONAL_ALL ITERATION {iteration_idx}")
        print(f"{'='*60}")

        iteration_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}")
        os.makedirs(iteration_dir, exist_ok=True)
        # Run phonon analysis to get current soft modes
        check_dir = os.path.join(iteration_dir, "guidance_check")
        os.makedirs(check_dir, exist_ok=True)

        try:
            next_softest_modes_info_list, _, _, tracked_k_points_data = run_single_phonon_analysis(
                current_primitive_atoms.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, check_dir,
                prefix=f"guidance_check_main_iter_{iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )

            # Identify all soft modes
            all_soft_modes = identify_all_soft_modes_from_phonon_analysis(
                next_softest_modes_info_list, tracked_k_points_data, negative_phonon_threshold_thz
            )

            if len(all_soft_modes) < 2:
                print(f"Less than 2 soft modes found in iteration {iteration_idx}. Stopping iterations.")
                break

            # Generate pairings of softest mode with all other soft modes
            mode_pairings = generate_soft_mode_pairings(all_soft_modes)

            if not mode_pairings:
                print(f"No mode pairings possible in iteration {iteration_idx}. Stopping iterations.")
                break

            print(f"Processing {len(mode_pairings)} mode pairings in iteration {iteration_idx}")

        except Exception as e:
            print(f"Error during phonon analysis in iteration {iteration_idx}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Process each mode pairing in separate subfolders with mode swapping
        pairing_results = []
        all_structures_info = []  # Track all structures (successful and failed)

        for pairing_idx, (softest_mode, other_mode) in enumerate(mode_pairings):
            if resume_from_checkpoint and checkpoint_state.get('iteration', 0) == iteration_idx:
                if pairing_idx < checkpoint_state.get('pairing_index', 0):
                    continue
            try:
                print(f"\n--- Processing Pairing {pairing_idx+1}/{len(mode_pairings)}: {softest_mode.get('label', 'unknown')} + {other_mode.get('label', 'unknown')} ---")

                # Validate that both modes have the required keys
                if 'raw_displacements' not in softest_mode:
                    print(f"  Error: Softest mode ({softest_mode.get('label', 'unknown')}) missing raw_displacements. Skipping pairing.")
                    continue
                if 'raw_displacements' not in other_mode:
                    print(f"  Error: Other mode ({other_mode.get('label', 'unknown')}) missing raw_displacements. Skipping pairing.")
                    continue

                # Define both original and swapped configurations with band indices
                softest_label = softest_mode.get('label', 'unknown')
                softest_band_idx = softest_mode.get('band_index', 'unknown')
                other_label = other_mode.get('label', 'unknown')
                other_band_idx = other_mode.get('band_index', 'unknown')

                # Create configurations - avoid duplicates when modes are identical
                configurations = [
                    {
                        'name': f"pairing_{pairing_idx+1}_{softest_label}_idx{softest_band_idx}_{other_label}_idx{other_band_idx}",
                        'modes_info': [softest_mode, other_mode],
                        'primary_mode': softest_mode,
                        'secondary_mode': other_mode,
                        'description': f"Original: {softest_label} (idx{softest_band_idx}) as primary, {other_label} (idx{other_band_idx}) as secondary"
                    }
                ]

                # Only add swapped configuration if it would be different from the original
                # Check if modes are different by comparing their key properties
                modes_are_different = (
                    softest_mode.get('coordinate') != other_mode.get('coordinate') or
                    softest_mode.get('band_index') != other_mode.get('band_index') or
                    softest_mode.get('label') != other_mode.get('label')
                )

                if modes_are_different:
                    configurations.append({
                        'name': f"pairing_{pairing_idx+1}_{other_label}_idx{other_band_idx}_{softest_label}_idx{softest_band_idx}",
                        'modes_info': [other_mode, softest_mode],
                        'primary_mode': other_mode,
                        'secondary_mode': softest_mode,
                        'description': f"Swapped: {other_label} (idx{other_band_idx}) as primary, {softest_label} (idx{softest_band_idx}) as secondary"
                    })
                else:
                    print(f"    Skipping swapped configuration for pairing {pairing_idx+1} - modes are identical")

            except Exception as e:
                print(f"  Error setting up pairing {pairing_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Continuing with next pairing...")
                continue

            # Process both original and swapped configurations
            for config_idx, config in enumerate(configurations):
                if resume_from_checkpoint and checkpoint_state.get('iteration', 0) == iteration_idx and \
                  checkpoint_state.get('pairing_index', 0) == pairing_idx:
                    if config_idx < checkpoint_state.get('config_index', 0):
                        continue
                try:
                    config_name = config['name']
                    config_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}", config_name)
                    os.makedirs(config_dir, exist_ok=True)

                    print(f"\n  Configuration {config_idx+1}/2: {config['description']}")

                    # Validate that both modes have raw_displacements
                    modes_valid = True
                    for mode_idx, mode in enumerate(config['modes_info']):
                        if 'raw_displacements' not in mode:
                            print(f"    Error: Mode {mode_idx+1} ({mode.get('label', 'unknown')}) missing raw_displacements. Skipping configuration.")
                            modes_valid = False
                            break

                    if not modes_valid:
                        print(f"    Skipping configuration {config_idx+1} due to missing raw_displacements")
                        # Track failed configuration
                        all_structures_info.append({
                            'config_name': config_name,
                            'configuration': config['description'],
                            'status': 'FAILED',
                            'failure_reason': 'Missing raw_displacements',
                            'energy_per_atom': None,
                            'sample_id': None,
                            'iteration': iteration_idx,
                            'pairing': config_name
                        })
                        continue

                    # Determine supercell variants based on the primary mode's q-point
                    primary_mode = config['primary_mode']
                    if 'coordinate' in primary_mode:
                        q_point_for_supercell = primary_mode['coordinate']
                        supercell_variants = [estimate_commensurate_supercell_size(q_point_for_supercell)]
                        print(f"    Using commensurate supercell size {supercell_variants[0]} based on primary mode q-point {q_point_for_supercell}.")
                    else:
                        supercell_variants = [(2,2,2)]  # Default
                        print("    No q-point information for primary mode. Defaulting to (2,2,2) supercell.")

                    # Generate and relax structures for this configuration using traditional grid search
                    sample_counter = 0
                    config_sample_results = []

                except Exception as e:
                    print(f"    Error setting up configuration {config_idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"    Continuing with next configuration...")
                    continue

                for disp_scale in soft_mode_displacement_scales:
                    for cell_scale in cell_scale_factors:
                        for ratio_mode2_to_mode1 in mode2_ratio_scales:
                            sample_counter += 1
                            if resume_from_checkpoint:
                                current_state_check = {
                                    'iteration': iteration_idx,
                                    'pairing_index': pairing_idx,
                                    'config_index': config_idx,
                                    'sample_index': sample_counter
                                }
                                if should_skip_sample(checkpoint_state, current_state_check, 'traditional_all'):
                                    print(f"      ⏭ Skipping sample {sample_counter} (already completed)")
                                    continue
                            cell_transformation_vector = (cell_scale, cell_scale, cell_scale, 0.0, 0.0, 0.0)

                            sample_output_dir = os.path.join(config_dir, f"sample_{sample_counter}")
                            os.makedirs(sample_output_dir, exist_ok=True)

                            print(f"      Sample {sample_counter}: Mode1 Scale: {disp_scale:.3f}, Mode2 Ratio: {ratio_mode2_to_mode1:.3f}, Cell Transform: {cell_transformation_vector}")

                            try:
                                # Validate that modes have required raw_displacements
                                modes_valid = True
                                for mode_idx, mode in enumerate(config['modes_info']):
                                    if 'raw_displacements' not in mode:
                                        print(f"        Error: Mode {mode_idx+1} ({mode.get('label', 'unknown')}) missing raw_displacements. Skipping sample {sample_counter}.")
                                        modes_valid = False
                                        break

                                if not modes_valid:
                                    continue

                                # Generate displaced supercells using the configuration's mode order
                                # Use different prefix for swapped configurations to prevent overwriting
                                config_prefix = original_prefix
                                if "Swapped:" in config['description']:
                                    config_prefix = f"{original_prefix}_swapped"

                                generated_cif_paths = generate_displaced_supercells(
                                    current_primitive_atoms.copy(),
                                    config['modes_info'],  # Use the configuration's mode order
                                    disp_scale,
                                    ratio_mode2_to_mode1,
                                    supercell_variants,
                                    sample_output_dir,
                                    iteration_idx,
                                    config_prefix,
                                    cell_transformation_vector,
                                    use_phase_factor=True,
                                    mutation_data=None,
                                    max_atoms_per_supercell=getattr(args, 'max_atoms_per_supercell', None)
                                )

                                if not generated_cif_paths:
                                    print(f"        No structures generated for sample {sample_counter}. Skipping.")
                                    continue

                                # Relax the generated structures
                                relaxed_structures_info = relax_structures_in_folder(
                                    sample_output_dir, calculator, args.engine, args.fmax,
                                    relaxation_patience=getattr(args, 'relaxation_patience', 5),
                                    volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
                                )

                                if relaxed_structures_info:
                                    # Find the lowest energy structure
                                    lowest_energy_info = min(relaxed_structures_info, key=lambda x: x['energy_per_atom'])

                                    # Extract detailed soft mode information
                                    modes_info = config['modes_info']
                                    primary_mode = modes_info[0] if len(modes_info) > 0 else {}
                                    secondary_mode = modes_info[1] if len(modes_info) > 1 else {}

                                    sample_result = {
                                        'sample_id': sample_counter,
                                        'energy_per_atom': lowest_energy_info['energy_per_atom'],
                                        'relaxed_atoms': lowest_energy_info['relaxed_atoms'],
                                        'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),
                                        'iteration': iteration_idx,
                                        'pairing': config_name,
                                        'configuration': config['description'],
                                        # Add structural information
                                        'num_atoms': lowest_energy_info.get('num_atoms', 'N/A'),
                                        'international_symbol': lowest_energy_info.get('international_symbol', 'N/A'),
                                        'crystal_system': lowest_energy_info.get('crystal_system', 'N/A'),
                                        # Add soft mode information
                                        'primary_mode': {
                                            'label': primary_mode.get('label', 'Unknown'),
                                            'k_point': primary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': primary_mode.get('frequency', 0.0),
                                            'band_index': primary_mode.get('band_index', 0)
                                        },
                                        'secondary_mode': {
                                            'label': secondary_mode.get('label', 'Unknown'),
                                            'k_point': secondary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': secondary_mode.get('frequency', 0.0),
                                            'band_index': secondary_mode.get('band_index', 0)
                                        }
                                    }
                                    config_sample_results.append(sample_result)

                                    # Track successful structure with detailed soft mode information
                                    all_structures_info.append({
                                        'config_name': config_name,
                                        'configuration': config['description'],
                                        'status': 'SUCCESS',
                                        'failure_reason': None,
                                        'energy_per_atom': lowest_energy_info['energy_per_atom'],
                                        'sample_id': sample_counter,
                                        'iteration': iteration_idx,
                                        'pairing': config_name,
                                        'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),
                                        'relaxed_atoms': lowest_energy_info['relaxed_atoms'],
                                        # Structural information
                                        'num_atoms': lowest_energy_info.get('num_atoms', 'N/A'),
                                        'international_symbol': lowest_energy_info.get('international_symbol', 'N/A'),
                                        'crystal_system': lowest_energy_info.get('crystal_system', 'N/A'),
                                        # Detailed soft mode information
                                        'primary_mode': {
                                            'label': primary_mode.get('label', 'Unknown'),
                                            'k_point': primary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': primary_mode.get('frequency', 0.0),
                                            'band_index': primary_mode.get('band_index', 0)
                                        },
                                        'secondary_mode': {
                                            'label': secondary_mode.get('label', 'Unknown'),
                                            'k_point': secondary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': secondary_mode.get('frequency', 0.0),
                                            'band_index': secondary_mode.get('band_index', 0)
                                        }
                                    })

                                    print(f"        Sample {sample_counter} completed. Energy: {lowest_energy_info['energy_per_atom']:.6f} eV/atom")
                                    
                                    # Save checkpoint after each successful sample
                                    try:
                                        checkpoint_manager.save_checkpoint_traditional_all(
                                            iteration=iteration_idx,
                                            pairing_index=pairing_idx,
                                            config_index=config_idx,
                                            sample_index=sample_counter,
                                            all_iterations_results=all_iterations_results + [sample_result],
                                            current_primitive_atoms=current_primitive_atoms,
                                            current_softest_modes=current_softest_modes_info_list
                                        )
                                        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
                                    except Exception as e:
                                        print(f"        ⚠ Warning: Failed to save checkpoint: {e}")
                                else:
                                    print(f"        Sample {sample_counter} failed during relaxation.")
                                    # Extract detailed soft mode information for failed structure
                                    modes_info = config['modes_info']
                                    primary_mode = modes_info[0] if len(modes_info) > 0 else {}
                                    secondary_mode = modes_info[1] if len(modes_info) > 1 else {}

                                    # Track failed structure
                                    all_structures_info.append({
                                        'config_name': config_name,
                                        'configuration': config['description'],
                                        'status': 'FAILED',
                                        'failure_reason': 'Relaxation failed',
                                        'energy_per_atom': None,
                                        'sample_id': sample_counter,
                                        'iteration': iteration_idx,
                                        'pairing': config_name,
                                        'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),
                                        'relaxed_atoms': None,
                                        # Structural information (N/A for failed structures)
                                        'num_atoms': 'N/A',
                                        'international_symbol': 'N/A',
                                        'crystal_system': 'N/A',
                                        # Detailed soft mode information
                                        'primary_mode': {
                                            'label': primary_mode.get('label', 'Unknown'),
                                            'k_point': primary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': primary_mode.get('frequency', 0.0),
                                            'band_index': primary_mode.get('band_index', 0)
                                        },
                                        'secondary_mode': {
                                            'label': secondary_mode.get('label', 'Unknown'),
                                            'k_point': secondary_mode.get('coordinate', [0, 0, 0]),
                                            'frequency': secondary_mode.get('frequency', 0.0),
                                            'band_index': secondary_mode.get('band_index', 0)
                                        }
                                    })

                            except Exception as e:
                                print(f"        Error processing sample {sample_counter}: {e}")
                                import traceback
                                traceback.print_exc()
                                print(f"        Continuing with next sample...")
                                # Extract detailed soft mode information for failed structure
                                modes_info = config['modes_info']
                                primary_mode = modes_info[0] if len(modes_info) > 0 else {}
                                secondary_mode = modes_info[1] if len(modes_info) > 1 else {}

                                # Track failed structure
                                all_structures_info.append({
                                    'config_name': config_name,
                                    'configuration': config['description'],
                                    'status': 'FAILED',
                                    'failure_reason': f'Exception: {str(e)}',
                                    'energy_per_atom': None,
                                    'sample_id': sample_counter,
                                    'iteration': iteration_idx,
                                    'pairing': config_name,
                                    'params': (disp_scale, ratio_mode2_to_mode1, cell_transformation_vector),
                                    'relaxed_atoms': None,
                                    # Structural information (N/A for failed structures)
                                    'num_atoms': 'N/A',
                                    'international_symbol': 'N/A',
                                    'crystal_system': 'N/A',
                                    # Detailed soft mode information
                                    'primary_mode': {
                                        'label': primary_mode.get('label', 'Unknown'),
                                        'k_point': primary_mode.get('coordinate', [0, 0, 0]),
                                        'frequency': primary_mode.get('frequency', 0.0),
                                        'band_index': primary_mode.get('band_index', 0)
                                    },
                                    'secondary_mode': {
                                        'label': secondary_mode.get('label', 'Unknown'),
                                        'k_point': secondary_mode.get('coordinate', [0, 0, 0]),
                                        'frequency': secondary_mode.get('frequency', 0.0),
                                        'band_index': secondary_mode.get('band_index', 0)
                                    }
                                })
                                continue

                # Store results for this configuration
                if config_sample_results:
                    # Find the best structure from this configuration
                    best_config_result = min(config_sample_results, key=lambda x: x['energy_per_atom'])
                    pairing_results.append(best_config_result)
                    print(f"    Best result for {config['description']}: {best_config_result['energy_per_atom']:.6f} eV/atom")
                else:
                    print(f"    No valid results for {config['description']}")

        # Add all pairing results to overall results
        all_iterations_results.extend(pairing_results)

        if not pairing_results:
            print(f"No valid results from any pairing configuration in iteration {iteration_idx}. Stopping iterations.")
            break

        # Find the overall best structure from this iteration
        best_iteration_result = min(pairing_results, key=lambda x: x['energy_per_atom'])
        print(f"\nBest structure from iteration {iteration_idx}: {best_iteration_result['energy_per_atom']:.6f} eV/atom")
        print(f"  Configuration: {best_iteration_result.get('configuration', 'Unknown')}")

        # Update current primitive atoms for next iteration
        current_primitive_atoms = best_iteration_result['relaxed_atoms'].copy()

        # Calculate total configurations processed (2 per pairing due to mode swapping)
        total_configurations = len(mode_pairings) * 2
        print(f"Iteration {iteration_idx} completed. Processed {len(mode_pairings)} pairings ({total_configurations} configurations with mode swapping) with {len(pairing_results)} successful results.")

        # Create detailed iteration summary
        iteration_summary_path = os.path.join(iteration_dir, f"iteration_{iteration_idx}_summary.txt")
        with open(iteration_summary_path, 'w') as f:
            f.write(f"--- Iteration {iteration_idx} Summary ---\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total mode pairings processed: {len(mode_pairings)}\n")
            f.write(f"Total configurations (with swapping): {total_configurations}\n")
            f.write(f"Total structures attempted: {len(all_structures_info)}\n")
            f.write(f"Successful results: {len(pairing_results)}\n")
            f.write(f"Failed results: {len(all_structures_info) - len(pairing_results)}\n")
            f.write(f"Success rate: {len(pairing_results)/len(all_structures_info)*100:.1f}%\n\n")

            f.write("Settings used:\n")
            f.write(f"  Displacement scales: {soft_mode_displacement_scales}\n")
            f.write(f"  Mode2 ratios: {mode2_ratio_scales}\n")
            f.write(f"  Cell scale factors: {cell_scale_factors}\n")
            f.write(f"  Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n\n")

            if all_structures_info:
                f.write("All Structures (Successful and Failed) with Structural Information:\n")
                f.write("=" * 280 + "\n")
                f.write(f"{'Sample ID':<10} {'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<20} {'Status':<10} {'Configuration Description':<80} {'Failure Reason':<30}\n")
                f.write("=" * 280 + "\n")

                # Sort structures by sample_id for better organization
                sorted_structures = sorted(all_structures_info, key=lambda x: (x.get('sample_id', 0)))

                for structure in sorted_structures:
                    sample_id = structure.get('sample_id', 'N/A')
                    num_atoms = structure.get('num_atoms', 'N/A')
                    international_symbol = structure.get('international_symbol', 'N/A')
                    crystal_system = structure.get('crystal_system', 'N/A')
                    config_desc = structure.get('configuration', 'Unknown')
                    energy = structure.get('energy_per_atom')
                    status = structure.get('status', 'Unknown')
                    failure_reason = structure.get('failure_reason', '')

                    energy_str = f"{energy:.6f}" if energy is not None else "N/A"
                    failure_str = failure_reason if failure_reason else "N/A"

                    f.write(f"{str(sample_id):<10} {str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<20} {status:<10} {config_desc:<80} {failure_str:<30}\n")

                f.write("=" * 280 + "\n")

                # Add successful structures summary with structural information (matching GA format)
                if pairing_results:
                    f.write("\nSuccessful Structures (sorted by energy):\n")
                    f.write("=" * 320 + "\n")
                    f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<25} {'Sample ID':<10} {'Configuration Description':<80} {'Parameters':<50}\n")
                    f.write("=" * 320 + "\n")

                    # Sort successful results by energy
                    sorted_successful = sorted(pairing_results, key=lambda x: x['energy_per_atom'])

                    for result in sorted_successful:
                        num_atoms = result.get('num_atoms', 'N/A')
                        international_symbol = result.get('international_symbol', 'N/A')
                        crystal_system = result.get('crystal_system', 'N/A')
                        energy = result['energy_per_atom']
                        sample_id = result.get('sample_id', 'N/A')
                        config_desc = result.get('configuration', 'Unknown')
                        params = result.get('params', ('N/A', 'N/A', ('N/A',)))

                        # Format parameters
                        if isinstance(params, tuple) and len(params) >= 3:
                            cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
                            params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str})"
                        else:
                            params_str = str(params)

                        energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
                        f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(sample_id):<10} {config_desc:<80} {params_str:<50}\n")

                    f.write("=" * 320 + "\n")

                    f.write(f"\nBest result: {best_iteration_result['energy_per_atom']:.6f} eV/atom\n")
                    f.write(f"Best configuration: {best_iteration_result.get('configuration', 'Unknown')}\n")
            else:
                f.write("No structures processed in this iteration.\n")

        print(f"Iteration {iteration_idx} summary saved to: {iteration_summary_path}")

    print(f"\n--- Traditional All Optimization Complete after {iteration_idx} iterations ---")

    # Final analysis similar to GA algorithm
    if not all_iterations_results:
        print("No valid results found across all iterations. Exiting.")
        return

    print(f"\n--- Final Analysis: Processing {len(all_iterations_results)} total structures ---")

    # Identify unique structures based on energy tolerance
    print(f"\nIdentifying unique structures from all {len(all_iterations_results)} calculated structures...")
    unique_structures = []
    energy_tolerance = 5e-4  # 0.5 meV/atom tolerance

    # Sort all structures by energy
    sorted_all_structures = sorted(all_iterations_results, key=lambda x: x['energy_per_atom'])

    for result in sorted_all_structures:
        energy = result['energy_per_atom']
        is_unique = True

        # Check if this energy is unique within tolerance
        for existing_result in unique_structures:
            if abs(energy - existing_result['energy_per_atom']) < energy_tolerance:
                is_unique = False
                break

        if is_unique:
            unique_structures.append(result)

    print(f"Found {len(unique_structures)} unique structures (energy tolerance: {energy_tolerance:.1e} eV/atom)")

    # Find top structures across all iterations
    sorted_results = sorted(all_iterations_results, key=lambda x: x['energy_per_atom'])
    final_top_structures = []

    for i, result in enumerate(sorted_results[:num_top_structures_to_analyze]):
        final_top_structures.append((
            result['energy_per_atom'],
            result['relaxed_atoms'],
            result['params']
        ))

    # Identify additional unique structures beyond the top 5
    additional_unique_structures = []
    for result in unique_structures:
        # Check if this unique structure is not already in the top 5
        is_in_top_5 = False
        for top_energy, _, _ in final_top_structures:
            if abs(result['energy_per_atom'] - top_energy) < energy_tolerance:
                is_in_top_5 = True
                break

        if not is_in_top_5:
            additional_unique_structures.append(result)

    # Create final structures directory and save structures
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    # Create unique structures directory
    unique_structures_dir = os.path.join(base_output_dir, "unique_structures")
    os.makedirs(unique_structures_dir, exist_ok=True)

    print(f"\n--- Saving and Analyzing Final Top {len(final_top_structures)} Structures ---")

    # Save unique structures as CIF files
    print(f"\n--- Saving {len(unique_structures)} Unique Structures ---")
    for i, result in enumerate(unique_structures):
        try:
            unique_cif_filename = f"unique_structure_{i+1:03d}_energy_{result['energy_per_atom']:.6f}_eV_per_atom.cif"
            unique_cif_path = os.path.join(unique_structures_dir, unique_cif_filename)
            write(unique_cif_path, result['relaxed_atoms'])
            print(f"  Saved unique structure {i+1}: {unique_cif_filename}")
        except Exception as e:
            print(f"  Error saving unique structure {i+1}: {e}")

    # Create individual phonon analysis folders for top 5 structures (GA-style)
    print(f"\n--- Creating Individual Phonon Analysis Folders for Top {len(final_top_structures)} Structures ---")
    for i, (energy_per_atom, relaxed_atoms, params) in enumerate(final_top_structures):
        try:
            # Create energy string for folder name (similar to GA mode)
            energy_str = f"m{abs(energy_per_atom):.4f}".replace('.', 'p')

            final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_top_{i+1}_energy_{energy_str}")
            print(f"\n--- Running Final Phonon Analysis on Top Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")

            # Convert to primitive structure for phonon analysis (like GA mode)
            pmg_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

            # Run phonon analysis (same as GA mode)
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structures_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            print(f"  Completed phonon analysis for top structure {i+1}")

        except Exception as e:
            print(f"  Error creating phonon analysis for top structure {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Create individual phonon analysis folders for additional unique structures (GA-style)
    if additional_unique_structures:
        print(f"\n--- Creating Individual Phonon Analysis Folders for {len(additional_unique_structures)} Additional Unique Structures ---")
        for i, result in enumerate(additional_unique_structures):
            try:
                energy_per_atom = result['energy_per_atom']
                relaxed_atoms = result['relaxed_atoms']

                # Create energy string for folder name (similar to GA mode)
                energy_str = f"m{abs(energy_per_atom):.4f}".replace('.', 'p')

                final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_unique_{i+1}_energy_{energy_str}")
                print(f"\n--- Running Final Phonon Analysis on Unique Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")

                # Convert to primitive structure for phonon analysis (like GA mode)
                pmg_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
                primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

                # Run phonon analysis (same as GA mode)
                _, _, _, _ = run_single_phonon_analysis(
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                    args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",
                    phonon_path_npoints=phonon_path_npoints,
                    phonon_dos_grid=phonon_dos_grid,
                    traj_kT=default_traj_kT,
                    num_modes_to_return=num_modes_to_return,
                    final_structures_dir=final_structures_dir,
                    negative_phonon_threshold=negative_phonon_threshold_thz,
                    save_yaml=args.save_yaml
                )
                print(f"  Completed phonon analysis for unique structure {i+1}")

            except Exception as e:
                print(f"  Error creating phonon analysis for unique structure {i+1}: {e}")
                import traceback
                traceback.print_exc()

    for i, (energy, atoms, params) in enumerate(final_top_structures):
        try:
            # Save structure files using the existing helper function
            result_dict = {
                'relaxed_atoms': atoms,
                'energy_per_atom': energy,
                'iteration': 'final',
                'sample': i+1
            }
            _save_final_structure(result_dict, final_structures_dir, i+1, "final", original_prefix)

            # Run final phonon analysis
            final_structure_dir = os.path.join(final_structures_dir, f"final_structure_{i+1}")
            os.makedirs(final_structure_dir, exist_ok=True)

            print(f"  Running final phonon analysis for structure {i+1} (Energy: {energy:.6f} eV/atom)")

            run_single_phonon_analysis(
                atoms.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, final_structure_dir,
                prefix=f"final_structure_{i+1}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structure_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )

        except Exception as e:
            print(f"  Error during final analysis for structure {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Create overall summary
    print("\n--- Creating Overall Summary ---")
    overall_summary_filepath = os.path.join(base_output_dir, "overall_traditional_all_summary.txt")

    # Collect all structures from all iterations (including failed ones)
    all_structures_across_iterations = []
    successful_structures = 0
    failed_structures = 0

    # Count successful and failed structures
    for result in all_iterations_results:
        successful_structures += 1

    # Note: Failed structures are tracked per iteration but not aggregated globally yet
    # This will be enhanced when we have access to all_structures_info from each iteration

    with open(overall_summary_filepath, 'w') as f:
        f.write("--- Traditional All Soft Mode Optimization Summary (with Mode Swapping) ---\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total iterations completed: {iteration_idx}\n")
        f.write(f"Total successful structures: {len(all_iterations_results)}\n")
        f.write(f"Total unique structures identified: {len(unique_structures)}\n")
        f.write(f"Final structures analyzed: {len(final_top_structures)}\n\n")

        f.write("Optimization Settings:\n")
        f.write(f"  Engine: {args.engine}\n")
        f.write(f"  Units: {args.units}\n")
        f.write(f"  Supercell dimensions: {args.supercell_dims}\n")
        f.write(f"  Delta: {args.delta}\n")
        f.write(f"  Fmax: {args.fmax}\n")
        f.write(f"  Displacement scales: {soft_mode_displacement_scales}\n")
        f.write(f"  Mode2 ratios: {mode2_ratio_scales}\n")
        f.write(f"  Cell scale factors: {cell_scale_factors}\n")
        f.write(f"  Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")
        f.write(f"  Max iterations: {max_iterations}\n")
        f.write(f"  Top structures to analyze: {num_top_structures_to_analyze}\n\n")

        # All Successful Structures Table (with structural information and detailed soft mode info)
        f.write("All Successful Structures (sorted by energy):\n")
        f.write("=" * 320 + "\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<25} {'Iter':<5} {'Sample':<8} {'Primary Mode':<25} {'Secondary Mode':<25} {'Configuration Description':<80} {'Parameters':<50} {'Soft Mode Details':<80}\n")
        f.write("=" * 320 + "\n")

        for i, result in enumerate(sorted_results):
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result['energy_per_atom']
            iteration = result.get('iteration', 'N/A')
            sample_id = result.get('sample_id', 'N/A')
            config_desc = result.get('configuration', 'Unknown')  # Full description, no truncation
            params = result.get('params', ('N/A', 'N/A', ('N/A',)))

            # Extract soft mode information
            primary_mode = result.get('primary_mode', {})
            secondary_mode = result.get('secondary_mode', {})

            primary_str = f"{primary_mode.get('label', 'N/A')}({primary_mode.get('band_index', 'N/A')})"
            secondary_str = f"{secondary_mode.get('label', 'N/A')}({secondary_mode.get('band_index', 'N/A')})"

            # Format parameters
            if isinstance(params, tuple) and len(params) >= 3:
                cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str})"
            else:
                params_str = str(params)

            # Format soft mode details
            primary_k = primary_mode.get('k_point', [0, 0, 0])
            secondary_k = secondary_mode.get('k_point', [0, 0, 0])
            primary_freq = primary_mode.get('frequency', 0.0)
            secondary_freq = secondary_mode.get('frequency', 0.0)

            soft_mode_details = f"P:{primary_freq:.2f}THz@{primary_k}, S:{secondary_freq:.2f}THz@{secondary_k}"

            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(iteration):<5} {str(sample_id):<8} {primary_str:<25} {secondary_str:<25} {config_desc:<80} {params_str:<50} {soft_mode_details:<80}\n")

        f.write("=" * 300 + "\n")

        # Unique Structures Summary
        f.write(f"\nUnique Structures Summary (energy tolerance: {energy_tolerance:.1e} eV/atom):\n")
        f.write("-" * 150 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'Energy Difference (meV/atom)':<30} {'Configuration Description':<90}\n")
        f.write("-" * 150 + "\n")

        for i, result in enumerate(unique_structures):
            energy = result['energy_per_atom']
            config_desc = result.get('configuration', 'Unknown')

            # Calculate energy difference from the best structure
            if i == 0:
                energy_diff_mev = 0.0
            else:
                energy_diff_mev = (energy - unique_structures[0]['energy_per_atom']) * 1000  # Convert to meV

            f.write(f"{i+1:<6} {energy:<20.6f} {energy_diff_mev:<30.2f} {config_desc:<90}\n")

        f.write("-" * 150 + "\n")

        # Final Top Structures (for backward compatibility)
        f.write(f"\nFinal Top {len(final_top_structures)} Structures Selected for Analysis:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'Configuration':<50} {'Parameters':<24}\n")
        f.write("-" * 100 + "\n")

        for i, (energy, atoms, params) in enumerate(final_top_structures):
            # Try to get configuration info from the corresponding result
            config_info = "N/A"
            if i < len(sorted_results):
                config_info = sorted_results[i].get('configuration', 'N/A')[:48]  # Less truncation

            cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])
            params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}"
            f.write(f"{i+1:<6} {energy:<20.6f} {config_info:<50} {params_str:<24}\n")

        f.write("\n" + "-" * 100 + "\n")
        f.write("Method: Traditional All with Mode Swapping\n")
        f.write("  - Softest mode combined with all other soft modes\n")
        f.write("  - Each pairing explored in both original and swapped configurations\n")
        f.write("  - Original: softest as primary, other as secondary\n")
        f.write("  - Swapped: other as primary, softest as secondary\n")
        f.write(f"Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")

    print(f"Overall summary saved to: {overall_summary_filepath}")
    print(f"Final structures saved to: {final_structures_dir}")
    print(f"Unique structures saved to: {unique_structures_dir}")
    print(f"Total successful structures: {len(all_iterations_results)}")
    print(f"Total unique structures found: {len(unique_structures)} (energy tolerance: {energy_tolerance:.1e} eV/atom)")
    print(f"Final top structures analyzed: {len(final_top_structures)}")
    print(f"Total iterations completed: {iteration_idx}")
    print(f"Mode swapping enabled: Each pairing explored in both original and swapped configurations")
    print(f"Enhanced summaries with complete configuration descriptions and failure tracking enabled")

    # Display summary
    with open(overall_summary_filepath, 'r') as f:
        print("\n" + f.read())


def run_random_structure_search(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_modes_info_list, max_iterations,
                               soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, num_new_points_per_iteration,
                               random_displacement_bounds, random_cell_perturbation, random_seed, ga_cell_scale_bounds, ga_cell_angle_bounds):
    """
    Runs an iterative workflow to find low-energy structures using random atomic displacements,
    and then performs a final phonon analysis on the best candidates found across all iterations.

    Args:
        args: Command line arguments object
        base_output_dir (str): Base output directory for all results
        initial_atoms_for_soft_mode_analysis (ase.atoms.Atoms): Initial structure for optimization
        initial_softest_modes_info_list (list): Initial soft modes info (not used in random method but kept for compatibility)
        max_iterations (int): Maximum number of optimization iterations
        soft_mode_displacement_scales (list): Not used in random method but kept for compatibility
        cell_scale_factors (list): Not used in random method but kept for compatibility
        mode2_ratio_scales (list): Not used in random method but kept for compatibility
        num_top_structures_to_analyze (int): Number of top structures to analyze in final step
        negative_phonon_threshold_thz (float): Threshold for identifying soft modes
        phonon_path_npoints (int): Number of points for phonon path calculations
        phonon_dos_grid (tuple): Grid for phonon DOS calculations
        default_traj_kT (float): Temperature for trajectory calculations
        num_modes_to_return (int): Number of modes to return from phonon analysis
        num_new_points_per_iteration (int): Number of random structures to generate per iteration
        random_displacement_bounds (list): Base [min, max] displacement bounds in Angstroms (varied per sample)
        random_cell_perturbation (bool): Whether to apply random cell perturbations
        random_seed (int): Random seed for reproducibility
        ga_cell_scale_bounds (tuple): Bounds for random cell scale factor generation
        ga_cell_angle_bounds (tuple): Bounds for random cell angle change generation
    """
    print("\n--- Running Random Structure Search Optimization ---")
    print(f"Random displacement bounds: {random_displacement_bounds} Å")
    print(f"Random cell perturbation: {random_cell_perturbation}")
    print(f"Random seed: {random_seed}")
    print(f"Structures per iteration: {num_new_points_per_iteration}")
    print(f"Maximum iterations: {max_iterations}")

    # Initialize calculator
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)
    if calculator is None:
        print("Failed to initialize calculator. Exiting.")
        return

    # Generate comprehensive supercell variants - all permutations up to max size
    def generate_comprehensive_supercell_variants(max_size=2):
        """Generate all possible supercell combinations up to max_size."""
        variants = []
        for nx in range(1, max_size + 1):
            for ny in range(1, max_size + 1):
                for nz in range(1, max_size + 1):
                    variants.append((nx, ny, nz))
        return variants

    # Initialize supercell variants (will be generated only in first iteration)
    supercell_variants = None

    # Track all results across iterations
    all_iterations_results = []
    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_best_supercell = None  # Track supercell dimensions from best structure

    # Get original prefix for file naming
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(base_output_dir, 'opt_random')
    
    # Try to load existing checkpoint
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    checkpoint_state = None
    resume_from_checkpoint = False
    
    if checkpoint_data:
        checkpoint_full, checkpoint_state = checkpoint_data
        resume_from_checkpoint = True
        print("\n" + "="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        print(f"Iteration: {checkpoint_state.get('iteration', 1)}")
        print(f"Sample Index: {checkpoint_state.get('sample_index', 0)}")
        print(f"Completed Samples: {checkpoint_state.get('total_samples_completed', 0)}")
        print("="*80 + "\n")
    
    # Restore state from checkpoint if available
    if resume_from_checkpoint:
        all_iterations_results = checkpoint_full.get('results', [])
        current_best_supercell = checkpoint_full.get('current_best_supercell')
        if checkpoint_full.get('current_primitive_atoms'):
            current_primitive_atoms = checkpoint_full['current_primitive_atoms']
        print(f"Restored {len(all_iterations_results)} previous results from checkpoint")
        print(f"Restored best supercell: {current_best_supercell}")

    # Main iteration loop
    start_iteration = 1
    if resume_from_checkpoint:
        start_iteration = checkpoint_state.get('iteration', 1)
        print(f"Resuming from iteration {start_iteration}")
    
    for iteration_idx in range(start_iteration, max_iterations + 1):
        # Generate supercell variants only in the first iteration (iter_0 equivalent)
        if iteration_idx == 1:
            supercell_variants = generate_comprehensive_supercell_variants(max_size=2)
            print(f"FIRST ITERATION: Generated comprehensive supercell variants: {supercell_variants}")
        else:
            print(f"SUBSEQUENT ITERATION {iteration_idx}: Skipping supercell variant generation, using existing cell structure")

        # Determine phase and number of samples
        is_exploration_phase = (iteration_idx == 1)
        if is_exploration_phase:
            samples_this_iteration = 2 * num_new_points_per_iteration  # Double samples for exploration
            phase_name = "EXPLORATION"
        else:
            samples_this_iteration = num_new_points_per_iteration  # Standard samples for refinement
            phase_name = "REFINEMENT"

        print(f"\n=== Random Search Iteration {iteration_idx}/{max_iterations} - {phase_name} PHASE ===")
        print(f"Generating {samples_this_iteration} structures ({'2x standard' if is_exploration_phase else 'standard'} for {phase_name.lower()})")

        # Create iteration directory
        iteration_dir = os.path.join(base_output_dir, f"main_iter_{iteration_idx}")
        os.makedirs(iteration_dir, exist_ok=True)

        iteration_results = []

        # Generate random structures for this iteration
        for sample_idx in range(samples_this_iteration):
            sample_counter = sample_idx + 1

            if resume_from_checkpoint:
                current_state_check = {
                    'iteration': iteration_idx,
                    'sample_index': sample_counter
                }
                if should_skip_sample(checkpoint_state, current_state_check, 'opt_random'):
                    print(f"\n  ⏭ Skipping sample {sample_counter} (already completed in checkpoint)")
                    continue

            print(f"\n  Generating random structure {sample_counter}/{samples_this_iteration} (Iteration {iteration_idx} - {phase_name})")

            # Create sample directory
            sample_output_dir = os.path.join(iteration_dir, f"sample_{sample_counter}")
            os.makedirs(sample_output_dir, exist_ok=True)

            # Set random seed for this sample
            import random as py_random
            import numpy as np
            if random_seed is not None:
                py_random.seed(random_seed + iteration_idx * 1000 + sample_idx)
                np.random.seed(random_seed + iteration_idx * 1000 + sample_idx)

            # Supercell selection strategy based on iteration
            if iteration_idx == 1:
                # FIRST ITERATION: Use different supercell variants for diversity
                # Ensure each sample gets a different supercell variant when possible
                variant_index = sample_idx % len(supercell_variants)
                selected_supercell = supercell_variants[variant_index]
                print(f"    FIRST ITERATION: Selected supercell variant {variant_index + 1}/{len(supercell_variants)}: {selected_supercell}")
            else:
                # SUBSEQUENT ITERATIONS: Skip supercell variant generation, use existing cell structure
                # Use the best supercell from previous iteration or default to (1,1,1) for primitive cell
                if current_best_supercell is not None:
                    selected_supercell = current_best_supercell
                    print(f"    SUBSEQUENT ITERATION: Using best supercell from previous iteration: {selected_supercell}")
                else:
                    # Fallback to primitive cell if no best supercell is available
                    selected_supercell = (1, 1, 1)
                    print(f"    SUBSEQUENT ITERATION: Using primitive cell (1,1,1) as fallback")

            # Generate variable maximum displacement bounds for this sample
            min_disp = random_displacement_bounds[0]  # Keep minimum fixed
            base_max_disp = random_displacement_bounds[1]  # Base maximum
            # Vary maximum displacement by ±25% of the base value
            max_disp_variation = base_max_disp * 0.25
            sample_max_disp = py_random.uniform(base_max_disp - max_disp_variation,
                                              base_max_disp + max_disp_variation)
            sample_displacement_bounds = [min_disp, sample_max_disp]

            print(f"    Sample displacement bounds: {sample_displacement_bounds}")

            # Generate random cell transformation vector using GA bounds
            cell_scales = [py_random.uniform(*ga_cell_scale_bounds) for _ in range(3)]
            cell_angles = [py_random.uniform(*ga_cell_angle_bounds) for _ in range(3)]
            cell_transformation_vector = tuple(cell_scales + cell_angles)

            print(f"    Cell transformation: {cell_transformation_vector}")

            # Generate random displaced structures
            try:
                generated_cif_paths = generate_random_displaced_structures(
                    current_primitive_atoms.copy(),
                    sample_displacement_bounds,  # Use sample-specific bounds
                    [selected_supercell],  # Pass single supercell as list
                    sample_output_dir,
                    iteration_idx,
                    original_prefix,
                    cell_transformation_vector,
                    random_cell_perturbation,
                    random_seed + iteration_idx * 1000 + sample_idx if random_seed is not None else None,
                    max_atoms_per_supercell=getattr(args, 'max_atoms_per_supercell', None)
                )

                if not generated_cif_paths:
                    print(f"        No structures generated for sample {sample_counter}. Skipping.")
                    continue

                # Relax the generated structures
                relaxed_structures_info = relax_structures_in_folder(
                    sample_output_dir, calculator, args.engine, args.fmax,
                    relaxation_patience=getattr(args, 'relaxation_patience', 5),
                    volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
                )

                if not relaxed_structures_info:
                    print(f"        No structures successfully relaxed for sample {sample_counter}. Skipping.")
                    continue

                # Find the lowest energy structure from this sample
                lowest_energy_sample = find_lowest_energy_structures(relaxed_structures_info, num_to_select=1)
                if lowest_energy_sample:
                    best_result_for_sample = lowest_energy_sample[0]
                    iteration_results.append({
                        'params': {
                            'displacement_bounds': sample_displacement_bounds,
                            'cell_transformation_vector': cell_transformation_vector,
                            'cell_perturbation': random_cell_perturbation
                        },
                        'energy_per_atom': best_result_for_sample['energy_per_atom'],
                        'relaxed_atoms': best_result_for_sample['relaxed_atoms'],
                        'original_file': best_result_for_sample['original_file'],
                        'num_atoms': best_result_for_sample.get('num_atoms', 'N/A'),
                        'international_symbol': best_result_for_sample.get('international_symbol', 'N/A'),
                        'crystal_system': best_result_for_sample.get('crystal_system', 'N/A'),
                        'iteration': iteration_idx,
                        'sample': sample_counter,
                        'selected_supercell': selected_supercell
                    })

                    print(f"        Sample {sample_counter} completed: {best_result_for_sample['energy_per_atom']:.6f} eV/atom")
                    
                    # Save checkpoint after each successful sample
                    try:
                        checkpoint_manager.save_checkpoint_opt_random(
                            iteration=iteration_idx,
                            sample_index=sample_counter,
                            all_iterations_results=all_iterations_results + iteration_results,
                            current_primitive_atoms=current_primitive_atoms,
                            current_best_supercell=selected_supercell
                        )
                        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
                    except Exception as e:
                        print(f"        ⚠ Warning: Failed to save checkpoint: {e}")

            except Exception as e:
                print(f"        Error processing sample {sample_counter}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Add iteration results to overall results
        all_iterations_results.extend(iteration_results)

        if not iteration_results:
            print(f"No valid results from iteration {iteration_idx}. Continuing to next iteration.")
            continue

        # Find best structure from this iteration to use as base for next iteration
        best_iteration_result = min(iteration_results, key=lambda x: x['energy_per_atom'])
        print(f"\nBest structure from iteration {iteration_idx}: {best_iteration_result['energy_per_atom']:.6f} eV/atom")
        print(f"Best structure supercell: {best_iteration_result['selected_supercell']}")

        # Update current primitive atoms and supercell for next iteration
        current_primitive_atoms = best_iteration_result['relaxed_atoms'].copy()
        current_best_supercell = best_iteration_result['selected_supercell']

        print(f"Iteration {iteration_idx} completed. Generated {len(iteration_results)} successful structures.")

        # Save comprehensive iteration summary (matching traditional_all format)
        iteration_summary_path = os.path.join(iteration_dir, f"iteration_{iteration_idx}_summary.txt")
        with open(iteration_summary_path, 'w') as f:
            f.write(f"--- Random Search Iteration {iteration_idx} Summary ({phase_name} PHASE) ---\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Phase: {phase_name} ({'Exploration with diverse supercells' if is_exploration_phase else 'Refinement with fixed supercell'})\n")
            f.write(f"Total structures attempted: {samples_this_iteration}\n")
            f.write(f"Successful results: {len(iteration_results)}\n")
            f.write(f"Failed results: {samples_this_iteration - len(iteration_results)}\n")
            f.write(f"Success rate: {len(iteration_results)/samples_this_iteration*100:.1f}%\n\n")

            f.write("Settings used:\n")
            f.write(f"  Base random displacement bounds: {random_displacement_bounds} Å (varied per sample)\n")
            f.write(f"  Random cell perturbation: {random_cell_perturbation}\n")
            f.write(f"  Random seed: {random_seed}\n")
            if is_exploration_phase:
                f.write(f"  Supercell strategy: EXPLORATION - cycling through variants {supercell_variants}\n")
            else:
                f.write(f"  Supercell strategy: REFINEMENT - fixed supercell {current_best_supercell}\n")
            f.write(f"  GA cell scale bounds: {ga_cell_scale_bounds}\n")
            f.write(f"  GA cell angle bounds: {ga_cell_angle_bounds}\n")
            f.write(f"  Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n\n")

            # Add comprehensive structure information table (matching traditional_all format)
            if iteration_results:
                f.write("All Structures (Successful and Failed) with Structural Information:\n")
                f.write("=" * 280 + "\n")
                f.write(f"{'Sample ID':<10} {'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<20} {'Status':<10} {'Configuration Description':<80} {'Failure Reason':<30}\n")
                f.write("=" * 280 + "\n")

                # Create comprehensive structure list including failed ones
                all_structures_info = []

                # Add successful structures
                for result in iteration_results:
                    all_structures_info.append({
                        'sample_id': result.get('sample', 'N/A'),
                        'num_atoms': result.get('num_atoms', 'N/A'),
                        'international_symbol': result.get('international_symbol', 'N/A'),
                        'crystal_system': result.get('crystal_system', 'N/A'),
                        'energy_per_atom': result['energy_per_atom'],
                        'status': 'SUCCESS',
                        'configuration': f"Random displacement (supercell: {result.get('selected_supercell', 'N/A')})",
                        'failure_reason': ''
                    })

                # Add failed structures (estimated based on missing samples)
                successful_samples = {result.get('sample', 0) for result in iteration_results}
                for sample_id in range(1, samples_this_iteration + 1):
                    if sample_id not in successful_samples:
                        all_structures_info.append({
                            'sample_id': sample_id,
                            'num_atoms': 'N/A',
                            'international_symbol': 'N/A',
                            'crystal_system': 'N/A',
                            'energy_per_atom': None,
                            'status': 'FAILED',
                            'configuration': 'Random displacement (failed)',
                            'failure_reason': 'Relaxation or calculation failed'
                        })

                # Sort structures by sample_id for better organization
                sorted_structures = sorted(all_structures_info, key=lambda x: x.get('sample_id', 0))

                for structure in sorted_structures:
                    sample_id = structure.get('sample_id', 'N/A')
                    num_atoms = structure.get('num_atoms', 'N/A')
                    international_symbol = structure.get('international_symbol', 'N/A')
                    crystal_system = structure.get('crystal_system', 'N/A')
                    config_desc = structure.get('configuration', 'Unknown')
                    energy = structure.get('energy_per_atom')
                    status = structure.get('status', 'Unknown')
                    failure_reason = structure.get('failure_reason', '')

                    energy_str = f"{energy:.6f}" if energy is not None else "N/A"
                    failure_str = failure_reason if failure_reason else "N/A"

                    f.write(f"{str(sample_id):<10} {str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<20} {status:<10} {config_desc:<80} {failure_str:<30}\n")

                f.write("=" * 280 + "\n")

                # Add successful structures summary with structural information (matching traditional_all format)
                f.write("\nSuccessful Structures (sorted by energy):\n")
                f.write("=" * 320 + "\n")
                f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<25} {'Sample ID':<10} {'Configuration Description':<80} {'Parameters':<50}\n")
                f.write("=" * 320 + "\n")

                # Sort successful results by energy
                sorted_results = sorted(iteration_results, key=lambda x: x['energy_per_atom'])

                for result in sorted_results:
                    num_atoms = result.get('num_atoms', 'N/A')
                    international_symbol = result.get('international_symbol', 'N/A')
                    crystal_system = result.get('crystal_system', 'N/A')
                    energy = result['energy_per_atom']
                    sample_id = result.get('sample', 'N/A')
                    supercell = result.get('selected_supercell', 'N/A')
                    config_desc = f"Random displacement (supercell: {supercell})"

                    # Format parameters for random method
                    params = result.get('params', {})
                    if isinstance(params, dict):
                        displacement_bounds = params.get('displacement_bounds', 'N/A')
                        cell_perturbation = params.get('cell_perturbation', 'N/A')
                        params_str = f"Disp:{displacement_bounds}, Cell:{cell_perturbation}"
                    else:
                        params_str = str(params)

                    energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
                    f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(sample_id):<10} {config_desc:<80} {params_str:<50}\n")

                f.write("=" * 320 + "\n")

                f.write(f"\nBest result: {best_iteration_result['energy_per_atom']:.6f} eV/atom\n")
                f.write(f"Best supercell: {best_iteration_result.get('selected_supercell', 'Unknown')}\n")
            else:
                f.write("No successful results in this iteration.\n")

        print(f"Iteration {iteration_idx} summary saved to: {iteration_summary_path}")

    print(f"\n--- Random Structure Search Complete after {max_iterations} iterations ---")

    # Final analysis (similar to GA algorithm)
    if not all_iterations_results:
        print("No valid results found across all iterations. Exiting.")
        return

    print(f"\n--- Final Analysis: Processing {len(all_iterations_results)} total structures ---")

    # Identify unique structures based on energy tolerance
    print(f"\nIdentifying unique structures from all {len(all_iterations_results)} calculated structures...")
    unique_structures = []
    energy_tolerance = 5e-4  # 0.5 meV/atom tolerance

    # Sort all structures by energy
    sorted_all_structures = sorted(all_iterations_results, key=lambda x: x['energy_per_atom'])

    for result in sorted_all_structures:
        energy = result['energy_per_atom']
        is_unique = True

        # Check if this energy is unique within tolerance
        for existing_result in unique_structures:
            if abs(energy - existing_result['energy_per_atom']) < energy_tolerance:
                is_unique = False
                break

        if is_unique:
            unique_structures.append(result)

    print(f"Found {len(unique_structures)} unique structures (energy tolerance: {energy_tolerance:.1e} eV/atom)")

    # Find top structures across all iterations
    sorted_results = sorted(all_iterations_results, key=lambda x: x['energy_per_atom'])
    final_top_structures = []

    for i, result in enumerate(sorted_results[:num_top_structures_to_analyze]):
        final_top_structures.append((
            result['energy_per_atom'],
            result['relaxed_atoms'],
            result['params']
        ))

    # Create final structures directory and save structures
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    print(f"\n--- Saving and Analyzing Final Top {len(final_top_structures)} Structures ---")

    # Identify additional unique structures beyond the top structures
    additional_unique_structures = []
    for result in unique_structures:
        # Check if this unique structure is not already in the top structures
        is_in_top = False
        for top_energy, _, _ in final_top_structures:
            if abs(result['energy_per_atom'] - top_energy) < energy_tolerance:
                is_in_top = True
                break

        if not is_in_top:
            additional_unique_structures.append(result)

    # Save final structure files BEFORE phonon analysis (so they can get frequency suffixes)
    print(f"\n--- Saving Final Top {len(final_top_structures)} Structures ---")
    for i, (energy_per_atom, relaxed_atoms, params) in enumerate(final_top_structures):
        try:
            # Save structure files using the existing helper function
            result_dict = {
                'relaxed_atoms': relaxed_atoms,
                'energy_per_atom': energy_per_atom,
                'iteration': 'final',
                'sample': i+1
            }
            _save_final_structure(result_dict, final_structures_dir, i+1, "top", original_prefix)

        except Exception as e:
            print(f"  Error saving final top structure {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Save additional unique structure files BEFORE phonon analysis
    if additional_unique_structures:
        print(f"\n--- Saving {len(additional_unique_structures)} Additional Unique Structures ---")
        for i, result in enumerate(additional_unique_structures):
            try:
                _save_final_structure(result, final_structures_dir, i+1, "unique", original_prefix)

            except Exception as e:
                print(f"  Error saving unique structure {i+1}: {e}")
                import traceback
                traceback.print_exc()

    # Create individual phonon analysis folders for top structures
    print(f"\n--- Creating Individual Phonon Analysis Folders for Top {len(final_top_structures)} Structures ---")
    for i, (energy_per_atom, relaxed_atoms, params) in enumerate(final_top_structures):
        try:
            # Create energy string for folder name
            energy_str = f"m{abs(energy_per_atom):.4f}".replace('.', 'p')

            final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_top_{i+1}_energy_{energy_str}")
            print(f"\n--- Running Final Phonon Analysis on Top Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")

            # Convert to primitive structure for phonon analysis
            pmg_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

            # Run phonon analysis
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structures_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            print(f"  Completed phonon analysis for top structure {i+1}")

        except Exception as e:
            print(f"  Error creating phonon analysis for top structure {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Create individual phonon analysis folders for additional unique structures
    if additional_unique_structures:
        print(f"\n--- Creating Individual Phonon Analysis Folders for {len(additional_unique_structures)} Additional Unique Structures ---")
        for i, result in enumerate(additional_unique_structures):
            try:
                energy_per_atom = result['energy_per_atom']
                relaxed_atoms = result['relaxed_atoms']

                # Create energy string for folder name
                energy_str = f"m{abs(energy_per_atom):.4f}".replace('.', 'p')

                final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_unique_{i+1}_energy_{energy_str}")
                print(f"\n--- Running Final Phonon Analysis on Unique Structure #{i+1} (Energy: {energy_per_atom:.6f} eV/atom) ---")

                # Convert to primitive structure for phonon analysis
                pmg_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
                primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

                # Run phonon analysis
                _, _, _, _ = run_single_phonon_analysis(
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                    args.supercell_dims, args.delta, args.fmax, final_phonon_dir,
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",
                    phonon_path_npoints=phonon_path_npoints,
                    phonon_dos_grid=phonon_dos_grid,
                    traj_kT=default_traj_kT,
                    num_modes_to_return=num_modes_to_return,
                    final_structures_dir=final_structures_dir,
                    negative_phonon_threshold=negative_phonon_threshold_thz,
                    save_yaml=args.save_yaml
                )
                print(f"  Completed phonon analysis for unique structure {i+1}")

            except Exception as e:
                print(f"  Error creating phonon analysis for unique structure {i+1}: {e}")
                import traceback
                traceback.print_exc()



    # Create overall summary
    print("\n--- Creating Overall Summary ---")
    overall_summary_filepath = os.path.join(base_output_dir, "overall_random_search_summary.txt")

    with open(overall_summary_filepath, 'w') as f:
        f.write("--- Random Structure Search Optimization Summary ---\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total iterations completed: {max_iterations}\n")
        f.write(f"Total successful structures: {len(all_iterations_results)}\n")
        f.write(f"Total unique structures identified: {len(unique_structures)}\n")
        f.write(f"Final structures analyzed: {len(final_top_structures)}\n\n")

        f.write("Optimization Settings:\n")
        f.write(f"  Engine: {args.engine}\n")
        f.write(f"  Units: {args.units}\n")
        f.write(f"  Supercell dimensions: {args.supercell_dims}\n")
        f.write(f"  Delta: {args.delta}\n")
        f.write(f"  Fmax: {args.fmax}\n")
        f.write(f"  Base random displacement bounds: {random_displacement_bounds} Å (varied per sample)\n")
        f.write(f"  Random cell perturbation: {random_cell_perturbation}\n")
        f.write(f"  Random seed: {random_seed}\n")
        f.write(f"  Structures per iteration: {num_new_points_per_iteration} (2x in iteration 1)\n")
        f.write(f"  Supercell strategy: Exploration (iter 1) + Refinement (iter 2+)\n")
        f.write(f"  Available supercell variants: {supercell_variants}\n")
        f.write(f"  GA cell scale bounds: {ga_cell_scale_bounds}\n")
        f.write(f"  GA cell angle bounds: {ga_cell_angle_bounds}\n")
        f.write(f"  Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")
        f.write(f"  Max iterations: {max_iterations}\n")
        f.write(f"  Top structures to analyze: {num_top_structures_to_analyze}\n\n")

        # All Successful Structures Table (GA-style formatting with structural information)
        f.write("All Successful Structures (sorted by energy):\n")
        f.write("=" * 250 + "\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<25} {'Iter':<5} {'Sample':<8} {'Random Parameters':<80}\n")
        f.write("=" * 250 + "\n")

        for i, result in enumerate(sorted_results):
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result['energy_per_atom']
            iteration = result.get('iteration', 'N/A')
            sample_id = result.get('sample', 'N/A')
            selected_supercell = result.get('selected_supercell', 'N/A')
            params = result.get('params', ('N/A', 'N/A', 'N/A'))

            # Extract parameters in GA style
            if isinstance(params, tuple) and len(params) >= 3:
                disp_bounds = params[0] if isinstance(params[0], list) else 'N/A'
                cell_transform = params[1] if params[1] is not None else None
                cell_pert = params[2] if len(params) > 2 else 'N/A'

                # Format displacement bounds to show actual values used
                if isinstance(disp_bounds, list) and len(disp_bounds) == 2:
                    disp_bounds_str = f"Disp:[{disp_bounds[0]:.2f}, {disp_bounds[1]:.2f}]"
                else:
                    disp_bounds_str = f"Disp:{disp_bounds}" if disp_bounds != 'N/A' else 'Disp:N/A'

                supercell_str = f"SC:{selected_supercell}" if selected_supercell != 'N/A' else 'SC:N/A'

                if cell_transform is not None and len(cell_transform) >= 6:
                    # Format cell transformation compactly
                    cell_transform_str = ", ".join([f"{val:.3f}" for val in cell_transform])
                    cell_str = f"Cell:({cell_transform_str})"
                else:
                    cell_str = "Cell:N/A"

                cell_pert_str = f"Pert:{cell_pert}"
                params_str = f"{disp_bounds_str}, {cell_str}, {supercell_str}, {cell_pert_str}"
            else:
                params_str = str(params)

            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(iteration):<5} {str(sample_id):<8} {params_str:<80}\n")

        f.write("=" * 150 + "\n")

        # Unique Structures Summary
        f.write(f"\nUnique Structures Summary (energy tolerance: {energy_tolerance:.1e} eV/atom):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'Energy Difference (meV/atom)':<30} {'Iteration':<10} {'Sample':<8}\n")
        f.write("-" * 100 + "\n")

        for i, result in enumerate(unique_structures):
            energy = result['energy_per_atom']
            iteration = result.get('iteration', 'N/A')
            sample_id = result.get('sample', 'N/A')

            # Calculate energy difference from the best structure
            if i == 0:
                energy_diff_mev = 0.0
            else:
                energy_diff_mev = (energy - unique_structures[0]['energy_per_atom']) * 1000  # Convert to meV

            f.write(f"{i+1:<6} {energy:<20.6f} {energy_diff_mev:<30.2f} {iteration:<10} {sample_id:<8}\n")

        f.write("-" * 100 + "\n")

        # Final Top Structures (GA-style formatting)
        f.write(f"\nFinal Top {len(final_top_structures)} Structures Selected for Analysis:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Rank':<6} {'Energy (eV/atom)':<20} {'Displacement Bounds':<25} {'Cell Scale Factors':<30} {'Cell Angle Changes':<30}\n")
        f.write("-" * 120 + "\n")

        for i, (energy, _, params) in enumerate(final_top_structures):
            if isinstance(params, tuple) and len(params) >= 2:
                disp_bounds = params[0] if isinstance(params[0], list) else 'N/A'
                cell_transform = params[1] if params[1] is not None else None

                # Format displacement bounds to show actual values used
                if isinstance(disp_bounds, list) and len(disp_bounds) == 2:
                    disp_bounds_str = f"[{disp_bounds[0]:.2f}, {disp_bounds[1]:.2f}]"
                else:
                    disp_bounds_str = str(disp_bounds) if disp_bounds != 'N/A' else 'N/A'

                if cell_transform is not None and len(cell_transform) >= 6:
                    cell_scales = cell_transform[:3]
                    cell_angles = cell_transform[3:6]
                    cell_scales_str = f"[{', '.join([f'{val:.3f}' for val in cell_scales])}]"
                    cell_angles_str = f"[{', '.join([f'{val:.3f}' for val in cell_angles])}]"
                else:
                    cell_scales_str = "N/A"
                    cell_angles_str = "N/A"
            else:
                disp_bounds_str = 'N/A'
                cell_scales_str = 'N/A'
                cell_angles_str = 'N/A'

            f.write(f"{i+1:<6} {energy:<20.6f} {disp_bounds_str:<25} {cell_scales_str:<30} {cell_angles_str:<30}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Method: Random Structure Search\n")
        f.write("  - Random atomic displacements within specified bounds\n")
        f.write("  - Optional random cell parameter perturbations\n")
        f.write("  - Iterative optimization using best structure as base for next iteration\n")
        f.write(f"Negative phonon threshold: {negative_phonon_threshold_thz:.4f} THz\n")

    print(f"Overall summary saved to: {overall_summary_filepath}")
    print(f"Final structures saved to: {final_structures_dir}")
    print(f"Total successful structures: {len(all_iterations_results)}")
    print(f"Total unique structures found: {len(unique_structures)} (energy tolerance: {energy_tolerance:.1e} eV/atom)")
    print(f"Final top structures analyzed: {len(final_top_structures)}")
    print(f"Additional unique structures analyzed: {len(additional_unique_structures)}")
    print(f"Total iterations completed: {max_iterations} (1 exploration + {max_iterations-1} refinement)")
    print(f"Random displacement bounds: {random_displacement_bounds} Å (varied per sample)")
    print(f"Supercell strategy: Exploration (diverse) → Refinement (fixed)")
    print(f"GA cell scale bounds: {ga_cell_scale_bounds}")
    print(f"GA cell angle bounds: {ga_cell_angle_bounds}")
    print(f"Random cell perturbation enabled: {random_cell_perturbation}")

    # Display summary
    with open(overall_summary_filepath, 'r') as f:
        print("\n" + f.read())


def run_neb_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_modes_info_list, max_iterations,
                                  soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                                  phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, neb_num_images, neb_spring_constant,
                                  neb_max_iterations, neb_force_tolerance, final_cif_path):
    """
    Runs NEB optimization between initial and final structures, followed by phonon analysis.

    This function follows the same pattern as other optimization methods in VibroML but performs
    NEB path optimization instead of soft mode displacement optimization.
    """
    print("\n--- Running NEB Soft Mode Optimization ---")

    # Load final structure
    print(f"Loading final structure from: {final_cif_path}")
    try:
        _, final_atoms = load_structure(final_cif_path)
        if final_atoms is None:
            raise ValueError(f"Could not load final structure from {final_cif_path}")
    except Exception as e:
        print(f"Error loading final structure: {e}")
        return

    # Initialize calculator
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)

    # Create output directory for NEB optimization
    neb_output_dir = os.path.join(base_output_dir, "neb_optimization")
    os.makedirs(neb_output_dir, exist_ok=True)

    # Get original prefix for naming
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

    # Mandatory structure relaxation for both initial and final structures
    print(f"\n--- Mandatory Structure Relaxation for NEB ---")
    print("Relaxing initial structure...")
    initial_relax_dir = os.path.join(neb_output_dir, "initial_structure_relaxation")
    os.makedirs(initial_relax_dir, exist_ok=True)

    relaxed_initial_atoms = relax_structure(
        initial_atoms_for_soft_mode_analysis.copy(),
        calculator,
        args.engine,
        args.fmax,
        initial_relax_dir,
        args.cif,
        relaxation_patience=getattr(args, 'relaxation_patience', 5),
        volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
    )

    if relaxed_initial_atoms is None:
        print("Error: Initial structure relaxation failed. Cannot proceed with NEB.")
        return

    print("Relaxing final structure...")
    final_relax_dir = os.path.join(neb_output_dir, "final_structure_relaxation")
    os.makedirs(final_relax_dir, exist_ok=True)

    relaxed_final_atoms = relax_structure(
        final_atoms.copy(),
        calculator,
        args.engine,
        args.fmax,
        final_relax_dir,
        final_cif_path,
        relaxation_patience=getattr(args, 'relaxation_patience', 5),
        volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
    )

    if relaxed_final_atoms is None:
        print("Error: Final structure relaxation failed. Cannot proceed with NEB.")
        return

    print("Both structures successfully relaxed. Proceeding with NEB optimization.")

    print(f"\n--- Starting NEB Optimization ---")
    print(f"Initial structure: {len(relaxed_initial_atoms)} atoms (relaxed)")
    print(f"Final structure: {len(relaxed_final_atoms)} atoms (relaxed)")
    print(f"Number of intermediate images: {neb_num_images}")
    print(f"Spring constant: {neb_spring_constant} eV/Å²")
    print(f"Force tolerance: {neb_force_tolerance} eV/Å")
    print(f"Maximum iterations: {neb_max_iterations}")

    # Run NEB optimization
    try:
        neb_results = run_neb_optimization(
            initial_atoms=relaxed_initial_atoms,
            final_atoms=relaxed_final_atoms,
            calculator=calculator,
            num_images=neb_num_images,
            spring_constant=neb_spring_constant,
            max_iterations=neb_max_iterations,
            force_tolerance=neb_force_tolerance,
            output_dir=neb_output_dir,
            prefix=original_prefix,
            climbing_start_iteration=None  # Standard NEB
        )
    except Exception as e:
        print(f"Error during NEB optimization: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n--- NEB Optimization Results ---")
    print(f"Converged: {neb_results['converged']}")
    print(f"Final max force: {neb_results['final_max_force']:.6f} eV/Å")
    print(f"Iterations: {neb_results['iterations']}")
    print(f"Optimization time: {neb_results['optimization_time']:.2f} seconds")

    # Generate enhanced NEB summary with force information
    neb_params = {
        'num_images': neb_num_images,
        'spring_constant': neb_spring_constant,
        'force_tolerance': neb_force_tolerance,
        'max_iterations': neb_max_iterations
    }

    # Save summary in both locations
    neb_summary_paths = [
        os.path.join(neb_output_dir, "neb_summary.txt"),  # NEB optimization subdirectory
        os.path.join(base_output_dir, "neb_summary.txt")  # Main output directory
    ]

    generate_enhanced_neb_summary(
        results=neb_results,
        method_name="Standard NEB",
        args=args,
        final_cif_path=final_cif_path,
        neb_params=neb_params,
        output_paths=neb_summary_paths
    )

    # Export final optimized structures to final_structures directory
    print(f"\n--- Exporting Final NEB Structures ---")
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    optimized_images = neb_results['images']
    final_energies = neb_results['final_energies']

    for i, (image, energy) in enumerate(zip(optimized_images, final_energies)):
        # Create descriptive filename with energy
        if i == 0:
            structure_type = "initial"
        elif i == len(optimized_images) - 1:
            structure_type = "final"
        else:
            structure_type = f"intermediate_{i:02d}"

        filename = f"neb_{structure_type}_img{i:02d}_energy_{energy:.4f}eV.cif"
        output_path = os.path.join(final_structures_dir, filename)

        try:
            write(output_path, image)
            print(f"  Saved {structure_type} structure: {filename}")
        except Exception as e:
            print(f"  Error saving {structure_type} structure: {e}")

    print(f"Final NEB structures exported to: {final_structures_dir}")

    # Perform phonon analysis on selected images from the optimized path (if requested)
    if args.with_phonon:
        print(f"\n--- Running Phonon Analysis on NEB Images ---")
    else:
        print(f"\n--- Skipping Phonon Analysis (--with-phonon not provided) ---")
        print("\n--- NEB Soft Mode Optimization Complete ---")
        return

    # Select images for phonon analysis (initial, final, and highest energy image)
    images_for_phonon = []
    final_energies = neb_results['final_energies']
    optimized_images = neb_results['images']

    # Always include initial and final
    images_for_phonon.append((0, "initial", optimized_images[0]))
    images_for_phonon.append((len(optimized_images)-1, "final", optimized_images[-1]))

    # Find highest energy image (transition state candidate)
    if len(final_energies) > 2:
        # Only consider intermediate images for transition state
        intermediate_energies = final_energies[1:-1]
        max_energy_idx = np.argmax(intermediate_energies) + 1  # +1 to account for skipping first image
        images_for_phonon.append((max_energy_idx, "transition_state", optimized_images[max_energy_idx]))

    # Create final structures directory
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    # Run phonon analysis on selected images
    for img_idx, img_type, atoms in images_for_phonon:
        try:
            print(f"\n--- Phonon Analysis for {img_type} image (index {img_idx}) ---")

            # Create individual phonon analysis directory
            energy_str = f"m{abs(final_energies[img_idx]):.4f}".replace('.', 'p')
            phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_{img_type}_img{img_idx}_energy_{energy_str}")

            # Convert to primitive structure for phonon analysis
            pmg_structure = AseAtomsAdaptor.get_structure(atoms)
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

            # Run phonon analysis
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, phonon_dir,
                prefix=f"final_{original_prefix}_{img_type}_img{img_idx}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structures_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            print(f"  Completed phonon analysis for {img_type} image")

        except Exception as e:
            print(f"  Error during phonon analysis for {img_type} image: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- NEB Soft Mode Optimization Complete ---")


def run_ci_neb_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_modes_info_list, max_iterations,
                                     soft_mode_displacement_scales, cell_scale_factors, mode2_ratio_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, neb_num_images, neb_spring_constant,
                                     neb_max_iterations, neb_force_tolerance, neb_climbing_start_iteration, final_cif_path):
    """
    Runs CI-NEB optimization between initial and final structures, followed by phonon analysis.

    This function follows the same pattern as other optimization methods in VibroML but performs
    CI-NEB path optimization instead of soft mode displacement optimization.
    """
    print("\n--- Running CI-NEB Soft Mode Optimization ---")

    # Load final structure
    print(f"Loading final structure from: {final_cif_path}")
    try:
        _, final_atoms = load_structure(final_cif_path)
        if final_atoms is None:
            raise ValueError(f"Could not load final structure from {final_cif_path}")
    except Exception as e:
        print(f"Error loading final structure: {e}")
        return

    # Initialize calculator
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)

    # Create output directory for CI-NEB optimization
    ci_neb_output_dir = os.path.join(base_output_dir, "ci_neb_optimization")
    os.makedirs(ci_neb_output_dir, exist_ok=True)

    # Get original prefix for naming
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

    # Mandatory structure relaxation for both initial and final structures
    print(f"\n--- Mandatory Structure Relaxation for CI-NEB ---")
    print("Relaxing initial structure...")
    initial_relax_dir = os.path.join(ci_neb_output_dir, "initial_structure_relaxation")
    os.makedirs(initial_relax_dir, exist_ok=True)

    relaxed_initial_atoms = relax_structure(
        initial_atoms_for_soft_mode_analysis.copy(),
        calculator,
        args.engine,
        args.fmax,
        initial_relax_dir,
        args.cif,
        relaxation_patience=getattr(args, 'relaxation_patience', 5),
        volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
    )

    if relaxed_initial_atoms is None:
        print("Error: Initial structure relaxation failed. Cannot proceed with CI-NEB.")
        return

    print("Relaxing final structure...")
    final_relax_dir = os.path.join(ci_neb_output_dir, "final_structure_relaxation")
    os.makedirs(final_relax_dir, exist_ok=True)

    relaxed_final_atoms = relax_structure(
        final_atoms.copy(),
        calculator,
        args.engine,
        args.fmax,
        final_relax_dir,
        final_cif_path,
        relaxation_patience=getattr(args, 'relaxation_patience', 5),
        volume_expansion_threshold=getattr(args, 'volume_expansion_threshold', 2.5)
    )

    if relaxed_final_atoms is None:
        print("Error: Final structure relaxation failed. Cannot proceed with CI-NEB.")
        return

    print("Both structures successfully relaxed. Proceeding with CI-NEB optimization.")

    print(f"\n--- Starting CI-NEB Optimization ---")
    print(f"Initial structure: {len(relaxed_initial_atoms)} atoms (relaxed)")
    print(f"Final structure: {len(relaxed_final_atoms)} atoms (relaxed)")
    print(f"Number of intermediate images: {neb_num_images}")
    print(f"Spring constant: {neb_spring_constant} eV/Å²")
    print(f"Force tolerance: {neb_force_tolerance} eV/Å")
    print(f"Maximum iterations: {neb_max_iterations}")
    print(f"Climbing image starts at iteration: {neb_climbing_start_iteration}")

    # Run CI-NEB optimization
    try:
        ci_neb_results = run_neb_optimization(
            initial_atoms=relaxed_initial_atoms,
            final_atoms=relaxed_final_atoms,
            calculator=calculator,
            num_images=neb_num_images,
            spring_constant=neb_spring_constant,
            max_iterations=neb_max_iterations,
            force_tolerance=neb_force_tolerance,
            output_dir=ci_neb_output_dir,
            prefix=original_prefix,
            climbing_start_iteration=neb_climbing_start_iteration  # CI-NEB
        )
    except Exception as e:
        print(f"Error during CI-NEB optimization: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n--- CI-NEB Optimization Results ---")
    print(f"Converged: {ci_neb_results['converged']}")
    print(f"Final max force: {ci_neb_results['final_max_force']:.6f} eV/Å")
    print(f"Iterations: {ci_neb_results['iterations']}")
    print(f"Optimization time: {ci_neb_results['optimization_time']:.2f} seconds")
    if ci_neb_results['climbing_image_idx'] is not None:
        print(f"Climbing image index: {ci_neb_results['climbing_image_idx']}")

    # Generate enhanced CI-NEB summary with force information
    ci_neb_params = {
        'num_images': neb_num_images,
        'spring_constant': neb_spring_constant,
        'force_tolerance': neb_force_tolerance,
        'max_iterations': neb_max_iterations,
        'climbing_start_iteration': neb_climbing_start_iteration
    }

    # Save summary in both locations
    ci_neb_summary_paths = [
        os.path.join(ci_neb_output_dir, "ci_neb_summary.txt"),  # CI-NEB optimization subdirectory
        os.path.join(base_output_dir, "ci_neb_summary.txt")     # Main output directory
    ]

    generate_enhanced_neb_summary(
        results=ci_neb_results,
        method_name="CI-NEB",
        args=args,
        final_cif_path=final_cif_path,
        neb_params=ci_neb_params,
        output_paths=ci_neb_summary_paths
    )

    # Export final optimized structures to final_structures directory
    print(f"\n--- Exporting Final CI-NEB Structures ---")
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    optimized_images = ci_neb_results['images']
    final_energies = ci_neb_results['final_energies']
    climbing_idx = ci_neb_results['climbing_image_idx']

    for i, (image, energy) in enumerate(zip(optimized_images, final_energies)):
        # Create descriptive filename with energy and type
        if i == 0:
            structure_type = "initial"
        elif i == len(optimized_images) - 1:
            structure_type = "final"
        elif i == climbing_idx:
            structure_type = f"climbing_{i:02d}"
        else:
            structure_type = f"intermediate_{i:02d}"

        filename = f"ci_neb_{structure_type}_img{i:02d}_energy_{energy:.4f}eV.cif"
        output_path = os.path.join(final_structures_dir, filename)

        try:
            write(output_path, image)
            print(f"  Saved {structure_type} structure: {filename}")
        except Exception as e:
            print(f"  Error saving {structure_type} structure: {e}")

    print(f"Final CI-NEB structures exported to: {final_structures_dir}")

    # Perform phonon analysis on selected images from the optimized path (if requested)
    if args.with_phonon:
        print(f"\n--- Running Phonon Analysis on CI-NEB Images ---")
    else:
        print(f"\n--- Skipping Phonon Analysis (--with-phonon not provided) ---")
        print("\n--- CI-NEB Soft Mode Optimization Complete ---")
        return

    # Select images for phonon analysis (initial, final, and climbing image)
    images_for_phonon = []
    final_energies = ci_neb_results['final_energies']
    optimized_images = ci_neb_results['images']
    climbing_idx = ci_neb_results['climbing_image_idx']

    # Always include initial and final
    images_for_phonon.append((0, "initial", optimized_images[0]))
    images_for_phonon.append((len(optimized_images)-1, "final", optimized_images[-1]))

    # Include climbing image if it exists
    if climbing_idx is not None:
        images_for_phonon.append((climbing_idx, "climbing_transition_state", optimized_images[climbing_idx]))

    # Create final structures directory
    final_structures_dir = os.path.join(base_output_dir, "final_structures")
    os.makedirs(final_structures_dir, exist_ok=True)

    # Run phonon analysis on selected images
    for img_idx, img_type, atoms in images_for_phonon:
        try:
            print(f"\n--- Phonon Analysis for {img_type} image (index {img_idx}) ---")

            # Create individual phonon analysis directory
            energy_str = f"m{abs(final_energies[img_idx]):.4f}".replace('.', 'p')
            phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_{img_type}_img{img_idx}_energy_{energy_str}")

            # Convert to primitive structure for phonon analysis
            pmg_structure = AseAtomsAdaptor.get_structure(atoms)
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

            # Run phonon analysis
            _, _, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_dims, args.delta, args.fmax, phonon_dir,
                prefix=f"final_{original_prefix}_{img_type}_img{img_idx}_energy_{energy_str}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return,
                final_structures_dir=final_structures_dir,
                negative_phonon_threshold=negative_phonon_threshold_thz,
                save_yaml=args.save_yaml
            )
            print(f"  Completed phonon analysis for {img_type} image")

        except Exception as e:
            print(f"  Error during phonon analysis for {img_type} image: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- CI-NEB Soft Mode Optimization Complete ---")


def run_md_stability_analysis(args, base_output_dir, initial_atoms_for_analysis):
    """
    Run AIMD stability analysis using Machine Learning Interatomic Potentials.

    This function follows VibroML patterns for structure preparation, MLIP-powered
    MD simulation, and comprehensive stability analysis.

    Args:
        args: Command line arguments containing MD parameters
        base_output_dir: Base output directory for results
        initial_atoms_for_analysis: Initial structure for MD analysis
    """
    print("\n--- Running AIMD Stability Analysis ---")

    # Initialize calculator
    calculator = initialize_calculator(args.engine, model_name=args.model_name, checkpoint_path=args.checkpoint, nep_model_path=args.nep_model, checkpoint_model_path=args.checkpoint_model)
    if calculator is None:
        print("Failed to initialize calculator. Exiting MD stability analysis.")
        return

    # Get original prefix for naming
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

    # Create MD-specific output directory
    md_output_dir = os.path.join(base_output_dir, "md_stability_analysis")
    os.makedirs(md_output_dir, exist_ok=True)

    print(f"MD stability analysis output directory: {md_output_dir}")

    # --- Step 1: Structure Preparation ---
    print(f"\n--- Step 1: Structure Preparation ---")

    # Prepare supercell for MD simulation
    try:
        supercell_atoms = prepare_md_supercell(initial_atoms_for_analysis, args.supercell_size)
        print(f"Supercell prepared with {len(supercell_atoms)} atoms")

        # Save a copy of the initial supercell structure for trajectory analysis reference
        initial_supercell_atoms = supercell_atoms.copy()

        # Save initial supercell structure
        initial_supercell_path = os.path.join(md_output_dir, f"{original_prefix}_initial_supercell.cif")
        write(initial_supercell_path, supercell_atoms)
        print(f"Initial supercell saved to: {initial_supercell_path}")

    except Exception as e:
        print(f"Error preparing supercell: {e}")
        return

    # --- Step 2: MD Simulation Setup ---
    print(f"\n--- Step 2: MD Simulation Setup ---")

    # Calculate simulation parameters
    total_time_fs = args.time * 1000  # Convert ps to fs
    equilibration_time_fs = total_time_fs * args.equilibration_fraction
    production_time_fs = total_time_fs - equilibration_time_fs

    # Split equilibration into NVT (temperature) and NPT (pressure/volume) phases
    nvt_time_fs = equilibration_time_fs * 0.5  # 50% for temperature equilibration
    npt_equilibration_time_fs = equilibration_time_fs * 0.5  # 50% for pressure/volume equilibration

    # Use appropriate timestep (1 fs for most systems)
    timestep_fs = 1.0
    total_steps = int(total_time_fs / timestep_fs)
    nvt_steps = int(nvt_time_fs / timestep_fs)
    npt_equilibration_steps = int(npt_equilibration_time_fs / timestep_fs)
    production_steps = int(production_time_fs / timestep_fs)
    print(f"Total steps: {total_steps}")
    print(f"  NVT steps: {nvt_steps}")
    print(f"  NPT equilibration steps: {npt_equilibration_steps}")
    print(f"  Production steps: {production_steps}")

    # Temperature ramping parameters
    ramp_fraction = 0.7  # 70% of NVT time for temperature ramping
    ramp_steps = int(nvt_steps * ramp_fraction)

    print(f"MD simulation parameters:")
    print(f"  Total time: {args.time} ps ({total_time_fs} fs)")
    print(f"  Phase 1 - NVT equilibration: {nvt_time_fs} fs ({nvt_steps} steps)")
    print(f"    Temperature ramp: {ramp_steps} steps (0K → {args.temp}K)")
    print(f"    Temperature stabilization: {nvt_steps - ramp_steps} steps")
    print(f"  Phase 2 - NPT equilibration: {npt_equilibration_time_fs} fs ({npt_equilibration_steps} steps)")
    print(f"  Phase 3 - NPT production: {production_time_fs} fs ({production_steps} steps)")
    print(f"  Timestep: {timestep_fs} fs")
    print(f"  Total steps: {total_steps}")

    # Set up NVT ensemble for temperature equilibration
    try:
        nvt_dynamics = setup_nvt_ensemble(
            supercell_atoms,
            args.temp,
            calculator,
            timestep_fs
        )
        print("NVT ensemble setup completed")

    except Exception as e:
        print(f"Error setting up NVT ensemble: {e}")
        return

    # --- Step 3: MD Simulation Execution ---
    print(f"\n--- Step 3: MD Simulation Execution ---")

    # Set up trajectory file
    trajectory_file = os.path.join(md_output_dir, f"{original_prefix}_md_trajectory.traj")

    # Set up logging
    log_file = os.path.join(md_output_dir, f"{original_prefix}_md_log.txt")

    print(f"Starting two-stage MD equilibration protocol...")
    print(f"Trajectory will be saved to: {trajectory_file}")
    print(f"MD log will be saved to: {log_file}")

    start_time = time.time()

    try:
        # Import required ASE modules for MD
        from ase.io.trajectory import Trajectory
        from vibroml.utils.md_utils import StressMDLogger

        # Set up trajectory writer
        traj_writer = Trajectory(trajectory_file, 'w', supercell_atoms)

        # --- Phase 1: NVT Temperature Equilibration ---
        print(f"\n--- Phase 1: NVT Temperature Equilibration ---")
        print(f"Ramping temperature from 0K to {args.temp}K over {ramp_steps} steps")

        # Set up NVT logger
        nvt_log_file = os.path.join(md_output_dir, f"{original_prefix}_nvt_log.txt")
        nvt_logger = StressMDLogger(nvt_dynamics, supercell_atoms, nvt_log_file, header=True,
                                   stress=False, peratom=True, mode="w")

        # Attach trajectory writer and logger for NVT phase
        nvt_dynamics.attach(traj_writer.write, interval=max(1, nvt_steps // 500))  # Save frames
        nvt_dynamics.attach(nvt_logger, interval=max(1, nvt_steps // 50))  # Log frequently during equilibration

        # Temperature ramping callback
        def ramp_callback():
            step = nvt_dynamics.nsteps
            temperature_ramp_callback(nvt_dynamics, args.temp, step, ramp_steps)

        # Attach temperature ramping
        nvt_dynamics.attach(ramp_callback, interval=1)

        # Run NVT equilibration
        nvt_dynamics.run(nvt_steps)

        print(f"NVT equilibration completed. Final temperature: {nvt_dynamics.atoms.get_temperature():.1f} K")

        # --- Phase 2: NPT Pressure/Volume Equilibration ---
        print(f"\n--- Phase 2: NPT Pressure/Volume Equilibration ---")

        # Set up NPT ensemble using the equilibrated structure from NVT
        npt_dynamics = setup_npt_ensemble(
            supercell_atoms,  # Already has proper velocities from NVT
            args.temp,
            args.pressure,
            calculator,
            timestep_fs
        )

        # Set up NPT logger
        npt_log_file = os.path.join(md_output_dir, f"{original_prefix}_npt_log.txt")
        npt_logger = StressMDLogger(npt_dynamics, supercell_atoms, npt_log_file, header=True,
                                   stress=True, peratom=True, mode="w")

        # Attach trajectory writer and logger for NPT equilibration
        npt_dynamics.attach(traj_writer.write, interval=max(1, npt_equilibration_steps // 200))
        npt_dynamics.attach(npt_logger, interval=max(1, npt_equilibration_steps // 20))

        # Run NPT equilibration
        npt_dynamics.run(npt_equilibration_steps)

        print(f"NPT equilibration completed.")
        print(f"  Temperature: {npt_dynamics.atoms.get_temperature():.1f} K")
        print(f"  Volume: {npt_dynamics.atoms.get_volume():.2f} Å³")
        
        # --- Phase 3: NPT Production Run ---
        print(f"\n--- Phase 3: NPT Production Run ---")

        # Create separate NPT dynamics object for production to avoid log file reuse
        npt_production_dynamics = setup_npt_ensemble(
            supercell_atoms,
            args.temp,
            args.pressure,
            calculator,
            timestep_fs
        )

        # Set up separate production trajectory file for cleaner analysis
        production_trajectory_file = os.path.join(md_output_dir, f"{original_prefix}_production_trajectory.traj")
        production_traj_writer = Trajectory(production_trajectory_file, 'w', supercell_atoms)

        # Set up production logger (less frequent logging)
        prod_log_file = os.path.join(md_output_dir, f"{original_prefix}_production_log.txt")
        prod_logger = StressMDLogger(npt_production_dynamics, supercell_atoms, prod_log_file, header=True,
                                    stress=True, peratom=True, mode="w")

        # Attach trajectory writer and logger for production
        npt_production_dynamics.attach(production_traj_writer.write, interval=max(1, production_steps // 1000))  # Save ~1000 frames
        npt_production_dynamics.attach(prod_logger, interval=max(1, production_steps // 100))  # Log every ~1% of production

        # Run production phase
        npt_production_dynamics.run(production_steps)

        # Close trajectory files
        traj_writer.close()
        production_traj_writer.close()

        simulation_time = time.time() - start_time
        print(f"Complete MD simulation finished in {simulation_time:.2f} seconds")
        print(f"  NVT phase: {nvt_steps} steps")
        print(f"  NPT equilibration: {npt_equilibration_steps} steps")
        print(f"  NPT production: {production_steps} steps")

    except Exception as e:
        print(f"Error during MD simulation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 4: Trajectory Analysis ---
    print(f"\n--- Step 4: Trajectory Analysis ---")

    try:
        # Calculate production sampling interval for analysis
        prod_interval = max(1, production_steps // 1000) if production_steps > 0 else 1

        # Analyze trajectory stability using production-only trajectory file
        # This simplifies analysis since no equilibration frame skipping is needed
        stability_results = analyze_trajectory_stability(
            production_trajectory_file,  # Use production-only trajectory
            initial_supercell_atoms,     # Use initial supercell for comparison (same size as trajectory)
            timestep_fs,
            prod_interval,
            production_steps
        )

        # Analyze energy trajectory using production-only trajectory file
        energy_results = analyze_energy_trajectory(production_trajectory_file)

        print("Trajectory analysis completed")
        print(f"  Analysis performed on production phase ({production_steps} steps, {production_steps * timestep_fs:.1f} fs)")
        print(f"  Using separate production trajectory file for cleaner analysis")

    except Exception as e:
        print(f"Error during trajectory analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 5: Stability Assessment ---
    print(f"\n--- Step 5: Stability Assessment ---")

    try:
        # Determine stability verdict using configurable thresholds
        stability_verdict = determine_stability(
            stability_results,
            rmsd_threshold=args.rmsd_threshold,
            volume_threshold=args.volume_threshold,
            rdf_threshold=args.rdf_threshold
        )

        print(f"Stability analysis completed")
        print(f"Verdict: {stability_verdict['verdict']} (confidence: {stability_verdict['confidence']})")

    except Exception as e:
        print(f"Error during stability assessment: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 6: Generate Output Files ---
    print(f"\n--- Step 6: Generating Output Files ---")

    try:
        # Generate analysis plots
        plot_files = save_trajectory_analysis_plots(
            stability_results,
            energy_results,
            md_output_dir,
            original_prefix
        )

        # Save final structure from production trajectory
        final_production_trajectory = read(production_trajectory_file, index=':')
        final_structure = final_production_trajectory[-1]
        final_structure_path = os.path.join(md_output_dir, f"{original_prefix}_final_structure.cif")
        write(final_structure_path, final_structure)
        print(f"Final structure saved to: {final_structure_path}")

        # Create XYZ trajectory for visualization
        xyz_trajectory_path = os.path.join(md_output_dir, f"{original_prefix}_production_trajectory.xyz")
        write(xyz_trajectory_path, final_production_trajectory)
        print(f"XYZ trajectory saved to: {xyz_trajectory_path}")

        # Save trajectory in XYZ format for visualization
        xyz_trajectory_path = os.path.join(md_output_dir, f"{original_prefix}_trajectory.xyz")
        # Save every 10th frame to reduce file size
        xyz_frames = final_trajectory[::max(1, len(final_trajectory)//100)]
        write(xyz_trajectory_path, xyz_frames)
        print(f"Trajectory (XYZ format) saved to: {xyz_trajectory_path}")

    except Exception as e:
        print(f"Error generating output files: {e}")
        import traceback
        traceback.print_exc()

    # --- Step 7: Generate Comprehensive Report ---
    print(f"\n--- Step 7: Generating Comprehensive Report ---")

    try:
        # Generate comprehensive MD stability report
        report_path = os.path.join(md_output_dir, f"{original_prefix}_md_stability_report.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AIMD STABILITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Structure: {args.cif}\n")
            f.write(f"Engine: {args.engine}\n")
            if hasattr(args, 'model_name'):
                f.write(f"Model: {args.model_name}\n")
            f.write("\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Stability Verdict: {stability_verdict['verdict']}\n")
            f.write(f"Confidence Level: {stability_verdict['confidence']}\n")
            f.write("\n")

            # Simulation Parameters
            f.write("SIMULATION PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Temperature: {args.temp} K\n")
            f.write(f"Pressure: {args.pressure} GPa\n")
            f.write(f"Total simulation time: {args.time} ps\n")
            f.write(f"Supercell size: {args.supercell_size}\n")
            f.write(f"Equilibration fraction: {args.equilibration_fraction}\n")
            f.write(f"Number of atoms: {len(supercell_atoms)}\n")
            f.write(f"Timestep: {timestep_fs} fs\n")
            f.write(f"Total steps: {total_steps}\n")
            f.write(f"Production steps: {production_steps}\n")
            f.write(f"\n")
            f.write(f"Stability Assessment Thresholds (MLIP-optimized):\n")
            f.write(f"  RMSD threshold: {args.rmsd_threshold} Å\n")
            f.write(f"  Volume fluctuation threshold: ±{args.volume_threshold}%\n")
            f.write(f"  RDF correlation threshold: >{args.rdf_threshold}\n")
            f.write("\n")

            # Stability Metrics
            f.write("STABILITY METRICS\n")
            f.write("-" * 40 + "\n")
            for reason in stability_verdict['reasoning']:
                f.write(f"{reason}\n")
            f.write("\n")

            # Detailed Analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"RMSD Analysis:\n")
            f.write(f"  Mean RMSD: {stability_results['rmsd_mean']:.3f} Å\n")
            f.write(f"  RMSD std dev: {stability_results['rmsd_std']:.3f} Å\n")
            f.write(f"  RMSD trend: {stability_results['rmsd_trend']:.2e} Å/step\n")
            f.write(f"\n")

            f.write(f"Volume Analysis (MLIP-optimized approach):\n")
            f.write(f"  Reference volume (first {stability_results['volume_reference_time_fs']:.1f} fs): {stability_results['average_volume']:.2f} Å³\n")
            f.write(f"  Reference frames used: {stability_results['volume_reference_frames']}\n")
            f.write(f"  Mean volume fluctuation: {stability_results['volume_mean_change']:.2f}%\n")
            f.write(f"  Volume fluctuation std dev: {stability_results['volume_std_change']:.2f}%\n")
            f.write(f"  Max volume fluctuation: {stability_results['volume_max_change']:.2f}%\n")
            f.write(f"  Note: Fluctuations calculated relative to production phase average\n")
            f.write(f"\n")

            f.write(f"RDF Analysis:\n")
            f.write(f"  Initial vs final correlation: {stability_results['rdf_correlation']:.3f}\n")
            f.write(f"\n")

            if energy_results.get('has_energy_data', False):
                f.write(f"Energy Analysis:\n")
                f.write(f"  Mean energy: {energy_results['energy_mean']:.6f} eV/atom\n")
                f.write(f"  Energy std dev: {energy_results['energy_std']:.6f} eV/atom\n")
                f.write(f"  Energy drift: {energy_results['energy_drift']:.2e} eV/atom/fs\n")
                f.write(f"\n")

            # Generated Files
            f.write("GENERATED FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Combined trajectory file: {os.path.basename(trajectory_file)}\n")
            f.write(f"Production trajectory file: {os.path.basename(production_trajectory_file)}\n")
            f.write(f"Final structure: {os.path.basename(final_structure_path)}\n")
            f.write(f"XYZ trajectory: {os.path.basename(xyz_trajectory_path)}\n")
            f.write(f"MD log: {os.path.basename(log_file)}\n")
            for plot_file in plot_files:
                f.write(f"Analysis plot: {os.path.basename(plot_file)}\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if stability_verdict['verdict'] == 'STABLE':
                f.write("The structure appears to be dynamically stable under the given conditions.\n")
                f.write("Consider running longer simulations or different temperatures for validation.\n")
            else:
                f.write("The structure shows signs of instability under the given conditions.\n")
                f.write("Consider:\n")
                f.write("- Running at lower temperatures\n")
                f.write("- Checking for phase transitions\n")
                f.write("- Performing phonon analysis for comparison\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        print(f"Comprehensive report saved to: {report_path}")

        # Generate machine-readable summary
        summary_path = os.path.join(md_output_dir, f"{original_prefix}_md_stability_summary.csv")

        with open(summary_path, 'w') as f:
            f.write("structure,verdict,confidence,rmsd_mean,volume_max_change,rdf_correlation,")
            f.write("temperature,pressure,time_ps,supercell_size,num_atoms,")
            f.write("rmsd_threshold,volume_threshold,rdf_threshold,volume_ref_frames,volume_ref_time_fs\n")
            f.write(f"{original_prefix},{stability_verdict['verdict']},{stability_verdict['confidence']},")
            f.write(f"{stability_results['rmsd_mean']:.6f},{stability_results['volume_max_change']:.2f},")
            f.write(f"{stability_results['rdf_correlation']:.6f},{args.temp},{args.pressure},")
            f.write(f"{args.time},{args.supercell_size},{len(supercell_atoms)},")
            f.write(f"{args.rmsd_threshold},{args.volume_threshold},{args.rdf_threshold},")
            f.write(f"{stability_results['volume_reference_frames']},{stability_results['volume_reference_time_fs']:.1f}\n")

        print(f"Machine-readable summary saved to: {summary_path}")

        # Save configuration if requested
        if args.save_yaml:
            import yaml
            config_path = os.path.join(md_output_dir, f"{original_prefix}_md_stability_config.yaml")

            config_data = {
                'md_parameters': {
                    'temperature': args.temp,
                    'pressure': args.pressure,
                    'time': args.time,
                    'supercell_size': args.supercell_size,
                    'equilibration_fraction': args.equilibration_fraction
                },
                'simulation_details': {
                    'engine': args.engine,
                    'model_name': getattr(args, 'model_name', 'default'),
                    'timestep_fs': timestep_fs,
                    'total_steps': total_steps,
                    'production_steps': production_steps,
                    'num_atoms': len(supercell_atoms)
                },
                'stability_results': {
                    'verdict': stability_verdict['verdict'],
                    'confidence': stability_verdict['confidence'],
                    'rmsd_mean': float(stability_results['rmsd_mean']),
                    'volume_max_change': float(stability_results['volume_max_change']),
                    'rdf_correlation': float(stability_results['rdf_correlation'])
                }
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            print(f"Configuration saved to: {config_path}")

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n--- AIMD Stability Analysis Complete ---")
    print(f"Final verdict: {stability_verdict['verdict']} (confidence: {stability_verdict['confidence']})")
    print(f"All results saved to: {md_output_dir}")

    return stability_verdict