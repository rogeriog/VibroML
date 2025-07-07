import os
import sys
import json
import time

from .utils.structure_utils import initialize_calculator, generate_displaced_supercells, estimate_commensurate_supercell_size
from .utils.relaxation_utils import relax_structure, relax_structures_in_folder, find_lowest_energy_structures
from .utils.phonon_utils import run_single_phonon_analysis


from .utils.genetic_algorithm import GeneticAlgorithm

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write


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
        sc_n = supercell_ns[i]
        d = deltas[i]
        fm = fmax_values[i]
        print(f"\n--- Testing Supercell N: {sc_n}, Delta: {d}, Fmax: {fm} ---")
        current_output_dir = os.path.join(output_dir, f"N{sc_n}_D{d}_F{fm}")
        os.makedirs(current_output_dir, exist_ok=True)
        run_settings = vars(args).copy()
        run_settings['supercell_n'] = sc_n
        run_settings['delta'] = d
        run_settings['fmax'] = fm
        with open(os.path.join(current_output_dir, "run_settings.json"), 'w') as f:
            json.dump(run_settings, f, indent=4)
        print(f"\nAttempting to relax structure for Fmax={fm}...")  
        # Pass a copy of initial_atoms to relax_structure to avoid modifying it directly  
        relaxed_atoms_for_current_params = relax_structure(  
            initial_atoms.copy(), calculator, args.engine, fm, current_output_dir, args.cif 
        )
        if relaxed_atoms_for_current_params is None:  
            print(f"Skipping phonon analysis for N={sc_n}, D={d}, F={fm} due to failed relaxation.")  
            results.append({  
                "supercell_n": sc_n,  
                "delta": d,  
                "fmax": fm,  
                "negative_frequency_at_special_point": None,  
                "time_taken": 0,  
                "relaxation_status": "failed"  
            })  
            continue # Skip to the next iteration if relaxation fails  
        else:  
            print(f"Relaxation successful for N={sc_n}, D={d}, F={fm}. Proceeding with phonon analysis.")  
            results.append({  
                "supercell_n": sc_n,  
                "delta": d,  
                "fmax": fm,  
                "relaxation_status": "successful"  
            })
        # Expect a list of softest modes now
        softest_modes_info_current, neg_freq_at_special_point, time_taken = run_single_phonon_analysis(
            relaxed_atoms_for_current_params.copy(), calculator, args.engine, args.units, sc_n, d, fm, current_output_dir, prefix=original_prefix,
            phonon_path_npoints=phonon_path_npoints,
            phonon_dos_grid=phonon_dos_grid,
            traj_kT=default_traj_kT,
            num_modes_to_return=num_modes_to_return,
        )

        if neg_freq_at_special_point is not None:
            results[-1].update({  
                "negative_frequency_at_special_point": neg_freq_at_special_point,  
                "time_taken": time_taken  
            })
            if neg_freq_at_special_point > best_negative_frequency:
                best_negative_frequency = neg_freq_at_special_point
                best_settings = {"supercell_n": sc_n, "delta": d, "fmax": fm}
                best_softest_modes_info = softest_modes_info_current
                best_relaxed_atoms = relaxed_atoms_for_current_params.copy()
            improvement = best_negative_frequency - previous_best_negative_frequency
            if improvement < abs(threshold_in_current_units * 0.5):
                print(f"Improvement in negative frequency ({improvement:.4f} {args.units}) is less than {abs(threshold_in_current_units * 0.5):.4f} {args.units}. Stopping parameter sweep.")
                break
            previous_best_negative_frequency = best_negative_frequency

    print("\n--- Parameter sweep auto-optimization complete ---")
    print(f"Most negative frequency at a special point found: {best_negative_frequency:.4f} {args.units}")
    print(f"Optimal settings: Supercell N = {best_settings.get('supercell_n')}, Delta = {best_settings.get('delta')}, Fmax = {best_settings.get('fmax')}")

    with open(os.path.join(output_dir, "auto_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    return best_negative_frequency, best_settings, best_softest_modes_info, best_relaxed_atoms

def run_automatic_soft_mode_optimization(args, output_dir, best_negative_frequency, best_settings, best_softest_modes_info, best_relaxed_atoms, negative_phonon_threshold_thz,
                                     soft_mode_max_iterations, soft_mode_displacement_scales, mode2_ratio_scales, soft_mode_num_top_structures_to_analyze,
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, cell_scale_factors, num_modes_to_return,
                                     ga_population_size, ga_mutation_rate, num_new_points_per_iteration):
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
                num_new_points_per_iteration=num_new_points_per_iteration
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
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, ga_population_size, ga_mutation_rate, num_new_points_per_iteration):
    """
    Runs an iterative workflow to find low-energy structures using a Genetic Algorithm,
    and then performs a final phonon analysis on the best candidates found across all iterations.
    """
    print("\n--- Running Soft Mode Iterative Optimization (Genetic Algorithm) ---")

    # --- 1. Initial Setup ---
    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_modes_info_list = initial_softest_modes_info_list
    calculator = initialize_calculator(args.engine)
    if calculator is None:
        print("Failed to initialize calculator. Exiting.")
        return

    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]
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
  
    # Define decomposition threshold  
    decomposition_threshold = -0.5 # eV
                
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
    all_iterations_results = []

    # Define GA parameter bounds. These ranges should be configurable, perhaps from default_settings.json.
    disp_scale_bounds = (0.0, 8.0) # Example: 0 to 8 Angstrom displacement magnitude
    ratio_mode2_to_mode1_bounds = (-1.5, 1.5) # Example: 0 to 1 (0 means no mode 2, 1 means equal magnitude)
    cell_scale_bounds = (-0.30, 0.30) # Example: -30% to +30% change in a,b,c
    cell_angle_bounds = (-30.0, 30.0) # Example: -30 to +30 degrees change in alpha,beta,gamma

    # Initialize Genetic Algorithm
    ga = GeneticAlgorithm(
        population_size=ga_population_size,
        mutation_rate=ga_mutation_rate,
        displacement_scale_bounds=disp_scale_bounds,
        ratio_mode2_to_mode1_bounds=ratio_mode2_to_mode1_bounds,
        cell_scale_bounds=cell_scale_bounds,
        cell_angle_bounds=cell_angle_bounds,
        supercell_variants=supercell_variants, 
        num_offspring=num_new_points_per_iteration
    )

    # --- 2. Main Iterative Loop (Genetic Algorithm Driven) ---
    for main_iteration_idx in range(1, max_iterations + 1):  
        print(f"\n### Starting Main Iteration {main_iteration_idx} (3 GA generations + phonon check) ###")  
        
        # Run 3 GA generations for this main iteration  
        for ga_generation in range(1, 4):  # 3 generations  
            print(f"\n--- GA Generation {ga_generation} of Main Iteration {main_iteration_idx} ---")
        
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
                                    'params': (disp_scale, ratio_mode2, initial_cell_transform_vec, supercell_variant), # Add supercell gene  
                                    'fitness': None  
                                })  
                # Adds additional random samples to reach population_size 
                # The GA object now handles this internally when filling up the population  
                ga.initialize_population(initial_individuals=initial_ga_individuals)  
                new_offspring_params = [ind['params'] for ind in ga.population]
            else:  
                # Generations 2 and 3: normal GA evolution  
                current_generation_results = [r for r in all_iterations_results if r.get('main_iteration') == main_iteration_idx and r.get('ga_generation') == ga_generation - 1]  
                if not current_generation_results:  
                    print(f"No results from previous generation to evolve from. Using all available results.")  
                    current_generation_results = all_iterations_results[-ga_population_size:] if len(all_iterations_results) >= ga_population_size else all_iterations_results  
                
                ga_format_current = convert_results_for_ga(current_generation_results)
                if ga_format_current:  
                    new_offspring_params = ga.evolve(ga_format_current)  
                else:  
                    print("No valid results to evolve from. Generating random population.")  
                    ga.initialize_population()  
                    new_offspring_params = [ind['params'] for ind in ga.population]
                        
            if not new_offspring_params:
                print(f"GA did not generate any new offspring for main iteration {main_iteration_idx}, generation {ga_generation}. Stopping.")
                break

            iteration_results = [] # To store results for this specific GA iteration

            # Generate, Relax, and Evaluate each new individual
            for i, individual_params in enumerate(new_offspring_params):  
                scale_mode1, ratio_mode2_to_mode1, cell_transformation_vector, supercell_variant = individual_params  
        
                sample_output_dir = os.path.join(base_output_dir, f"main_iter_{main_iteration_idx}_gen_{ga_generation}", f"sample_{i+1}")  
                os.makedirs(sample_output_dir, exist_ok=True)  
        
                print(f"\n  Generating and relaxing structure for GA sample {i+1} (Main Iter {main_iteration_idx}, Gen {ga_generation}):")  
                # Log the supercell being used for this individual  
                print(f"    Mode1 Scale: {scale_mode1:.3f}, Mode2 Ratio: {ratio_mode2_to_mode1:.3f}, Cell Transform: {cell_transformation_vector}, Supercell: {supercell_variant}")  
        
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
                    cell_transformation_vector  
                )

                if not generated_cif_paths:
                    print(f"    No CIFs generated for sample {i+1}. Skipping relaxation.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None, # Indicate failure
                        'relaxed_atoms': None,
                        'original_file': None,
                        'main_iteration': main_iteration_idx,  
                        'ga_generation': ga_generation,  
                        'sample': i + 1  
                    })
                    continue

                # Relax the generated structures
                unique_supercell_folders = list(set(os.path.dirname(fpath) for fpath in generated_cif_paths))

                sample_relaxation_results = []
                for folder in unique_supercell_folders:
                    folder_relaxation_results = relax_structures_in_folder(folder, calculator, args.engine, args.fmax)
                    sample_relaxation_results.extend(folder_relaxation_results)

                if not sample_relaxation_results:
                    print(f"    No structures successfully relaxed for sample {i+1}.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None, # FIXED: Set to None for failed relaxations
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
                        print(f"    Sample {i+1} (Energy: {E_relax:.6f} eV/atom) flagged as DECOMPOSED (E_relax < E_ref {E_ref:.6f} + {decomposition_threshold} eV). Excluding from ranking and further GA steps.")
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
                            'sample': i + 1
                        })
                        print(f"    Sample {i+1} relaxed. Lowest energy: {best_result_for_sample['energy_per_atom']:.6f} eV/atom")
                else:
                    print(f"    Could not find lowest energy for sample {i+1}.")
                    iteration_results.append({
                        'params': individual_params,
                        'energy_per_atom': None, # Indicate failure
                        'relaxed_atoms': None,
                        'original_file': None,
                        'main_iteration': main_iteration_idx,  
                        'ga_generation': ga_generation,  
                        'sample': i + 1
                    })

            # Add this iteration's results to the master list for GA evolution in next iteration
            # Filter out failed relaxations (energy_per_atom is None) before adding to all_iterations_results
            valid_iteration_results = [r for r in iteration_results if r['energy_per_atom'] is not None]
            all_iterations_results.extend(valid_iteration_results)

            if not valid_iteration_results:
                print(f"No valid structures found in main iteration {main_iteration_idx}, generation {ga_generation}. Continuing anyway (exploratory mode).")

            # Create iteration-specific relaxation summary
            # FIXED: Create directory first
            generation_dir = os.path.join(base_output_dir, f"main_iter_{main_iteration_idx}_gen_{ga_generation}")
            os.makedirs(generation_dir, exist_ok=True)
            iteration_summary_filepath = os.path.join(generation_dir, "relaxation_summary_generation.txt")

            with open(iteration_summary_filepath, 'w') as f:
                f.write(f"--- Relaxation Summary for Main Iteration {main_iteration_idx}, GA Generation {ga_generation} ---\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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
                    if isinstance(params, tuple) and len(params) == 4:  
                        cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])  
                        sc_str = "x".join(map(str, params[3])) # e.g., (2,2,1) -> "2x2x1"  
                        params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str}), SC:{sc_str}"  
                    else:  
                        params_str = str(params)
                    energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)  
                    f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('main_iteration', 'N/A')):<10} {str(result.get('ga_generation', 'N/A')):<8} {str(result.get('sample', 'N/A')):<8} {params_str:<65}\n")
            
            print(f"Generation results saved to {iteration_summary_filepath}")
            
        # End of GA generation loop  
          
        # After 3 GA generations, perform phonon check and update primitive cell  
        print(f"\n--- Completed 3 GA generations for Main Iteration {main_iteration_idx} ---")  
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

            next_softest_modes_info_list, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
                args.supercell_n, args.delta, args.fmax, check_dir, # Use args.supercell_n, delta, fmax from initial config
                prefix=f"guidance_check_main_iter_{main_iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return
            )
            # Update the guiding atoms and soft mode list for the next loop
            current_primitive_atoms = primitive_atoms_for_next_iter.copy()
            current_softest_modes_info_list = next_softest_modes_info_list
            
            # Update supercell variants based on the new guiding mode's q-point
            if 'coordinate' in next_softest_modes_info_list[0]:
                q_point_for_supercell = next_softest_modes_info_list[0]['coordinate']
                supercell_variants = [estimate_commensurate_supercell_size(q_point_for_supercell)]
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
    energy_tolerance = 1e-4  # 0.0001 eV/atom tolerance

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
            run_single_phonon_analysis(  
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,  
                args.supercell_n, args.delta, args.fmax, final_phonon_dir,  
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",  
                phonon_path_npoints=phonon_path_npoints,  
                phonon_dos_grid=phonon_dos_grid,  
                traj_kT=default_traj_kT,  
                num_modes_to_return=num_modes_to_return  
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
                run_single_phonon_analysis(  
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,  
                    args.supercell_n, args.delta, args.fmax, final_phonon_dir,  
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",  
                    phonon_path_npoints=phonon_path_npoints,  
                    phonon_dos_grid=phonon_dos_grid,  
                    traj_kT=default_traj_kT,  
                    num_modes_to_return=num_modes_to_return  
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
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'Iter':<6} {'Sample':<8} {'GA Params':<65}\n")  
        f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*6:<6} {'-'*8:<8} {'-'*65:<65} \n")

        # Sort all_iterations_results by energy_per_atom for this summary
        sorted_overall_results = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])

        for result in sorted_overall_results:
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result.get('energy_per_atom', 'FAIL')
            params = result.get('params', ('N/A', 'N/A', ('N/A',)*6, 'N/A'))
            if isinstance(params, tuple) and len(params) == 4:  
                cell_transform_str = ", ".join([f"{val:.3f}" for val in params[2]])  
                sc_str = "x".join(map(str, params[3])) # e.g., (2,2,1) -> "2x2x1"  
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:({cell_transform_str}), SC:{sc_str}"  
            else:  
                params_str = str(params)
            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)  
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {str(result.get('main_iteration', 'N/A')):<10} {str(result.get('ga_generation', 'N/A')):<8} {str(result.get('sample', 'N/A')):<8} {params_str:<65}\n")
    print(f"Overall relaxation summary saved to: {overall_summary_filepath}")

    # Print the overall summary to screen
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
    calculator = initialize_calculator(args.engine)
    if calculator is None:
        print("Failed to initialize calculator. Exiting.")
        return
    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]
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

    all_iterations_results = [] # To store results for all iterations

    # --- 2. Main Iterative Loop (Traditional Grid Search) ---
    for iteration_idx in range(1, max_iterations + 1):
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
                        cell_transformation_vector  
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
                        folder_relaxation_results = relax_structures_in_folder(folder, calculator, args.engine, args.fmax)  
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

            next_softest_modes_info_list, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
                args.supercell_n, args.delta, args.fmax, check_dir,
                prefix=f"guidance_check_iter_{iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return
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
            run_single_phonon_analysis(  
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,  
                args.supercell_n, args.delta, args.fmax, final_phonon_dir,  
                prefix=f"final_{original_prefix}_top_{i+1}_energy_{energy_str}",  
                phonon_path_npoints=phonon_path_npoints,  
                phonon_dos_grid=phonon_dos_grid,  
                traj_kT=default_traj_kT,  
                num_modes_to_return=num_modes_to_return  
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
                run_single_phonon_analysis(  
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,  
                    args.supercell_n, args.delta, args.fmax, final_phonon_dir,  
                    prefix=f"final_{original_prefix}_unique_{i+1}_energy_{energy_str}",  
                    phonon_path_npoints=phonon_path_npoints,  
                    phonon_dos_grid=phonon_dos_grid,  
                    traj_kT=default_traj_kT,  
                    num_modes_to_return=num_modes_to_return  
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