# auto_optimize.py (Revised and Formatted)
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import re
import shutil

from .utils.structure_utils import load_structure, initialize_calculator, print_initial_structure_info, generate_displaced_supercells
from .utils.relaxation_utils import relax_structure, relax_structures_in_folder, find_lowest_energy_structures, create_displaced_supercell_summary
from .utils.phonon_utils import run_phonon_calculation, get_phonon_results, analyze_special_points_and_modes
from .utils.plotting_utils import plot_phonon_results, save_raw_data
from .utils.utils import clean_phonon_cache

from .utils.genetic_algorithm import GeneticAlgorithm # NEW IMPORT

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read, write


# analyze_and_resample function is removed as per plan.


def run_single_phonon_analysis(atoms, calculator, engine, units, supercell_n, delta, fmax, output_dir, prefix="phonon_run", phonon_path_npoints=100, phonon_dos_grid=(40,40,40), traj_kT=1.0, num_modes_to_return=2):
    """
    Runs a single phonon calculation step (calculate, plot, save) on a given atoms object.
    This function is a core component used by both parameter sweep and soft mode optimization.

    Args:
        atoms (ase.Atoms): The ASE Atoms object representing the structure.
        calculator (ase.calculators.calculator.Calculator): The ASE calculator to use.
        engine (str): The name of the DFT engine (e.g., "VASP", "QuantumEspresso").
        units (str): Units for frequencies (e.g., "THz", "cm-1").
        supercell_n (int): Supercell size (N,N,N).
        delta (float): Displacement delta for phonon calculation.
        fmax (float): Maximum force for relaxation convergence.
        output_dir (str): Directory to save output files.
        prefix (str): Prefix for output filenames (default "phonon_run").
        phonon_path_npoints (int): Number of points along the phonon path (default 100).
        phonon_dos_grid (tuple): Grid for DOS calculation (default (40,40,40)).
        traj_kT (float): Temperature for trajectory generation (default 0.1).
        num_modes_to_return (int): The number of softest modes to return (default 2).

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, each containing information about a softest mode,
                    including its raw displacements. Returns an empty list if no soft modes found.
            - float: The most negative frequency found (bsmin).
            - float: The total time taken for the analysis.
    """
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Running Single Phonon Analysis in: {output_dir} ---")

    print("\n--- Step 1: Running Phonon Calculation ---")
    ph = run_phonon_calculation(atoms, calculator, supercell_n, delta, output_dir)
    if ph is None:
        print("Error during phonon calculation setup.")
        return [], None, None # Return empty list for modes

    print("\n--- Step 2: Getting and Processing Phonon Results ---")
    bs, path, dos, bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, bsmin = get_phonon_results(ph, atoms, units, phonon_path_npoints, phonon_dos_grid)

    print("\n--- Step 3: Saving Raw Data ---")
    save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_n, delta, fmax, output_dir)

    print("\n--- Step 4: Plotting Results ---")
    struct_formula = atoms.get_chemical_formula()
    plot_phonon_results(bs_energies, dos, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, struct_formula, prefix, supercell_n, delta, fmax, output_dir)

    print(f"\n--- Step 5: Analyzing Special Points and Top {num_modes_to_return} Softest Modes ---")

    softest_modes_info_list = analyze_special_points_and_modes(
        ph, bs, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, traj_kT=traj_kT, num_modes_to_return=num_modes_to_return
    )
    time_taken = time.time() - start_time

    summary_filename = os.path.join(output_dir, f"{prefix}_phonon_run_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write(f"--- Phonon Run Summary ---\n")
        f.write(f"Structure Formula: {atoms.get_chemical_formula()}\n")
        f.write(f"Total Number of Atoms: {len(atoms)}\n")
        f.write(f"Engine Used: {engine}\n")
        f.write(f"Units: {units}\n")
        f.write(f"Supercell Size (N,N,N): ({supercell_n}, {supercell_n}, {supercell_n})\n")
        f.write(f"Displacement Delta: {delta}\n")
        f.write(f"Fmax for Relaxation: {fmax}\n")
        f.write(f"Most Negative Frequency (overall): {bsmin:.4f} {units}\n") # bsmin is still the overall minimum
        f.write(f"Time Taken for Phonon Analysis: {time_taken:.2f} seconds\n\n")
        try:
            energy = atoms.get_potential_energy()
            energy_per_atom = energy / len(atoms)
            f.write(f"Energy of Structure: {energy:.6f} eV\n")
            f.write(f"Energy per Atom: {energy_per_atom:.6f} eV/atom\n\n")
        except Exception as e:
            f.write(f"Could not retrieve energy for this structure: {e}\n\n")

        # Write information about the top N softest modes
        if softest_modes_info_list:
            f.write(f"--- Top {len(softest_modes_info_list)} Softest Modes Found at Special K-points ---\n")
            for i, mode_info in enumerate(softest_modes_info_list):
                f.write(f"Mode {i+1}:\n")
                f.write(f"   Label: {mode_info.get('label', 'N/A')}\n")
                f.write(f"   Coordinate: {mode_info.get('coordinate', 'N/A')}\n")
                f.write(f"   Frequency: {mode_info.get('frequency', 'N/A'):.4f} {units}\n")
                f.write(f"   Band Index: {mode_info.get('band_index', 'N/A')}\n")
                f.write("\n")
        else:
            f.write("No negative frequencies found at special k-points.\n\n")

        f.write("Path of K-points:\n")
        for i, (dist, label) in enumerate(zip(special_k_point_distances, special_k_point_labels)):
            f.write(f"  {label}: {dist:.4f}\n")
        f.write("\n")
        symmetry_file_path = os.path.join(output_dir, "relaxed_symmetry_analysis.txt")
        if not os.path.exists(symmetry_file_path):
            symmetry_file_path = os.path.join(output_dir, "initial_symmetry_analysis.txt")
        if os.path.exists(symmetry_file_path):
            try:
                with open(symmetry_file_path, 'r') as sym_f:
                    sym_content = sym_f.read()
                    sg_number_line = next((line for line in sym_content.splitlines() if "Space group number:" in line), None)
                    sg_symbol_line = next((line for line in sym_content.splitlines() if "International symbol:" in line), None)
                    if sg_number_line:
                        f.write(f"{sg_number_line.strip()}\n")
                    if sg_symbol_line:
                        f.write(f"{sg_symbol_line.strip()}\n")
            except Exception as e:
                f.write(f"Could not read space group from symmetry analysis file: {e}\n")
        else:
            f.write("Space Group: Not available (symmetry analysis file not found).\n")

    print(f"Comprehensive phonon run summary saved to: {summary_filename}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Total time taken for this phonon analysis: {time_taken:.2f} s')

    # Return the list of softest modes, the overall minimum frequency, and time taken
    return softest_modes_info_list, bsmin, time_taken


def run_parameter_sweep_optimization(args, output_dir, supercell_ns, deltas, fmax_values, negative_phonon_threshold_thz, soft_mode_max_iterations, soft_mode_displacement_scales, soft_mode_num_top_structures_to_analyze,
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, cell_scale_factors, num_modes_to_return, ga_population_size, ga_mutation_rate, num_new_points_per_iteration):
    """
    Runs the parameter sweep auto-optimization loop.
    If a soft mode is found in the best configuration, it triggers the soft mode optimization.
    """
    print("Running in parameter sweep auto mode to find optimal settings...")
    best_negative_frequency = -float('inf')
    best_settings = {}
    best_softest_modes_info = [] # Changed to list
    best_relaxed_atoms = None

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

    print("\n--- Loading and potentially relaxing initial structure for parameter sweep ---")
    struct, initial_atoms = load_structure(args.cif)
    if struct is None or initial_atoms is None:
        print("Failed to load initial structure.")
        return None

    calculator = initialize_calculator(args.engine)
    if calculator is None:
        print("Failed to initialize calculator.")
        return None

    relaxed_initial_atoms = initial_atoms.copy()
    if not args.no_relax:
        initial_relax_dir = os.path.join(output_dir, "initial_relaxation_for_sweep")
        os.makedirs(initial_relax_dir, exist_ok=True)
        relaxed_initial_atoms = relax_structure(initial_atoms.copy(), calculator, args.engine, args.fmax, initial_relax_dir, args.cif)
        if relaxed_initial_atoms is None:
            print("Initial relaxation failed. Exiting parameter sweep.")
            return None
    else:
        print("Skipping initial structure relaxation for parameter sweep.")

    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

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
        # Expect a list of softest modes now
        softest_modes_info_current, neg_freq_at_special_point, time_taken = run_single_phonon_analysis(
            relaxed_initial_atoms.copy(), calculator, args.engine, args.units, sc_n, d, fm, current_output_dir, prefix=original_prefix,
            phonon_path_npoints=phonon_path_npoints,
            phonon_dos_grid=phonon_dos_grid,
            traj_kT=default_traj_kT,
            num_modes_to_return=num_modes_to_return,
        )
        if neg_freq_at_special_point is not None:
            results.append({
                "supercell_n": sc_n,
                "delta": d,
                "fmax": fm,
                "negative_frequency_at_special_point": neg_freq_at_special_point,
                "time_taken": time_taken
            })
            if neg_freq_at_special_point > best_negative_frequency:
                best_negative_frequency = neg_freq_at_special_point
                best_settings = {"supercell_n": sc_n, "delta": d, "fmax": fm}
                best_softest_modes_info = softest_modes_info_current # Store the list
                best_relaxed_atoms = relaxed_initial_atoms.copy()
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

    # Pass the list of softest modes to soft mode optimization
    if best_negative_frequency < threshold_in_current_units and best_softest_modes_info and best_relaxed_atoms is not None:
        print(f"\nSoft mode detected ({best_negative_frequency:.4f} {args.units}) after parameter sweep. Initiating iterative soft mode optimization...")
        run_soft_mode_optimization(
            args,
            output_dir,
            best_relaxed_atoms,
            best_softest_modes_info, # Pass the list
            max_iterations=soft_mode_max_iterations,
            # displacement_scales and cell_scale_factors will be used for GA initialization
            soft_mode_displacement_scales=soft_mode_displacement_scales,
            cell_scale_factors=cell_scale_factors,
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
    else:
        print("\nNo significant soft mode detected after parameter sweep, or structure is stable enough. Skipping iterative soft mode optimization.")

    return best_settings


def run_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_modes_info_list, max_iterations,
                               soft_mode_displacement_scales, cell_scale_factors, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, num_modes_to_return, ga_population_size, ga_mutation_rate, num_new_points_per_iteration):
    """
    Runs an iterative workflow to find low-energy structures using a Genetic Algorithm,
    and then performs a final phonon analysis on the best candidates found across all iterations.
    """
    print("\n--- Running Soft Mode Iterative Optimization (Genetic Algorithm) ---")

    # --- 1. Initial Setup ---
    current_primitive_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_modes_info_list = initial_softest_modes_info_list # This is now a list
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

    # Determine supercell variants based on cell lengths  
    cell = current_primitive_atoms.cell  
    lengths = np.linalg.norm(cell, axis=1)  
    
    # Check for equality of lengths  
    is_all_equal = np.allclose(lengths, lengths[0], rtol=1e-3)  
    is_two_equal = (  
        (np.allclose(lengths[0], lengths[1], rtol=1e-3) and not np.allclose(lengths[1], lengths[2], rtol=1e-3)) or  
        (np.allclose(lengths[0], lengths[2], rtol=1e-3) and not np.allclose(lengths[0], lengths[1], rtol=1e-3)) or  
        (np.allclose(lengths[1], lengths[2], rtol=1e-3) and not np.allclose(lengths[0], lengths[1], rtol=1e-3))  
    )  
    is_all_different = not is_all_equal and not is_two_equal  
    
    if is_all_equal:  
        supercell_variants = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)]  
    elif is_two_equal:  
        supercell_variants = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 2, 2)]  
    else: # is_all_different  
        supercell_variants = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 2, 2), (2, 2, 2)]
    all_iterations_results = [] # Stores {'params': (disp_scale, ratio, cell_vec), 'fitness': energy, 'relaxed_atoms': atoms_obj, 'original_file': path}

    # Define GA parameter bounds
    # These ranges should be configurable, perhaps from default_settings.json
    disp_scale_bounds = (0.0, 100.0) # Example: 0 to 100 Angstrom displacement magnitude
    ratio_mode2_to_mode1_bounds = (0.0, 1.0) # Example: 0 to 1 (0 means no mode 2, 1 means equal magnitude)
    cell_scale_bounds = (-0.2, 0.2) # Example: -20% to +20% change in a,b,c
    cell_angle_bounds = (-30.0, 30.0) # Example: -30 to +30 degrees change in alpha,beta,gamma

    # Initialize Genetic Algorithm
    # num_new_points (30 samples per iteration) will be the num_offspring for GA
    ga = GeneticAlgorithm(
        population_size=ga_population_size, # New arg for GA population size
        mutation_rate=ga_mutation_rate,     # New arg for GA mutation rate
        displacement_scale_bounds=disp_scale_bounds,
        ratio_mode2_to_mode1_bounds=ratio_mode2_to_mode1_bounds,
        cell_scale_bounds=cell_scale_bounds,
        cell_angle_bounds=cell_angle_bounds,
        num_offspring=num_new_points_per_iteration 
    )

    # --- 2. Main Iterative Loop (Genetic Algorithm Driven) ---
    for iteration_idx in range(1, max_iterations + 1):
        print(f"\n### Starting GA Iteration {iteration_idx} ###")

        # Check if the guiding soft mode is valid
        if not current_softest_modes_info_list or 'raw_displacements' not in current_softest_modes_info_list[0]:
            print(f"No softest mode information to guide iteration {iteration_idx}. Stopping.")
            break

        # Check if the guiding soft mode is already stable enough to stop
        if current_softest_modes_info_list[0]['frequency'] >= threshold_in_current_units:  
            print(f"\nGuiding soft mode frequency ({current_softest_modes_info_list[0]['frequency']:.4f} {args.units}) is positive/stable. Continuing iterative search as requested.")  
        else:  
            print(f"Using soft mode at {current_softest_modes_info_list[0].get('label', 'unknown')} ({current_softest_modes_info_list[0]['frequency']:.4f} {args.units}) to guide displacements.")


        # Prepare initial population for the first iteration, or evolve for subsequent
        if iteration_idx == 1:
            # For the first iteration, generate initial individuals based on soft_mode_displacement_scales
            # and cell_scale_factors. We need to convert these into the GA individual format.
            initial_ga_individuals = []
            # Create a grid of initial parameters from soft_mode_displacement_scales and cell_scale_factors
            # This will form the initial population for the GA
            for disp_scale in soft_mode_displacement_scales:
                for cell_scale in cell_scale_factors:
                    # Convert simple cell_scale to a 6-element vector (only a,b,c scaled equally)
                    initial_cell_transform_vec = (cell_scale, cell_scale, cell_scale, 0.0, 0.0, 0.0)
                    # For initial population, assume ratio_mode2_to_mode1 is 0 (only mode 1)
                    initial_ga_individuals.append({
                        'params': (disp_scale, 0.0, initial_cell_transform_vec),
                        'fitness': None # Fitness will be calculated after relaxation
                    })
            # If the number of initial_ga_individuals is less than ga.population_size, GA will fill the rest randomly
            ga.initialize_population(initial_individuals=initial_ga_individuals)
            # For the first iteration, we use the initial_ga_individuals as the "new_offspring_params"
            # to kick off the relaxation process.
            new_offspring_params = [ind['params'] for ind in ga.population]
        else:
            # For subsequent iterations, evolve the population based on previous results
            # all_iterations_results contains {'params': ..., 'fitness': ..., 'relaxed_atoms': ..., 'original_file': ...}
            # We need to pass the 'params' and 'fitness' to the GA's evolve method
            current_population_for_ga = [{'params': r['params'], 'fitness': r['energy_per_atom']} for r in all_iterations_results if 'energy_per_atom' in r and r['energy_per_atom'] is not None]
            if not current_population_for_ga:
                print("No valid structures with energy found in previous iteration to evolve from. Stopping.")
                break
            new_offspring_params = ga.evolve(current_population_for_ga)

        if not new_offspring_params:
            print(f"GA did not generate any new offspring for iteration {iteration_idx}. Stopping.")
            break

        iteration_results = [] # To store results for this specific GA iteration

        # Generate, Relax, and Evaluate each new individual
        for i, individual_params in enumerate(new_offspring_params):
            scale_mode1, ratio_mode2_to_mode1, cell_transformation_vector = individual_params
            
            sample_output_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}", f"sample_{i+1}")
            os.makedirs(sample_output_dir, exist_ok=True)

            print(f"\n  Generating and relaxing structure for GA sample {i+1} (Iter {iteration_idx}):")
            print(f"    Mode1 Scale: {scale_mode1:.3f}, Mode2 Ratio: {ratio_mode2_to_mode1:.3f}, Cell Transform: {cell_transformation_vector}")

            # Generate displaced supercells using the new function signature
            generated_cif_paths = generate_displaced_supercells(
                current_primitive_atoms.copy(),
                current_softest_modes_info_list, # Pass the list of modes
                scale_mode1,
                ratio_mode2_to_mode1,
                supercell_variants,
                sample_output_dir, # Use sample-specific directory
                iteration_idx,
                original_prefix,
                cell_transformation_vector
            )

            if not generated_cif_paths:
                print(f"    No CIFs generated for sample {i+1}. Skipping relaxation.")
                iteration_results.append({
                    'params': individual_params,
                    'energy_per_atom': None, # Indicate failure
                    'relaxed_atoms': None,
                    'original_file': None
                })
                continue

            # Relax the generated structures
            # Assuming generate_displaced_supercells creates files in sample_output_dir/supercell_NxNxN
            # We need to find the actual folder where CIFs were written
            # The generate_displaced_supercells function now creates a structure like:
            # base_output_dir/iteration_idx/supercell_NxNxN/filename.cif
            # So, the folder to relax is base_output_dir/iteration_idx/supercell_NxNxN
            # Let's get the unique supercell folders from generated_cif_paths
            unique_supercell_folders = list(set(os.path.dirname(fpath) for fpath in generated_cif_paths))

            sample_relaxation_results = []
            for folder in unique_supercell_folders:
                folder_relaxation_results = relax_structures_in_folder(folder, calculator, args.engine, args.fmax)
                sample_relaxation_results.extend(folder_relaxation_results)

            if not sample_relaxation_results:
                print(f"    No structures successfully relaxed for sample {i+1}.")
                iteration_results.append({
                    'params': individual_params,
                    'energy_per_atom': None, # Indicate failure
                    'relaxed_atoms': None,
                    'original_file': None
                })
                continue

            # Find the lowest energy structure among the relaxed ones for this sample
            lowest_energy_sample = find_lowest_energy_structures(sample_relaxation_results, num_to_select=1)
            if lowest_energy_sample:
                best_result_for_sample = lowest_energy_sample[0]
                iteration_results.append({
                    'params': individual_params,
                    'energy_per_atom': best_result_for_sample['energy_per_atom'],
                    'relaxed_atoms': best_result_for_sample['relaxed_atoms'],
                    'original_file': best_result_for_sample['original_file'],
                    'num_atoms': best_result_for_sample.get('num_atoms', 'N/A'),  
                    'international_symbol': best_result_for_sample.get('international_symbol', 'N/A'),  
                    'crystal_system': best_result_for_sample.get('crystal_system', 'N/A')
                })
                print(f"    Sample {i+1} relaxed. Lowest energy: {best_result_for_sample['energy_per_atom']:.6f} eV/atom")
            else:
                print(f"    Could not find lowest energy for sample {i+1}.")
                iteration_results.append({
                    'params': individual_params,
                    'energy_per_atom': None, # Indicate failure
                    'relaxed_atoms': None,
                    'original_file': None
                })

        # Add this iteration's results to the master list for GA evolution in next iteration
        # Filter out failed relaxations (energy_per_atom is None) before adding to all_iterations_results
        valid_iteration_results = [r for r in iteration_results if r['energy_per_atom'] is not None]
        all_iterations_results.extend(valid_iteration_results)

        if not valid_iteration_results:
            print(f"No valid structures found in iteration {iteration_idx}. Stopping GA optimization.")
            break

        # NEW SNIPPET START: Create iteration-specific relaxation summary  
        iteration_summary_filepath = os.path.join(base_output_dir, f"iter_{iteration_idx}", "relaxation_summary_iter.txt")
          
        with open(iteration_summary_filepath, 'w') as f:    
            f.write(f"--- Relaxation Summary for Iteration {iteration_idx} ---\n")    
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")    
            f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'GA Params':<50}\n")    
            f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*50:<50}\n")    
    
            # Sort valid_iteration_results by energy_per_atom for this summary    
            sorted_iter_results = sorted(valid_iteration_results, key=lambda x: x['energy_per_atom'])    
    
            for result in sorted_iter_results:    
                num_atoms = result.get('num_atoms', 'N/A')  
                international_symbol = result.get('international_symbol', 'N/A')  
                crystal_system = result.get('crystal_system', 'N/A')  
                energy = result.get('energy_per_atom', 'FAIL')    
                params = result.get('params', ('N/A', 'N/A', 'N/A'))    
                params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:{params[2]}" if isinstance(params, tuple) else str(params)    
    
                energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)    
                f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {params_str:<50}\n")    
        print(f"Iteration {iteration_idx} relaxation summary saved to: {iteration_summary_filepath}")
        # NEW SNIPPET END

        # --- Prepare for Next Iteration: Find the soft mode of the overall best structure found so far ---
        # This is crucial for guiding the next GA generation with relevant phonon information.
        # Sort all valid results collected so far to find the absolute best structure
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
            primitive_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())

            # Run a minimal phonon check just to get the next guiding mode(s)
            check_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}_guidance_phonon_check")
            os.makedirs(check_dir, exist_ok=True)
            print(f"\nPerforming guidance phonon check on best structure from Iteration {iteration_idx} to find next soft modes...")
            next_softest_modes_info_list, _, _ = run_single_phonon_analysis(
                primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
                args.supercell_n, args.delta, args.fmax, check_dir, # Use args.supercell_n, delta, fmax from initial config
                prefix=f"guidance_check_iter_{iteration_idx}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return # Ensure we get top N modes
            )
            # Update the guiding atoms and soft mode list for the next loop
            current_primitive_atoms = primitive_atoms_for_next_iter.copy()
            current_softest_modes_info_list = next_softest_modes_info_list
            if not current_softest_modes_info_list:
                print(f"No soft modes found in best structure from iteration {iteration_idx}. Ending GA optimization.")
                break # Stop if no soft modes are found to continue guiding
        except Exception as e:
            print(f"Could not prepare for next iteration due to an error during phonon analysis: {e}")
            import traceback
            traceback.print_exc()
            break


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
        params = result.get('params', ('N/A', 'N/A', 'N/A'))
        print(f"  {idx}. {original_file} (Disp1: {params[0]:.3f}, Ratio21: {params[1]:.3f}, CellTrans: {params[2]}): {energy:.6f} eV/atom")


    print(f"\nSelected the top {len(final_top_structures)} structures from all iterations for final analysis:")
    for i, result in enumerate(final_top_structures):
        original_file_base = os.path.splitext(os.path.basename(result['original_file']))[0]
        params = result.get('params', ('N/A', 'N/A', 'N/A'))
        print(f"  {i+1}. {original_file_base} (Disp1: {params[0]:.3f}, Ratio21: {params[1]:.3f}, CellTrans: {params[2]}) (Energy: {result['energy_per_atom']:.6f} eV/atom)")

    for i, result in enumerate(final_top_structures):
        top_structure_relaxed_supercell = result['relaxed_atoms']
        original_file_base = os.path.splitext(os.path.basename(result['original_file']))[0]
        final_phonon_dir = os.path.join(base_output_dir, f"final_phonon_analysis_top_{i+1}")
        print(f"\n--- Running Final Phonon Analysis on Top Structure #{i+1} ({original_file_base}) ---")
        try:
            pmg_structure = AseAtomsAdaptor.get_structure(top_structure_relaxed_supercell)
            primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())
            run_single_phonon_analysis(
                primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units,
                args.supercell_n, args.delta, args.fmax, final_phonon_dir,
                prefix=f"final_{original_prefix}_top_{i+1}",
                phonon_path_npoints=phonon_path_npoints,
                phonon_dos_grid=phonon_dos_grid,
                traj_kT=default_traj_kT,
                num_modes_to_return=num_modes_to_return
            )
        except Exception as e:
            print(f"  Error during final phonon analysis for {original_file_base}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Soft Mode Iterative Optimization Complete ---")
    overall_summary_filepath = os.path.join(base_output_dir, "overall_relaxation_summary.txt")  
      
    print("\n--- Creating Overall Relaxation Summary (All Iterations) ---")    
    with open(overall_summary_filepath, 'w') as f:    
        f.write(f"--- Overall Relaxation Summary (All Iterations) ---\n")    
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")    
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy per Atom (eV/atom)':<25} {'GA Params':<50}\n")    
        f.write(f"{'-'*12:<12} {'-'*15:<15} {'-'*18:<18} {'-'*25:<25} {'-'*50:<50}\n")    
    
        # Sort all_iterations_results by energy_per_atom for this summary    
        sorted_overall_results = sorted(valid_overall_results, key=lambda x: x['energy_per_atom'])    
    
        for result in sorted_overall_results:    
            num_atoms = result.get('num_atoms', 'N/A')  
            international_symbol = result.get('international_symbol', 'N/A')  
            crystal_system = result.get('crystal_system', 'N/A')  
            energy = result.get('energy_per_atom', 'FAIL')    
            params = result.get('params', ('N/A', 'N/A', 'N/A'))    
            params_str = f"D1:{params[0]:.3f}, R21:{params[1]:.3f}, Cell:{params[2]}" if isinstance(params, tuple) else str(params)    
    
            energy_str = f"{energy:.6f}" if isinstance(energy, (int, float)) else str(energy)    
            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy_str:<25} {params_str:<50}\n")    
    print(f"Overall relaxation summary saved to: {overall_summary_filepath}")    
    
    # Print the overall summary to screen    
    with open(overall_summary_filepath, 'r') as f:    
        print("\n" + f.read())