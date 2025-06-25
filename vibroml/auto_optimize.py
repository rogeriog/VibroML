# auto_optimize.py (Revised)
import os
import sys
import json
import time
import numpy as np
# Updated imports to reflect the 'utils' package
from utils.structure_utils import load_structure, initialize_calculator, print_initial_structure_info, generate_displaced_supercells
from utils.relaxation_utils import relax_structure, relax_structures_in_folder, find_lowest_energy_structures
from utils.phonon_utils import run_phonon_calculation, get_phonon_results, analyze_special_points_and_modes
from utils.plotting_utils import plot_phonon_results, save_raw_data
from utils.utils import clean_phonon_cache

def run_single_phonon_analysis(atoms, calculator, engine, units, supercell_n, delta, fmax, output_dir, prefix="phonon_run", phonon_path_npoints=100, phonon_dos_grid=(40,40,40), traj_kT=1.0):
   """
   Runs a single phonon calculation step (calculate, plot, save) on a given atoms object.
   This function is a core component used by both parameter sweep and soft mode optimization.
   """
   start_time = time.time()

   # Ensure output directory exists
   os.makedirs(output_dir, exist_ok=True)
   print(f"\n--- Running Single Phonon Analysis in: {output_dir} ---")

   # --- Step 1: Run Phonon Calculation ---
   print("\n--- Step 1: Running Phonon Calculation ---")
   ph = run_phonon_calculation(atoms, calculator, supercell_n, delta, output_dir)
   if ph is None:
      print("Error during phonon calculation setup.")
      return None, None, None # Error during phonon calculation setup

   # --- Step 2: Get and Process Phonon Results ---
   print("\n--- Step 2: Getting and Processing Phonon Results ---")
   bs, path, dos, bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, bsmin = get_phonon_results(ph, atoms, units, phonon_path_npoints, phonon_dos_grid)
   # --- Step 3: Save Raw Data ---
   print("\n--- Step 3: Saving Raw Data ---")
   save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_n, delta, fmax, output_dir)

   # --- Step 4: Plot Results ---
   print("\n--- Step 4: Plotting Results ---")
   struct_formula = atoms.get_chemical_formula()
   plot_phonon_results(bs_energies, dos, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, struct_formula, prefix, supercell_n, delta, fmax, output_dir)

   # --- Step 5: Analyze Special Points and Softest Mode ---
   print("\n--- Step 5: Analyzing Special Points and Softest Mode ---")
   softest_mode_info = analyze_special_points_and_modes(  
      ph, bs, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, traj_kT=traj_kT  
   )
   # --- Step 6: Save comprehensive run details to a readable file ---
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
       f.write(f"Most Negative Frequency: {bsmin:.4f} {units}\n")  
       f.write(f"Time Taken for Phonon Analysis: {time_taken:.2f} seconds\n\n")  
  
       # Add energy per atom (assuming it's available from a previous relaxation step or can be calculated)  
       try:  
           energy = atoms.get_potential_energy()  
           energy_per_atom = energy / len(atoms)  
           f.write(f"Energy of Structure: {energy:.6f} eV\n")  
           f.write(f"Energy per Atom: {energy_per_atom:.6f} eV/atom\n\n")  
       except Exception as e:  
           f.write(f"Could not retrieve energy for this structure: {e}\n\n")  
  
       f.write("Path of K-points:\n")  
       for i, (dist, label) in enumerate(zip(special_k_point_distances, special_k_point_labels)):  
           f.write(f"  {label}: {dist:.4f}\n")  
       f.write("\n")  
  
       # Attempt to read space group from the symmetry analysis file  
       symmetry_file_path = os.path.join(output_dir, "relaxed_symmetry_analysis.txt") # Assuming relaxed structure was analyzed  
       if not os.path.exists(symmetry_file_path):  
           # If not found, try initial symmetry analysis file  
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

   # Return softest_mode_info and the most negative frequency found in the band structure
   return softest_mode_info, bsmin, time_taken


def run_parameter_sweep_optimization(args, output_dir, supercell_ns, deltas, fmax_values, negative_phonon_threshold_thz, soft_mode_max_iterations, soft_mode_displacement_scales, soft_mode_num_top_structures_to_analyze):
    """
    Runs the parameter sweep auto-optimization loop.
    If a soft mode is found in the best configuration, it triggers the soft mode optimization.
    """
    print("Running in parameter sweep auto mode to find optimal settings...")
    best_negative_frequency = -float('inf') # We want to find the LEAST negative (closest to zero or positive)
    best_settings = {}
    best_softest_mode_info = None # To store softest mode info for the best run
    best_relaxed_atoms = None # To store the relaxed atoms for the best run


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
            
    # Load and potentially relax the initial structure once
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

        # Call the unified phonon analysis function
        softest_mode_info_current, neg_freq_at_special_point, time_taken = run_single_phonon_analysis(  
        relaxed_initial_atoms.copy(), calculator, args.engine, args.units, sc_n, d, fm, current_output_dir, prefix=original_prefix,  
        phonon_path_npoints=args.phonon_path_npoints, # Pass from args (which gets from defaults)  
        phonon_dos_grid=args.phonon_dos_grid,       # Pass from args (which gets from defaults)  
        traj_kT=args.default_traj_kT                # Pass from args (which gets from defaults)  
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
                best_softest_mode_info = softest_mode_info_current # Store the softest mode info
                best_relaxed_atoms = relaxed_initial_atoms.copy() # Store the atoms object for the best run

                
                improvement = best_negative_frequency - previous_best_negative_frequency
                

                # Note: We are looking for improvement, so a *less* negative frequency is better (closer to zero)
                # If the improvement is too small (i.e., current best is not significantly better than previous best)
                if improvement < abs(threshold_in_current_units * 0.5): # Use a fraction of the threshold for early stopping
                    print(f"Improvement in negative frequency ({improvement:.4f} {args.units}) is less than {abs(threshold_in_current_units * 0.5):.4f} {args.units}. Stopping parameter sweep.")
                    break

            previous_best_negative_frequency = best_negative_frequency

    print("\n--- Parameter sweep auto-optimization complete ---")
    print(f"Most negative frequency at a special point found: {best_negative_frequency:.4f} {args.units}")
    print(f"Optimal settings: Supercell N = {best_settings.get('supercell_n')}, Delta = {best_settings.get('delta')}, Fmax = {best_settings.get('fmax')}")

    with open(os.path.join(output_dir, "auto_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    if best_negative_frequency < threshold_in_current_units and best_softest_mode_info is not None and best_relaxed_atoms is not None:
        print(f"\nSoft mode detected ({best_negative_frequency:.4f} {args.units}) after parameter sweep. Initiating iterative soft mode optimization...")
        # Pass the best relaxed atoms and its softest mode info to the iterative function
        run_soft_mode_optimization(
            args, # Pass all original arguments
            output_dir,
            best_relaxed_atoms, # The structure that had the soft mode
            best_softest_mode_info # The softest mode info from that structure
        )
    else:
        print("\nNo significant soft mode detected after parameter sweep, or structure is stable enough. Skipping iterative soft mode optimization.")

    return best_settings


def run_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_mode_info, is_primitive_run=False, max_iterations=1, displacement_scales=None, num_top_structures_to_analyze=3, negative_phonon_threshold_thz=-0.1):  
    """  
    Runs the iterative soft mode displacement and relaxation workflow.  
  
    Args:  
        args (argparse.Namespace): Command line arguments.  
        base_output_dir (str): The main output directory for the entire run.  
        initial_atoms_for_soft_mode_analysis (ase.atoms.Atoms): The starting structure  
                                                                 (e.g., relaxed initial structure or best from sweep).  
        initial_softest_mode_info (dict): The softest mode information from the initial phonon analysis  
                                          of `initial_atoms_for_soft_mode_analysis`.  
        is_primitive_run (bool): If True, indicates this is a run specifically for primitive cell displacements.  
        max_iterations (int): Maximum number of iterations for soft mode optimization.  
        displacement_scales (list): List of scaling factors for the raw displacements.  
        num_top_structures_to_analyze (int): Number of lowest energy structures to analyze phonons for.  
        negative_phonon_threshold_thz (float): Threshold for considering a phonon frequency as "negative" (unstable) in THz.  
    """
    print("\n--- Running Soft Mode Iterative Optimization ---")

    # Define supercell variants based on cell symmetry
    current_atoms = initial_atoms_for_soft_mode_analysis.copy()
    cell = current_atoms.cell
    lengths = np.linalg.norm(cell, axis=1)
    angles = np.array([
        np.arccos(np.dot(cell[1], cell[2] )/(lengths[1]*lengths[2])),
        np.arccos(np.dot(cell[0], cell[2] )/(lengths[0]*lengths[2])),
        np.arccos(np.dot(cell[0], cell[1] )/(lengths[0]*lengths[1]))
    ])
    
    # Check if cell is cubic (all angles ~90° and all lengths equal)
    is_cubic = (np.allclose(angles, np.pi/2, atol=1e-3) and 
                np.allclose(lengths, lengths[0], rtol=1e-3))
    
    # Check if cell is tetragonal (all angles 90° and two equal lengths)
    is_tetragonal = (np.allclose(angles, np.pi/2, atol=1e-3) and
                    any(np.allclose(lengths[i], lengths[j], rtol=1e-3)
                        for i, j in [(0,1), (1,2), (0,2)]))
    
    if is_cubic:
        # For cubic, we only need unique combinations
        supercell_variants = [(1,1,1), (2,2,2), (2,2,1), (2,1,1)]
    elif is_tetragonal:
        # For tetragonal, we need more combinations but can still reduce
        supercell_variants = [(1,1,1), (2,2,2), (2,2,1), (2,1,2), (2,1,1)]
    else:
        # For lower symmetry, use all combinations
        supercell_variants = [(1,1,1), (2,1,1), (1,2,1), (1,1,2), 
                            (2,2,1), (2,1,2), (1,2,2), 
                            (2,2,2)]    # Exponentially increasing displacement scales
    
    if displacement_scales is None: # Fallback if not provided  
        displacement_scales = np.exp(np.linspace(np.log(10), np.log(100), 5)).tolist()  
        displacement_scales = [round(s, 3) for s in displacement_scales]

    print(f"Using displacement scales: {displacement_scales}")


    current_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_mode_info = initial_softest_mode_info

    calculator = initialize_calculator(args.engine)
    if calculator is None:
        print("Failed to initialize calculator for soft mode optimization. Exiting.")
        return

    original_prefix = os.path.splitext(os.path.basename(args.cif))[0]

    # Convert threshold from THz to current units
    threshold_in_current_units = negative_phonon_threshold_thz # Use from arguments  
    if args.units == "cm-1":  
        threshold_in_current_units *= 33.35641  
    elif args.units == "eV":  
        threshold_in_current_units *= 4.135667696e-3

    for iteration_idx in range(1, max_iterations + 1):
        print(f"\n### Starting Soft Mode Iteration {iteration_idx} ###")

        if current_softest_mode_info is None or 'raw_displacements' not in current_softest_mode_info:
            print(f"No softest mode information available for iteration {iteration_idx}. Stopping.")
            break

        # Check if the frequency of the current softest mode is already above threshold
        # This check is crucial for early stopping if the previous iteration already stabilized it.
        if current_softest_mode_info['frequency'] >= threshold_in_current_units:
            print(f"\nSoftest mode frequency ({current_softest_mode_info['frequency']:.4f} {args.units}) is already above the threshold ({threshold_in_current_units:.4f} {args.units}). Structure is considered stable enough. Stopping iterative optimization.")
            break

        print(f"Softest mode identified at {current_softest_mode_info.get('label', 'unknown')} with frequency {current_softest_mode_info['frequency']:.4f} {args.units}")


        # --- Step 1: Generate Displaced Supercells ---
        print(f"\n--- Iteration {iteration_idx}: Generating Displaced Supercells ---")
        generated_cif_paths = generate_displaced_supercells(
            current_atoms.copy(), # Use the current (primitive) structure as the base
            current_softest_mode_info,
            supercell_variants,
            displacement_scales,
            base_output_dir, # Use the main output directory
            iteration_idx,
            original_prefix
        )

        if not generated_cif_paths:
            print(f"No displaced structures generated in iteration {iteration_idx}. Stopping.")
            break

        # --- Step 2: Relax Displaced Structures ---
        print(f"\n--- Iteration {iteration_idx}: Relaxing Displaced Structures ---")
        all_relaxation_results = []
        supercell_folders = {}
        for fpath in generated_cif_paths:
            folder = os.path.dirname(fpath)
            if folder not in supercell_folders:
                supercell_folders[folder] = []
            supercell_folders[folder].append(fpath)

        for folder in supercell_folders:
             folder_relaxation_results = relax_structures_in_folder(folder, calculator, args.engine, args.fmax)
             all_relaxation_results.extend(folder_relaxation_results)

        if not all_relaxation_results:
            print(f"No structures were successfully relaxed in iteration {iteration_idx}. Stopping.")
            break

        # --- Step 3: Find Lowest Energy Structures ---
        print(f"\n--- Iteration {iteration_idx}: Finding Lowest Energy Structures ---")
        lowest_energy_structures = find_lowest_energy_structures(all_relaxation_results, num_to_select=num_top_structures_to_analyze)

        if not lowest_energy_structures:
            print(f"Could not find lowest energy structures in iteration {iteration_idx}. Stopping.")
            break

        # --- Step 4: Run Full Phonon Analysis on Top Structures ---
        print(f"\n--- Iteration {iteration_idx}: Running Full Phonon Analysis on Top {len(lowest_energy_structures)} Structures ---")
        best_neg_freq_in_top_structures = -float('inf') # Track the best frequency found in this iteration
        best_atoms_for_next_iter = None
        best_softest_mode_info_for_next_iter = None
        stop_iterative_loop = False

        for i, result in enumerate(lowest_energy_structures):
            top_structure_relaxed_supercell_atoms = result['relaxed_atoms']
            original_file_base = os.path.splitext(os.path.basename(result['original_file']))[0]

            # Create a specific output directory for the phonon analysis of this top structure  
            top_structure_phonon_dir = os.path.join(base_output_dir, f"soft_mode_iter_{iteration_idx}_top_structure_{i+1}_phonon_analysis")  
            os.makedirs(top_structure_phonon_dir, exist_ok=True)  
  
            print(f"\nAnalyzing phonon for top structure {i+1} ({original_file_base})...")  
  
            # Find the primitive cell of the relaxed supercell structure for phonon analysis  
            try:  
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  
                from pymatgen.io.ase import AseAtomsAdaptor 
                from ase.io import read, write
 
  
                pmg_structure = AseAtomsAdaptor.get_structure(top_structure_relaxed_supercell_atoms)  
                analyzer = SpacegroupAnalyzer(pmg_structure)  
                primitive_pmg = analyzer.get_primitive_standard_structure()  
                primitive_atoms_for_phonon = AseAtomsAdaptor.get_atoms(primitive_pmg)  
                print(f"  Found primitive cell for relaxed structure ({len(primitive_atoms_for_phonon)} atoms).")  
  
                # --- NEW: Save the primitive_atoms_for_phonon to its phonon analysis folder ---  
                primitive_cif_path = os.path.join(top_structure_phonon_dir, f"{original_prefix}_iter{iteration_idx}_top{i+1}_primitive.cif")  
                write(primitive_cif_path, primitive_atoms_for_phonon)  
                print(f"  Saved primitive structure for phonon analysis to: {primitive_cif_path}")

                # Run phonon analysis on this primitive cell
                # Use args.supercell_n and args.delta for the phonon calculation on this primitive cell
                softest_mode_info_top, most_negative_freq_top, time_taken_top = run_single_phonon_analysis(  
                    primitive_atoms_for_phonon.copy(), calculator, args.engine, args.units, args.supercell_n, args.delta, args.fmax, top_structure_phonon_dir, prefix=f"{original_prefix}_iter{iteration_idx}_top{i+1}_final_phonon",  
                    phonon_path_npoints=args.phonon_path_npoints, # Pass from args (which gets from defaults)  
                    phonon_dos_grid=args.phonon_dos_grid,       # Pass from args (which gets from defaults)  
                    traj_kT=args.default_traj_kT                # Pass from args (which gets from defaults)  
                )

                if most_negative_freq_top is not None:
                    # Save the final phonon analysis results with the requested naming convention
                    final_output_filename_base = f"{original_prefix}_soft_mode_{iteration_idx}_{current_softest_mode_info.get('label', 'unknown')}_supercell_{original_file_base.split('supercell_')[1].split('/')[0]}_sufix"
                    # The plot_phonon_results and save_raw_data functions already handle saving.
                    # We just need to ensure the output directory is correct.
                    # The prefix in run_single_phonon_analysis handles the naming.

                    # Check if this structure is stable enough
                    if most_negative_freq_top >= threshold_in_current_units:
                        print(f"  Structure {original_file_base} is stable enough ({most_negative_freq_top:.4f} {args.units}).")
                        stop_iterative_loop = True # Found a stable structure, stop the main loop

                    # Keep track of the best (least negative) frequency among the top structures in this iteration
                    if most_negative_freq_top > best_neg_freq_in_top_structures:
                        best_neg_freq_in_top_structures = most_negative_freq_top
                        best_atoms_for_next_iter = primitive_atoms_for_phonon.copy() # Store the primitive atoms of the best structure
                        best_softest_mode_info_for_next_iter = softest_mode_info_top # Store its softest mode info

            except Exception as e:
                print(f"  Error finding primitive cell or running phonon analysis for {original_file_base}: {e}")
                import traceback
                traceback.print_exc()


        # --- Step 5: Check Stopping Condition and Prepare for Next Iteration ---
        if stop_iterative_loop:
            print("\nStable structure found among the top candidates. Stopping iterative optimization.")
            break # Exit the main iteration loop

        if iteration_idx < max_iterations:
            # If not stopping, prepare for the next iteration
            if best_atoms_for_next_iter is not None and best_softest_mode_info_for_next_iter is not None:
                print(f"\nIteration {iteration_idx} did not yield a stable structure.")
                print(f"Proceeding to Iteration {iteration_idx + 1} using the softest mode from the best structure found in this iteration (frequency: {best_neg_freq_in_top_structures:.4f} {args.units}).")
                current_atoms = best_atoms_for_next_iter.copy() # Use the primitive cell of the best structure as the starting point for the next iteration
                current_softest_mode_info = best_softest_mode_info_for_next_iter # Use the softest mode from this best structure
            else:
                print(f"\nCould not identify a best structure or its softest mode for the next iteration. Stopping.")
                break # Stop if we can't continue

        else:
            print(f"\nMaximum number of iterations ({max_iterations}) reached. Stopping iterative optimization.")

    print("\n--- Soft Mode Iterative Optimization Complete ---")