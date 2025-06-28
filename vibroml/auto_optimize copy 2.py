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

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read, write


def analyze_and_resample(summary_file_path, num_new_points=60, num_best_points_to_sample_from=3):  
    """  
    Analyzes the relaxation summary, finds the best performing displacement/cell-scaling  
    parameters, and samples new points around them for the next iteration using Gaussian sampling.  
  
    Args:  
        summary_file_path (str): Path to the summary_displaced_supercell.txt file.  
        num_new_points (int): The total number of new points to sample.  
        num_best_points_to_sample_from (int): The number of top results to use as centers for sampling.  
  
    Returns:  
        tuple: A tuple containing (new_displacement_scales, new_cell_scale_factors).  
        Returns (None, None) if analysis fails.  
    """  
    if not os.path.exists(summary_file_path):  
        print(f"Warning: Summary file not found at {summary_file_path}. Cannot resample for next iteration.")  
        return None, None  
  
    data = []  
    try:  
        with open(summary_file_path, 'r') as f:  
            lines = f.readlines()  
  
        # Skip header lines (first two lines)  
        data_lines = lines[2:]  
  
        # Regex to capture the fields. This regex assumes:  
        # - Name: anything ending in .cif  
        # - Number_of_atoms: one or more digits  
        # - Final_energy_per_atom: a number (float) or 'FAIL'  
        # - Crystal_system: word characters or 'N/A'  
        # - International_symbol: word characters or 'N/A'  
        # It also accounts for potential multiple entries on one line by using a non-greedy match for Name  
        # and then looking for the next pattern.  
        # This regex needs to be robust to the concatenated lines you showed.  
        # Let's try to capture each block of data.  
        # The pattern seems to be: Name (ending .cif) + Number + Energy + Crystal + Symbol  
        # And then it repeats.  
        # We need to find all occurrences of this pattern on each line.  
        # Let's refine the regex to capture each record.  
        # It seems like the Name is the most distinct starting point.  
        # Name: ([\w\d\._-]+?\.cif)\s+  
        # Number_of_atoms: (\d+)\s+  
        # Final_energy_per_atom: (-?\d+\.\d+|FAIL)\s+  
        # Crystal_system: (\w+|N/A)\s+  
        # International_symbol: (\w+|N/A)  
  
        # This regex will try to find all matches on a single line  
        # It's crucial to make sure the regex is non-greedy where appropriate (e.g., for Name)  
        # and accounts for the spaces between columns.  
        # Based on your example, the columns are quite distinct.  
        # Let's assume a pattern like:  
        # FILENAME INT FLOAT STRING STRING  
        # We need to capture each of these groups.  
        # The `Name` field seems to be the key to separating records.  
        # Let's try to match each full record.  
        record_pattern = re.compile(  
            r'([\w\d\._-]+?\.cif)\s+'  # Name (non-greedy, ends with .cif)  
            r'(\d+)\s+'               # Number_of_atoms  
            r'(-?\d+\.\d+|FAIL)\s+'   # Final_energy_per_atom (float or FAIL)  
            r'(\w+|N/A)\s+'           # Crystal_system (word chars or N/A)  
            r'(\w+|N/A)'              # International_symbol (word chars or N/A)  
        )  
  
        for line in data_lines:  
            # Find all non-overlapping matches of the record pattern in the line  
            matches = record_pattern.finditer(line)  
            for match in matches:  
                name, num_atoms, energy_str, crystal_sys, int_symbol = match.groups()  
                # Convert energy to numeric, handling 'FAIL'  
                final_energy = float(energy_str) if energy_str != 'FAIL' else np.nan  
                data.append({  
                    'Name': name,  
                    'Number_of_atoms': int(num_atoms),  
                    'Final_energy_per_atom': final_energy,  
                    'Crystal_system': crystal_sys,  
                    'International_symbol': int_symbol  
                })  
  
        df = pd.DataFrame(data)  
  
        # Drop rows where Final_energy_per_atom is NaN (from 'FAIL' or other parsing issues)  
        df.dropna(subset=['Final_energy_per_atom'], inplace=True)  
  
        # Ensure 'Final_energy_per_atom' is numeric  
        df['Final_energy_per_atom'] = pd.to_numeric(df['Final_energy_per_atom'])  
  
    except Exception as e:  
        print(f"Error reading or parsing summary file {summary_file_path}: {e}")  
        import traceback  
        traceback.print_exc() # Print full traceback for more detailed debugging  
        return None, None  
  
    if df.empty:  
        print("Warning: No valid data found in summary file after parsing. Cannot resample for next iteration.")  
        return None, None  
  
    def extract_params(name):  
        disp_match = re.search(r'_d([\d\.]+)_c', name)  
        cell_match = re.search(r'_c(m|p)(\d+)\.cif', name)  
        displacement = float(disp_match.group(1)) if disp_match else None  
        cell_scaling = None  
        if cell_match:          
            sign = -1 if cell_match.group(1) == 'm' else 1          
            value = float(cell_match.group(2)) / 100.0  # Ensure it's a float division          
            cell_scaling = sign * value  
        return displacement, cell_scaling  
  
    df[['displacement', 'cell_scaling']] = df['Name'].apply(lambda x: pd.Series(extract_params(x)))  
  
    df.dropna(subset=['displacement', 'cell_scaling'], inplace=True)  
  
    if df.empty:  
        print("Warning: Could not extract displacement/scaling parameters from any file names. Cannot resample.")  
        return None, None  
  
    df_sorted = df.sort_values(by='Final_energy_per_atom').reset_index(drop=True)  
  
    best_points = df_sorted.head(num_best_points_to_sample_from)  
    if best_points.empty:  
        print("Warning: No best points found after filtering. Cannot resample for next iteration.")  
        return None, None  
  
    print("\n--- Resampling for Next Iteration ---")  
    print(f"Using top {len(best_points)} configurations as sampling centers:")  
    print(best_points[['Name', 'Final_energy_per_atom', 'displacement', 'cell_scaling']])  
  
    new_displacements = []  
    new_cell_scalings = []  
    points_per_center = num_new_points // len(best_points)  
  
    # Calculate the range of displacement and cell_scaling among the best points  
    # This helps in setting a dynamic standard deviation for sampling  
    disp_values = best_points['displacement'].values  
    cell_values = best_points['cell_scaling'].values  
  
    print(f"DEBUG: Best points displacement values: {disp_values}")  
    print(f"DEBUG: Best points cell scaling values: {cell_values}")  
  
    disp_range = np.max(disp_values) - np.min(disp_values) if len(disp_values) > 1 else 0.0  
    cell_range = np.max(cell_values) - np.min(cell_values) if len(cell_values) > 1 else 0.0  
  
    print(f"DEBUG: Displacement range among best points: {disp_range}")  
    print(f"DEBUG: Cell scaling range among best points: {cell_range}")  
  
    # Define minimum standard deviations to ensure some exploration even if best points are identical  
    min_std_disp = 0.05  
    min_std_cell = 0.005  
  
    for i, row in best_points.iterrows():  
        mean_disp = row['displacement']  
        mean_cell = row['cell_scaling']  
        # Set standard deviation based on the range of best points, with a minimum floor  
        std_disp = max(disp_range * 0.3, min_std_disp) # Sample within 30% of the best range, or min_std  
        std_cell = max(cell_range * 0.3, min_std_cell) # Sample within 30% of the best range, or min_std  
        print(f"DEBUG: For best point {i+1}: Mean Disp={mean_disp}, Mean Cell={mean_cell}")  
        print(f"DEBUG: Calculated Std Disp={std_disp}, Std Cell={std_cell}")  
        sampled_disps = np.random.normal(loc=mean_disp, scale=std_disp, size=points_per_center)  
        new_displacements.extend(np.abs(sampled_disps)) # Displacement should always be positive  
        sampled_cells = np.random.normal(loc=mean_cell, scale=std_cell, size=points_per_center)  
        new_cell_scalings.extend(sampled_cells)  
        print(f"DEBUG: Sampled Displacements for point {i+1}: {sampled_disps}")  
        print(f"DEBUG: Sampled Cell Scalings for point {i+1}: {sampled_cells}")  
  
  
    # Round to a reasonable number of decimal places for practical use and uniqueness  
    final_displacements = sorted(list(set([round(d, 4) for d in new_displacements])))  
    final_cell_scalings = sorted(list(set([round(c, 4) for c in new_cell_scalings])))  
  
    # Ensure 0.0 is always an option for cell scaling if not already present  
    if 0.0 not in final_cell_scalings:  
        final_cell_scalings.append(0.0)  
        final_cell_scalings.sort()  
  
    print(f"DEBUG: Final unique generated displacement scales: {final_displacements}")  
    print(f"DEBUG: Final unique generated cell scale factors: {final_cell_scalings}")  
  
    print(f"Generated {len(final_displacements)} new unique displacement scales for next iteration.")  
    print(f"Generated {len(final_cell_scalings)} new unique cell scale factors for next iteration.")  
    return final_displacements, final_cell_scalings


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
                                     phonon_path_npoints, phonon_dos_grid, default_traj_kT, cell_scale_factors, num_modes_to_return):
    """
    Runs the parameter sweep auto-optimization loop.
    If a soft mode is found in the best configuration, it triggers the soft mode optimization.
    """
    print("Running in parameter sweep auto mode to find optimal settings...")
    best_negative_frequency = -float('inf')
    best_settings = {}
    best_softest_mode_info = None
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
        softest_mode_info_current, neg_freq_at_special_point, time_taken = run_single_phonon_analysis(
            relaxed_initial_atoms.copy(), calculator, args.engine, args.units, sc_n, d, fm, current_output_dir, prefix=original_prefix,
            phonon_path_npoints=phonon_path_npoints,
            phonon_dos_grid=phonon_dos_grid,
            traj_kT=default_traj_kT
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
                best_softest_mode_info = softest_mode_info_current
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

    if best_negative_frequency < threshold_in_current_units and best_softest_mode_info is not None and best_relaxed_atoms is not None:
        print(f"\nSoft mode detected ({best_negative_frequency:.4f} {args.units}) after parameter sweep. Initiating iterative soft mode optimization...")
        run_soft_mode_optimization(
            args,
            output_dir,
            best_relaxed_atoms,
            best_softest_mode_info,
            max_iterations=soft_mode_max_iterations,
            displacement_scales=soft_mode_displacement_scales,
            num_top_structures_to_analyze=soft_mode_num_top_structures_to_analyze,
            negative_phonon_threshold_thz=negative_phonon_threshold_thz,
            phonon_path_npoints=phonon_path_npoints,
            phonon_dos_grid=phonon_dos_grid,
            default_traj_kT=default_traj_kT,
            cell_scale_factors=cell_scale_factors,
            num_modes_to_return=num_modes_to_return
        )
    else:
        print("\nNo significant soft mode detected after parameter sweep, or structure is stable enough. Skipping iterative soft mode optimization.")

    return best_settings


def run_soft_mode_optimization(args, base_output_dir, initial_atoms_for_soft_mode_analysis, initial_softest_mode_info, max_iterations,
                               displacement_scales, num_top_structures_to_analyze, negative_phonon_threshold_thz,
                               phonon_path_npoints, phonon_dos_grid, default_traj_kT, cell_scale_factors, num_modes_to_return):
    """
    Runs an iterative workflow to find low-energy structures and then performs a final
    phonon analysis on the best candidates found across all iterations.
    """
    print("\n--- Running Soft Mode Iterative Optimization ---")

    # --- 1. Initial Setup ---
    current_atoms = initial_atoms_for_soft_mode_analysis.copy()
    current_softest_mode_info = initial_softest_mode_info
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

    # Determine supercell variants based on symmetry
    cell = current_atoms.cell
    lengths = np.linalg.norm(cell, axis=1)
    angles = np.array([
        np.arccos(np.dot(cell[1], cell[2]) / (lengths[1] * lengths[2])),
        np.arccos(np.dot(cell[0], cell[2]) / (lengths[0] * lengths[2])),
        np.arccos(np.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]))
    ])
    is_cubic = (np.allclose(angles, np.pi / 2, atol=1e-3) and np.allclose(lengths, lengths[0], rtol=1e-3))
    is_tetragonal = (np.allclose(angles, np.pi / 2, atol=1e-3) and any(np.allclose(lengths[i], lengths[j], rtol=1e-3) for i, j in [(0, 1), (1, 2), (0, 2)]))
    if is_cubic:
        supercell_variants = [(2,1,1)]# [(1, 1, 1), (2, 2, 2), (2, 2, 1), (2, 1, 1)]
    elif is_tetragonal:
        supercell_variants = [(1, 1, 1), (2, 2, 2), (2, 2, 1), (2, 1, 2), (2, 1, 1)]
    else:
        supercell_variants = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]

    current_displacement_scales = list(displacement_scales)
    current_cell_scale_factors = list(cell_scale_factors)
    all_iterations_results = []

    # --- 2. Main Iterative Loop (Displacement and Relaxation) ---
    for iteration_idx in range(1, max_iterations + 1):
        print(f"\n### Starting Relaxation Iteration {iteration_idx} ###")
        # Check if the guiding soft mode is valid
        if current_softest_mode_info is None or 'raw_displacements' not in current_softest_mode_info:
            print(f"No softest mode information to guide iteration {iteration_idx}. Stopping.")
            break
        # Check if the guiding soft mode is already stable enough to stop
        if current_softest_mode_info['frequency'] >= threshold_in_current_units:
            print(f"\nGuiding soft mode frequency ({current_softest_mode_info['frequency']:.4f} {args.units}) is stable. Ending iterative search.")
            break
        print(f"Using soft mode at {current_softest_mode_info.get('label', 'unknown')} ({current_softest_mode_info['frequency']:.4f} {args.units}) to guide displacements.")
        # Generate, Relax, and Find Lowest Energy Structures for this iteration
        generated_cif_paths = generate_displaced_supercells(current_atoms.copy(), current_softest_mode_info, supercell_variants, current_displacement_scales, base_output_dir, iteration_idx, original_prefix, current_cell_scale_factors)
        if not generated_cif_paths:
            break
        all_relaxation_results = []
        supercell_folders = {os.path.dirname(fpath): [] for fpath in generated_cif_paths}
        for fpath in generated_cif_paths:
            supercell_folders[os.path.dirname(fpath)].append(fpath)
        for folder in supercell_folders:
            folder_relaxation_results = relax_structures_in_folder(folder, calculator, args.engine, args.fmax)
            all_relaxation_results.extend(folder_relaxation_results)
        if not all_relaxation_results:
            break
        lowest_energy_this_iter = find_lowest_energy_structures(all_relaxation_results, num_to_select=num_top_structures_to_analyze)
        if not lowest_energy_this_iter:
            break
        # Add this iteration's best results to the master list
        all_iterations_results.extend(lowest_energy_this_iter)
        # Create summary file and resample for the next iteration
        soft_mode_label = current_softest_mode_info.get('label', 'unknown')
        current_mode_folder_path = os.path.join(base_output_dir, f"soft_mode_{iteration_idx}_{soft_mode_label}")
        if os.path.exists(current_mode_folder_path):
            create_displaced_supercell_summary(current_mode_folder_path)
            if iteration_idx < max_iterations:
                summary_file = os.path.join(current_mode_folder_path, "summary_displaced_supercell.txt")
                new_displacements, new_cell_factors = analyze_and_resample(summary_file)
                print(f"New displacement scales: {new_displacements}")
                print(f"New cell scale factors: {new_cell_factors}")
                if new_displacements and new_cell_factors:
                    print("Updating displacement and cell scales for the next iteration.")
                    current_displacement_scales = new_displacements
                    current_cell_scale_factors = new_cell_factors
        # --- Prepare for Next Iteration ---
        # We must find the soft mode of this iteration's BEST structure to guide the NEXT iteration.
        # best_candidate_for_next_iter = lowest_energy_this_iter[0]
        # try:
        #     relaxed_supercell = best_candidate_for_next_iter['relaxed_atoms']
        #     pmg_structure = AseAtomsAdaptor.get_structure(relaxed_supercell)
        #     primitive_atoms_for_next_iter = AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(pmg_structure).get_primitive_standard_structure())
        #     # Run a minimal phonon check just to get the next guiding mode
        #     check_dir = os.path.join(base_output_dir, f"iter_{iteration_idx}_guidance_check")
        #     print(f"\nPerforming guidance check on best structure from Iteration {iteration_idx}...")
        #     next_softest_mode_info, _, _ = run_single_phonon_analysis(
        #         primitive_atoms_for_next_iter.copy(), calculator, args.engine, args.units,
        #         args.supercell_n, args.delta, args.fmax, check_dir,
        #         prefix=f"guidance_check_iter_{iteration_idx}"
        #     )
        #     # Update the guiding atoms and soft mode for the next loop
        #     current_atoms = primitive_atoms_for_next_iter.copy()
        #     current_softest_mode_info = next_softest_mode_info
        # except Exception as e:
        #     print(f"Could not prepare for next iteration due to an error: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     break

    # --- 3. Final Phonon Analysis on Overall Best Structures ---
    print("\n\n--- All relaxation iterations complete. ---")
    print("--- Analyzing overall best structures for final phonon properties. ---")

    if not all_iterations_results:
        print("No structures were successfully relaxed to perform final analysis on.")
        return

    # Filter the master list to ensure all entries are valid dictionaries with the required key before sorting.
    # This prevents a crash if a failed relaxation result (None or a malformed dict) is present.
    valid_overall_results = [
        r for r in all_iterations_results
        if isinstance(r, dict) and 'energy_per_atom' in r and isinstance(r['energy_per_atom'], (int, float))
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
        print(f"  {idx}. {original_file}: {energy:.6f} eV/atom")

    print(f"\nSelected the top {len(final_top_structures)} structures from all iterations for final analysis:")
    for i, result in enumerate(final_top_structures):
        original_file_base = os.path.splitext(os.path.basename(result['original_file']))[0]
        print(f"  {i+1}. {original_file_base} (Energy: {result['energy_per_atom']:.6f} eV/atom)")

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