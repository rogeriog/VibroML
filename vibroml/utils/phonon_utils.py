# utils/phonon_utils.py
import os
import numpy as np
from ase.phonons import Phonons
# Ensure these imports are correct based on your project structure
from .config import EV_TO_THZ_FACTOR, THZ_TO_CM_FACTOR, EV_TO_CM_FACTOR
import json
from math import pi, sqrt
from ase.io import read, write
from ase.atoms import Atoms # <--- ADD THIS IMPORT
import sys # NEW IMPORT  
import io  # NEW IMPORT  
import re  # NEW IMPORT
def run_phonon_calculation(atoms, calculator, supercell_n, delta, output_dir):
   """Sets up and runs the phonon calculation."""
   atoms.set_calculator(calculator) # Ensure calculator is set for phonon calculation

   print("\n### Structure for Phonon Calculation ###")
   print("   The phonon band structure will be calculated using the current structure.")
   print(f"   Using supercell size: ({supercell_n}, {supercell_n}, {supercell_n})")
   print(f"   Using displacement delta: {delta}")

   # Phonon calculator
   supercell = (supercell_n, supercell_n, supercell_n)
   ph = Phonons(atoms, calculator, supercell=supercell, delta=delta)
   ph.run()

   # Read forces and assemble the dynamical matrix
   ph.read(acoustic=True)
   ph.clean()

   print("Phonon calculation completed.")
   return ph

def get_phonon_results(ph, atoms, units, phonon_path_npoints=100, phonon_dos_grid=(40,40,40)): # NEW ARGS
   """Gets band structure and DOS results and converts units."""
   # Get the band path object
   path = atoms.cell.bandpath(npoints=phonon_path_npoints) # Use phonon_path_npoints

   # --- NEW: Capture stderr to suppress ASE warnings and summarize ---  
   old_stderr = sys.stderr  
   sys.stderr = captured_stderr = io.StringIO()


   try:  
       # Get band structure and DOS in default energy units (eV)  
       bs = ph.get_band_structure(path, verbose=False)  
       dos = ph.get_dos(kpts=phonon_dos_grid, verbose=False).sample_grid(npts=200, width=1e-4)  
   finally:  
       sys.stderr = old_stderr # Restore stderr immediately  
  
   # Process captured warnings  
   captured_warnings = captured_stderr.getvalue()  
   imaginary_freq_warnings = []  
   for line in captured_warnings.splitlines():  
       if "WARNING, " in line and "imaginary frequencies at q =" in line:  
           imaginary_freq_warnings.append(line.strip())  
  
   if imaginary_freq_warnings:  
       print("\n--- ASE Phonon Warnings Summary ---")  
       print(f"Detected {len(imaginary_freq_warnings)} instances of imaginary frequencies during band structure calculation.")  
       # Optionally, print the first few or unique warnings  
       unique_warnings = sorted(list(set(imaginary_freq_warnings)))  
       if unique_warnings:  
           print("Examples of warnings (first 3 unique):")  
           for i, warning in enumerate(unique_warnings[:3]):  
               print(f"  - {warning}")  
       print("-----------------------------------\n")  
   # --- END NEW BLOCK ---
  
   # Convert energies based on chosen units
   if units == "THz":
      bs_energies = bs.energies[0] / EV_TO_THZ_FACTOR # Access the first (and only) spin channel
      dos_energies = dos.get_energies() / EV_TO_THZ_FACTOR
      y_label = "Frequency (THz)"
   elif units == "cm-1":
      bs_energies = bs.energies[0] / EV_TO_THZ_FACTOR * THZ_TO_CM_FACTOR # Access the first (and only) spin channel
      dos_energies = dos.get_energies() / EV_TO_THZ_FACTOR * THZ_TO_CM_FACTOR
      y_label = "Frequency (cm⁻¹)"
   elif units == "eV":
      bs_energies = bs.energies[0] # Access the first (and only) spin channel
      dos_energies = dos.get_energies()
      y_label = "Energy (eV)"
   else:
      print(f"Warning: Unknown units '{units}'. Using eV.")
      bs_energies = bs.energies[0] # Access the first (and only) spin channel
      dos_energies = dos.get_energies()
      y_label = "Energy (eV)"


   # Correctly get k-point distances and labels for plotting
   all_k_point_distances, special_k_point_distances, special_k_point_labels = bs.get_labels()

   bsmin = np.min(bs_energies) if bs_energies.size > 0 else 0

   print(f'Most negative frequency: {bsmin:.4f} {units}')

   # Return bs and path objects as well for further analysis
   return bs, path, dos, bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, bsmin

def analyze_special_points_and_modes(ph, bs, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, traj_kT=1, num_modes_to_return=2):  
   """Analyzes frequencies at special points and identifies the softest modes.  
  
   Args:  
      ph (ase.phonons.Phonons): The Phonons object.  
      bs (ase.spectrum.band_structure.BandStructure): The BandStructure object.  
      path (ase.dft.kpoints.BandPath): The BandPath object.  
      bs_energies (np.ndarray): The phonon band energies (frequencies).  
      special_k_point_distances (list): Distances of special k-points.  
      special_k_point_labels (list): Labels of special k-points.  
      units (str): Units for frequencies (e.g., "THz", "cm-1").  
      output_dir (str): Directory to save output files.  
      traj_kT (float): Temperature for trajectory generation (default 0.1).  
      num_modes_to_return (int): The number of softest modes to return (default 2).  
   """  
   print("\n--- Analyzing Special K-points ---")  
  
   special_point_analysis = []  
   all_negative_modes_info = [] # To store all negative modes found at special points  
  
   special_point_coords = path.special_points # Dictionary mapping label to coordinate  
  
   for label, distance in zip(special_k_point_labels, special_k_point_distances):  
      coord = special_point_coords.get(label)  
      if coord is None:  
         print(f"Warning: Could not find coordinate for special point label '{label}'. Skipping.")  
         continue  
  
      # Find the index in path.kpts that matches this coordinate  
      kpt_index = None  
      for i, kpt in enumerate(path.kpts):  
         if np.allclose(kpt, coord):  
               kpt_index = i  
               break  
  
      if kpt_index is None:  
            print(f"Warning: Could not find k-point index for coordinate {coord} (label '{label}'). Skipping.")  
            continue  
  
      # Get frequencies at this k-point index  
      freqs_at_kpt = bs_energies[kpt_index]  
  
      # Iterate through all bands at this k-point to find negative frequencies  
      for band_idx, freq in enumerate(freqs_at_kpt):  
         if freq < 0: # Only consider negative frequencies  
            mode_info = {  
                  "label": label,  
                  "coordinate": [float(c) for c in coord], # Convert numpy array to list for JSON  
                  "frequency": float(freq),  
                  "band_index": int(band_idx),  
                  "kpoint_index_in_path": int(kpt_index) # Store index in path.kpts  
            }  
            all_negative_modes_info.append(mode_info)  
  
      # Also add the minimum frequency at this k-point to special_point_analysis (even if positive)  
      min_freq_at_kpt = np.min(freqs_at_kpt)  
      min_band_index_at_kpt = np.argmin(freqs_at_kpt)  
      special_point_analysis.append({  
         "label": label,  
         "coordinate": [float(c) for c in coord],  
         "min_frequency": float(min_freq_at_kpt),  
         "band_index": int(min_band_index_at_kpt)  
      })  
      print(f"   Special Point: {label} ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}) - Minimum Frequency: {min_freq_at_kpt:.4f} {units}")  
  
  
   # Sort all found negative modes by frequency (most negative first)  
   all_negative_modes_info.sort(key=lambda x: x['frequency'])  
  
   # Select the top N softest modes  
   top_n_softest_modes = []  
   processed_modes_keys = set() # To avoid duplicate modes (same k-point, same band)  
  
   for mode_info in all_negative_modes_info:  
      mode_key = (mode_info['kpoint_index_in_path'], mode_info['band_index'])  
      if mode_key not in processed_modes_keys:  
         top_n_softest_modes.append(mode_info)  
         processed_modes_keys.add(mode_key)  
         if len(top_n_softest_modes) >= num_modes_to_return:  
            break  
  
   # Save special point analysis (this includes all special points, not just negative modes)  
   analysis_filename = os.path.join(output_dir, "special_point_analysis.json")  
   with open(analysis_filename, 'w') as f:  
      json.dump(special_point_analysis, f, indent=4)  
   print(f"\nSpecial point analysis saved to {analysis_filename}")  
  
   # Report and get displacements for the top N softest modes  
   all_soft_modes_with_displacements = []  
   if top_n_softest_modes:  
      print(f"\n--- Top {len(top_n_softest_modes)} Softest Modes Analysis ---")  
      for i, softest_mode_info in enumerate(top_n_softest_modes):  
         print(f"\nMode {i+1}:")  
         print(f"   Label: {softest_mode_info['label']}")  
         print(f"   Coordinate: ({softest_mode_info['coordinate'][0]:.4f}, {softest_mode_info['coordinate'][1]:.4f}, {softest_mode_info['coordinate'][2]:.4f})")  
         print(f"   Frequency: {softest_mode_info['frequency']:.4f} {units}")  
         print(f"   Band Index: {softest_mode_info['band_index']}")  
  
         try:  
            q_c = np.array(softest_mode_info['coordinate'])  
            band_index = softest_mode_info['band_index']  
  
            omega_at_q, u_at_q = ph.band_structure([q_c], modes=True)  
            eigenvector = u_at_q[0, band_index].real  
            num_atoms_primitive = len(ph.atoms)  
            raw_displacements = eigenvector.reshape(num_atoms_primitive, 3)  
  
            softest_mode_info['raw_displacements'] = raw_displacements.tolist()  
            all_soft_modes_with_displacements.append(softest_mode_info)  
  
            max_disp_magnitude = np.max(np.linalg.norm(raw_displacements, axis=1))  
            if max_disp_magnitude > 1e-6:  
                  scaled_displacements_for_viz = raw_displacements / max_disp_magnitude * 0.5  
            else:  
                  scaled_displacements_for_viz = raw_displacements  
  
            print("Displacements for this mode (in primitive cell coordinates, scaled for visualization):")  
            primitive_atoms = ph.atoms  
            for j, (atom, disp) in enumerate(zip(primitive_atoms, scaled_displacements_for_viz)):  
                  print(f"   Atom {j+1} ({atom.symbol}): dx={disp[0]:.6f}, dy={disp[1]:.6f}, dz={disp[2]:.6f} Å")  
  
            displacements_filename = os.path.join(output_dir, f"softest_mode_{i+1}_{softest_mode_info['label']}_band{band_index}_displacements.txt")  
            with open(displacements_filename, 'w') as f:  
                  f.write(f"Softest mode {i+1} at special point {softest_mode_info['label']} ({softest_mode_info['coordinate']})\n")  
                  f.write(f"Frequency: {softest_mode_info['frequency']:.4f} {units}\n")  
                  f.write(f"Band Index: {softest_mode_info['band_index']}\n\n")  
                  f.write("Displacements (in primitive cell coordinates, scaled for visualization):\n")  
                  for j, (atom, disp) in enumerate(zip(primitive_atoms, scaled_displacements_for_viz)):  
                     f.write(f"Atom {j+1} ({atom.symbol}): dx={disp[0]:.6f}, dy={disp[1]:.6f}, dz={disp[2]:.6f} Å\n")  
            print(f"Displacements for mode {i+1} saved to {displacements_filename}")  
  
            original_ph_name = ph.name  
            ph.name = os.path.join(output_dir, f"softest_mode_{i+1}_{softest_mode_info['label']}_band{band_index}")  
            ph.write_modes(  
                  q_c,  
                  branches=[band_index],  
                  kT=traj_kT,  
                  repeat=(2, 2, 2),  
                  center=True  
            )  
            traj_filename_generated = f"{ph.name}.mode.{band_index}.traj"  
            print(f"Mode {i+1} animation (ASE .traj) saved to {traj_filename_generated}")  
            ph.name = original_ph_name  
  
            xyz_filename = os.path.join(output_dir, f"softest_mode_{i+1}_{softest_mode_info['label']}_band{band_index}.xyz")  
            try:  
                  frames = read(traj_filename_generated, index=':')  
                  write(xyz_filename, frames)  
                  print(f"Converted trajectory to XYZ: {xyz_filename}")  
            except Exception as e:  
                  print(f"Error converting trajectory to XYZ for mode {i+1}: {e}")  
                  import traceback  
                  traceback.print_exc()  
  
         except Exception as e:  
            print(f"Error retrieving displacements or writing trajectory for mode {i+1}: {e}")  
            import traceback  
            traceback.print_exc()  
  
   else:  
      print("\nNo negative frequencies found at special points.")  
  
   # Return the list of softest mode info dictionaries  
   return all_soft_modes_with_displacements
