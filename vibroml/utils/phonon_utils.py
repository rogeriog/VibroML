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

def get_phonon_results(ph, atoms, units):
   """Gets band structure and DOS results and converts units."""
   # Get the band path object
   path = atoms.cell.bandpath(npoints=100) # Use default path or specify if needed

   # Get band structure and DOS in default energy units (eV)
   bs = ph.get_band_structure(path)
   dos = ph.get_dos(kpts=(40, 40, 40)).sample_grid(npts=200, width=1e-4)

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

def analyze_special_points_and_modes(ph, bs, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, traj_kT=1):
   """Analyzes frequencies at special points and identifies the softest mode.

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
   """
   print("\n--- Analyzing Special K-points ---")

   special_point_analysis = []
   most_negative_freq = float('inf')
   softest_mode_info = None

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

      # Find the minimum frequency and its band index at this k-point
      min_freq_at_kpt = np.min(freqs_at_kpt)
      min_band_index_at_kpt = np.argmin(freqs_at_kpt)

      print(f"   Special Point: {label} ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}) - Minimum Frequency: {min_freq_at_kpt:.4f} {units}")

      special_point_analysis.append({
         "label": label,
         "coordinate": [float(c) for c in coord], # Convert numpy array to list for JSON
         "min_frequency": float(min_freq_at_kpt),
         "band_index": int(min_band_index_at_kpt)
      })

      # Check if this is the most negative frequency found so far
      if min_freq_at_kpt < most_negative_freq:
         most_negative_freq = min_freq_at_kpt
         softest_mode_info = {
               "label": label,
               "coordinate": [float(c) for c in coord],
               "frequency": float(min_freq_at_kpt),
               "band_index": int(min_band_index_at_kpt),
               "kpoint_index_in_path": int(kpt_index) # Store index in path.kpts
         }

   # Save special point analysis
   analysis_filename = os.path.join(output_dir, "special_point_analysis.json")
   with open(analysis_filename, 'w') as f:
      json.dump(special_point_analysis, f, indent=4)
   print(f"\nSpecial point analysis saved to {analysis_filename}")

   # Report the most negative special point and get displacements
   if softest_mode_info:
      print("\n--- Softest Mode Analysis ---")
      print(f"Most negative frequency found at a special point:")
      print(f"   Label: {softest_mode_info['label']}")
      print(f"   Coordinate: ({softest_mode_info['coordinate'][0]:.4f}, {softest_mode_info['coordinate'][1]:.4f}, {softest_mode_info['coordinate'][2]:.4f})")
      print(f"   Frequency: {softest_mode_info['frequency']:.4f} {units}")
      print(f"   Band Index: {softest_mode_info['band_index']}")

      try:
         # q_c is the k-point coordinate for the softest mode
         q_c = np.array(softest_mode_info['coordinate'])
         band_index = softest_mode_info['band_index']

         # Call ph.band_structure to get frequencies and eigenvectors for this specific k-point
         # The result is (omega_l, u_l) where omega_l is (1, num_branches) and u_l is (1, num_branches, num_atoms_primitive * 3)
         omega_at_q, u_at_q = ph.band_structure([q_c], modes=True)

         # Extract the eigenvector for the specific band_index
         # u_at_q[0] gives (num_branches, num_atoms_primitive * 3)
         # u_at_q[0, band_index] gives the 1D eigenvector for that mode
         eigenvector = u_at_q[0, band_index].real # Take the real part for displacements

         # Reshape the eigenvector into (num_atoms, 3) for displacements
         num_atoms_primitive = len(ph.atoms) # Number of atoms in the primitive cell used by Phonons object
         raw_displacements = eigenvector.reshape(num_atoms_primitive, 3)

         # Store the raw displacements in softest_mode_info
         softest_mode_info['raw_displacements'] = raw_displacements.tolist() # Convert to list for JSON

         # Normalize displacements for better visualization if needed (e.g., to max 0.5 Å)
         max_disp_magnitude = np.max(np.linalg.norm(raw_displacements, axis=1))
         if max_disp_magnitude > 1e-6: # Avoid division by zero
               scaled_displacements_for_viz = raw_displacements / max_disp_magnitude * 0.5 # Scale to a reasonable amplitude, e.g., 0.5 Å
         else:
               scaled_displacements_for_viz = raw_displacements # No scaling if magnitude is zero

         print("\nDisplacements for the softest mode (in primitive cell coordinates, scaled for visualization):")
         primitive_atoms = ph.atoms # This is the primitive cell used by the Phonons object
         for i, (atom, disp) in enumerate(zip(primitive_atoms, scaled_displacements_for_viz)):
               print(f"   Atom {i+1} ({atom.symbol}): dx={disp[0]:.6f}, dy={disp[1]:.6f}, dz={disp[2]:.6f} Å")

         # Save displacements to a file
         displacements_filename = os.path.join(output_dir, "softest_mode_displacements.txt")
         with open(displacements_filename, 'w') as f:
               f.write(f"Softest mode at special point {softest_mode_info['label']} ({softest_mode_info['coordinate']})\n")
               f.write(f"Frequency: {softest_mode_info['frequency']:.4f} {units}\n")
               f.write(f"Band Index: {softest_mode_info['band_index']}\n\n")
               f.write("Displacements (in primitive cell coordinates, scaled for visualization):\n")
               primitive_atoms = ph.atoms
               for i, (atom, disp) in enumerate(zip(primitive_atoms, scaled_displacements_for_viz)):
                  f.write(f"Atom {i+1} ({atom.symbol}): dx={disp[0]:.6f}, dy={disp[1]:.6f}, dz={disp[2]:.6f} Å\n")
         print(f"Softest mode displacements saved to {displacements_filename}")

         # --- Generate .traj file ---
         original_ph_name = ph.name
         ph.name = os.path.join(output_dir, f"softest_mode_{softest_mode_info['label']}_band{band_index}")
         ph.write_modes(
               q_c,
               branches=[band_index],
               kT=traj_kT,
               repeat=(2, 2, 2),
               center=True
         )
         traj_filename_generated = f"{ph.name}.mode.{band_index}.traj"
         print(f"Softest mode animation (ASE .traj) saved to {traj_filename_generated}")
         ph.name = original_ph_name # Restore original ph.name

         # --- NEW: Convert the generated .traj file to .xyz ---
         xyz_filename = os.path.join(output_dir, f"softest_mode_{softest_mode_info['label']}_band{band_index}.xyz")
         try:
               # Read the trajectory from the .traj file
               frames = read(traj_filename_generated, index=':') # Read all frames
               # Write to XYZ format
               write(xyz_filename, frames)
               print(f"Converted trajectory to XYZ: {xyz_filename}")
         except Exception as e:
               print(f"Error converting trajectory to XYZ: {e}")
               import traceback
               traceback.print_exc()


      except Exception as e:
         print(f"Error retrieving softest mode displacements or writing trajectory: {e}")
         import traceback
         traceback.print_exc()

   else:
      print("\nNo special points found or analyzed.")

   # Return the softest mode info dictionary
   return softest_mode_info
