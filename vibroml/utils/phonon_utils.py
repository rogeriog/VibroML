# utils/phonon_utils.py
import os
import sys
import shutil
import io
import time
import json
import numpy as np
from math import pi, sqrt
from ase.phonons import Phonons
from ase.io import read, write
from ase.atoms import Atoms
from ase.build import make_supercell
from ase.dft.kpoints import monkhorst_pack
# --- START FIX: Handle older ASE versions missing ase.spectrum.dos ---
try:
    from ase.spectrum.dos import DOS
except ImportError:
    # Fallback implementation for older ASE versions
    class DOS:
        def __init__(self, energies, weights=None, width=0.1, npts=200):
            """
            Simple Gaussian smearing DOS implementation to replace ase.spectrum.dos.DOS
            """
            self.energies = np.asarray(energies).flatten()
            if weights is None:
                self.weights = np.ones_like(self.energies)
            else:
                self.weights = np.asarray(weights).flatten()
            self.width = width
            
            # Determine energy grid range (min - 3sigma to max + 3sigma)
            min_e = self.energies.min() - 3.0 * width
            max_e = self.energies.max() + 3.0 * width
            self.energy_grid = np.linspace(min_e, max_e, npts)
            
        def get_energies(self):
            return self.energy_grid
            
        def get_weights(self):
            # Vectorized Gaussian smearing
            # delta shape: (npts, n_energies)
            delta = self.energy_grid[:, None] - self.energies[None, :]
            # Gaussian formula: (1/(sigma*sqrt(2pi))) * exp(-0.5 * (x/sigma)^2)
            prefactor = 1.0 / (self.width * np.sqrt(2 * np.pi))
            gaussians = prefactor * np.exp(-0.5 * (delta / self.width)**2)
            # Sum weighted gaussians
            return np.dot(gaussians, self.weights)
# --- END FIX ---

# Ensure these imports are correct based on your project structure
from .config import EV_TO_THZ_FACTOR, THZ_TO_CM_FACTOR, EV_TO_CM_FACTOR
from .plotting_utils import plot_phonon_results, plot_pdos
from .utils import save_raw_data
from .genetic_algorithm import GeneticAlgorithm
from .structure_utils import estimate_commensurate_supercell_size
def calculate_and_save_pdos(ph, dos_grid, width, npts, output_dir, prefix, units, conversion_divisor, struct_formula):
   """
   Calculates Projected Density of States (PDOS) for phonons and saves to JSON.
   
   Args:
      ph (Phonons): ASE Phonons object
      dos_grid (tuple): grid for q-point sampling (e.g. (40,40,40))
      width (float): Gaussian smearing width in target units
      npts (int): Number of energy points
      output_dir (str): Output directory
      prefix (str): File prefix
      units (str): Unit label (e.g. 'THz')
      conversion_divisor (float): Factor to divide eV by to get target units
      struct_formula (str): Chemical formula of the structure
   """
   print("\n--- Calculating Projected DOS (PDOS) ---")
   start_time = time.time()
   
   # 1. Generate Q-points for the full Brillouin zone
   qpoints = monkhorst_pack(dos_grid)
   
   # 2. Get frequencies and eigenvectors at these q-points
   # This step can be time consuming as it requires diagonalization at every q-point
   print(f"  Diagonalizing dynamical matrix at {len(qpoints)} q-points...")
   frequencies_ev, modes = ph.band_structure(qpoints, modes=True, verbose=False)
   
   # Convert frequencies to the requested unit (e.g., THz)
   frequencies = frequencies_ev / conversion_divisor
   
   # 3. Calculate weights: |e_i|^2 for each atom i
   # modes shape: (n_q, n_bands, n_atoms, 3) (complex)
   # weights shape: (n_q, n_bands, n_atoms) (real)
   # We sum the squared magnitudes of the x, y, z components
   weights = np.linalg.norm(modes, axis=3)**2 
   
   # 4. Generate DOS curves
   atom_pdos_list = []
   frequency_points = None
   
   # We iterate over atoms to create a DOS for each
   for i, atom in enumerate(ph.atoms):
      # Extract weights for this specific atom across all q-points and bands
      # Shape becomes (n_q, n_bands) which ASE DOS accepts
      atom_weights = weights[:, :, i]
      
      # Create DOS object to handle smearing
      # Note: We use the converted frequencies directly
      dos_object = DOS(frequencies, weights=atom_weights, width=width, npts=npts)
      
      # Store the energy grid (frequency points) from the first iteration
      if frequency_points is None:
         frequency_points = dos_object.get_energies().tolist()
         
      atom_pdos_list.append({
         "index": i,
         "symbol": atom.symbol,
         "pdos": dos_object.get_weights().tolist()
      })
   
   # 5. Save structure to JSON
   pdos_data = {
      "units": units,
      "smearing_width": width,
      "frequency_points": frequency_points,
      "atoms": atom_pdos_list
   }
   
   pdos_filename = os.path.join(output_dir, f"{prefix}_pdos.json")
   try:
      with open(pdos_filename, 'w') as f:
         json.dump(pdos_data, f) # No indent to keep file size smaller, or use indent=2
      print(f"  PDOS data saved to: {pdos_filename}")
   except Exception as e:
      print(f"  Error saving PDOS data: {e}")
      
   print(f"  PDOS calculation took {time.time() - start_time:.2f} s")
   # 6. Plot PDOS
   try:
      plot_pdos(
            atom_pdos_list=atom_pdos_list,
            frequency_points=frequency_points,
            units=units,
            output_dir=output_dir,
            prefix=prefix,
            struct_formula=struct_formula
      )
   except Exception as e:
      print(f"  Error plotting PDOS: {e}")
      import traceback
      traceback.print_exc()
      
   print(f"  PDOS calculation took {time.time() - start_time:.2f} s")
def run_single_phonon_analysis(atoms, calculator, engine, units, supercell_dims, delta, fmax, output_dir, prefix="phonon_run", phonon_path_npoints=100, phonon_dos_grid=(40,40,40), traj_kT=1.0, num_modes_to_return=2, ph_obj_for_specific_mode=None, q_point_for_specific_mode=None, band_idx_for_specific_mode=None, displacement_magnitude=1.0, preloaded_eigenmode_data=None, final_structures_dir=None, negative_phonon_threshold=None, save_yaml=False, compute_pdos=False):
   """
   Runs a single phonon calculation step (calculate, plot, save) on a given atoms object.

   Args:
      atoms (ase.Atoms): The ASE Atoms object representing the structure.
      calculator (ase.calculators.calculator.Calculator): The ASE calculator to use.
      engine (str): The name of the DFT engine (e.g., "VASP", "QuantumEspresso").
      units (str): Units for frequencies (e.g., "THz", "cm-1").
      supercell_n (int): Supercell size (N,N,N).
      delta (float): Displacement delta for phonon calculation.
      fmax (float): Float: Maximum force for relaxation convergence.
      output_dir (str): Directory to save output files.
      prefix (str): Prefix for output filenames (default "phonon_run").
      phonon_path_npoints (int): Number of points along the phonon path (default 100).
      phonon_dos_grid (tuple): Grid for DOS calculation (default (40,40,40)).
      traj_kT (float): Temperature for trajectory generation (default 0.1).
      num_modes_to_return (int): The number of softest modes to return (default 2).
      ph_obj_for_specific_mode (ase.phonons.Phonons, optional): Pre-calculated Phonons object for specific mode analysis.
      q_point_for_specific_mode (np.array, optional): User-defined q-point for specific mode analysis.
      band_idx_for_specific_mode (int, optional): User-defined mode index for specific mode analysis.
      preloaded_eigenmode_data (dict, optional): Pre-loaded eigenmode data from band.yaml file.
                                                 If provided, skips phonon calculation and uses this data directly.
      negative_phonon_threshold (float, optional): Threshold for soft mode detection. If provided,
                                                   when no modes below this threshold are found,
                                                   the function will select highest frequency modes instead.

   Returns:
      tuple: A tuple containing:
         - list: A list of dictionaries, each containing information about a softest mode,
                  including its raw displacements. Returns an empty list if no soft modes found.
         - float: The most negative frequency found (bsmin).
         - float: The total time taken for the analysis.
         - dict: Comprehensive tracking data for GA mode replacement containing:
           {
             'soft_modes': [list of all soft modes below threshold],
             'highest_freq_modes': [list of highest frequency modes at special k-points],
             'lowest_freq_modes': [list of lowest frequency modes at special k-points, excluding Gamma]
           }
   """
   start_time = time.time()

   os.makedirs(output_dir, exist_ok=True)
   print(f"\n--- Running Single Phonon Analysis in: {output_dir} ---")

   # Check if we have preloaded eigenmode data from band.yaml
   if preloaded_eigenmode_data is not None:
      print("\n--- Using Preloaded Eigenmode Data from band.yaml ---")
      print("Skipping phonon calculation and using provided eigenmode data.")

      # Extract data from preloaded eigenmode
      frequency = preloaded_eigenmode_data['frequency']
      eigenvector = preloaded_eigenmode_data['eigenvector']
      q_point = preloaded_eigenmode_data['q_point']
      band_idx = preloaded_eigenmode_data['band_idx']

      # Convert frequency from THz (band.yaml format) to the requested units
      if units == "THz":
         converted_frequency = frequency
      elif units == "cm-1":
         converted_frequency = frequency * THZ_TO_CM_FACTOR
      elif units == "eV":
         converted_frequency = frequency * EV_TO_THZ_FACTOR
      else:
         print(f"Warning: Unknown units '{units}'. Using THz.")
         converted_frequency = frequency
         units = "THz"

      # Generate displaced supercell using the preloaded eigenmode
      print(f"\n--- Generating Displaced Supercell from Preloaded Eigenmode ---")
      print(f"Q-point: {q_point}")
      print(f"Band index: {band_idx}")
      print(f"Frequency: {converted_frequency:.4f} {units}")

      # Create a mock ph object for supercell generation (we only need atoms)
      class MockPhonons:
         def __init__(self, atoms):
            self.atoms = atoms

      mock_ph = MockPhonons(atoms)

      # Generate the displaced supercell
      # Note: max_atoms_per_supercell is not available in this context, so we pass None (no limit)
      displaced_supercell = generate_and_visualize_specific_mode_supercell_from_eigenmode(
         atoms, eigenvector, q_point, band_idx, displacement_magnitude, output_dir, prefix, converted_frequency, units,
         max_atoms_per_supercell=None
      )

      # Create a summary of the preloaded mode
      mode_info = {
         'label': 'preloaded_from_yaml',
         'coordinate': q_point.tolist(),
         'frequency': converted_frequency,
         'band_index': band_idx,
         'raw_displacements': eigenvector.tolist()
      }

      # Save summary
      summary_filename = os.path.join(output_dir, f"{prefix}_preloaded_eigenmode_summary.txt")
      with open(summary_filename, 'w') as f:
         f.write(f"--- Preloaded Eigenmode Summary ---\n")
         f.write(f"Source: {preloaded_eigenmode_data.get('source', 'band.yaml file')}\n")
         f.write(f"Structure Formula: {atoms.get_chemical_formula()}\n")
         f.write(f"Total Number of Atoms: {len(atoms)}\n")
         f.write(f"Units: {units}\n")
         f.write(f"Q-point: ({q_point[0]:.4f}, {q_point[1]:.4f}, {q_point[2]:.4f})\n")
         f.write(f"Band Index: {band_idx}\n")
         f.write(f"Frequency: {converted_frequency:.4f} {units}\n")
         f.write(f"Displacement Magnitude: {displacement_magnitude:.3f} Å\n")

         try:
            energy = atoms.get_potential_energy()
            energy_per_atom = energy / len(atoms)
            f.write(f"Energy of Structure: {energy:.6f} eV\n")
            f.write(f"Energy per Atom: {energy_per_atom:.6f} eV/atom\n")
         except Exception as e:
            f.write(f"Could not retrieve energy for this structure: {e}\n")

      print(f"Preloaded eigenmode summary saved to: {summary_filename}")

      time_taken = time.time() - start_time
      print(f'Total time taken for preloaded eigenmode processing: {time_taken:.2f} s')

      # Return the mode info and frequency
      return [mode_info], converted_frequency, time_taken

   print("\n--- Step 1: Running Phonon Calculation ---")
   ph = run_phonon_calculation(atoms, calculator, supercell_dims, delta, output_dir)
   if ph is None:
      print("Error during phonon calculation setup.")
      return [], None, None # Return empty list for modes
   
   print("\n--- Step 2: Getting and Processing Phonon Results ---")
   bs, path, dos, bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, discontinuities, y_label, bsmin = get_phonon_results(ph, atoms, units, phonon_path_npoints, phonon_dos_grid)

   # Determine conversion divisor based on units
   # We divide eV by this number to get the target unit
   if units == "THz":
      conversion_divisor = EV_TO_THZ_FACTOR
   elif units == "cm-1":
      # eV -> cm-1: eV * (THZ_TO_CM / EV_TO_THZ) = eV * EV_TO_CM
      # So divisor = 1 / EV_TO_CM = EV_TO_THZ / THZ_TO_CM
      conversion_divisor = EV_TO_THZ_FACTOR / THZ_TO_CM_FACTOR
   elif units == "eV":
      conversion_divisor = 1.0
   else:
      conversion_divisor = 1.0
   struct_formula = atoms.get_chemical_formula()
   if compute_pdos:
      # Calculate and save PDOS 
      calculate_and_save_pdos(
         ph=ph, 
         dos_grid=phonon_dos_grid, 
         width=1e-4,  # Standard width matching get_phonon_results
         npts=200,    # Standard npts matching get_phonon_results
         output_dir=output_dir, 
         prefix=prefix, 
         units=units, 
         conversion_divisor=conversion_divisor,
         struct_formula=struct_formula
      )
   else:
       print("--- Skipping PDOS Calculation (use --compute-pdos to enable) ---")
   print("\n--- Step 3: Saving YAML and Raw Data ---")
   if save_yaml:
       save_phonopy_band_yaml(ph, path, bs, all_k_point_distances, output_dir, prefix)
       print("   YAML file saved.")
   else:
       print("   YAML file saving skipped (--save-yaml not provided).")
   dos_data_combined = np.column_stack([dos_energies, dos.get_weights()])
   dos_filename = os.path.join(output_dir, f"{prefix}_dos_curve.txt")
   np.savetxt(dos_filename, dos_data_combined, header=f"Frequency({units})  DOS_Intensity")
   print(f"    Saved full DOS curve to: {dos_filename}")
   
   save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_dims, delta, fmax, output_dir)

   print("\n--- Step 4: Plotting Results ---")
   struct_formula = atoms.get_chemical_formula()
   plot_phonon_results(bs_energies, dos, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, discontinuities, y_label, struct_formula, prefix, supercell_dims, delta, fmax, output_dir)

   print(f"\n--- Step 5: Analyzing Special Points and Top {num_modes_to_return} Softest Modes ---")

   softest_modes_info_list, tracked_k_points_data = analyze_special_points_and_modes(
      ph, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, prefix, traj_kT=traj_kT, num_modes_to_return=num_modes_to_return,
      target_q_point=q_point_for_specific_mode, target_band_idx=band_idx_for_specific_mode, displacement_magnitude=displacement_magnitude, negative_phonon_threshold=negative_phonon_threshold
   )
   time_taken = time.time() - start_time

   summary_filename = os.path.join(output_dir, f"{prefix}_phonon_run_summary.txt")
   with open(summary_filename, 'w') as f:
      f.write(f"--- Phonon Run Summary ---\n")
      f.write(f"Structure Formula: {atoms.get_chemical_formula()}\n")
      f.write(f"Total Number of Atoms: {len(atoms)}\n")
      f.write(f"Engine Used: {engine}\n")
      f.write(f"Units: {units}\n")
      supercell_str = f"({supercell_dims[0]}, {supercell_dims[1]}, {supercell_dims[2]})" if isinstance(supercell_dims, (list, tuple)) else f"({supercell_dims}, {supercell_dims}, {supercell_dims})"  
      f.write(f"Supercell Size: {supercell_str}\n")
      f.write(f"Displacement Delta: {delta}\n")
      f.write(f"Fmax for Relaxation: {fmax}\n")
      f.write(f"Most Negative Frequency (overall): {bsmin:.4f} {units}\n")
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
               f.write(f"   Coordinate (q-point): {mode_info.get('coordinate', 'N/A')}\n") # Changed label to q-point
               f.write(f"   Frequency: {mode_info.get('frequency', 'N/A'):.4f} {units}\n")
               f.write(f"   Band Index: {mode_info.get('band_index', 'N/A')}\n")
               f.write("\n")
      else:
         f.write("No negative frequencies found at special k-points.\n\n")

      # Print user-defined mode if available
      if q_point_for_specific_mode is not None and band_idx_for_specific_mode is not None:
         try:
               # Get frequencies at the user-defined q-point
               # First, find the index of the user's q-point in the band path
               user_q_point_in_path_idx = None
               for i, kpt in enumerate(path.kpts):
                  if np.allclose(kpt, q_point_for_specific_mode):
                     user_q_point_in_path_idx = i
                     break
               
               if user_q_point_in_path_idx is not None:
                  freqs_at_user_q = bs_energies[user_q_point_in_path_idx]
                  if 0 <= band_idx_for_specific_mode < len(freqs_at_user_q):
                     user_mode_freq = freqs_at_user_q[band_idx_for_specific_mode]
                     f.write(f"--- User-Defined Mode Analysis ---\n")
                     f.write(f"User-Defined Q-point: ({q_point_for_specific_mode[0]:.4f}, {q_point_for_specific_mode[1]:.4f}, {q_point_for_specific_mode[2]:.4f})\n")
                     f.write(f"User-Defined Mode Index: {band_idx_for_specific_mode}\n")
                     f.write(f"Frequency of User-Defined Mode: {user_mode_freq:.4f} {units}\n\n")
                  else:
                     f.write(f"--- User-Defined Mode Analysis ---\n")
                     f.write(f"Warning: User-defined mode index {band_idx_for_specific_mode} is out of bounds for q-point {q_point_for_specific_mode}.\n\n")
               else:
                  f.write(f"--- User-Defined Mode Analysis ---\n")
                  f.write(f"Warning: User-defined q-point {q_point_for_specific_mode} not found in the calculated band path.\n\n")
         except Exception as e:
               f.write(f"--- User-Defined Mode Analysis ---\n")
               f.write(f"Error analyzing user-defined mode: {e}\n\n")


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

   # Copy structure files to phonon analysis directory and add frequency to filenames if final_structures_dir is provided
   if final_structures_dir and os.path.exists(final_structures_dir):
       print(f"\n--- Processing Structure Files ---")
       # Extract analysis identifier from prefix (e.g., "final_BaHfSe3_4.530_top_1_energy_m6p0976" -> "top_1_energy_m6p0976")
       analysis_id_parts = prefix.split('_')
       analysis_id = None
       if len(analysis_id_parts) >= 3:
           # Look for patterns like "top_1_energy_..." or "unique_1_energy_..."
           for i, part in enumerate(analysis_id_parts):
               if part in ['top', 'unique'] and i + 2 < len(analysis_id_parts) and analysis_id_parts[i + 2] == 'energy':
                   analysis_id = '_'.join(analysis_id_parts[i:])
                   break

       if analysis_id:
           # Add frequency information to structure filenames
           if softest_modes_info_list:
               softest_frequency = softest_modes_info_list[0]['frequency']
               print(f"  Adding frequency information to structure filenames...")
               add_frequency_to_structure_filenames(final_structures_dir, analysis_id, softest_frequency, units)

           # Copy structure files to phonon analysis directory
           print(f"  Copying structure files to phonon analysis directory...")
           copy_structure_files_to_phonon_analysis_dir(output_dir, final_structures_dir, analysis_id)
       else:
           print(f"  Could not extract analysis_id from prefix: {prefix}")

   # Return the list of softest modes, the overall minimum frequency, time taken, and tracked k-points data
   return softest_modes_info_list, bsmin, time_taken, tracked_k_points_data


def run_phonon_analysis_with_uq(
    atoms,
    engine,
    model_name,
    units,
    supercell_dims,
    delta,
    fmax,
    output_dir,
    prefix="phonon_run",
    phonon_path_npoints=100,
    uq_plot_modes=("ensemble", "mean+std"),
):
    """
    Compute phonon band structure with uncertainty quantification (UQ) using
    uqphonon + i-PI with the LLPR committee ensemble embedded in PET-MAD models.

    Only available with ``--engine pet`` and UQ-capable models (pet-mad-xs v1.5.0
    and pet-mad-s v1.5.0).  The function saves one PNG per requested plot mode and
    a plain-text UQ summary, then returns a tuple compatible with
    :func:`run_single_phonon_analysis`.

    Parameters
    ----------
    atoms : ase.Atoms
        Pre-relaxed structure (primitive cell recommended).
    engine : str
        Must be ``"pet"`` — raises a warning and returns ``None`` otherwise.
    model_name : str
        PET model string, e.g. ``"pet-mad-xs"`` or ``"pet-mad-s"``.
    units : str
        Frequency units: ``"THz"`` (default), ``"cm-1"``, or ``"eV"``.
        The plot always uses ``"THz"`` or ``"cm-1"``; ``"eV"`` falls back to ``"THz"``.
    supercell_dims : int, str, or tuple
        Supercell dimensions accepted by :func:`parse_supercell_dimensions`.
    delta : float
        Displacement amplitude for finite differences (Å).
    fmax : float
        Force convergence threshold used during relaxation (logged only).
    output_dir : str
        Directory where plots and the summary file are written.
    prefix : str
        Filename prefix for output files.
    phonon_path_npoints : int
        Number of q-points per path segment.
    uq_plot_modes : iterable of str
        Plot modes to generate — any subset of ``("ensemble", "mean+std", "spectral")``.

    Returns
    -------
    tuple : (softest_modes_info_list, bsmin, time_taken, tracked_k_points_data)
        Compatible with :func:`run_single_phonon_analysis`.
        ``softest_modes_info_list`` and ``tracked_k_points_data`` are always empty
        (soft-mode analysis is not performed in UQ mode).
    """
    import time
    import tempfile

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if engine != "pet":
        print(
            f"Warning: --phonon-uq is only supported with --engine pet, not '{engine}'. "
            "Skipping UQ calculation."
        )
        return None

    # ── imports (guarded so VibroML still loads when packages absent) ──────────
    try:
        import upet
        from upet._version import UPET_UQ_SUPPORTED_MODELS
    except ImportError as exc:
        print(f"Error: upet is not installed – cannot run phonon UQ ({exc})")
        return None

    try:
        from uqphonon import PhononEnsemble
    except ImportError as exc:
        print(
            f"Error: uqphonon is not installed – cannot run phonon UQ ({exc})\n"
            "Install with: pip install git+https://github.com/ppegolo/uqphonon.git@v0.1.0"
        )
        return None

    try:
        from uqphonon._plot import plot_bands as _uq_plot_bands
    except ImportError as exc:
        print(f"Error: uqphonon plotting unavailable ({exc})")
        return None

    # ── parse model name into base + size ──────────────────────────────────────
    # e.g. "pet-mad-xs" → ("pet-mad", "xs");  "pet-omat-s" → ("pet-omat", "s")
    if "-" not in model_name:
        print(f"Error: Cannot parse model_name '{model_name}' into base+size for UQ.")
        return None
    model_base, model_size = model_name.rsplit("-", 1)

    # ── check UQ support ───────────────────────────────────────────────────────
    # UPET_UQ_SUPPORTED_MODELS entries look like "pet-mad-xs-v1.5.0"
    uq_base_names = set()
    for m in UPET_UQ_SUPPORTED_MODELS:
        parts = m.rsplit("-v", 1)
        if len(parts) == 2:
            uq_base_names.add(parts[0])  # e.g. "pet-mad-xs"
    if model_name not in uq_base_names:
        print(
            f"Warning: '{model_name}' may not support UQ ensemble predictions.\n"
            f"  UQ-capable models: {sorted(uq_base_names)}\n"
            "  Continuing – i-PI run may fail or return single-member output."
        )

    # ── export TorchScript model ───────────────────────────────────────────────
    model_pt = os.path.join(output_dir, f"{model_base.replace('-','_')}_{model_size}.pt")
    if not os.path.exists(model_pt):
        print(f"\n--- Exporting PET model to TorchScript: {model_pt} ---")
        try:
            upet.save_upet(model=model_base, size=model_size, output=model_pt)
        except Exception as exc:
            print(f"Error exporting model: {exc}")
            return None
    else:
        print(f"   Reusing cached TorchScript model: {model_pt}")

    # ── device ────────────────────────────────────────────────────────────────
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    # ── supercell matrix ──────────────────────────────────────────────────────
    from .utils import parse_supercell_dimensions
    sc = parse_supercell_dimensions(supercell_dims)
    supercell_matrix = np.diag(sc)

    # ── run PhononEnsemble ────────────────────────────────────────────────────
    print(f"\n--- Running PhononEnsemble UQ: {atoms.get_chemical_formula()} ---")
    print(f"   Model: {model_name}  Supercell: {sc}  Delta: {delta} Å  Device: {device}")

    ensemble = PhononEnsemble(
        atoms=atoms,
        supercell_matrix=supercell_matrix,
        model=model_pt,
        device=device,
        primitive_matrix=np.eye(3),
    )
    ensemble.compute_displacements(distance=delta)

    workdir = os.path.join(output_dir, f"_uqphonon_{prefix}")
    print(f"   i-PI working directory: {workdir}")
    try:
        ensemble.run_forces(workdir=workdir)
    except Exception as exc:
        print(f"Error running i-PI forces: {exc}")
        return None

    n_disp, n_ens, n_at, _ = ensemble.forces.shape
    print(f"   {n_ens} ensemble members | {n_disp} displacements | {n_at} atoms/supercell")

    ensemble.compute_bands(npoints=phonon_path_npoints)

    # ── bsmin (from mean phonon, in requested units) ───────────────────────────
    mean_ph = ensemble.mean_phonon
    all_freqs_thz = np.concatenate(
        [seg.ravel() for seg in mean_ph.band_structure.frequencies]
    )
    # Conversion from phonopy-native THz to the requested units
    from .config import THZ_TO_CM_FACTOR, EV_TO_THZ_FACTOR
    if units == "cm-1":
        freq_scale = THZ_TO_CM_FACTOR
    elif units == "eV":
        freq_scale = 1.0 / EV_TO_THZ_FACTOR
    else:
        freq_scale = 1.0
    bsmin = float(all_freqs_thz.min() * freq_scale)

    # ── plot unit (uqphonon only accepts "THz", "cm-1", "meV") ────────────────
    uq_unit = units if units in ("THz", "cm-1") else "THz"

    # ── generate plots ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    formula = atoms.get_chemical_formula()
    color_map = {"ensemble": "tab:blue", "mean+std": "tab:blue", "spectral": None}

    for mode in uq_plot_modes:
        fig, ax = plt.subplots(figsize=(10, 6))
        kwargs = {}
        if mode != "spectral":
            kwargs["color"] = color_map.get(mode, "tab:blue")
        kwargs["std_alpha"] = 0.25
        try:
            ensemble.plot(ax=ax, mode=mode, unit=uq_unit, **kwargs)
        except Exception as exc:
            print(f"   Warning: plot mode '{mode}' failed ({exc}); skipping.")
            plt.close(fig)
            continue
        if bsmin < -0.1:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        title_mode = mode.replace("+", " ± ") if mode == "mean+std" else mode
        ax.set_title(f"{formula} — {model_name} ({title_mode})")
        plt.tight_layout()
        safe_mode = mode.replace("+", "plus").replace(" ", "_")
        plot_path = os.path.join(output_dir, f"{prefix}_phonon_uq_{safe_mode}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"   Saved: {plot_path}")

    # ── text summary ──────────────────────────────────────────────────────────
    time_taken = time.time() - start_time
    summary_path = os.path.join(output_dir, f"{prefix}_phonon_uq_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("--- Phonon UQ Run Summary ---\n")
        fh.write(f"Structure Formula: {formula}\n")
        fh.write(f"Number of Atoms (primitive cell): {len(atoms)}\n")
        fh.write(f"Engine: {engine}  |  Model: {model_name}\n")
        fh.write(f"Units: {units}\n")
        fh.write(f"Supercell: ({sc[0]}, {sc[1]}, {sc[2]})\n")
        fh.write(f"Displacement delta: {delta} Å\n")
        fh.write(f"Ensemble members: {n_ens}\n")
        fh.write(f"Displacements: {n_disp}\n")
        fh.write(f"Atoms per supercell: {n_at}\n")
        fh.write(f"Most negative mean frequency: {bsmin:.4f} {units}\n")
        stable_threshold = -0.1  # THz — below this is a genuine imaginary mode
        fh.write(f"Dynamically stable: {'Yes' if bsmin > stable_threshold else 'No (imaginary modes present)'}\n")
        fh.write(f"Time taken: {time_taken:.2f} s\n")
    print(f"   UQ summary saved: {summary_path}")
    print(f"\n» Phonon UQ done in {time_taken:.2f} s  |  min freq = {bsmin:.4f} {units}")

    # Compatible return with run_single_phonon_analysis
    return [], bsmin, time_taken, {}


def copy_structure_files_to_phonon_analysis_dir(phonon_analysis_dir, final_structures_dir, analysis_id):
   """
   Copy all structure file variants (CIF, XYZ, conventional, primitive) from final_structures/
   directory to the corresponding phonon analysis directory for better organization and traceability.

   Args:
       phonon_analysis_dir (str): Path to the phonon analysis directory
       final_structures_dir (str): Path to the final_structures directory
       analysis_id (str): Analysis identifier to match files (e.g., "top_1_energy_m6p0976")
   """
   import shutil
   import glob

   if not os.path.exists(final_structures_dir):
       print(f"  Warning: final_structures directory not found: {final_structures_dir}")
       return

   # Create a structures subdirectory in the phonon analysis directory
   structures_subdir = os.path.join(phonon_analysis_dir, "structure_files")
   os.makedirs(structures_subdir, exist_ok=True)

   # Find all structure files that match the analysis_id
   pattern = os.path.join(final_structures_dir, f"*{analysis_id}*")
   matching_files = glob.glob(pattern)

   # If no matches, try a more flexible approach using energy part
   if not matching_files and 'energy_' in analysis_id:
       energy_part = analysis_id.split('energy_')[1]
       pattern = os.path.join(final_structures_dir, f"*energy_{energy_part}*")
       matching_files = glob.glob(pattern)

   copied_files = []
   for src_file in matching_files:
       if src_file.endswith(('.cif', '.xyz')):
           filename = os.path.basename(src_file)
           dest_file = os.path.join(structures_subdir, filename)
           try:
               shutil.copy2(src_file, dest_file)
               copied_files.append(filename)
               print(f"    Copied structure file: {filename}")
           except Exception as e:
               print(f"    Warning: Could not copy {filename}: {e}")

   if copied_files:
       print(f"  ✅ Copied {len(copied_files)} structure files to {structures_subdir}")
   else:
       print(f"  ⚠️  No matching structure files found for analysis_id: {analysis_id}")


def add_frequency_to_structure_filenames(final_structures_dir, analysis_id, softest_frequency, units="THz"):
   """
   Rename structure files in final_structures/ directory to include the softest phonon frequency.

   Args:
       final_structures_dir (str): Path to the final_structures directory
       analysis_id (str): Analysis identifier to match files (e.g., "top_1_energy_m6p0976")
       softest_frequency (float): The softest phonon frequency
       units (str): Units for frequency (default: "THz")
   """
   import glob

   if not os.path.exists(final_structures_dir):
       print(f"  Warning: final_structures directory not found: {final_structures_dir}")
       return

   # Format frequency for filename (replace . with p, - with m)
   freq_str = f"{abs(softest_frequency):.4f}".replace('.', 'p')
   if softest_frequency < 0:
       freq_str = 'm' + freq_str
   else:
       freq_str = 'p' + freq_str

   # Find all structure files that match the analysis_id
   pattern = os.path.join(final_structures_dir, f"*{analysis_id}*")
   matching_files = glob.glob(pattern)

   # If no matches, try a more flexible approach using energy part
   if not matching_files and 'energy_' in analysis_id:
       energy_part = analysis_id.split('energy_')[1]
       pattern = os.path.join(final_structures_dir, f"*energy_{energy_part}*")
       matching_files = glob.glob(pattern)

   renamed_files = []
   for src_file in matching_files:
       if src_file.endswith(('.cif', '.xyz')):
           # Check if frequency is already in filename
           if '_freq' in os.path.basename(src_file):
               print(f"    Frequency already in filename: {os.path.basename(src_file)}")
               continue

           # Create new filename with frequency
           base_name = os.path.basename(src_file)
           name_parts = base_name.rsplit('.', 1)  # Split filename and extension
           if len(name_parts) == 2:
               new_name = f"{name_parts[0]}_freq{freq_str}{units}.{name_parts[1]}"
           else:
               new_name = f"{base_name}_freq{freq_str}{units}"

           new_path = os.path.join(final_structures_dir, new_name)

           try:
               os.rename(src_file, new_path)
               renamed_files.append((os.path.basename(src_file), new_name))
               print(f"    Renamed: {os.path.basename(src_file)} → {new_name}")
           except Exception as e:
               print(f"    Warning: Could not rename {os.path.basename(src_file)}: {e}")

   if renamed_files:
       print(f"  ✅ Renamed {len(renamed_files)} structure files to include frequency ({softest_frequency:.4f} {units})")
   else:
       print(f"  ⚠️  No structure files renamed for analysis_id: {analysis_id}")



def run_phonon_calculation_with_custom_supercell(atoms, calculator, supercell_dims, delta, output_dir):
   """Sets up and runs the phonon calculation with custom supercell dimensions."""
   from .utils import parse_supercell_dimensions
   start = time.time()
   atoms.set_calculator(calculator)

   # Parse supercell dimensions
   supercell = parse_supercell_dimensions(supercell_dims)

   print("\n### Structure for Phonon Calculation ###")
   print("   The phonon band structure will be calculated using the current structure.")
   print(f"   Using supercell size: {supercell}")
   print(f"   Using displacement delta: {delta}")

   # Phonon calculator
   phonon_calc_path = os.path.join(output_dir, "phonon_files_temp")       
   if os.path.exists(phonon_calc_path):  
      print(f"   Clearing existing phonon calculation directory: {phonon_calc_path}")  
      shutil.rmtree(phonon_calc_path)
   ph = Phonons(atoms, calculator, supercell=supercell, delta=delta, name=phonon_calc_path)
   ph.run()

   # Read forces and assemble the dynamical matrix
   ph.read(acoustic=True)
   ph.clean()
   
   print("Phonon calculation completed.")
   end = time.time()
   print(f"Phonon calculation took {end - start:.2f} seconds")
   return ph

def run_phonon_calculation(atoms, calculator, supercell_dims, delta, output_dir):
   """Sets up and runs the phonon calculation."""
   from .utils import parse_supercell_dimensions
   start = time.time()
   atoms.set_calculator(calculator) # Ensure calculator is set for phonon calculation

   # Handle both old supercell_n (integer) and new custom supercell formats
   if isinstance(supercell_dims, tuple):  
      supercell = supercell_dims  
   else:  
      supercell = parse_supercell_dimensions(supercell_dims)

   print("\n### Structure for Phonon Calculation ###")
   print("   The phonon band structure will be calculated using the current structure.")
   print(f"   Using supercell size: {supercell}")
   print(f"   Using displacement delta: {delta}")

   # Phonon calculator
   
   # Isolate the phonon calculation files by setting a unique path for this run.  
   # This prevents reusing results from other steps in the GA.  
   phonon_calc_path = os.path.join(output_dir, "phonon_files_temp")       
   if os.path.exists(phonon_calc_path):  
       print(f"   Clearing existing phonon calculation directory: {phonon_calc_path}")  
       shutil.rmtree(phonon_calc_path)
   ph = Phonons(atoms, calculator, supercell=supercell, delta=delta, name=phonon_calc_path)
   ph.run()

   # Read forces and assemble the dynamical matrix
   ph.read(acoustic=True)
   ph.clean()

   print("Phonon calculation completed.")
   end = time.time()
   print(f"Phonon calculation took {end - start:.2f} seconds")
   return ph

def save_phonon_results(ph, output_dir, units, bs, dos, dos_energies):
   # Save band structure data
   try:
      import pickle
      with open(os.path.join(output_dir, "bands.pkl"), "wb") as f:
         pickle.dump(bs, f)
      print(f"  - Saved band structure to {os.path.join(output_dir, 'bands.pkl')}")
   except Exception as e:
      print(f"  - Could not save band structure: {e}")

   # Save DOS data
   if dos is not None and dos_energies is not None:
      try:
         dos_data = {
            "units": units,
            "frequencies": dos_energies.tolist(), # Use the unit-converted energies
            "dos_values": dos.get_weights().tolist() # Use the weights from the DOS object
         }
         with open(os.path.join(output_dir, "dos.json"), "w") as f:
            json.dump(dos_data, f, indent=4)
         print(f"  - Saved DOS to {os.path.join(output_dir, 'dos.json')}")
      except Exception as e:
         print(f"  - Could not save DOS: {e}")
   else:
      print("  - DOS data not provided, skipping DOS save.")

   # Optionally, save the primitive cell structure itself
   try:
      write(os.path.join(output_dir, "primitive_cell.cif"), ph.atoms)
      print(f"  - Saved primitive cell structure to {os.path.join(output_dir, 'primitive_cell.cif')}")
   except Exception as e:
      print(f"  - Could not save primitive cell structure: {e}")

def get_phonon_results(ph, atoms, units, phonon_path_npoints=100, phonon_dos_grid=(40,40,40)):
   """Gets band structure and DOS results and converts units."""
   # Get the band path object
   path = atoms.cell.bandpath(npoints=phonon_path_npoints) # Use phonon_path_npoints

   # --- Capture stderr to suppress ASE warnings and summarize ---
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
   
   # Check for discontinuities in special k-point distances
   discontinuities = []
   for i in range(len(special_k_point_distances) - 1):
         if abs(special_k_point_distances[i] - special_k_point_distances[i+1]) < 1e-6:
            discontinuities.append(i)
   
   if discontinuities:
         print("\nDetected band structure discontinuities at:")
         for idx in discontinuities:
            print(f"  - Between {special_k_point_labels[idx]} and {special_k_point_labels[idx+1]}"
                  f" at distance {special_k_point_distances[idx]:.4f}")
   
   bsmin = np.min(bs_energies) if bs_energies.size > 0 else 0

   print(f'Most negative frequency: {bsmin:.4f} {units}')

   # Return bs and path objects as well for further analysis
   return bs, path, dos, bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, discontinuities, y_label, bsmin

def get_eigenvector_for_q_and_band_index(ph, q_point, band_idx):
   """
   Retrieves the eigenvector (raw displacements) for a specific q-point and mode index.

   Args:
      ph (ase.phonons.Phonons): The Phonons object.
      q_point (np.array): The q-point in fractional reciprocal coordinates.
      band_idx (int): The index of the mode (0-indexed).

   Returns:
      np.ndarray: The raw displacements (eigenvector) for the specified mode,
                  or None if the mode cannot be retrieved.
   """
   try:
      # ph.band_structure returns (frequencies, eigenvectors) for the given q-points
      # The eigenvectors are (num_q_points, num_bands, num_atoms_primitive * 3)
      omega_at_q, u_at_q = ph.band_structure([q_point], modes=True)

      # Check if band_idx is valid
      num_bands = u_at_q.shape[1]
      if not (0 <= band_idx < num_bands):
         print(f"Error: Mode index {band_idx} is out of bounds. Available modes: 0 to {num_bands-1}.")
         return None

      eigenvector = u_at_q[0, band_idx] # Select the first q-point (only one requested) and the specific mode
      num_atoms_primitive = len(ph.atoms)
      raw_displacements = eigenvector.reshape(num_atoms_primitive, 3)
      return raw_displacements
   except Exception as e:
      print(f"Error retrieving eigenvector for q-point {q_point} and mode {band_idx}: {e}")
      import traceback
      traceback.print_exc()
      return None

def _process_and_save_mode_data(ph, q_point, band_idx, frequency, units, output_dir, traj_kT, prefix=""):
   """
   Helper function to process, save, and visualize data for a given phonon mode.
   This consolidates the repetitive logic for both softest modes and target modes.

   Args:
      ph (ase.phonons.Phonons): The Phonons object.
      q_point (np.array): The q-point (fractional coordinates) of the mode.
      band_idx (int): The index of the mode (0-indexed).
      frequency (float): The frequency of the mode.
      units (str): Units for frequency (e.g., "THz", "cm-1").
      output_dir (str): Directory to save output files.
      traj_kT (float): Temperature for trajectory generation.
      prefix (str): A prefix for output filenames (e.g., "softest_mode_1", "target_q0.5_0_0").
   """
   q_point_str_for_filename = "_".join([f"{c:.3f}" for c in q_point]).replace('.', 'p').replace('-', 'm')  
   base_filename = f"{prefix}_q{q_point_str_for_filename}_mode{band_idx}"  
   print(f"\n--- Processing Mode: {prefix} at Q-point: {q_point}, Mode Index: {band_idx} ---")  
   print(f"   Frequency: {frequency:.4f} {units}")  
  
   try:  
      raw_displacements = get_eigenvector_for_q_and_band_index(ph, q_point, band_idx)  
      if raw_displacements is None:  
         print(f"  - Could not retrieve raw displacements for {base_filename}.")  
         return  
  
      # Ensure raw_displacements is a NumPy array before processing  
      raw_displacements = np.asarray(raw_displacements)  
  
      # Save eigenvectors to JSON  
      eigenvector_data = {  
         "q_point": list(q_point),  
         "band_idx": band_idx,  
         "frequency": float(frequency),  
         # Convert complex NumPy array to a list of lists of dictionaries for JSON serialization  
         "eigenvectors": [[{"real": x.real, "imag": x.imag} for x in row] for row in raw_displacements]  
      }  
      eigenvector_filename = os.path.join(output_dir, f"{base_filename}_eigenvectors.json")  
      with open(eigenvector_filename, "w") as f:  
         json.dump(eigenvector_data, f, indent=4)  
      print(f"  - Saved eigenvectors to {eigenvector_filename}")

      # Save displacements to TXT
      max_disp_magnitude = np.max(np.linalg.norm(np.real(raw_displacements), axis=1))
      if max_disp_magnitude > 1e-6:
         scaled_displacements_for_viz = np.real(raw_displacements) / max_disp_magnitude * 0.5
      else:
         scaled_displacements_for_viz = np.real(raw_displacements)

      displacements_filename = os.path.join(output_dir, f"{base_filename}_displacements.txt")
      with open(displacements_filename, 'w') as f:
         f.write(f"Displacements for Q-point: {list(q_point)} (fractional coordinates)\n")
         f.write(f"Mode Index: {band_idx}\n")
         f.write(f"Frequency: {frequency:.4f} {units}\n\n")

         f.write("--- Raw Complex Displacements (Eigenvectors) ---\n")
         f.write("Atom Index | Element | Displacement (x, y, z) (complex values)\n")
         f.write("------------------------------------------------------------------\n")
         for i, (disp, symbol) in enumerate(zip(raw_displacements, ph.atoms.symbols)):
               f.write(f"{i:<10} | {symbol:<7} | ({disp[0]:.6f}, {disp[1]:.6f}, {disp[2]:.6f})\n")
         f.write("\n")

         f.write("--- Scaled Real Displacements for Visualization ---\n")
         f.write("(These are the real parts of the eigenvectors, scaled for visual clarity)\n")
         f.write("Atom Index | Element | Displacement (x, y, z) (real values, scaled)\n")
         f.write("------------------------------------------------------------------\n")
         for i, (disp_scaled, symbol) in enumerate(zip(scaled_displacements_for_viz, ph.atoms.symbols)):
               f.write(f"{i:<10} | {symbol:<7} | dx={disp_scaled[0]:.6f}, dy={disp_scaled[1]:.6f}, dz={disp_scaled[2]:.6f} Å\n")
         f.write("\n")
      print(f"  - Displacements saved to: {displacements_filename}")

      # Generate trajectory
      original_ph_name = ph.name # Save original name to restore later
      ph.name = os.path.join(output_dir, base_filename)
      ph.write_modes(
         q_point,
         branches=[band_idx],
         kT=traj_kT,
         repeat=(2, 2, 2),
         center=True
      )
      traj_filename_generated = f"{ph.name}.mode.{band_idx}.traj"
      print(f"  - Mode animation (ASE .traj) saved to {traj_filename_generated}")
      ph.name = original_ph_name # Restore original name

      # Convert trajectory to XYZ
      xyz_filename = os.path.join(output_dir, f"{base_filename}.xyz")
      try:
         frames = read(traj_filename_generated, index=':')
         write(xyz_filename, frames)
         print(f"  - Converted trajectory to XYZ: {xyz_filename}")
      except Exception as e:
         print(f"  - Error converting trajectory to XYZ for {base_filename}: {e}")
         import traceback
         traceback.print_exc()

   except Exception as e:
      print(f"Error processing mode {base_filename}: {e}")
      import traceback
      traceback.print_exc()


def analyze_special_points_and_modes(ph, path, bs_energies, special_k_point_distances, special_k_point_labels, units, output_dir, prefix, traj_kT=1, num_modes_to_return=2, target_q_point=None, target_band_idx=None, displacement_magnitude=1.0, negative_phonon_threshold=None):
   """Analyzes frequencies at special points and identifies the softest modes.
   Optionally, it can also process and save displacements for a specific q-point and mode.
   Now also saves band structure and DOS data.

   When no soft modes below the threshold are found, selects the highest frequency modes
   from special k-points (optical modes) instead of the lowest frequency modes.

   Args:
      ph (ase.phonons.Phonons): The Phonons object.
      bs (ase.spectrum.band_structure.BandStructure): The BandStructure object.
      path (ase.dft.kpoints.BandPath): The BandPath object.
      bs_energies (np.ndarray): The phonon band energies (frequencies).
      special_k_point_distances (list): Distances of special k-points.
      special_k_point_labels (list): Labels of special k-points.
      units (str): Units for frequencies (e.g., "THz", "cm-1").
      output_dir (str): Directory to save output files.
      prefix (str): Prefix for output files.
      traj_kT (float): Temperature for trajectory generation (default 0.1).
      num_modes_to_return (int): The number of softest modes to return (default 2).
      target_q_point (list or np.array, optional): A specific q-point for which to retrieve and save eigenvectors.
                                                   If None, only special points are analyzed.
      target_band_idx (int, optional): The index of the mode (0-indexed) for the target_q_point.
                                       Required if target_q_point is provided.
      negative_phonon_threshold (float, optional): Threshold for soft mode detection. If provided,
                                                   when no modes below this threshold are found,
                                                   the function will select highest frequency modes instead.

   Returns:
      tuple: (softest_modes_info_list, tracked_k_points_data) where:
         - softest_modes_info_list: List of top N softest modes as before
         - tracked_k_points_data: Dict containing comprehensive tracking data for GA mode replacement:
           {
             'soft_modes': [list of all soft modes below threshold],
             'highest_freq_modes': [list of highest frequency modes at special k-points],
             'lowest_freq_modes': [list of lowest frequency modes at special k-points, excluding Gamma]
           }
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

      if kpt_index == None:
         print(f"Warning: Could not find k-point index for coordinate {coord} (label '{label}'). Skipping.")
         continue

      # Get frequencies at this k-point index
      freqs_at_kpt = bs_energies[kpt_index]

      # Iterate through all bands at this k-point to find negative frequencies
      for band_idx, freq in enumerate(freqs_at_kpt):
         # Skip None values or non-numeric values (can occur with some calculators)
         if freq is None:
            continue
         try:
            freq_val = float(freq)
         except (TypeError, ValueError):
            continue

         # Only consider negative frequencies if threshold is provided
         if negative_phonon_threshold is not None and freq_val < negative_phonon_threshold:
               mode_info = {
                  "label": label,
                  "coordinate": [float(c) for c in coord], # Convert numpy array to list for JSON
                  "frequency": freq_val,
                  "band_index": int(band_idx),
                  "kpoint_index_in_path": int(kpt_index) # Store index in path.kpts
               }
               all_negative_modes_info.append(mode_info)

      # Also add the minimum frequency at this k-point to special_point_analysis (even if positive)
      # Filter out None values before computing min
      valid_freqs = [f for f in freqs_at_kpt if f is not None]
      if valid_freqs:
         min_freq_at_kpt = np.min(valid_freqs)
         min_band_index_at_kpt = np.argmin(freqs_at_kpt)  # Use original array for index
         special_point_analysis.append({
            "label": label,
            "coordinate": [float(c) for c in coord],
            "min_frequency": float(min_freq_at_kpt),
            "band_index": int(min_band_index_at_kpt)
         })
         print(f"   Special Point: {label} ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}) - Minimum Frequency: {min_freq_at_kpt:.4f} {units}")
      else:
         print(f"   Special Point: {label} ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}) - No valid frequencies found")


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
         raw_displacements = get_eigenvector_for_q_and_band_index(  
             ph,  
             np.array(softest_mode_info['coordinate']),  
             softest_mode_info['band_index']  
         )
         if raw_displacements is not None:  
            # Add raw_displacements to the mode_info dictionary  
            softest_mode_info['raw_displacements'] = raw_displacements.tolist()
         # Call the helper function for each softest mode
         _process_and_save_mode_data(
               ph=ph,
               q_point=np.array(softest_mode_info['coordinate']),
               band_idx=softest_mode_info['band_index'],
               frequency=softest_mode_info['frequency'],
               units=units,
               output_dir=output_dir,
               traj_kT=traj_kT,
               prefix=f"softest_mode_{i+1}_{softest_mode_info['label']}"
         )
         # You might want to append the processed info to all_soft_modes_with_displacements
         # if you need to return it, but the helper function handles saving.
         # For now, we'll just add the basic info back.
         all_soft_modes_with_displacements.append(softest_mode_info)
   else:
      print("\nNo negative frequencies found at special points.")

      # If threshold is provided and no soft modes found, select highest frequency modes instead
      if negative_phonon_threshold is not None:
         print(f"No soft modes below threshold ({negative_phonon_threshold:.4f} {units}) found.")
         print("Implementing optical mode selection: selecting highest frequency modes from special k-points.")

         # Collect all positive frequency modes at special points
         all_positive_modes_info = []
         special_point_coords = path.special_points # Dictionary mapping label to coordinate

         for label, coord in special_point_coords.items():
            # Find the index of this special point in the path
            kpt_index = None
            for i, kpt in enumerate(path.kpts):
               if np.allclose(kpt, coord, atol=1e-6):
                  kpt_index = i
                  break

            if kpt_index is None:
               print(f"Warning: Could not find k-point index for special point {label}")
               continue

            # Get frequencies at this k-point index
            freqs_at_kpt = bs_energies[kpt_index]

            # Iterate through all bands at this k-point to find positive frequencies
            for band_idx, freq in enumerate(freqs_at_kpt):
               if freq > 0: # Only consider positive frequencies
                  mode_info = {
                     "label": label,
                     "coordinate": [float(c) for c in coord], # Convert numpy array to list for JSON
                     "frequency": float(freq),
                     "band_index": int(band_idx),
                     "kpoint_index_in_path": int(kpt_index) # Store index in path.kpts
                  }
                  all_positive_modes_info.append(mode_info)

         # Sort all positive modes by frequency (highest first)
         all_positive_modes_info.sort(key=lambda x: x['frequency'], reverse=True)

         # Select the top N highest frequency modes
         top_n_highest_modes = []
         processed_modes_keys = set() # To avoid duplicate modes (same k-point, same band)

         for mode_info in all_positive_modes_info:
            mode_key = (mode_info['kpoint_index_in_path'], mode_info['band_index'])
            if mode_key not in processed_modes_keys:
               top_n_highest_modes.append(mode_info)
               processed_modes_keys.add(mode_key)
               if len(top_n_highest_modes) >= num_modes_to_return:
                  break

         # Process the highest frequency modes similar to soft modes
         if top_n_highest_modes:
            print(f"\n--- Top {len(top_n_highest_modes)} Highest Frequency (Optical) Modes Analysis ---")
            for i, highest_mode_info in enumerate(top_n_highest_modes):
               print(f"   Optical Mode {i+1}: {highest_mode_info['label']} - Frequency: {highest_mode_info['frequency']:.4f} {units}")

               raw_displacements = get_eigenvector_for_q_and_band_index(
                   ph,
                   np.array(highest_mode_info['coordinate']),
                   highest_mode_info['band_index']
               )
               if raw_displacements is not None:
                  # Add raw_displacements to the mode_info dictionary
                  highest_mode_info['raw_displacements'] = raw_displacements.tolist()

               # Call the helper function for each highest frequency mode
               _process_and_save_mode_data(
                     ph=ph,
                     q_point=np.array(highest_mode_info['coordinate']),
                     band_idx=highest_mode_info['band_index'],
                     frequency=highest_mode_info['frequency'],
                     units=units,
                     output_dir=output_dir,
                     traj_kT=traj_kT,
                     prefix=f"optical_mode_{i+1}_{highest_mode_info['label']}"
               )

               all_soft_modes_with_displacements.append(highest_mode_info)
         else:
            print("No positive frequency modes found at special points.")
      else:
         print("No threshold provided - returning empty list.")

   # --- Handle specific target q-point and mode if provided ---
   if target_q_point is not None and target_band_idx is not None:
      print(f"\n--- Processing Specific Target Mode ---")
      try:
         # Get frequency of the target mode
         omega_at_q, _ = ph.band_structure([target_q_point], modes=True)
         freq_ev = omega_at_q[0, target_band_idx]
         
         if units == "THz":
             freq_at_mode = freq_ev / EV_TO_THZ_FACTOR
         elif units == "cm-1":
             freq_at_mode = (freq_ev / EV_TO_THZ_FACTOR) * THZ_TO_CM_FACTOR
         elif units == "eV":
             freq_at_mode = freq_ev
         else:
             freq_at_mode = freq_ev / EV_TO_THZ_FACTOR # Default to THz
             
         print(f"Manual Mode Frequency: {freq_ev:.6f} eV -> {freq_at_mode:.4f} {units}")

         # Create a mode_info dict for the target mode to return to main.py for optimization
         target_mode_info = {
             "label": "User-Target",
             "coordinate": [float(c) for c in target_q_point],
             "frequency": float(freq_at_mode),
             "band_index": int(target_band_idx),
             # Use 0 as a placeholder for kpoint index if not in standard path
             "kpoint_index_in_path": -1 
         }
         # Get raw displacements for this target mode
         raw_disp = get_eigenvector_for_q_and_band_index(ph, np.array(target_q_point), target_band_idx)
         if raw_disp is not None:
             target_mode_info['raw_displacements'] = raw_disp.tolist()
         
         all_soft_modes_with_displacements.insert(0, target_mode_info)
    
         

         # Call the helper function for the target mode
         _process_and_save_mode_data(
               ph=ph,
               q_point=np.array(target_q_point),
               band_idx=target_band_idx,
               frequency=freq_at_mode,
               units=units,
               output_dir=output_dir,
               traj_kT=traj_kT,
               prefix="target_mode" # A generic prefix for the target mode
         )
         # Note: max_atoms_per_supercell is not available in this context, so we pass None (no limit)
         generate_and_visualize_specific_mode_supercell(
            ph.atoms, # This is the primitive cell (potentially relaxed)
            ph,
            np.array(target_q_point),
            target_band_idx,
            displacement_magnitude,
            output_dir,
            prefix,
            max_atoms_per_supercell=None
        )
      except Exception as e:
         print(f"Error processing target q-point {target_q_point}, mode {target_band_idx}: {e}")
         import traceback
         traceback.print_exc()
   
   # --- Collect comprehensive tracking data for GA mode replacement ---
   tracked_k_points_data = {
       'soft_modes': [],
       'highest_freq_modes': [],
       'lowest_freq_modes': []
   }

   # 1. Collect all soft modes (negative frequencies below threshold)
   if negative_phonon_threshold is not None:
       # Add raw_displacements to all soft modes for structure generation
       soft_modes_with_displacements = []
       for mode_info in all_negative_modes_info:
           mode_with_displacements = mode_info.copy()
           raw_displacements = get_eigenvector_for_q_and_band_index(
               ph,
               np.array(mode_info['coordinate']),
               mode_info['band_index']
           )
           if raw_displacements is not None:
               mode_with_displacements['raw_displacements'] = raw_displacements.tolist()
           else:
               print(f"Warning: Could not get raw_displacements for soft mode {mode_info.get('label', 'unknown')} at {mode_info.get('coordinate', 'unknown')}")
           soft_modes_with_displacements.append(mode_with_displacements)

       tracked_k_points_data['soft_modes'] = soft_modes_with_displacements
       print(f"Tracked {len(tracked_k_points_data['soft_modes'])} soft modes for GA mode replacement")

   # 2. Collect highest frequency modes at special k-points
   all_highest_freq_modes = []
   for label, coord in special_point_coords.items():
       # Find the index of this special point in the path
       kpt_index = None
       for i, kpt in enumerate(path.kpts):
           if np.allclose(kpt, coord, atol=1e-6):
               kpt_index = i
               break

       if kpt_index is None:
           continue

       # Get frequencies at this k-point index
       freqs_at_kpt = bs_energies[kpt_index]

       # Find the highest frequency mode at this k-point
       max_freq_idx = np.argmax(freqs_at_kpt)
       max_freq = freqs_at_kpt[max_freq_idx]

       mode_info = {
           "label": label,
           "coordinate": [float(c) for c in coord],
           "frequency": float(max_freq),
           "band_index": int(max_freq_idx),
           "kpoint_index_in_path": int(kpt_index)
       }
       all_highest_freq_modes.append(mode_info)

   # Sort by frequency (highest first) and store
   all_highest_freq_modes.sort(key=lambda x: x['frequency'], reverse=True)
   tracked_k_points_data['highest_freq_modes'] = all_highest_freq_modes
   print(f"Tracked {len(tracked_k_points_data['highest_freq_modes'])} highest frequency modes for GA mode replacement")

   # 3. Collect lowest frequency modes at special k-points (excluding Gamma point)
   all_lowest_freq_modes = []
   for label, coord in special_point_coords.items():
       # Skip Gamma point (usually labeled as 'G' or at origin)
       if label.upper() == 'G' or np.allclose(coord, [0.0, 0.0, 0.0], atol=1e-6):
           continue

       # Find the index of this special point in the path
       kpt_index = None
       for i, kpt in enumerate(path.kpts):
           if np.allclose(kpt, coord, atol=1e-6):
               kpt_index = i
               break

       if kpt_index is None:
           continue

       # Get frequencies at this k-point index
       freqs_at_kpt = bs_energies[kpt_index]

       # Find the lowest frequency mode at this k-point
       min_freq_idx = np.argmin(freqs_at_kpt)
       min_freq = freqs_at_kpt[min_freq_idx]

       mode_info = {
           "label": label,
           "coordinate": [float(c) for c in coord],
           "frequency": float(min_freq),
           "band_index": int(min_freq_idx),
           "kpoint_index_in_path": int(kpt_index)
       }
       all_lowest_freq_modes.append(mode_info)

   # Sort by frequency (lowest first) and store
   all_lowest_freq_modes.sort(key=lambda x: x['frequency'])
   tracked_k_points_data['lowest_freq_modes'] = all_lowest_freq_modes
   print(f"Tracked {len(tracked_k_points_data['lowest_freq_modes'])} lowest frequency modes (excluding Gamma) for GA mode replacement")

   # Return both the original softest modes list and the comprehensive tracking data
   return all_soft_modes_with_displacements, tracked_k_points_data

def load_eigenmode_from_band_yaml(band_yaml_path, target_q_point, target_band_idx):
    """
    Loads eigenmode data from an existing band.yaml file for a specific q-point and band index.

    Args:
        band_yaml_path (str): Path to the band.yaml file.
        target_q_point (np.array): The target q-point in fractional reciprocal coordinates.
        target_band_idx (int): The target band index (0-indexed).

    Returns:
        tuple: (frequency, eigenvector, lattice, natom) where:
            - frequency (float): The frequency in THz (as stored in band.yaml)
            - eigenvector (np.ndarray): Complex eigenvector array of shape (natom, 3)
            - lattice (np.ndarray): The lattice vectors from the band.yaml
            - natom (int): Number of atoms
        Returns (None, None, None, None) if the mode is not found or file cannot be read.
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not found. Cannot read band.yaml file.")
        print("Please install it using: pip install pyyaml")
        return None, None, None, None

    try:
        with open(band_yaml_path, 'r') as f:
            band_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading band.yaml file {band_yaml_path}: {e}")
        return None, None, None, None

    # Extract basic information
    natom = band_data.get('natom')
    lattice = np.array(band_data.get('lattice', []))
    points = band_data.get('points', [])

    if natom is None or len(lattice) == 0 or len(points) == 0:
        print(f"Error: Invalid band.yaml format in {band_yaml_path}")
        return None, None, None, None

    # Find the matching q-point
    target_q_point = np.array(target_q_point)
    tolerance = 1e-6

    for point in points:
        q_point = np.array(point.get('q-point', []))
        if len(q_point) == 3 and np.allclose(q_point, target_q_point, atol=tolerance):
            bands = point.get('bands', [])

            # Check if the target band index is valid
            if 0 <= target_band_idx < len(bands):
                band = bands[target_band_idx]
                frequency = band.get('frequency')
                eigenvector_data = band.get('eigenvector', [])

                if frequency is None or len(eigenvector_data) == 0:
                    print(f"Error: Missing frequency or eigenvector data for q-point {target_q_point}, band {target_band_idx}")
                    return None, None, None, None

                # Convert eigenvector data to complex numpy array
                if len(eigenvector_data) != natom:
                    print(f"Error: Eigenvector data length ({len(eigenvector_data)}) doesn't match natom ({natom})")
                    return None, None, None, None

                eigenvector = np.zeros((natom, 3), dtype=complex)
                for atom_idx, atom_data in enumerate(eigenvector_data):
                    if len(atom_data) != 3:
                        print(f"Error: Invalid eigenvector format for atom {atom_idx}")
                        return None, None, None, None

                    for coord_idx, coord_data in enumerate(atom_data):
                        if len(coord_data) != 2:
                            print(f"Error: Invalid complex number format for atom {atom_idx}, coordinate {coord_idx}")
                            return None, None, None, None

                        real_part, imag_part = coord_data
                        eigenvector[atom_idx, coord_idx] = complex(real_part, imag_part)

                print(f"Successfully loaded eigenmode from {band_yaml_path}")
                print(f"  Q-point: {q_point}")
                print(f"  Band index: {target_band_idx}")
                print(f"  Frequency: {frequency:.4f} THz")

                return frequency, eigenvector, lattice, natom
            else:
                print(f"Error: Band index {target_band_idx} is out of bounds. Available bands: 0 to {len(bands)-1}")
                return None, None, None, None

    print(f"Error: Q-point {target_q_point} not found in {band_yaml_path}")
    return None, None, None, None

def save_phonopy_band_yaml(ph, path, bs, all_k_point_distances, output_dir, prefix):
    """
    Generates and saves a Phonopy-standard band.yaml file from ASE Phonons results.

    Args:
        ph (ase.phonons.Phonons): The calculated ASE Phonons object.
        path (ase.dft.kpoints.BandPath): The band path object used for calculation.
        bs (ase.spectrum.band_structure.BandStructure): The calculated band structure object.
        all_k_point_distances (np.ndarray): Array of distances for each k-point along the path.
        output_dir (str): Directory to save the output file.
        prefix (str): Prefix for the output filename.
    """
    try:
        import yaml
    except ImportError:
        print("\nWarning: PyYAML not found. Cannot generate band.yaml.")
        print("Please install it using: pip install pyyaml")
        return

    print("\n--- Step 3.5: Generating Phonopy band.yaml ---")

    # Get frequencies and eigenvectors directly from the Phonons object
    omega_kl_ev, u_kl = ph.band_structure(path.kpts, modes=True, verbose=False)

    # Convert frequencies from eV (ASE default) to THz (Phonopy default)
    omega_kl_thz = omega_kl_ev / EV_TO_THZ_FACTOR

    # Prepare the 'points' data structure for band.yaml
    phonon_data = []
    for k_idx, kpt_coord in enumerate(path.kpts):
        frequencies_at_k = omega_kl_thz[k_idx]
        eigenvectors_at_k = u_kl[k_idx]

        bands_data = []
        for band_idx in range(frequencies_at_k.shape[0]):
            freq = frequencies_at_k[band_idx]
            eigenvec = eigenvectors_at_k[band_idx]

            eigenvector_list = []
            for atom_idx in range(eigenvec.shape[0]):
                x, y, z = eigenvec[atom_idx]
                eigenvector_list.append([
                    [float(x.real), float(x.imag)],
                    [float(y.real), float(y.imag)],
                    [float(z.real), float(z.imag)]
                ])

            bands_data.append({
                'frequency': float(freq),
                'eigenvector': eigenvector_list
            })

        phonon_data.append({
            'q-point': [float(c) for c in kpt_coord],
            # --- THIS IS THE CORRECTED LINE ---
            'distance': float(all_k_point_distances[k_idx]),
            'bands': bands_data
        })

    # Prepare the 'path' data structure for band.yaml
    _, special_k_point_distances, special_k_point_labels = bs.get_labels()

    phonopy_path_data = []
    unique_labels = set()
    for label, dist in zip(special_k_point_labels, special_k_point_distances):
        if label and label not in unique_labels:
            coord = path.special_points.get(label)
            if coord is not None:
                phonopy_path_data.append({
                    'q-position': [float(c) for c in coord],
                    'distance': float(dist),
                    'label': label
                })
                unique_labels.add(label)

    # Assemble the final dictionary for band.yaml
    band_yaml_dict = {
        'natom': len(ph.atoms),
        'lattice': ph.atoms.cell.tolist(),
        'points': phonon_data,
        'path': phonopy_path_data
    }

    # Write the dictionary to a YAML file
    band_yaml_path = os.path.join(output_dir, f"{prefix}_band.yaml")
    try:
        with open(band_yaml_path, 'w') as f:
            yaml.dump(band_yaml_dict, f, default_flow_style=False, sort_keys=False)
        print(f"   Successfully saved Phonopy band structure to: {band_yaml_path}")
    except Exception as e:
        print(f"   Error saving band.yaml file: {e}")
        import traceback
        traceback.print_exc()     
def generate_and_visualize_specific_mode_supercell(primitive_atoms, ph_obj, q_point, band_idx, displacement_magnitude, output_base_dir, original_prefix, max_atoms_per_supercell=None):
   """
   Generates a single displaced supercell for a specific q-point and mode,
   and visualizes the mode.

   Args:
      primitive_atoms (ase.atoms.Atoms): The primitive cell structure (potentially relaxed).
      ph_obj (ase.phonons.Phonons): The Phonons object from the calculation.
      q_point (np.array): The q-point in fractional reciprocal coordinates.
      band_idx (int): The index of the mode (0-indexed).
      displacement_magnitude (float): The desired magnitude of displacement in Angstroms.
      output_base_dir (str): The main output directory.
      original_prefix (str): The base filename prefix of the original structure.
      max_atoms_per_supercell (int): Maximum number of atoms allowed in supercells (None = no limit).

   Returns:
      ase.atoms.Atoms: The generated displaced supercell atoms object, or None if generation fails.
   """
   print(f"\n--- Generating and Visualizing Displaced Supercell for Mode {band_idx} at q={q_point} ---")

   # 1. Get the raw eigenvector for the specified q-point and mode
   raw_displacements_primitive = get_eigenvector_for_q_and_band_index(ph_obj, q_point, band_idx)
   if raw_displacements_primitive is None:
      print("Failed to retrieve eigenvector. Exiting mode generation.")
      return None # Return None on failure

   # Normalize the eigenvector to get unit displacement vectors
   max_raw_disp_magnitude = np.max(np.linalg.norm(np.real(raw_displacements_primitive), axis=1))
   if max_raw_disp_magnitude < 1e-6:
      print("Warning: Eigenvector displacements are zero or very small. Cannot generate meaningful displacement.")
      normalized_displacements_primitive = raw_displacements_primitive # Keep as is if zero
   else:
      normalized_displacements_primitive = raw_displacements_primitive / max_raw_disp_magnitude

   # 2. Estimate commensurate supercell size
   supercell_dims = estimate_commensurate_supercell_size(q_point)

   # Apply atom count constraint if specified
   if max_atoms_per_supercell is not None:
      from .structure_utils import filter_supercells_by_atom_count
      supercell_dims = filter_supercells_by_atom_count(primitive_atoms, [supercell_dims], max_atoms_per_supercell)[0]

   sc_n1, sc_n2, sc_n3 = supercell_dims
   supercell_matrix = np.diag(np.array(supercell_dims))

   # Create a dedicated output directory for this specific mode generation
   q_point_str = "_".join([f"{c:.3f}" for c in q_point]).replace('.', 'p').replace('-', 'm') # For filename safety
   mode_gen_output_dir = os.path.join(output_base_dir, f"mode_gen_q_{q_point_str}_mode_{band_idx}_disp_{displacement_magnitude:.3f}")
   os.makedirs(mode_gen_output_dir, exist_ok=True)
   print(f"Output directory for mode generation: {mode_gen_output_dir}")

   # 3. Generate the supercell from the primitive cell
   # make_supercell with wrap=False to get unwrapped positions for phase factor application
   supercell_atoms_unwrapped = make_supercell(primitive_atoms, supercell_matrix, wrap=False)

   # Manually determine atom_map and cell_shifts_primitive_units for the supercell atoms
   num_atoms_supercell = len(supercell_atoms_unwrapped)
   atom_map = np.zeros(num_atoms_supercell, dtype=int)
   cell_shifts_primitive_units = np.zeros((num_atoms_supercell, 3), dtype=int)

   primitive_positions = primitive_atoms.get_positions()
   primitive_cell_inv = np.linalg.inv(primitive_atoms.get_cell())

   for i_sc in range(num_atoms_supercell):
      pos_sc_unwrapped = supercell_atoms_unwrapped.get_positions()[i_sc]
      frac_pos_in_primitive_basis = np.dot(pos_sc_unwrapped, primitive_cell_inv)

      min_dist = float('inf')
      best_prim_idx = -1
      for j_prim in range(len(primitive_atoms)):
         prim_frac_pos = np.dot(primitive_positions[j_prim], primitive_cell_inv)
         diff_frac = frac_pos_in_primitive_basis - prim_frac_pos
         diff_frac_wrapped = diff_frac - np.round(diff_frac)
         dist = np.linalg.norm(diff_frac_wrapped)
         if dist < min_dist:
               min_dist = dist
               best_prim_idx = j_prim

      atom_map[i_sc] = best_prim_idx
      prim_frac_pos_of_matched_atom = np.dot(primitive_positions[best_prim_idx], primitive_cell_inv)
      cell_shifts_primitive_units[i_sc] = np.round(frac_pos_in_primitive_basis - prim_frac_pos_of_matched_atom).astype(int)

   # Now, wrap the supercell atoms for actual structure representation (base for animation)
   supercell_atoms_base = supercell_atoms_unwrapped.copy()
   supercell_atoms_base.wrap(pbc=True)

   # 4. Apply displacements with phase factors to create the *single* displaced structure
   displaced_supercell_atoms = supercell_atoms_unwrapped.copy() # Start with unwrapped supercell for displacement application
   total_displacements_supercell_complex = np.zeros_like(supercell_atoms_unwrapped.get_positions(), dtype=complex)

   for i_sc in range(num_atoms_supercell):
      prim_idx = atom_map[i_sc]
      cell_shift_primitive_units_i_sc = cell_shifts_primitive_units[i_sc]

      # Calculate phase factor
      dot_product = np.dot(q_point, cell_shift_primitive_units_i_sc)
      phase_factor = np.exp(1j * 2 * np.pi * dot_product)

      # Apply scaled and phased displacement
      disp_vector_complex = normalized_displacements_primitive[prim_idx] * displacement_magnitude * phase_factor
      total_displacements_supercell_complex[i_sc] = disp_vector_complex

   # Apply the real part of the total displacements to get the final displaced structure
   displaced_supercell_atoms.set_positions(displaced_supercell_atoms.get_positions() + np.real(total_displacements_supercell_complex))
   displaced_supercell_atoms.wrap(pbc=True) # Wrap the final displaced positions

   # 5. Save the displaced supercell
   filename_base = f"{original_prefix}_q_{q_point_str}_mode_{band_idx}_disp_{displacement_magnitude:.3f}_sc_{sc_n1}x{sc_n2}x{sc_n3}"
   cif_path = os.path.join(mode_gen_output_dir, f"{filename_base}.cif")
   xyz_path = os.path.join(mode_gen_output_dir, f"{filename_base}.xyz")

   write(cif_path, displaced_supercell_atoms)
   write(xyz_path, displaced_supercell_atoms)
   print(f"Generated displaced supercell: {cif_path} and {xyz_path}")

   # 6. Generate visualization (trajectory) for the mode in the supercell
   # Create a trajectory of the oscillation around the *undisplaced* supercell
   num_frames = 200 # Number of frames for the animation
   traj_frames = []
   for i in range(num_frames):
      phase = 2 * np.pi * i / num_frames
      # Apply displacement with phase to the original supercell (supercell_atoms_base)
      current_frame_atoms = supercell_atoms_base.copy()
      # Lets remove the displacement factor, I want the animation with a smaller displacement
      displacements_for_animation = total_displacements_supercell_complex / displacement_magnitude
      # Calculate instantaneous displacement for this phase
      instantaneous_displacements = np.real(displacements_for_animation * np.exp(1j * phase))
      current_frame_atoms.set_positions(current_frame_atoms.get_positions() + instantaneous_displacements)
      current_frame_atoms.wrap(pbc=True)
      traj_frames.append(current_frame_atoms)

   traj_filename = os.path.join(mode_gen_output_dir, f"{filename_base}_mode_animation.traj")
   xyz_animation_filename = os.path.join(mode_gen_output_dir, f"{filename_base}_mode_animation.xyz")

   write(traj_filename, traj_frames)
   write(xyz_animation_filename, traj_frames)
   print(f"Generated mode animation trajectory: {traj_filename} and {xyz_animation_filename}")

   # Optional: Open visualization (requires GUI, might not work in all environments)
   # try:
   #     print("\nAttempting to open ASE viewer for mode animation (requires GUI environment)...")
   #     view(traj_frames)
   #     print("ASE viewer opened. Close the viewer to continue.")
   # except Exception as e:
   #     print(f"Could not open ASE viewer: {e}. Ensure X server is running and ASE GUI dependencies are installed.")
   print("You can view the generated .traj and .xyz files using external visualization tools.")

   print("\n--- Specific Mode Supercell Generation and Visualization Complete ---")
   return displaced_supercell_atoms # Return the generated displaced supercell

def generate_and_visualize_specific_mode_supercell_from_eigenmode(primitive_atoms, eigenvector, q_point, band_idx, displacement_magnitude, output_base_dir, original_prefix, frequency, units, max_atoms_per_supercell=None):
   """
   Generates a single displaced supercell for a specific eigenmode loaded from band.yaml,
   and visualizes the mode.

   Args:
      primitive_atoms (ase.atoms.Atoms): The primitive cell structure.
      eigenvector (np.ndarray): The complex eigenvector array of shape (natom, 3).
      q_point (np.array): The q-point in fractional reciprocal coordinates.
      band_idx (int): The index of the mode (0-indexed).
      displacement_magnitude (float): The desired magnitude of displacement in Angstroms.
      output_base_dir (str): The main output directory.
      original_prefix (str): The base filename prefix of the original structure.
      frequency (float): The frequency of the mode.
      units (str): Units for the frequency.
      max_atoms_per_supercell (int): Maximum number of atoms allowed in supercells (None = no limit).

   Returns:
      ase.atoms.Atoms: The generated displaced supercell atoms object, or None if generation fails.
   """
   print(f"\n--- Generating and Visualizing Displaced Supercell from Preloaded Eigenmode ---")
   print(f"Mode {band_idx} at q={q_point}, frequency={frequency:.4f} {units}")

   # Normalize the eigenvector to get unit displacement vectors
   max_raw_disp_magnitude = np.max(np.linalg.norm(np.real(eigenvector), axis=1))
   if max_raw_disp_magnitude < 1e-6:
      print("Warning: Eigenvector displacements are zero or very small. Cannot generate meaningful displacement.")
      normalized_displacements_primitive = eigenvector # Keep as is if zero
   else:
      normalized_displacements_primitive = eigenvector / max_raw_disp_magnitude

   # Estimate commensurate supercell size
   supercell_dims = estimate_commensurate_supercell_size(q_point)

   # Apply atom count constraint if specified
   if max_atoms_per_supercell is not None:
      from .structure_utils import filter_supercells_by_atom_count
      supercell_dims = filter_supercells_by_atom_count(primitive_atoms, [supercell_dims], max_atoms_per_supercell)[0]

   sc_n1, sc_n2, sc_n3 = supercell_dims
   supercell_matrix = np.diag(np.array(supercell_dims))

   # Create a dedicated output directory for this specific mode generation
   q_point_str = "_".join([f"{c:.3f}" for c in q_point]).replace('.', 'p').replace('-', 'm') # For filename safety
   mode_gen_output_dir = os.path.join(output_base_dir, f"preloaded_mode_q_{q_point_str}_mode_{band_idx}_disp_{displacement_magnitude:.3f}")
   os.makedirs(mode_gen_output_dir, exist_ok=True)
   print(f"Output directory for mode generation: {mode_gen_output_dir}")

   # Generate the supercell from the primitive cell
   # make_supercell with wrap=False to get unwrapped positions for phase factor application
   supercell_atoms_unwrapped = make_supercell(primitive_atoms, supercell_matrix, wrap=False)

   # Manually determine atom_map and cell_shifts_primitive_units for the supercell atoms
   num_atoms_supercell = len(supercell_atoms_unwrapped)
   atom_map = np.zeros(num_atoms_supercell, dtype=int)
   cell_shifts_primitive_units = np.zeros((num_atoms_supercell, 3), dtype=int)

   primitive_positions = primitive_atoms.get_positions()
   primitive_cell_inv = np.linalg.inv(primitive_atoms.get_cell())

   for i_sc in range(num_atoms_supercell):
      pos_sc_unwrapped = supercell_atoms_unwrapped.get_positions()[i_sc]
      frac_pos_in_primitive_basis = np.dot(pos_sc_unwrapped, primitive_cell_inv)

      min_dist = float('inf')
      best_prim_idx = -1
      for j_prim in range(len(primitive_atoms)):
         prim_frac_pos = np.dot(primitive_positions[j_prim], primitive_cell_inv)
         diff_frac = frac_pos_in_primitive_basis - prim_frac_pos
         diff_frac_wrapped = diff_frac - np.round(diff_frac)
         dist = np.linalg.norm(diff_frac_wrapped)
         if dist < min_dist:
               min_dist = dist
               best_prim_idx = j_prim

      atom_map[i_sc] = best_prim_idx
      prim_frac_pos_of_matched_atom = np.dot(primitive_positions[best_prim_idx], primitive_cell_inv)
      cell_shifts_primitive_units[i_sc] = np.round(frac_pos_in_primitive_basis - prim_frac_pos_of_matched_atom).astype(int)

   # Now, wrap the supercell atoms for actual structure representation (base for animation)
   supercell_atoms_base = supercell_atoms_unwrapped.copy()
   supercell_atoms_base.wrap(pbc=True)

   # Apply displacements with phase factors to create the *single* displaced structure
   displaced_supercell_atoms = supercell_atoms_unwrapped.copy() # Start with unwrapped supercell for displacement application
   total_displacements_supercell_complex = np.zeros_like(supercell_atoms_unwrapped.get_positions(), dtype=complex)

   for i_sc in range(num_atoms_supercell):
      prim_idx = atom_map[i_sc]
      cell_shift_primitive_units_i_sc = cell_shifts_primitive_units[i_sc]

      # Calculate phase factor
      dot_product = np.dot(q_point, cell_shift_primitive_units_i_sc)
      phase_factor = np.exp(1j * 2 * np.pi * dot_product)

      # Apply scaled and phased displacement
      disp_vector_complex = normalized_displacements_primitive[prim_idx] * displacement_magnitude * phase_factor
      total_displacements_supercell_complex[i_sc] = disp_vector_complex

   # Apply the real part of the total displacements to get the final displaced structure
   displaced_supercell_atoms.set_positions(displaced_supercell_atoms.get_positions() + np.real(total_displacements_supercell_complex))
   displaced_supercell_atoms.wrap(pbc=True) # Wrap the final displaced positions

   # Save the displaced supercell
   filename_base = f"{original_prefix}_preloaded_q_{q_point_str}_mode_{band_idx}_disp_{displacement_magnitude:.3f}_sc_{sc_n1}x{sc_n2}x{sc_n3}"
   cif_path = os.path.join(mode_gen_output_dir, f"{filename_base}.cif")
   xyz_path = os.path.join(mode_gen_output_dir, f"{filename_base}.xyz")

   write(cif_path, displaced_supercell_atoms)
   write(xyz_path, displaced_supercell_atoms)
   print(f"Generated displaced supercell: {cif_path} and {xyz_path}")

   # Generate visualization (trajectory) for the mode in the supercell
   # Create a trajectory of the oscillation around the *undisplaced* supercell
   num_frames = 200 # Number of frames for the animation
   traj_frames = []
   for i in range(num_frames):
      phase = 2 * np.pi * i / num_frames
      # Apply displacement with phase to the original supercell (supercell_atoms_base)
      current_frame_atoms = supercell_atoms_base.copy()
      # Lets remove the displacement factor, I want the animation with a smaller displacement
      displacements_for_animation = total_displacements_supercell_complex / displacement_magnitude
      # Calculate instantaneous displacement for this phase
      instantaneous_displacements = np.real(displacements_for_animation * np.exp(1j * phase))
      current_frame_atoms.set_positions(current_frame_atoms.get_positions() + instantaneous_displacements)
      current_frame_atoms.wrap(pbc=True)
      traj_frames.append(current_frame_atoms)

   traj_filename = os.path.join(mode_gen_output_dir, f"{filename_base}_mode_animation.traj")
   xyz_animation_filename = os.path.join(mode_gen_output_dir, f"{filename_base}_mode_animation.xyz")

   write(traj_filename, traj_frames)
   write(xyz_animation_filename, traj_frames)
   print(f"Generated mode animation trajectory: {traj_filename} and {xyz_animation_filename}")

   print("You can view the generated .traj and .xyz files using external visualization tools.")

   print("\n--- Preloaded Eigenmode Supercell Generation and Visualization Complete ---")
   return displaced_supercell_atoms # Return the generated displaced supercell