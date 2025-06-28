import os
import sys
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from m3gnet.models import M3GNet, M3GNetCalculator, Potential
import numpy as np
from .utils import HAVE_MACE, get_mace_device, mace_mp
from ase.build import make_supercell 
from ase.io import read, write
from ase.atoms import Atoms

def load_structure(cif_path):
   """Loads a structure from a CIF file and converts it to an ASE Atoms object."""
   try:
      struct = Structure.from_file(cif_path)
      atoms = AseAtomsAdaptor().get_atoms(struct)
      print(f"Successfully loaded structure from {cif_path}")
      return struct, atoms
   except Exception as e:
      print(f"Error reading CIF file {cif_path}: {e}")
      return None, None

def initialize_calculator(engine, model_name="medium-omat-0"):
   """Initializes and returns the appropriate calculator (M3GNet or MACE)."""
   calculator = None
   if engine == "m3gnet":
      print("Initializing M3GNet calculator...")
      potential = Potential(M3GNet.load())
      calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
   elif engine == "mace":
      if not HAVE_MACE:
         sys.exit("MACE not found – `pip install mace-torch` or use --engine m3gnet")

      device = get_mace_device()
      print(f"Initializing MACE calculator on device: {device}...")
      calculator = mace_mp(model=model_name,
                           dispersion=False,
                           default_dtype="float64",
                           device=device,
                           stress=True)
   else:
      print(f"Error: Engine '{engine}' not supported yet.")
      return None

   print(f"{engine.upper()} calculator initialized.")
   return calculator

def print_initial_structure_info(atoms):
   """Prints basic information about the initial structure."""  
   print("\n--- Initial Structure Information ---")  
   print(f"Formula: {atoms.get_chemical_formula()}")  
   print(f"Number of atoms: {len(atoms)}")  
   print(f"Cell volume: {atoms.get_volume():.4f} Å³")  
   print("Cell parameters (a, b, c, alpha, beta, gamma):")  
   cell = atoms.get_cell()  
   print(f"  a={cell.lengths()[0]:.4f}, b={cell.lengths()[1]:.4f}, c={cell.lengths()[2]:.4f}")  
   print(f"  alpha={cell.angles()[0]:.2f}, beta={cell.angles()[1]:.2f}, gamma={cell.angles()[2]:.2f}")  
   print("-------------------------------------")
   initial_fractional_coords = atoms.get_scaled_positions()
   print("   Initial Fractional Coordinates:")
   for i, pos in enumerate(initial_fractional_coords):
      print(f"      Atom {i+1}: {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}")

def print_final_structure_info(initial_atoms, final_atoms, initial_stress, final_stress):
   """Prints comparison information between initial and final ASE Atoms objects after relaxation."""
   initial_cell = initial_atoms.get_cell()
   initial_positions = initial_atoms.get_positions()
   initial_cell_params = initial_cell.cellpar()

   final_cell = final_atoms.get_cell()
   final_positions = final_atoms.get_positions()
   final_fractional_coords = final_atoms.get_scaled_positions()
   final_cell_params = final_cell.cellpar()

   print("\n### Relaxation Results ###")
   print("   Cell Parameters Change:")
   print(f"      Initial (a, b, c, alpha, beta, gamma): {initial_cell_params[0]:.4f}, {initial_cell_params[1]:.4f}, {initial_cell_params[2]:.4f}, {initial_cell_params[3]:.2f}, {initial_cell_params[4]:.2f}, {initial_cell_params[5]:.2f}")
   print(f"      Final   (a, b, c, alpha, beta, gamma): {final_cell_params[0]:.4f}, {final_cell_params[1]:.4f}, {final_cell_params[2]:.4f}, {final_cell_params[3]:.2f}, {final_cell_params[4]:.2f}, {final_cell_params[5]:.2f}")
   print(f"      Difference (a, b, c): {final_cell_params[0]-initial_cell_params[0]:.4f}, {final_cell_params[1]-initial_cell_params[1]:.4f}, {final_cell_params[2]-initial_cell_params[2]:.4f}")
   print(f"      Difference (alpha, beta, gamma): {final_cell_params[3]-initial_cell_params[3]:.2f}, {final_cell_params[4]-initial_cell_params[4]:.2f}, {final_cell_params[5]-initial_cell_params[5]:.2f}")

   print("\n   Fractional Coordinates After Relaxation:")
   for i, pos in enumerate(final_fractional_coords):
      print(f"      Atom {i+1}: {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}")

   # Calculate maximum atomic displacement
   displacements = final_positions - initial_positions
   max_displacement = np.max(np.linalg.norm(displacements, axis=1))
   print(f"\n   Maximum atomic displacement during relaxation: {max_displacement:.4f} Å")

   # Print initial and final stress for sanity check
   if initial_stress is not None:
      print(f"\n   Initial stress (GPa):\n{initial_stress/1e9}")
   if final_stress is not None:
      print(f"   Final stress (GPa):\n{final_stress/1e9}")

def save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir, suffix=""):
   """Saves the relaxed structure as a CIF file."""
   from pymatgen.io.ase import AseAtomsAdaptor

   cif_filename_base = os.path.splitext(os.path.basename(original_cif_path))[0]
   relaxed_cif_filename = f"{cif_filename_base}_relaxed_{engine}_f{fmax}{suffix}.cif"
   relaxed_cif_path = os.path.join(output_dir, relaxed_cif_filename)

   relaxed_struct = AseAtomsAdaptor().get_structure(relaxed_atoms)
   relaxed_struct.to(filename=relaxed_cif_path)
   print(f"Relaxed structure saved to: {relaxed_cif_path}")

import numpy as np
import os
from ase import Atoms
from ase.build import make_supercell
from ase.io import write
from ase.cell import Cell # Import Cell for creating new cell matrices

def generate_displaced_supercells(primitive_atoms,
                                  softest_modes_info_list, # Changed to a list of mode infos
                                  scale_mode1,             # New: Single scale for mode 1
                                  ratio_mode2_to_mode1,    # Single ratio for mode 2
                                  supercell_variants,
                                  output_base_dir,
                                  iteration_idx,
                                  original_prefix,
                                  cell_transformation_vector): # 6-element vector
   """
   Generates supercells displaced along a combination of soft phonon modes
   with flexible cell parameter transformations.

   Args:
      primitive_atoms (ase.atoms.Atoms): The primitive cell structure.
      softest_modes_info_list (list): A list containing dictionaries for the softest mode
                                       (index 0) and potentially the second softest mode (index 1).
                                       Each dict includes 'raw_displacements'.
      scale_mode1 (float): The scaling factor for the raw displacements of the first softest mode.
      ratio_mode2_to_mode1 (float): Ratio of displacement magnitude of mode 2 to mode 1.
      supercell_variants (list): List of tuples defining supercell sizes, e.g., [(2,1,1), (2,2,2)].
      output_base_dir (str): The main output directory.
      iteration_idx (int): The current iteration number.
      original_prefix (str): The base filename prefix of the original structure.
      cell_transformation_vector (tuple): A 6-element tuple (scale_a, scale_b, scale_c,
                                          scale_alpha, scale_beta, scale_gamma) for cell transformation.

   Returns:
      list: A list of paths to the generated displaced CIF files.
   """
   print(f"\n--- Generating Displaced Supercells (Iteration {iteration_idx}) ---")

   # Extract mode info from the list
   softest_mode_info_1 = softest_modes_info_list[0] if softest_modes_info_list else None
   softest_mode_info_2 = softest_modes_info_list[1] if len(softest_modes_info_list) > 1 else None

   if softest_mode_info_1 is None:
       print("Error: No softest mode information provided. Cannot generate displaced structures.")
       return []

   soft_mode_1_label = softest_mode_info_1.get('label', 'unknown_1')
   soft_mode_2_label = softest_mode_info_2.get('label', 'unknown_2') if softest_mode_info_2 else 'none'

   # Create a more descriptive directory name for the iteration
   iteration_dir = os.path.join(output_base_dir, f"iteration_{iteration_idx}")
   os.makedirs(iteration_dir, exist_ok=True)
   print(f"Created directory for iteration {iteration_idx}: {iteration_dir}")

   raw_displacements_1 = np.array(softest_mode_info_1['raw_displacements'])
   num_atoms_primitive = len(primitive_atoms)

   max_raw_disp_magnitude_1 = np.max(np.linalg.norm(raw_displacements_1, axis=1))

   if max_raw_disp_magnitude_1 < 1e-6:
      print("Warning: Softest mode 1 displacements are zero or very small. Cannot generate displaced structures.")
      # If mode 1 is zero, and mode 2 is also zero or not present, return empty
      if softest_mode_info_2 is None or np.max(np.linalg.norm(np.array(softest_mode_info_2['raw_displacements']), axis=1)) < 1e-6:
          return []

   normalized_displacements_1 = raw_displacements_1 / max_raw_disp_magnitude_1

   normalized_displacements_2 = None
   max_raw_disp_magnitude_2 = 0.0
   if softest_mode_info_2:
       raw_displacements_2 = np.array(softest_mode_info_2['raw_displacements'])
       max_raw_disp_magnitude_2 = np.max(np.linalg.norm(raw_displacements_2, axis=1))
       if max_raw_disp_magnitude_2 > 1e-6:
           normalized_displacements_2 = raw_displacements_2 / max_raw_disp_magnitude_2
       else:
           print("Warning: Second softest mode displacements are zero or very small. Will not combine.")


   generated_files = []

   # Ensure primitive_atoms is an ASE Atoms object
   if not isinstance(primitive_atoms, Atoms):
       print(f"Attempting forceful conversion of primitive_atoms from {type(primitive_atoms)} to ase.atoms.Atoms...")
       primitive_atoms = Atoms(
           symbols=primitive_atoms.get_chemical_symbols(),
           positions=primitive_atoms.get_positions(),
           cell=primitive_atoms.get_cell(),
           pbc=primitive_atoms.get_pbc()
       )
       print(f"Forceful conversion complete. New primitive_atoms type: {type(primitive_atoms)}")

   # Apply cell transformation to the primitive cell first
   transformed_primitive_atoms = primitive_atoms.copy()
   original_cell_params = transformed_primitive_atoms.get_cell().cellpar() # (a, b, c, alpha, beta, gamma)

   scale_a, scale_b, scale_c, scale_alpha, scale_beta, scale_gamma = cell_transformation_vector

   new_a = original_cell_params[0] * (1.0 + scale_a)
   new_b = original_cell_params[1] * (1.0 + scale_b)
   new_c = original_cell_params[2] * (1.0 + scale_c)
   new_alpha = original_cell_params[3] + scale_alpha
   new_beta = original_cell_params[4] + scale_beta
   new_gamma = original_cell_params[5] + scale_gamma

   # Ensure angles are within reasonable bounds (e.g., 0 to 180 degrees)
   new_alpha = np.clip(new_alpha, 1e-6, 179.999)
   new_beta = np.clip(new_beta, 1e-6, 179.999)
   new_gamma = np.clip(new_gamma, 1e-6, 179.999)

   new_cell_params = (new_a, new_b, new_c, new_alpha, new_beta, new_gamma)
   new_cell_matrix = Cell.fromcellpar(new_cell_params)

   transformed_primitive_atoms.set_cell(new_cell_matrix, scale_atoms=True) # scale_atoms=True moves atoms proportionally

   # Create labels for the cell transformation vector for filename
   cell_transform_labels = []
   for val in cell_transformation_vector:
       if val == 0:
           cell_transform_labels.append("000")
       else:
           # Format as 'p050' for +0.05, 'm020' for -0.02, 'p005' for +0.005
           # Using 3 decimal places for precision in filename
           cell_transform_labels.append(f"{'m' if val < 0 else 'p'}{abs(int(val*1000)):03d}")
   cell_transform_str = "_".join(cell_transform_labels)


   for supercell_variant in supercell_variants:
      sc_n1, sc_n2, sc_n3 = supercell_variant
      supercell_dir = os.path.join(iteration_dir, f"supercell_{sc_n1}x{sc_n2}x{sc_n3}")
      os.makedirs(supercell_dir, exist_ok=True)
      print(f"  Created directory for supercell variant: {supercell_dir}")

      supercell_variant_matrix = np.diag(np.array(supercell_variant))
      # Create supercell from the *transformed* primitive atoms
      supercell_atoms_base = make_supercell(transformed_primitive_atoms, supercell_variant_matrix)
      num_atoms_supercell = len(supercell_atoms_base)

      primitive_positions = transformed_primitive_atoms.get_positions()
      primitive_cell = transformed_primitive_atoms.get_cell()
      primitive_symbols = transformed_primitive_atoms.get_chemical_symbols()

      primitive_to_supercell_indices = [[] for _ in range(num_atoms_primitive)]

      # Map supercell atoms back to their original primitive cell atom index
      # This is crucial for applying primitive cell displacements to supercell atoms
      for i_sc in range(num_atoms_supercell):
          pos_sc = supercell_atoms_base.get_positions()[i_sc]
          symbol_sc = supercell_atoms_base.get_chemical_symbols()[i_sc]

          # Use scaled_positions to map supercell atom back to primitive cell fractional coordinates
          f_prim = primitive_cell.scaled_positions(pos_sc)

          # Wrap fractional coordinates to [0, 1) range
          f_prim_wrapped = f_prim % 1.0
          # Handle floating point inaccuracies near 1.0, map to 0.0
          f_prim_wrapped = np.where(np.isclose(f_prim_wrapped, 1.0), 0.0, f_prim_wrapped)

          found_match = False
          for i_prim in range(num_atoms_primitive):
              pos_prim_orig = primitive_positions[i_prim]
              symbol_prim_orig = primitive_symbols[i_prim]

              if symbol_sc == symbol_prim_orig:
                  f_prim_orig = primitive_cell.scaled_positions(pos_prim_orig)
                  # Compare wrapped fractional coordinates
                  if np.allclose(f_prim_wrapped, f_prim_orig % 1.0, atol=1e-4): # Compare wrapped original too
                      primitive_to_supercell_indices[i_prim].append(i_sc)
                      found_match = True
                      break

          if not found_match:
               print(f"Warning: Could not map supercell atom {i_sc} ({symbol_sc}) at {pos_sc} to a primitive atom.")


      displaced_atoms = supercell_atoms_base.copy()
      total_displacements_for_this_sample = np.zeros_like(supercell_atoms_base.get_positions())

      for i_prim in range(num_atoms_primitive):
           # Calculate displacement for mode 1
           disp_mode1_vector = normalized_displacements_1[i_prim] * scale_mode1 * max_raw_disp_magnitude_1

           # Calculate displacement for mode 2, if available
           disp_mode2_vector = np.zeros(3)
           if normalized_displacements_2 is not None:
               # Scale mode 2 displacement by ratio_mode2_to_mode1 and also by scale_mode1
               # Note: The problem statement implies ratio_mode2_to_mode1 scales the *magnitude*
               # of mode 2 relative to mode 1's *magnitude*.
               # So, total_disp = (scale_mode1 * normalized_disp_mode1) + (scale_mode1 * ratio_mode2_to_mode1 * normalized_disp_mode2)
               # This means the effective scale for mode 2 is (scale_mode1 * ratio_mode2_to_mode1)
               disp_mode2_vector = normalized_displacements_2[i_prim] * (scale_mode1 * ratio_mode2_to_mode1) * max_raw_disp_magnitude_2

           # Combine displacements
           combined_disp_vector = disp_mode1_vector + disp_mode2_vector

           for i_sc in primitive_to_supercell_indices[i_prim]:
                total_displacements_for_this_sample[i_sc] = combined_disp_vector

      displaced_atoms.set_positions(displaced_atoms.get_positions() + total_displacements_for_this_sample)

      # Filename convention update
      # d1 for displacement scale of mode 1
      # r21 for ratio of mode 2 to mode 1
      # c_ for cell transformation vector
      # Using f-strings for precise formatting of floats in filename
      filename = (f"{original_prefix}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}.cif")
      filepath = os.path.join(supercell_dir, filename)
      write(filepath, displaced_atoms) # displaced_atoms already has the cell transformation applied
      generated_files.append(filepath)

      filename_xyz = (f"{original_prefix}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}.xyz")
      filepath_xyz = os.path.join(supercell_dir, filename_xyz)
      write(filepath_xyz, displaced_atoms)
      print(f"  Generated {filename} and {filename_xyz}")

   print("Finished generating displaced supercells with combined modes and cell transformations.")
   return generated_files