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
from ase.cell import Cell
from ase.geometry.cell import cellpar_to_cell
from fractions import Fraction # For estimating supercell size
from ase.visualize import view 


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

def estimate_commensurate_supercell_size(q_point_frac, max_denominator=10):
    """
    Estimates the smallest integer supercell dimensions (N1, N2, N3)
    that are commensurate with the given q-point in fractional reciprocal coordinates.
    This means the q-point will fold to the Gamma point in the supercell.

    Args:
        q_point_frac (list or np.array): The q-point in fractional reciprocal coordinates
                                          of the primitive cell, e.g., [0.5, 0.0, 0.0].
        max_denominator (int): Maximum denominator to consider when converting to fraction.
                                Helps prevent extremely large supercells for irrational-like q-points.

    Returns:
        tuple: A tuple (N1, N2, N3) representing the suggested supercell dimensions.
               Returns (1,1,1) if q_point is [0,0,0] or very close to it.
    """
    supercell_dims = [1, 1, 1]
    q_point_frac = np.array(q_point_frac)

    if np.allclose(q_point_frac, [0.0, 0.0, 0.0], atol=1e-6):
        print("Q-point is Gamma (0,0,0). Smallest commensurate supercell is (1,1,1).")
        return (1, 1, 1)

    for i, q_comp in enumerate(q_point_frac):
        # Handle components very close to 0 or 1 (or other integers)
        if np.isclose(q_comp % 1.0, 0.0, atol=1e-6):
            supercell_dims[i] = 1
            continue

        # Convert to fraction to find the smallest denominator
        try:
            fraction = Fraction(q_comp).limit_denominator(max_denominator)
            supercell_dims[i] = fraction.denominator
        except OverflowError:
            print(f"Warning: Could not find a simple fraction for q-component {q_comp}. "
                  f"Consider increasing max_denominator or checking q-point validity.")
            supercell_dims[i] = max_denominator # Fallback to max_denominator
        except Exception as e:
            print(f"Error processing q-component {q_comp}: {e}. Setting supercell dim to 1.")
            supercell_dims[i] = 1 # Default to 1 on error

    print(f"Estimated smallest commensurate supercell for q-point {q_point_frac}: {tuple(supercell_dims)}")
    return tuple(supercell_dims)


def generate_displaced_supercells(primitive_atoms,
                                  softest_modes_info_list, # Changed to a list of mode infos
                                  scale_mode1,             # Scaling factor for mode 1 displacements
                                  ratio_mode2_to_mode1,    # Single ratio for mode 2
                                  supercell_variants,
                                  output_base_dir,
                                  iteration_idx,
                                  original_prefix,
                                  cell_transformation_vector): # 6-element vector
   """
   Generates supercells displaced along a combination of soft phonon modes
   with flexible cell parameter transformations, considering q-point phase factors.

   Args:
      primitive_atoms (ase.atoms.Atoms): The primitive cell structure.
      softest_modes_info_list (list): A list containing dictionaries for the softest mode
                                       (index 0) and potentially the second softest mode (index 1).
                                       Each dict MUST include 'raw_displacements' and 'coordinate' (q_point).
      scale_mode1 (float): The scaling factor for the raw displacements of the first softest mode.
      ratio_mode2_to_mode1 (float): Ratio of displacement magnitude of mode 2 to mode 1.
      supercell_variants (list): List of tuples defining supercell sizes, e.g., [(2,1,1), (2,2,2)].
                                 It is recommended to use `estimate_commensurate_supercell_size`
                                 to determine appropriate supercell sizes based on the q-points.
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

   

   raw_displacements_1 = np.array(softest_mode_info_1['raw_displacements'])
   # Use 'coordinate' key for q_point, as set in phonon_utils.py
   q_point_frac_1 = np.array(softest_mode_info_1.get('coordinate', [0.0, 0.0, 0.0]))
   if 'coordinate' not in softest_mode_info_1:
       print(f"Warning: 'coordinate' (q_point) not found for mode 1. Assuming Gamma point (q=[0,0,0]).")

   num_atoms_primitive = len(primitive_atoms)

   max_raw_disp_magnitude_1 = np.max(np.linalg.norm(raw_displacements_1, axis=1))

   if max_raw_disp_magnitude_1 < 1e-6:
      print("Warning: Softest mode 1 displacements are zero or very small. Cannot generate displaced structures.")
      # If mode 1 is zero, and mode 2 is also zero or not present, return empty
      if softest_mode_info_2 is None or np.max(np.linalg.norm(np.array(softest_mode_info_2['raw_displacements']), axis=1)) < 1e-6:
          return []

   # Normalize displacements. These can be complex.
   normalized_displacements_1 = raw_displacements_1 / max_raw_disp_magnitude_1

   normalized_displacements_2 = None
   max_raw_disp_magnitude_2 = 0.0
   q_point_frac_2 = np.array([0.0, 0.0, 0.0])
   if softest_mode_info_2:
       raw_displacements_2 = np.array(softest_mode_info_2['raw_displacements'])
       q_point_frac_2 = np.array(softest_mode_info_2.get('coordinate', [0.0, 0.0, 0.0]))
       if 'coordinate' not in softest_mode_info_2:
           print(f"Warning: 'coordinate' (q_point) not found for mode 2. Assuming Gamma point (q=[0,0,0]).")

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

   # --- ROBUST ANGLE HANDLING ---
   # Ensure angles are strictly positive and less than 180, with a margin
   # This is crucial to avoid numerical issues that lead to cz_sqr < 0
   angle_min_bound = 5.0 # Keep angles at least 5 degrees away from 0
   angle_max_bound = 175.0 # Keep angles at least 5 degrees away from 180

   new_alpha = np.clip(new_alpha, angle_min_bound, angle_max_bound)
   new_beta = np.clip(new_beta, angle_min_bound, angle_max_bound)
   new_gamma = np.clip(new_gamma, angle_min_bound, angle_max_bound)

   new_cell_params = (new_a, new_b, new_c, new_alpha, new_beta, new_gamma)

   try:
       # Attempt to create the cell matrix. This is where the AssertionError happens.
       new_cell_matrix = cellpar_to_cell(new_cell_params)
   except AssertionError:
       print(f"Warning: Generated cell parameters are unphysical for sample {original_prefix} "
             f"with cell_transformation_vector {cell_transformation_vector}. "
             f"Resulting parameters: {new_cell_params}. Skipping this sample.")
       # Return an empty list, indicating no structures were generated for this sample.
       # The GA should then assign a very high (bad) fitness to this sample.
       return []
   except Exception as e:
       print(f"An unexpected error occurred while creating cell matrix for sample {original_prefix}: {e}. Skipping this sample.")
       return []

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
      

      supercell_variant_matrix = np.diag(np.array(supercell_variant))

      # Generate supercell without atom_map
      supercell_atoms_unwrapped = make_supercell(transformed_primitive_atoms, supercell_variant_matrix, wrap=False)

      # Now, manually determine atom_map and cell_shifts_primitive_units
      num_atoms_supercell = len(supercell_atoms_unwrapped)
      atom_map = np.zeros(num_atoms_supercell, dtype=int)
      # Initialize cell_shifts_primitive_units correctly
      cell_shifts_primitive_units = np.zeros((num_atoms_supercell, 3), dtype=int)

      # Get primitive cell positions and inverse cell matrix for fractional coordinates
      primitive_positions = transformed_primitive_atoms.get_positions()
      primitive_cell_inv = np.linalg.inv(transformed_primitive_atoms.get_cell())

      for i_sc in range(num_atoms_supercell):
          pos_sc_unwrapped = supercell_atoms_unwrapped.get_positions()[i_sc]

          # Convert supercell atom position to fractional coordinates in the primitive cell basis
          # This gives us (primitive_atom_frac_pos + cell_shift_frac_pos)
          frac_pos_in_primitive_basis = np.dot(pos_sc_unwrapped, primitive_cell_inv)

          # Find the closest primitive atom by checking the fractional part
          # The fractional part should be close to one of the primitive atom's fractional positions
          min_dist = float('inf')
          best_prim_idx = -1
          for j_prim in range(num_atoms_primitive):
              prim_frac_pos = np.dot(primitive_positions[j_prim], primitive_cell_inv)
              # Calculate the difference in fractional coordinates, wrapping around 0 and 1
              diff_frac = frac_pos_in_primitive_basis - prim_frac_pos
              diff_frac_wrapped = diff_frac - np.round(diff_frac) # This gives the "wrapped" difference

              dist = np.linalg.norm(diff_frac_wrapped)
              if dist < min_dist:
                  min_dist = dist
                  best_prim_idx = j_prim

          atom_map[i_sc] = best_prim_idx

          # Calculate the integer cell shift (R_n) for this supercell atom
          # R_n = (supercell_atom_unwrapped_pos - primitive_atom_original_pos) in primitive cell basis
          # This is the integer part of frac_pos_in_primitive_basis - primitive_atom_frac_pos
          prim_frac_pos_of_matched_atom = np.dot(primitive_positions[best_prim_idx], primitive_cell_inv)
          # Corrected assignment: use the initialized array name
          cell_shifts_primitive_units[i_sc] = np.round(frac_pos_in_primitive_basis - prim_frac_pos_of_matched_atom).astype(int)


      # Now, wrap the supercell atoms for actual structure representation
      supercell_atoms_base = supercell_atoms_unwrapped.copy()
      supercell_atoms_base.wrap(pbc=True) # Wrap positions back into the cell

      displaced_atoms = supercell_atoms_base.copy()
      total_displacements_for_this_sample = np.zeros_like(supercell_atoms_base.get_positions(), dtype=complex) # Use complex for intermediate calculations

      for i_sc in range(num_atoms_supercell):
           prim_idx = atom_map[i_sc] # Get the original primitive atom index for this supercell atom

           # Get the pre-calculated integer cell shift (n1, n2, n3) for this supercell atom
           # Corrected access: use the initialized array name
           cell_shift_primitive_units_i_sc = cell_shifts_primitive_units[i_sc]

           # Calculate phase factor for mode 1
           dot_product_1 = np.dot(q_point_frac_1, cell_shift_primitive_units_i_sc)
           phase_factor_1 = np.exp(1j * 2 * np.pi * dot_product_1)

           # Calculate displacement for mode 1 (can be complex)
           disp_mode1_vector_complex = normalized_displacements_1[prim_idx] * scale_mode1 * max_raw_disp_magnitude_1 * phase_factor_1

           # Calculate displacement for mode 2, if available (can be complex)
           disp_mode2_vector_complex = np.zeros(3, dtype=complex)
           if normalized_displacements_2 is not None:
               dot_product_2 = np.dot(q_point_frac_2, cell_shift_primitive_units_i_sc)
               phase_factor_2 = np.exp(1j * 2 * np.pi * dot_product_2)

               disp_mode2_vector_complex = normalized_displacements_2[prim_idx] * (scale_mode1 * ratio_mode2_to_mode1) * max_raw_disp_magnitude_2 * phase_factor_2

           # Combine complex displacements
           combined_disp_vector_complex = disp_mode1_vector_complex + disp_mode2_vector_complex

           # Store the real part for the final physical displacement
           total_displacements_for_this_sample[i_sc] = combined_disp_vector_complex

      # Apply the real part of the total displacements to the supercell atoms
      displaced_atoms.set_positions(displaced_atoms.get_positions() + np.real(total_displacements_for_this_sample))

      # Filename convention update
      # d1 for displacement scale of mode 1
      # r21 for ratio of mode 2 to mode 1
      # c_ for cell transformation vector
      # Using f-strings for precise formatting of floats in filename
      filename = (f"{original_prefix}_sc_{sc_n1}x{sc_n2}x{sc_n3}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}.cif")
      filepath = os.path.join(output_base_dir, filename)
      write(filepath, displaced_atoms) # displaced_atoms already has the cell transformation applied
      generated_files.append(filepath)

      filename_xyz = (f"{original_prefix}_sc_{sc_n1}x{sc_n2}x{sc_n3}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}.xyz")
      filepath_xyz = os.path.join(output_base_dir, filename_xyz)
      write(filepath_xyz, displaced_atoms)
      print(f"  Generated {filename} and {filename_xyz}")

   print("Finished generating displaced supercells with combined modes and cell transformations.")
   return generated_files

