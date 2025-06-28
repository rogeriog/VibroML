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

def save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir):
   """Saves the relaxed structure as a CIF file."""
   from pymatgen.io.ase import AseAtomsAdaptor # Import here to avoid circular dependency if structure_utils imports relaxation_utils

   cif_filename_base = os.path.splitext(os.path.basename(original_cif_path))[0]
   relaxed_cif_filename = f"{cif_filename_base}_relaxed_{engine}_f{fmax}.cif"
   relaxed_cif_path = os.path.join(output_dir, relaxed_cif_filename)

   # Convert ASE Atoms object back to Pymatgen Structure for CIF writing
   relaxed_struct = AseAtomsAdaptor().get_structure(relaxed_atoms)
   relaxed_struct.to(filename=relaxed_cif_path)
   print(f"Relaxed structure saved to: {relaxed_cif_path}")


def generate_displaced_supercells(primitive_atoms, softest_mode_info, supercell_variants, displacement_scales, output_base_dir, iteration_idx, original_prefix, cell_scale_factors):
   """
   Generates supercells displaced along the softest phonon mode with cell parameter variations.

   Args:
      primitive_atoms (ase.atoms.Atoms): The primitive cell structure.
      softest_mode_info (dict): Dictionary containing softest mode information,
                                 including 'raw_displacements'.
      supercell_variants (list): List of tuples defining supercell sizes, e.g., [(2,1,1), (2,2,2)].
      displacement_scales (list): List of scaling factors for the raw displacements.
      output_base_dir (str): The main output directory.
      iteration_idx (int): The current iteration number.
      original_prefix (str): The base filename prefix of the original structure.

   Returns:
      list: A list of paths to the generated displaced CIF files.
   """
   print(f"\n--- Generating Displaced Supercells (Iteration {iteration_idx}) ---")

   soft_mode_label = softest_mode_info.get('label', 'unknown')
   soft_mode_dir = os.path.join(output_base_dir, f"soft_mode_{iteration_idx}_{soft_mode_label}")
   os.makedirs(soft_mode_dir, exist_ok=True)
   print(f"Created directory for soft mode {iteration_idx}: {soft_mode_dir}")

   raw_displacements = np.array(softest_mode_info['raw_displacements'])
   num_atoms_primitive = len(primitive_atoms)

   # Calculate the magnitude of the raw displacement vector for normalization
   max_raw_disp_magnitude = np.max(np.linalg.norm(raw_displacements, axis=1))

   if max_raw_disp_magnitude < 1e-6:
      print("Warning: Softest mode displacements are zero or very small. Cannot generate displaced structures.")
      return []

   # Normalize the raw displacements
   normalized_displacements = raw_displacements / max_raw_disp_magnitude

   # Define cell parameter scaling labels
   cell_scale_labels = [f"{'m' if x < 0 else 'p'}{abs(int(x*100)):02d}" if x != 0 else "00" for x in cell_scale_factors] 

   generated_files = []

   # --- FORCEFUL CONVERSION HERE ---
   # Regardless of its current type, create a new pure ase.atoms.Atoms object.
   print(f"Attempting forceful conversion of primitive_atoms from {type(primitive_atoms)} to ase.atoms.Atoms...")
   primitive_atoms = Atoms(
       symbols=primitive_atoms.get_chemical_symbols(),
       positions=primitive_atoms.get_positions(),
       cell=primitive_atoms.get_cell(),
       pbc=primitive_atoms.get_pbc()
   )
   print(f"Forceful conversion complete. New primitive_atoms type: {type(primitive_atoms)}")
   # --- END FORCEFUL CONVERSION ---


   for supercell_variant in supercell_variants:
      sc_n1, sc_n2, sc_n3 = supercell_variant
      supercell_dir = os.path.join(soft_mode_dir, f"supercell_{sc_n1}x{sc_n2}x{sc_n3}")
      os.makedirs(supercell_dir, exist_ok=True)
      print(f"  Created directory for supercell variant: {supercell_dir}")

      # --- Debug prints (keep these for now to confirm the type) ---
      print("\n--- Debugging make_supercell inputs (after forceful conversion) ---")
      print(f"  primitive_atoms type: {type(primitive_atoms)}")
      print(f"  primitive_atoms.cell type: {type(primitive_atoms.cell)}")
      print(f"  primitive_atoms.cell value:\n{primitive_atoms.cell}")
      print(f"  primitive_atoms.cell.array shape: {primitive_atoms.cell.array.shape}")
      print(f"  primitive_atoms.positions type: {type(primitive_atoms.positions)}")
      print(f"  primitive_atoms.positions shape: {primitive_atoms.positions.shape}")
      print(f"  supercell_variant type: {type(supercell_variant)}")
      print(f"  supercell_variant value: {supercell_variant}")
      print("---------------------------------------")
      # --- End debug prints ---
      
      # Create the supercell
      supercell_variant_matrix = np.diag(np.array(supercell_variant))
      supercell_atoms = make_supercell(primitive_atoms, supercell_variant_matrix)
      num_atoms_supercell = len(supercell_atoms)

      # Map primitive cell indices to supercell indices and displacements
      primitive_positions = primitive_atoms.get_positions()
      primitive_cell = primitive_atoms.get_cell()
      primitive_symbols = primitive_atoms.get_chemical_symbols()

      supercell_positions = supercell_atoms.get_positions()
      supercell_symbols = supercell_atoms.get_chemical_symbols()

      primitive_to_supercell_indices = [[] for _ in range(num_atoms_primitive)]

      # Iterate through each atom in the supercell to map it back to its primitive origin
      for i_sc in range(num_atoms_supercell):
          pos_sc = supercell_positions[i_sc]
          symbol_sc = supercell_symbols[i_sc]

          # Convert supercell position to fractional coordinates relative to the primitive cell
          f_prim = primitive_cell.scaled_positions(pos_sc)[0]

          # Wrap fractional coordinates into the [0, 1) range
          f_prim_wrapped = f_prim % 1.0
          f_prim_wrapped = np.where(np.isclose(f_prim_wrapped, 1.0), 0.0, f_prim_wrapped)

          # Find which primitive atom this corresponds to
          found_match = False
          for i_prim in range(num_atoms_primitive):
              pos_prim_orig = primitive_positions[i_prim]
              symbol_prim_orig = primitive_symbols[i_prim]

              if symbol_sc == symbol_prim_orig:
                  f_prim_orig = primitive_cell.scaled_positions(pos_prim_orig)[0]
                  if np.allclose(f_prim_wrapped, f_prim_orig, atol=1e-4):
                      primitive_to_supercell_indices[i_prim].append(i_sc)
                      found_match = True
                      break

          if not found_match:
               print(f"Warning: Could not map supercell atom {i_sc} ({symbol_sc}) at {pos_sc} to a primitive atom.")


      # Now apply scaled displacements with cell parameter variations
      for scale in displacement_scales:
          displaced_atoms = supercell_atoms.copy()
          # Create a zero array for displacements in the supercell
          displacements_for_this_scale = np.zeros_like(supercell_atoms.get_positions())

          for i_prim in range(num_atoms_primitive):
               # The displacement vector for a primitive atom
               disp_prim_vector = normalized_displacements[i_prim] * scale * max_raw_disp_magnitude # Scale by the desired amplitude

               # Apply this displacement to all corresponding atoms in the supercell
               for i_sc in primitive_to_supercell_indices[i_prim]:
                    displacements_for_this_scale[i_sc] = disp_prim_vector

          displaced_atoms.set_positions(displaced_atoms.get_positions() + displacements_for_this_scale)

          # Generate variants with different cell parameter scalings
          for cell_scale, cell_label in zip(cell_scale_factors, cell_scale_labels):
              # Create a copy of the displaced atoms for this cell scaling
              cell_scaled_atoms = displaced_atoms.copy()
              
              # Get the original cell and scale it
              original_cell = displaced_atoms.get_cell()
              scaled_cell = original_cell * (1.0 + cell_scale)
              
              # Set the new scaled cell
              cell_scaled_atoms.set_cell(scaled_cell, scale_atoms=True)
              
              # Save the displaced and cell-scaled structure
              filename = f"{original_prefix}_d{scale:.3f}_c{cell_label}.cif"
              filepath = os.path.join(supercell_dir, filename)
              write(filepath, cell_scaled_atoms)
              generated_files.append(filepath)

              # --- NEW: Save as .xyz as well ---  
              filename_xyz = f"{original_prefix}_d{scale:.3f}_c{cell_label}.xyz"  
              filepath_xyz = os.path.join(supercell_dir, filename_xyz)  
              write(filepath_xyz, cell_scaled_atoms)  
              # No need to add to generated_files as we only track CIFs for relaxation  
              print(f"  Generated {filename} (displacement: {scale:.3f}, cell scale: {cell_scale:+.1%}) and {filename_xyz}")

   print("Finished generating displaced supercells with cell parameter variations.")
   return generated_files