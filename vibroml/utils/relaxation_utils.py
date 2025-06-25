import os
import time
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from m3gnet.models import Relaxer
from pymatgen.io.ase import AseAtomsAdaptor
from .structure_utils import print_final_structure_info, save_relaxed_structure
from ase.io import read, write
from ase.calculators.calculator import Calculator
import numpy as np

import spglib
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def relax_structure(atoms, calculator, engine, fmax, output_dir, original_cif_path):
   """Performs structure relaxation using the specified engine."""
   print(f"\n» {engine.upper()} relaxation starting…")
   start_time = time.time()

   initial_atoms = atoms.copy()

   print("\n--- Analyzing Initial Structure Symmetry ---")
   analyze_symmetry(initial_atoms, output_dir, prefix="initial", auto_tune_symprec=True)
   print("------------------------------------------")

   atoms.set_calculator(calculator)

   initial_stress = None
   initial_energy = None
   initial_energy_per_atom = None
   try:
      initial_stress = atoms.get_stress()
      initial_energy = atoms.get_potential_energy()
      initial_energy_per_atom = initial_energy / len(atoms)
      print(f"   Initial energy: {initial_energy:.6f} eV")
      print(f"   Initial energy per atom: {initial_energy_per_atom:.6f} eV/atom")
   except Exception as e:
      print(f"\n   Could not retrieve initial energy/stress: {e}")

   relaxed_atoms = None
   relax_traj_path = os.path.join(output_dir, "relax.traj")

   if engine == "mace":
      ucf = UnitCellFilter(atoms)
      relax_log_path = os.path.join(output_dir, "relax.log")
      print(f"   MACE relaxation log will be written to: {relax_log_path}")
      print(f"   MACE relaxation trajectory will be written to: {relax_traj_path}")
      opt = BFGS(ucf, logfile=relax_log_path, trajectory=relax_traj_path)
      opt.run(fmax=fmax, steps=100)
      relaxed_atoms = ucf.atoms.copy()
   elif engine == "m3gnet":
      relaxer = Relaxer()
      print(f"   M3GNet relaxation will generate a trajectory at: {relax_traj_path}")
      relax_results = relaxer.relax(atoms, fmax=fmax, verbose=True)

      relaxed_atoms = AseAtomsAdaptor().get_atoms(relax_results['final_structure'])

      if 'trajectory' in relax_results and relax_results['trajectory']:
          ase_trajectory_atoms = [AseAtomsAdaptor().get_atoms(s) for s in relax_results['trajectory']]
          if not np.allclose(ase_trajectory_atoms[0].positions, initial_atoms.positions):
              ase_trajectory_atoms.insert(0, initial_atoms)
          if not np.allclose(ase_trajectory_atoms[-1].positions, relaxed_atoms.positions):
              ase_trajectory_atoms.append(relaxed_atoms)

          try:
              write(relax_traj_path, ase_trajectory_atoms)
              print(f"   M3GNet relaxation trajectory saved to {relax_traj_path}")
          except Exception as e:
              print(f"   Error writing M3GNet trajectory: {e}")
              import traceback
              traceback.print_exc()
      else:
          print("   No intermediate trajectory frames found for M3GNet relaxation.")
          try:
              write(relax_traj_path, [initial_atoms, relaxed_atoms])
              print(f"   M3GNet relaxation trajectory (initial + final) saved to {relax_traj_path}")
          except Exception as e:
              print(f"   Error writing M3GNet initial/final trajectory: {e}")
              import traceback
              traceback.print_exc()

   end_time = time.time()
   print(f"» {engine.upper()} relaxation finished in {end_time - start_time:.2f} seconds.")

   relaxed_atoms.set_calculator(calculator)
   final_stress = None
   final_energy = None
   final_energy_per_atom = None
   try:
      final_stress = relaxed_atoms.get_stress()
      final_energy = relaxed_atoms.get_potential_energy()
      final_energy_per_atom = final_energy / len(relaxed_atoms)
      print(f"   Final energy: {final_energy:.6f} eV")
      print(f"   Final energy per atom: {final_energy_per_atom:.6f} eV/atom")
   except Exception as e:
      print(f"\n   Could not retrieve final stress after relaxation: {e}")

   # Save energy information to file
   energy_info_path = os.path.join(output_dir, "energy_info.txt")
   with open(energy_info_path, 'w') as f:
       f.write("=== Energy Information ===\n")
       f.write(f"Number of atoms: {len(relaxed_atoms)}\n")
       if initial_energy is not None:
           f.write(f"Initial energy: {initial_energy:.6f} eV\n")
           f.write(f"Initial energy per atom: {initial_energy_per_atom:.6f} eV/atom\n")
       if final_energy is not None:
           f.write(f"Final energy: {final_energy:.6f} eV\n")
           f.write(f"Final energy per atom: {final_energy_per_atom:.6f} eV/atom\n")
       if initial_energy is not None and final_energy is not None:
           energy_change = final_energy - initial_energy
           energy_change_per_atom = energy_change / len(relaxed_atoms)
           f.write(f"Energy change: {energy_change:.6f} eV\n")
           f.write(f"Energy change per atom: {energy_change_per_atom:.6f} eV/atom\n")


   print_final_structure_info(initial_atoms, relaxed_atoms, initial_stress, final_stress)
   save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir)

   relax_xyz_path = os.path.join(output_dir, "relax.xyz")
   try:
       frames = read(relax_traj_path, index=':')
       write(relax_xyz_path, frames)
       print(f"Converted relaxation trajectory to XYZ: {relax_xyz_path}")
   except Exception as e:
       print(f"Error converting relaxation trajectory to XYZ: {e}")
       import traceback
       traceback.print_exc()

   print("\nStructure relaxed.")

   print("\n--- Analyzing Relaxed Structure Symmetry ---")
   analyze_symmetry(relaxed_atoms, output_dir, prefix="relaxed", auto_tune_symprec=True)
   print("------------------------------------------")

   return relaxed_atoms

def relax_structures_in_folder(folder_path: str, calculator: Calculator, engine: str, fmax: float):    
   """    
   Relaxes all CIF structures found in a given folder and outputs a summary file.    
  
   Args:    
      folder_path (str): Path to the folder containing CIF files.    
      calculator (ase.calculators.calculator.Calculator): The ASE calculator to use.    
      engine (str): Name of the calculation engine.    
      fmax (float): Maximum force tolerance for relaxation.    
  
   Returns:    
      list: A list of dictionaries, each containing original_file, energy, and relaxed_atoms.    
   """    
   print(f"\n--- Relaxing structures in folder: {folder_path} ---")    
   relaxation_results = []    
   cif_files = [f for f in os.listdir(folder_path) if f.endswith(".cif")]    
  
   if not cif_files:    
      print(f"No CIF files found in {folder_path}.")    
      return []    
  
   summary_filepath = os.path.join(folder_path, "relaxation_summary.txt")  
   with open(summary_filepath, 'w') as summary_f:  
      summary_f.write(f"--- Relaxation Summary for Folder: {folder_path} ---\n")  
      summary_f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")  
  
      for cif_file in cif_files:    
         original_filepath = os.path.join(folder_path, cif_file)    
         print(f"  Relaxing {cif_file}...")    
         summary_f.write(f"### Structure: {cif_file} ###\n")  
         summary_f.write(f"Original File: {original_filepath}\n")  
         summary_f.write(f"Total Number of Atoms: {len(atoms)}\n")
         try:    
            # Load the structure    
            atoms = read(original_filepath)    
            atoms.set_calculator(calculator) # Ensure calculator is set    
  
            # Define the optimizer with UnitCellFilter for cell relaxation  
            log_filename = cif_file.replace(".cif", "_relaxation.log")    
            traj_filename = cif_file.replace(".cif", "_relaxation.traj")    
            ucf = UnitCellFilter(atoms)  
            optimizer = BFGS(ucf, logfile=os.path.join(folder_path, log_filename), trajectory=os.path.join(folder_path, traj_filename))    
  
            # Run the optimization    
            optimizer.run(fmax=fmax)    
            print(f"  Relaxation of {cif_file} completed.")    
  
            # Get the final energy    
            energy = atoms.get_potential_energy()    
            energy_per_atom = energy / len(atoms)  
            print(f"  Final energy: {energy:.4f} eV")   
            print(f"  Final energy per atom: {energy_per_atom:.6f} eV/atom")   
  
            summary_f.write(f"Relaxation Status: SUCCESS\n")  
            summary_f.write(f"Final Energy: {energy:.6f} eV\n")  
            summary_f.write(f"Final Energy per Atom: {energy_per_atom:.6f} eV/atom\n")  
  
            # Save the relaxed structure (optional, can overwrite or save with suffix)    
            relaxed_filepath = original_filepath.replace(".cif", "_relaxed.cif")    
            write(relaxed_filepath, atoms)    
            print(f"  Relaxed structure saved to {relaxed_filepath}")    
            summary_f.write(f"Relaxed File: {relaxed_filepath}\n")  
  
            cif_results = {    
                  'original_file': original_filepath,    
                  'relaxed_file': relaxed_filepath,    
                  'energy': energy,    
                  'energy_per_atom': energy_per_atom,  
                  'relaxed_atoms': atoms.copy() # Store a copy of the relaxed atoms    
            }  
            relaxation_results.append(cif_results)    
              
            # Perform symmetry analysis for the relaxed structure  
            summary_f.write("\n  --- Relaxed Structure Symmetry Analysis ---\n")  
            cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())  
              
            # Auto-tune symprec for symmetry analysis  
            symprec = 1e-3 # Default  
            symprec_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  
            best_spacegroup_num = 0  
            best_symprec_found = symprec  
            best_dataset = None  
  
            for test_symprec in symprec_values:  
                dataset = spglib.get_symmetry_dataset(cell, symprec=test_symprec)  
                if dataset and dataset['number'] > best_spacegroup_num:  
                    best_spacegroup_num = dataset['number']  
                    best_symprec_found = test_symprec  
                    best_dataset = dataset  
              
            if best_dataset:  
                summary_f.write(f"  Symmetry Precision (Auto-tuned): {best_symprec_found:.4e}\n")  
                summary_f.write(f"  Space Group Number: {best_dataset['number']}\n")  
                summary_f.write(f"  International Symbol: {best_dataset['international']}\n")  
                summary_f.write(f"  Hall Symbol: {best_dataset['hall']}\n")  
                summary_f.write(f"  Point Group Symbol: {best_dataset['pointgroup']}\n")  
                  
                # Try to get crystal system using Pymatgen  
                crystal_system = "N/A"  
                try:  
                    pmg_structure = AseAtomsAdaptor().get_structure(atoms)  
                    sga = SpacegroupAnalyzer(pmg_structure, symprec=best_symprec_found)  
                    crystal_system = sga.get_crystal_system()  
                except Exception as e:  
                    summary_f.write(f"  Warning: Could not determine crystal system using Pymatgen: {e}\n")  
                summary_f.write(f"  Crystal System: {crystal_system}\n")  
                  
                if 'lattice_type' in best_dataset:  
                    summary_f.write(f"  Lattice Type: {best_dataset['lattice_type']}\n")  
                else:  
                    summary_f.write("  Lattice Type: Not directly available from spglib dataset\n")  
                  
                summary_f.write(f"  Number of atoms in primitive cell: {len(best_dataset['std_types'])}\n")  
                summary_f.write(f"  Transformation matrix to primitive cell:\n{np.array2string(best_dataset['transformation_matrix'], separator=', ')}\n")  
                summary_f.write(f"  Origin shift: {np.array2string(best_dataset['origin_shift'], separator=', ')}\n")  
            else:  
                summary_f.write("  No symmetry found for the relaxed structure at any tested precision.\n")  
            summary_f.write("  -------------------------------------------\n\n")  
  
         except Exception as e:    
            print(f"  Error relaxing {cif_file}: {e}")    
            summary_f.write(f"Relaxation Status: FAILED\n")  
            summary_f.write(f"Error: {e}\n\n")  
            # Optionally, store error info or skip this structure    
  
      print(f"Finished relaxing structures in {folder_path}.")    
      print(f"Detailed relaxation summary saved to: {summary_filepath}")  
   return relaxation_results

def find_lowest_energy_structures(all_relaxation_results: list, num_to_select: int = 3):  
   """  
   Finds the structures with the lowest energies from a list of relaxation results.  

   Args:  
      all_relaxation_results (list): A list of dictionaries from relax_structures_in_folder.  
      num_to_select (int): The number of lowest energy structures to select.  

   Returns:  
      list: A list of the top num_to_select relaxation result dictionaries, sorted by energy.  
   """  
   print(f"\n--- Finding the {num_to_select} lowest energy structures ---")  

   if not all_relaxation_results:  
      print("No relaxation results available.")  
      return []  

   # Sort the results by energy  
   sorted_results = sorted(all_relaxation_results, key=lambda x: x['energy_per_atom'])  

   # Select the top N  
   lowest_energy_structures = sorted_results[:num_to_select]  

   print("Top lowest energy structures found:")  
   for i, result in enumerate(lowest_energy_structures):  
      print(f"  {i+1}. Energy per atom: {result['energy_per_atom']:.4f} eV/atom, File: {os.path.basename(result['original_file'])}")  

   return lowest_energy_structures


def analyze_symmetry(atoms, output_dir, prefix="", symprec=1e-3, auto_tune_symprec=False):
    """
    Analyzes the symmetry of an ASE Atoms object using spglib and saves the results to a file.
    Can optionally auto-tune symprec to find the highest symmetry.

    Args:
        atoms (ase.Atoms): The ASE Atoms object to analyze.
        output_dir (str): The directory to save the symmetry analysis file.
        prefix (str): A prefix for the output filename (e.g., "initial", "relaxed").
        symprec (float): Symmetry precision for a single run, or starting point for auto-tuning.
        auto_tune_symprec (bool): If True, attempts to find the highest symmetry by varying symprec.
    """
    print(f"\n» Analyzing {prefix} structure symmetry with spglib…")

    cell = atoms.get_cell()
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_scaled_positions()

    filename = f"{prefix}_symmetry_analysis.txt" if prefix else "symmetry_analysis.txt"
    symmetry_file_path = os.path.join(output_dir, filename)

    best_dataset = None
    best_symprec = symprec
    
    symprec_values_to_check = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    if auto_tune_symprec:
        print(f"   Attempting to auto-tune symprec to find highest symmetry...")
        
        found_symmetries = []
        for current_symprec in symprec_values_to_check:
            dataset = spglib.get_symmetry_dataset((cell, positions, numbers), symprec=current_symprec)
            if dataset:
                found_symmetries.append((dataset['number'], current_symprec, dataset))
        
        if found_symmetries:
            found_symmetries.sort(key=lambda x: (x[0], x[1]))
            best_dataset = found_symmetries[0][2]
            best_symprec = found_symmetries[0][1]
            print(f"   Highest symmetry found (Space Group {best_dataset['number']}) at symprec = {best_symprec:.4e}")
        else:
            print("   No symmetry found across the tested symprec range.")
            best_dataset = None
            best_symprec = None
    else:
        best_dataset = spglib.get_symmetry_dataset((cell, positions, numbers), symprec=symprec)
        best_symprec = symprec

    # --- DEBUGGING: Print available keys/attributes of best_dataset ---
    print("\n--- DEBUG: Contents of best_dataset ---")
    if best_dataset:
        if isinstance(best_dataset, dict):
            print("best_dataset is a dictionary. Keys available:")
            for key in best_dataset.keys():
                print(f"  - {key}")
            print("Full dictionary:")
            print(best_dataset)
        else:
            print("best_dataset is an object. Attempting to list attributes:")
            print(dir(best_dataset))
            print("Attempting to convert to dict (if it's a dataclass):")
            try:
                import dataclasses
                print(dataclasses.asdict(best_dataset))
            except Exception as e:
                print(f"  Could not convert to dict: {e}")
    else:
        print("best_dataset is None (no symmetry found).")
    print("---------------------------------------")
    # --- END DEBUGGING ---


    crystal_system = "N/A"
    if best_dataset:
        try:
            pmg_structure = AseAtomsAdaptor().get_structure(atoms)
            sga = SpacegroupAnalyzer(pmg_structure, symprec=best_symprec)
            crystal_system = sga.get_crystal_system()
        except Exception as e:
            print(f"   Warning: Could not determine crystal system using Pymatgen: {e}")
            crystal_system = "Error"


    with open(symmetry_file_path, 'w') as f:
        f.write(f"### {prefix.capitalize()} Structure Symmetry Analysis ###\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if auto_tune_symprec:
            f.write(f"Symmetry Precision (symprec) - Auto-tuned: {best_symprec:.4e}\n")
            f.write(f"Symprec values checked: {', '.join([f'{s:.1e}' for s in symprec_values_to_check])}\n\n")
        else:
            f.write(f"Symmetry Precision (symprec): {symprec:.4e}\n\n")


        if best_dataset:
            f.write(f"Space group number: {best_dataset['number']}\n")
            f.write(f"International symbol: {best_dataset['international']}\n")
            f.write(f"Hall symbol: {best_dataset['hall']}\n")
            f.write(f"Point group symbol: {best_dataset['pointgroup']}\n")
            f.write(f"Crystal system: {crystal_system}\n")
            # --- COMMENTED OUT THE PROBLEMATIC LINE FOR NOW ---
            # f.write(f"Lattice type: {best_dataset['lattice_type']}\n")
            
            # --- NEW: Add a placeholder or conditional write for lattice_type ---
            # We will re-enable this once we know the correct key or method
            if 'lattice_type' in best_dataset: # Check if the key exists
                f.write(f"Lattice type: {best_dataset['lattice_type']}\n")
            else:
                f.write("Lattice type: Not directly available from spglib dataset (check debug output for alternatives)\n")


            f.write(f"Number of atoms in primitive cell: {len(best_dataset['std_types'])}\n")
            f.write(f"Transformation matrix to primitive cell:\n{np.array2string(best_dataset['transformation_matrix'], separator=', ')}\n")
            f.write(f"Origin shift: {np.array2string(best_dataset['origin_shift'], separator=', ')}\n")
            print(f"   {prefix.capitalize()} symmetry analysis saved to: {symmetry_file_path}")
        else:
            f.write("Could not determine symmetry for any tested symprec value.\n")
            print(f"   Could not determine {prefix} symmetry for any tested symprec value.")

    print(f"» {prefix.capitalize()} symmetry analysis complete.")