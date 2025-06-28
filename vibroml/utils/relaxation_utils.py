import os, sys, io, re
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

    try:
        if engine == "mace":
            ucf = UnitCellFilter(atoms)
            relax_log_path = os.path.join(output_dir, "relax.log")
            print(f"   MACE relaxation log will be written to: {relax_log_path}")
            print(f"   MACE relaxation trajectory will be written to: {relax_traj_path}")
            opt = BFGS(ucf, logfile=relax_log_path, trajectory=relax_traj_path)
            opt.run(fmax=fmax, steps=1000)
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
    except Exception as e:
        print(f"   ERROR: An exception occurred during the relaxation process: {e}")
        import traceback
        traceback.print_exc()
        relaxed_atoms = None # Ensure it's None if relaxation itself fails

    end_time = time.time()
    print(f"» {engine.upper()} relaxation finished in {end_time - start_time:.2f} seconds.")

    # Check for convergence
    converged = False
    if relaxed_atoms:
        try:
            relaxed_atoms.set_calculator(calculator)
            forces = relaxed_atoms.get_forces()
            max_force = np.sqrt((forces**2).sum(axis=1).max())
            if max_force <= fmax:
                converged = True
                print(f"   Relaxation converged! Max force ({max_force:.4f}) <= fmax ({fmax:.4f})")
            else:
                print(f"   WARNING: Relaxation did not converge! Max force ({max_force:.4f}) > fmax ({fmax:.4f})")
        except Exception as e:
            print(f"   ERROR: Could not get forces to check convergence: {e}")
    
    if not converged:
        energy_info_path = os.path.join(output_dir, "energy_info.txt")
        with open(energy_info_path, 'w') as f:
            f.write("RELAXATION FAILED (did not converge or error during relaxation)\n")
        if relaxed_atoms:
            save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir, suffix="_unconverged")
        return None

    # If converged, proceed
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
       list: A list of dictionaries, each containing original_file, energy, and relaxed_atoms for CONVERGED structures.
    """    
    print(f"\n--- Relaxing structures in folder: {folder_path} ---")    
    relaxation_results = []    
    cif_files = [f for f in os.listdir(folder_path) if f.endswith(".cif") and not f.endswith(("_relaxed.cif", "_unconverged.cif"))]
   
    if not cif_files:    
       print(f"No new CIF files to relax in {folder_path}.")    
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
          
          try:    
             atoms = read(original_filepath)    
             atoms.set_calculator(calculator)    
             summary_f.write(f"Total Number of Atoms: {len(atoms)}\n")
             log_filename = cif_file.replace(".cif", "_relaxation.log")    
             traj_filename = cif_file.replace(".cif", "_relaxation.traj")    
             ucf = UnitCellFilter(atoms)  
             optimizer = BFGS(ucf, logfile=os.path.join(folder_path, log_filename), trajectory=os.path.join(folder_path, traj_filename))    
   
             optimizer.run(fmax=fmax, steps=1000)    
             print(f"  Relaxation of {cif_file} completed.")    
   
             # Check for convergence
             converged = False
             final_energy = None
             final_energy_per_atom = None
             try:
                 forces = atoms.get_forces()
                 max_force = np.sqrt((forces**2).sum(axis=1).max())
                 if max_force <= fmax:
                     converged = True
                     final_energy = atoms.get_potential_energy()
                     final_energy_per_atom = final_energy / len(atoms)
                     print(f"  Convergence met. Max force: {max_force:.4f} <= {fmax:.4f}")
                     print(f"  Final energy: {final_energy:.4f} eV")   
                     print(f"  Final energy per atom: {final_energy_per_atom:.6f} eV/atom")   
                 else:
                     print(f"  WARNING: Relaxation of {cif_file} did not converge! Max force ({max_force:.4f}) > fmax ({fmax:.4f})")
             except Exception as e:
                 print(f"  ERROR: Could not get forces/energy for {cif_file} to check convergence: {e}")
 
             if converged:
                 summary_f.write(f"Relaxation Status: SUCCESS\n")  
                 summary_f.write(f"Final Energy: {final_energy:.6f} eV\n")  
                 summary_f.write(f"Final Energy per Atom: {final_energy_per_atom:.6f} eV/atom\n")  
   
                 relaxed_filepath = original_filepath.replace(".cif", "_relaxed.cif")    
                 write(relaxed_filepath, atoms)    
                 print(f"  Relaxed structure saved to {relaxed_filepath}")    
                 summary_f.write(f"Relaxed File: {relaxed_filepath}\n")  
   
                 relaxed_xyz_filepath = original_filepath.replace(".cif", "_relaxed.xyz")  
                 write(relaxed_xyz_filepath, atoms)  
                 print(f"  Relaxed structure saved to {relaxed_xyz_filepath}")  
                 summary_f.write(f"Relaxed XYZ File: {relaxed_xyz_filepath}\n")  
 
                   
                 # Perform symmetry analysis for the relaxed structure using the dedicated function  
                 # This will also write the symmetry analysis to a file in the folder_path  
                 best_dataset_for_relaxed, crystal_system_for_relaxed = analyze_symmetry(atoms, folder_path, prefix="relaxed_in_folder", auto_tune_symprec=True)  
                   
                 # Now, use these returned values in cif_results  
                 cif_results = {      
                       'original_file': original_filepath,      
                       'relaxed_file': relaxed_filepath,      
                       'energy': final_energy,      
                       'energy_per_atom': final_energy_per_atom,    
                       'relaxed_atoms': atoms.copy(),  
                       'num_atoms': len(atoms),    
                       'international_symbol': best_dataset_for_relaxed['international'] if best_dataset_for_relaxed else 'N/A',    
                       'crystal_system': crystal_system_for_relaxed if best_dataset_for_relaxed else 'N/A'  
                 }    
                 relaxation_results.append(cif_results)      
                     
                 # Add symmetry info to the summary_f for this specific structure  
                 summary_f.write("\n  --- Relaxed Structure Symmetry Analysis ---\n")  
                 if best_dataset_for_relaxed:  
                     summary_f.write(f"  Symmetry Precision (Auto-tuned): {best_dataset_for_relaxed.get('symprec_found', 'N/A')}\n") # Assuming symprec_found is added to dataset  
                     summary_f.write(f"  Space Group Number: {best_dataset_for_relaxed['number']}\n")  
                     summary_f.write(f"  International Symbol: {best_dataset_for_relaxed['international']}\n")  
                     summary_f.write(f"  Hall Symbol: {best_dataset_for_relaxed['hall']}\n")  
                     summary_f.write(f"  Point Group Symbol: {best_dataset_for_relaxed['pointgroup']}\n")  
                     summary_f.write(f"  Crystal System: {crystal_system_for_relaxed}\n")  
                     if 'lattice_type' in best_dataset_for_relaxed:  
                         summary_f.write(f"  Lattice Type: {best_dataset_for_relaxed['lattice_type']}\n")  
                     else:  
                         summary_f.write("  Lattice Type: Not directly available from spglib dataset\n")  
                     summary_f.write(f"  Number of atoms in primitive cell: {len(best_dataset_for_relaxed['std_types'])}\n")  
                 else:  
                     summary_f.write("  No symmetry found for the relaxed structure at any tested precision.\n")  
                 summary_f.write("  -------------------------------------------\n\n")
             else: # Not converged
                 summary_f.write(f"Relaxation Status: FAIL (did not converge)\n")
                 summary_f.write(f"Final Energy: FAIL\n")
                 summary_f.write(f"Final Energy per Atom: FAIL\n")
                 summary_f.write(f"Relaxed File: N/A\n")
                 summary_f.write(f"Relaxed XYZ File: N/A\n")
                 unconverged_filepath = original_filepath.replace(".cif", "_unconverged.cif")
                 write(unconverged_filepath, atoms)
                 summary_f.write(f"Unconverged File: {unconverged_filepath}\n")
                 summary_f.write("\n  --- Relaxed Structure Symmetry Analysis ---\n")
                 summary_f.write("  Skipped due to relaxation failure.\n")
                 summary_f.write("  -------------------------------------------\n\n")
   
          except Exception as e:    
             print(f"  Error relaxing {cif_file}: {e}")    
             summary_f.write(f"Relaxation Status: FAILED (error)\n")  
             summary_f.write(f"Error: {e}\n\n")  
   
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
      print("No successfully relaxed structures available to select from.")  
      return []  

   # Sort the results by energy per atom
   sorted_results = sorted(all_relaxation_results, key=lambda x: x['energy_per_atom'])  

   # Select the top N  
   lowest_energy_structures = sorted_results[:num_to_select]  

   print("Top lowest energy structures found:")  
   for i, result in enumerate(lowest_energy_structures):  
      print(f"  {i+1}. Energy per atom: {result['energy_per_atom']:.6f} eV/atom, File: {os.path.basename(result['original_file'])}")  

   return lowest_energy_structures


# In analyze_symmetry function, at the very beginning of the function:
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

    best_dataset = None # Initialize best_dataset here
    best_symprec = symprec
    
    symprec_values_to_check = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    old_stderr = sys.stderr  
    sys.stderr = captured_stderr = io.StringIO()  

    try:  
        if auto_tune_symprec:  
            print(f"   Attempting to auto-tune symprec to find highest symmetry...")  

            found_symmetries = []  
            for current_symprec in symprec_values_to_check:  
                dataset = spglib.get_symmetry_dataset((cell, positions, numbers), symprec=current_symprec)  
                if dataset:  
                    found_symmetries.append((dataset['number'], current_symprec, dataset))  

            if found_symmetries:  
                # Sort by space group number (descending) then symprec (ascending)
                found_symmetries.sort(key=lambda x: (-x[0], x[1])) # Changed sort key
                best_dataset = found_symmetries[0][2]  
                best_symprec = found_symmetries[0][1]  
                print(f"   Highest symmetry found (Space Group {best_dataset['number']}) at symprec = {best_symprec:.4e}")  
            else:  
                print("   No symmetry found across the tested symprec range.")  
                # best_dataset remains None
                best_symprec = None  
        else:  
            best_dataset = spglib.get_symmetry_dataset((cell, positions, numbers), symprec=symprec)  
            best_symprec = symprec  
    finally:  
        sys.stderr = old_stderr 
    spglib_warnings = captured_stderr.getvalue()  
    if spglib_warnings:  
        print("\n--- Spglib Warnings Summary ---")  
        print("Spglib generated warnings during symmetry analysis. These are often related to precision issues.")  
        print("-------------------------------\n")  

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
            print(f"Space Group: {best_dataset['number']}")
            print(f"International Symbol: {best_dataset['international']}")
            print(f"Hall Symbol: {best_dataset['hall']}")
            print(f"Point Group: {best_dataset['pointgroup']}")
    else:
        print("best_dataset is None (no symmetry found).")
    print("---------------------------------------")

    crystal_system = "N/A"
    if best_dataset and best_symprec is not None: # Added check for best_symprec
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
            if 'lattice_type' in best_dataset:
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
    return best_dataset, crystal_system

def create_displaced_supercell_summary(mode_folder_path: str):  
    """  
    Reads relaxation summaries from all supercell subfolders within a mode folder  
    and creates a consolidated summary file.  
  
    Args:  
        mode_folder_path (str): Path to the soft_mode_{idx}_{label} folder.  
    """  
    print(f"\n--- Creating consolidated summary for displaced supercells in: {mode_folder_path} ---")  
  
    consolidated_summary_filepath = os.path.join(mode_folder_path, "summary_displaced_supercell.txt")  
    summary_data = []  
  
    print(f"DEBUG: Listing contents of {mode_folder_path}: {os.listdir(mode_folder_path)}")  
  
    for supercell_dir_name in os.listdir(mode_folder_path):  
        supercell_dir_path = os.path.join(mode_folder_path, supercell_dir_name)  
  
        if os.path.isdir(supercell_dir_path) and supercell_dir_name.startswith("supercell_"):  
            relaxation_summary_path = os.path.join(supercell_dir_path, "relaxation_summary.txt")  
            print(f"DEBUG: Found supercell directory: {supercell_dir_name}")  
            print(f"DEBUG: Checking for relaxation summary at: {relaxation_summary_path}")  
  
            if os.path.exists(relaxation_summary_path):  
                print(f"  Reading summary from: {relaxation_summary_path}")  
                with open(relaxation_summary_path, 'r') as f:  
                    content = f.read()  
                print(f"DEBUG: Content read from {relaxation_summary_path} (first 500 chars):\n{content[:500]}...")  
  
                structure_blocks = re.findall(r"### Structure: (.*?\.cif) ###\n(.*?)(?=(?:### Structure:|$))", content, re.DOTALL)  
                print(f"DEBUG: Number of structure blocks found by regex: {len(structure_blocks)}")  
  
                for filename, block_content in structure_blocks:  
                    entry = {  
                        "Name": filename,  
                        "Number_of_atoms": "N/A",  
                        "Final_energy_per_atom": "FAIL",  
                        "Crystal_system": "N/A",  
                        "International_symbol": "N/A"  
                    }  
                    print(f"DEBUG: Processing block for file: {filename}")  
  
                    match = re.search(r"Total Number of Atoms: (\d+)", block_content)  
                    if match:  
                        entry["Number_of_atoms"] = int(match.group(1))  
                        print(f"DEBUG: Extracted Number_of_atoms: {entry['Number_of_atoms']}")  
                    else:  
                        print("DEBUG: Number_of_atoms regex failed to match.")  
  
                    match = re.search(r"Final Energy per Atom: (.*)", block_content)
                    if match:
                        val = match.group(1).strip()
                        if "FAIL" in val or "N/A" in val:
                            entry["Final_energy_per_atom"] = "FAIL"
                        else:
                            try:
                                entry["Final_energy_per_atom"] = float(val.replace("eV/atom", "").strip())
                                print(f"DEBUG: Extracted Final_energy_per_atom: {entry['Final_energy_per_atom']}")
                            except (ValueError, TypeError):
                                entry["Final_energy_per_atom"] = "FAIL"
                                print(f"DEBUG: Could not parse energy value '{val}'")
                    else:
                        print("DEBUG: Final_energy_per_atom regex failed to match.")
  
                    match = re.search(r"\s*Crystal System: (\w+)", block_content)  
                    if match:  
                        entry["Crystal_system"] = match.group(1)  
                        print(f"DEBUG: Extracted Crystal_system: {entry['Crystal_system']}")  
                    else:  
                        print("DEBUG: Crystal_system regex failed to match.")  
  
                    match = re.search(r"\s*International Symbol: (\S+)", block_content)  
                    if match:  
                        entry["International_symbol"] = match.group(1)  
                        print(f"DEBUG: Extracted International_symbol: {entry['International_symbol']}")  
                    else:  
                        print("DEBUG: International_symbol regex failed to match.")  
  
                    summary_data.append(entry)  
            else:  
                print(f"  No relaxation_summary.txt found in {supercell_dir_path}")  
  
    if not summary_data:  
        print("DEBUG: No data found to create consolidated summary. summary_data is empty.")  
        return  
  
    def sort_key(entry):
        energy = entry.get("Final_energy_per_atom")
        if isinstance(energy, (int, float)):
            return energy
        return float('inf')
    summary_data.sort(key=sort_key)
  
    with open(consolidated_summary_filepath, 'w') as f:  
        f.write(f"{'Name':<30} {'Number_of_atoms':<15} {'Final_energy_per_atom':<25} {'Crystal_system':<20} {'International_symbol':<25}\n")  
        f.write(f"{'-'*30:<30} {'-'*15:<15} {'-'*25:<25} {'-'*20:<20} {'-'*25:<25}\n")  
  
        for entry in summary_data:  
            energy_val = entry['Final_energy_per_atom']
            energy_str = f"{energy_val:<25.6f}" if isinstance(energy_val, (int, float)) else f"{str(energy_val):<25}"  
            f.write(f"{entry['Name']:<30} {entry['Number_of_atoms']:<15} {energy_str} {entry['Crystal_system']:<20} {entry['International_symbol']:<25}\n")  
  
    print(f"Consolidated summary saved to: {consolidated_summary_filepath}")