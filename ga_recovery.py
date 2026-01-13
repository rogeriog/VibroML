import os
import time
import sys
import numpy as np
import re
import json

# Ensure we can import from the local vibroml package
sys.path.append(os.getcwd())

from vibroml.checkpointing import CheckpointManager
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# --- Helper Functions ---

def sanitize_filename(name):
    """Sanitize strings for filenames (e.g., space group symbols)."""
    name = name.replace('/', '_').replace(' ', '')
    return re.sub(r'[^\w\-\.]', '', name)

def format_energy_string(energy):
    """Format energy for filenames (vibroml convention)."""
    s = f"{energy:.6f}"
    if energy < 0:
        return 'm' + s[1:].replace('.', 'p')
    return 'p' + s.replace('.', 'p')

def get_best_unique_structures(results, energy_tolerance=5e-4):
    """Groups structures by energy and selects the one with the fewest atoms."""
    sorted_by_energy = sorted(results, key=lambda x: x['energy_per_atom'])
    
    if not sorted_by_energy:
        return []
        
    energy_groups = []
    current_group = [sorted_by_energy[0]]
    
    for i in range(1, len(sorted_by_energy)):
        struct = sorted_by_energy[i]
        prev_struct = current_group[0]
        
        if abs(struct['energy_per_atom'] - prev_struct['energy_per_atom']) < energy_tolerance:
            current_group.append(struct)
        else:
            energy_groups.append(current_group)
            current_group = [struct]
    
    if current_group:
        energy_groups.append(current_group)
        
    unique_representatives = []
    for group in energy_groups:
        # Sort by atom count (ascending), then energy
        group.sort(key=lambda x: (len(x['relaxed_atoms']), x['energy_per_atom']))
        unique_representatives.append(group[0])
        
    return unique_representatives

# --- Main Logic ---

def recover_all(run_directory):
    print(f"--- VibroML Recovery Tool (Full) ---")
    print(f"Target Directory: {run_directory}")
    
    # 1. Load Checkpoint
    try:
        cm = CheckpointManager(run_directory, 'ga')
        checkpoint_data = cm.load_latest_checkpoint()
    except Exception as e:
        print(f"Error initializing CheckpointManager: {e}")
        return

    if not checkpoint_data:
        print("Error: No valid checkpoint found.")
        return

    full_data, state = checkpoint_data
    all_results = full_data.get('results', [])
    
    # Filter valid results
    valid_results = [
        r for r in all_results
        if isinstance(r, dict) and r.get('energy_per_atom') is not None
    ]
    
    if not valid_results:
        print("Checkpoint loaded, but no valid relaxed structures found.")
        return

    print(f"Loaded {len(valid_results)} valid structures from checkpoint.")

    # ---------------------------------------------------------
    # PART 1: Generate the Overall Summary Text File
    # ---------------------------------------------------------
    print("\n[1/2] Generating Overall GA Summary...")
    sorted_results = sorted(valid_results, key=lambda x: x['energy_per_atom'])
    summary_path = os.path.join(run_directory, "overall_ga_summary_recovered.txt")
    
    with open(summary_path, 'w') as f:
        f.write("--- Genetic Algorithm Summary (Recovered) ---\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total structures: {len(valid_results)}\n\n")

        # Table Header
        f.write("All Successful Structures (sorted by energy):\n")
        f.write("=" * 180 + "\n")
        f.write(f"{'Num Atoms':<12} {'Int. Symbol':<15} {'Crystal System':<18} {'Energy (eV/atom)':<20} {'Iter':<5} {'Gen':<5} {'Sample':<8} {'GA Parameters':<80}\n")
        f.write("=" * 180 + "\n")

        # Table Rows
        for result in sorted_results:
            num_atoms = result.get('num_atoms', 'N/A')
            international_symbol = result.get('international_symbol', 'N/A')
            crystal_system = result.get('crystal_system', 'N/A')
            energy = result['energy_per_atom']
            
            # Extract Parameters safely
            params = result.get('params', 'N/A')
            if isinstance(params, (list, tuple)) and len(params) >= 4:
                cell_str = "N/A"
                if hasattr(params[2], '__iter__'):
                    cell_str = ",".join([f"{v:.2f}" for v in params[2][:3]]) # Just show scales for brevity
                params_str = f"D1:{params[0]:.2f}, R21:{params[1]:.2f}, Cell:({cell_str}), SC:{params[3]}"
            else:
                params_str = str(params)

            f.write(f"{str(num_atoms):<12} {international_symbol:<15} {crystal_system:<18} {energy:.6f}{'':<14} {str(result.get('main_iteration','?')):<5} {str(result.get('ga_generation','?')):<5} {str(result.get('sample','?')):<8} {params_str:<80}\n")
            
        f.write("=" * 180 + "\n")
    print(f"  -> Saved: {summary_path}")

    # ---------------------------------------------------------
    # PART 2: Generate Unique Structure Files
    # ---------------------------------------------------------
    print("\n[2/2] Generating Unique Structure Files...")
    
    unique_structures = get_best_unique_structures(valid_results)
    output_dir = os.path.join(run_directory, "final_structures_recovered")
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_path = os.path.join(output_dir, "recovered_structures_manifest.txt")
    manifest_lines = []
    manifest_lines.append(f"{'ID':<5} {'Space Group':<15} {'Atoms':<8} {'Energy (eV/atom)':<20} {'Prim. Atoms':<15} {'Filename'}")
    manifest_lines.append("-" * 100)

    for idx, result in enumerate(unique_structures):
        atoms = result['relaxed_atoms']
        energy = result['energy_per_atom']
        
        # Symmetry Analysis
        pmg_structure = AseAtomsAdaptor.get_structure(atoms)
        try:
            sga = SpacegroupAnalyzer(pmg_structure, symprec=0.01)
            sg_symbol = sga.get_space_group_symbol()
            primitive_structure = sga.get_primitive_standard_structure()
            primitive_atoms = AseAtomsAdaptor.get_atoms(primitive_structure)
            prim_count = len(primitive_atoms)
        except Exception as e:
            sg_symbol = "NA"
            primitive_atoms = atoms.copy()
            prim_count = len(atoms)

        # File Naming
        safe_sg = sanitize_filename(sg_symbol)
        en_str = format_energy_string(energy)
        base_name = f"unique_{idx+1}_sg_{safe_sg}_energy_{en_str}"
        
        # Save Files
        write(os.path.join(output_dir, f"{base_name}_relaxed.cif"), atoms)
        write(os.path.join(output_dir, f"{base_name}_primitive.cif"), primitive_atoms)
        
        manifest_lines.append(f"{idx+1:<5} {sg_symbol:<15} {len(atoms):<8} {energy:<20.6f} {prim_count:<15} {base_name}")

    # Write Manifest
    with open(manifest_path, 'w') as f:
        f.write("\n".join(manifest_lines))
        
    print(f"  -> Saved {len(unique_structures)} unique structures to: {output_dir}")
    print(f"  -> Manifest saved: {manifest_path}")
    print("\nRecovery Complete.")

if __name__ == "__main__":
    # Your target directory
    target_dir = "./examples/Bi2Sn2O7/Sn2Bi2O7_MACE_GA_phonon_output_20260107-205131"
    recover_all(target_dir)