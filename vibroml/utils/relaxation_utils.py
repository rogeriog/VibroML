import os
import sys
import json
import time
import subprocess
import tempfile
import shutil

from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read, write
from ase.calculators.calculator import Calculator
import numpy as np

import spglib
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import re
import io

# Optional M3GNet import
try:
    from m3gnet.models import Relaxer
    HAVE_M3GNET = True
except ImportError:
    HAVE_M3GNET = False

# Optional eSEN/UMA (fairchem-core) imports
# Support both old API (1.10.0) and new API (2.12.0+)
HAVE_ESEN = False
HAVE_UMA = False
FAIRCHEM_API_VERSION = None

try:
    # Try old API first (fairchem-core 1.10.0)
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    HAVE_ESEN = True
    FAIRCHEM_API_VERSION = "old"
except ImportError:
    try:
        # Try new API (fairchem-core 2.12.0+)
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit
        HAVE_UMA = True
        FAIRCHEM_API_VERSION = "new"
    except ImportError:
        pass

# Check for GPUMD
from .utils import HAVE_GPUMD

class MinimumIterationOptimizer:
    """
    A wrapper for ASE optimizers that enforces a minimum number of iterations
    before allowing convergence based on force criteria.
    """
    def __init__(self, optimizer, min_iterations=5):
        self.optimizer = optimizer
        self.min_iterations = min_iterations
        self.iteration_count = 0
        self.original_converged = None

        # Store original converged method
        if hasattr(optimizer, 'converged'):
            self.original_converged = optimizer.converged
            # Replace with our custom converged method
            optimizer.converged = self._custom_converged

    def _custom_converged(self, forces=None):
        """Custom convergence check that requires minimum iterations."""
        self.iteration_count += 1

        # Always return False if we haven't reached minimum iterations
        if self.iteration_count < self.min_iterations:
            print(f"   Iteration {self.iteration_count}: Minimum iterations not reached ({self.min_iterations} required)")
            return False

        # After minimum iterations, use original convergence criteria
        if self.original_converged:
            result = self.original_converged(forces)
            if result:
                print(f"   Iteration {self.iteration_count}: Converged after minimum iterations requirement met")
            return result
        else:
            # Fallback: check forces manually if no original converged method
            if forces is not None:
                max_force = np.sqrt((forces**2).sum(axis=1)).max()
                fmax = getattr(self.optimizer, 'fmax', 0.05)  # Default fmax if not set
                result = max_force < fmax
                if result:
                    print(f"   Iteration {self.iteration_count}: Converged (max force {max_force:.6f} < {fmax:.6f})")
                return result
            return False

    def run(self, fmax=0.05, steps=None):
        """Run the optimizer with minimum iteration enforcement."""
        self.optimizer.fmax = fmax
        return self.optimizer.run(fmax=fmax, steps=steps)

    def attach(self, callback):
        """Attach callback to the underlying optimizer."""
        return self.optimizer.attach(callback)

class EnergyVolumeStopper:
    def __init__(self, optimizer, energy_increase_threshold=0.5,
                 energy_decrease_threshold=-5.0, volume_threshold=2.5,
                 max_steps=1000, min_iterations=5):
        """
        A logger to stop optimization based on energy and volume criteria.

        Stops if:
        1. Energy increases dramatically (> energy_increase_threshold eV/atom from initial)
        2. Energy decreases dramatically (< energy_decrease_threshold eV/atom from initial)
           (e.g., -5.0 eV/atom, indicating likely decomposition or unphysical state)
        3. Volume increases > volume_threshold times initial volume
        4. Maximum steps reached

        But only after min_iterations have been completed.
        """
        self.optimizer = optimizer
        self.energy_increase_threshold = energy_increase_threshold  # eV/atom
        self.energy_decrease_threshold = energy_decrease_threshold  # eV/atom (negative value)
        self.volume_threshold = volume_threshold  # multiplier
        self.max_steps = max_steps
        self.min_iterations = min_iterations  # NEW: minimum iterations before applying stopping criteria

        # Internal state
        self.initial_energy_per_atom = None
        self.initial_volume = None
        self.step_count = 0
          
    def __call__(self):  
        self.step_count += 1  
        optimizable_object = self.optimizer.atoms  
        if isinstance(optimizable_object, UnitCellFilter):  
            # If it's a filter, the real Atoms object is an attribute  
            atoms = optimizable_object.atoms  
        else:  
            # Otherwise, the object is the Atoms object itself  
            atoms = optimizable_object
          
        try:
            current_energy = atoms.get_potential_energy()
            if current_energy is None:
                print(f"   Step {self.step_count}: Could not retrieve energy, skipping energy/volume monitoring")
                return
            current_energy_per_atom = current_energy / len(atoms)
            current_volume = atoms.get_volume()
            if current_volume is None:
                print(f"   Step {self.step_count}: Could not retrieve volume, skipping energy/volume monitoring")
                return

            # Initialize on first step
            if self.initial_energy_per_atom is None and self.step_count > self.min_iterations :
                self.initial_energy_per_atom = current_energy_per_atom
                self.initial_volume = current_volume
                print(f"   Step 1: Initial energy per atom: {self.initial_energy_per_atom:.6f} eV/atom")
                print(f"   Step 1: Initial volume: {self.initial_volume:.3f} Å³")
                return

            print(f"   Step {self.step_count}: Energy/atom={current_energy_per_atom:.6f} eV/atom, "
                  f"Volume={current_volume:.3f} Å³")

            # Calculate energy change relative to initial - add None check to prevent TypeError
            if self.initial_energy_per_atom is None:
                print(f"   Step {self.step_count}: Initial energy not yet set, skipping energy change checks")
                return

            energy_change_from_initial = current_energy_per_atom - self.initial_energy_per_atom

            # Don't apply stopping criteria until minimum iterations are reached
            if self.step_count <= self.min_iterations:
                print(f"   Step {self.step_count}: Minimum iterations not reached ({self.min_iterations} required)")
                return

            # Check 1: Dramatic energy increase
            if energy_change_from_initial > self.energy_increase_threshold:  
                print(f"\n   STOPPING: Energy increased dramatically!")  
                print(f"   Energy change from initial: {energy_change_from_initial:.6f} eV/atom > threshold {self.energy_increase_threshold} eV/atom")  
                raise StopIteration  
              
            # Check 2: Dramatic energy decrease (unphysical drop)
            if energy_change_from_initial < self.energy_decrease_threshold:
                print(f"\n   STOPPING: Energy decreased dramatically (unphysical drop)!")
                print(f"   Energy change from initial: {energy_change_from_initial:.6f} eV/atom < threshold {self.energy_decrease_threshold} eV/atom")
                print(f"   This often indicates decomposition or an ill-posed structure.")
                print(f"\n   WARNING: If this structure failed during initial relaxation due to large energy changes,")
                print(f"   consider increasing the --relaxation-patience parameter (current: {self.min_iterations}).")
                print(f"   Try using --relaxation-patience 30 or higher to allow more initial relaxation steps.")
                raise StopIteration
              
            # Check 3: Volume expansion - add None check to prevent TypeError
            if self.initial_volume is None:
                print(f"   Step {self.step_count}: Initial volume not yet set, skipping volume expansion check")
            else:
                volume_ratio = current_volume / self.initial_volume
                if volume_ratio > self.volume_threshold:
                    print(f"\n   STOPPING: Volume expanded too much!")
                    print(f"   Volume ratio: {volume_ratio:.2f} > threshold {self.volume_threshold}")
                    raise StopIteration
              
            # Check 4: Max steps  
            if self.step_count >= self.max_steps:  
                print(f"\n   STOPPING: Max steps ({self.max_steps}) reached.")  
                raise StopIteration  
                  
        except Exception as e:  
            if isinstance(e, StopIteration): # Check if it's the StopIteration we raised  
                raise  # Re-raise StopIteration to halt the optimizer  
            print(f"\n   ERROR in energy/volume monitoring: {e}")


def write_gpumd_model_xyz(atoms, filepath):
    """
    Write ASE Atoms object to GPUMD's model.xyz format (extended XYZ).

    Format:
        Line 1: Number of atoms
        Line 2: pbc="T T T" Lattice="..." Properties=species:S:1:pos:R:3:mass:R:1
        Lines 3+: species x y z mass

    Args:
        atoms: ASE Atoms object
        filepath: Path to write model.xyz file
    """
    with open(filepath, 'w') as f:
        # Line 1: Number of atoms
        f.write(f"{len(atoms)}\n")

        # Line 2: Lattice and properties
        cell = atoms.get_cell()
        # GPUMD uses row-major format: a_x a_y a_z b_x b_y b_z c_x c_y c_z
        lattice_str = " ".join([f"{x:.10f}" for x in cell.flatten()])
        f.write(f'pbc="T T T" Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:mass:R:1\n')

        # Lines 3+: Atomic data
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        masses = atoms.get_masses()

        for i in range(len(atoms)):
            f.write(f"{symbols[i]} {positions[i,0]:.10f} {positions[i,1]:.10f} {positions[i,2]:.10f} {masses[i]:.10f}\n")


def create_gpumd_run_in(nep_path, fmax, max_steps, output_dir):
    """
    Create GPUMD run.in file for structure relaxation.

    Args:
        nep_path: Path to NEP potential file
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum number of minimization steps
        output_dir: Directory to write run.in file

    Returns:
        Path to created run.in file
    """
    run_in_path = os.path.join(output_dir, 'run.in')

    with open(run_in_path, 'w') as f:
        f.write(f'# GPUMD native relaxation\n')
        f.write(f'# Generated by VibroML\n\n')
        f.write(f'potential   {nep_path}\n\n')
        f.write(f'# Minimize energy using FIRE method\n')
        f.write(f'# Syntax: minimize <method> <force_tolerance> <max_steps> <box_change> <hydrostatic_strain>\n')
        f.write(f'# method: fire or sd\n')
        f.write(f'# force_tolerance: convergence criterion in eV/Å\n')
        f.write(f'# max_steps: maximum number of steps\n')
        f.write(f'# box_change: 1 = allow box to change, 0 = fixed box\n')
        f.write(f'# hydrostatic_strain: 1 = use hydrostatic pressure, 0 = full stress tensor\n')
        f.write(f'minimize    fire {fmax} {max_steps} 1 0\n\n')
        f.write(f'# Output the relaxed structure\n')
        f.write(f'ensemble    nve\n')
        f.write(f'time_step   0\n')
        f.write(f'dump_thermo 1\n')
        f.write(f'dump_xyz    -1 0 1 relaxed.xyz\n')
        f.write(f'run         1\n')

    return run_in_path


def parse_gpumd_output(output_text):
    """
    Parse GPUMD console output to extract relaxation information.

    Args:
        output_text: String containing GPUMD console output

    Returns:
        dict with keys: 'converged', 'steps', 'final_fmax', 'final_energy', 'final_pressure'
    """
    info = {
        'converged': False,
        'steps': 0,
        'final_fmax': None,
        'final_energy': None,
        'final_pressure': None
    }

    lines = output_text.split('\n')

    for line in lines:
        # Look for "Energy minimization finished"
        if 'Energy minimization finished' in line:
            info['converged'] = True

        # Parse step lines: "    step 58: total_potential = -9.7125881673 eV, f_max = 0.0099799096 eV/A, pressure = -1.1701930498 GPa."
        if 'step' in line and 'total_potential' in line:
            try:
                # Extract step number
                step_match = re.search(r'step\s+(\d+):', line)
                if step_match:
                    info['steps'] = int(step_match.group(1)) + 1  # +1 because steps are 0-indexed

                # Extract energy
                energy_match = re.search(r'total_potential\s*=\s*([-\d.]+)\s*eV', line)
                if energy_match:
                    info['final_energy'] = float(energy_match.group(1))

                # Extract fmax
                fmax_match = re.search(r'f_max\s*=\s*([-\d.]+)\s*eV/A', line)
                if fmax_match:
                    info['final_fmax'] = float(fmax_match.group(1))

                # Extract pressure
                pressure_match = re.search(r'pressure\s*=\s*([-\d.]+)\s*GPa', line)
                if pressure_match:
                    info['final_pressure'] = float(pressure_match.group(1))
            except (ValueError, AttributeError) as e:
                pass  # Continue parsing other lines

    return info


def relax_with_gpumd_native(atoms, nep_path, gpumd_binary, fmax, max_steps, output_dir):
    """
    Relax structure using GPUMD's native FIRE minimizer.

    This bypasses ASE's optimizer and uses GPUMD's built-in minimization,
    which is optimized for NEP potentials and shows better convergence.

    GPUMD multi-GPU NEP requires box dimensions >= 3 * cutoff (typically 18 Å for NEP89).
    If the input cell is too small, this function automatically creates a supercell,
    relaxes it, and extracts the primitive cell from the relaxed supercell.

    Args:
        atoms: ASE Atoms object to relax
        nep_path: Path to NEP potential file
        gpumd_binary: Path to GPUMD executable
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum number of minimization steps
        output_dir: Directory for GPUMD input/output files

    Returns:
        tuple: (relaxed_atoms, info_dict)
            relaxed_atoms: ASE Atoms object with relaxed structure (or None if failed)
            info_dict: Dictionary with relaxation information
    """
    print(f"   Using GPUMD native FIRE minimizer")
    print(f"   NEP potential: {nep_path}")
    print(f"   Force tolerance: {fmax} eV/Å")
    print(f"   Max steps: {max_steps}")

    # Create temporary directory for GPUMD files
    gpumd_dir = os.path.join(output_dir, 'gpumd_native_relax')
    os.makedirs(gpumd_dir, exist_ok=True)

    try:
        # Write model.xyz
        model_xyz_path = os.path.join(gpumd_dir, 'model.xyz')
        write_gpumd_model_xyz(atoms, model_xyz_path)
        print(f"   Wrote model.xyz with {len(atoms)} atoms")

        # Create run.in
        run_in_path = create_gpumd_run_in(nep_path, fmax, max_steps, gpumd_dir)
        print(f"   Created run.in")

        # Run GPUMD
        print(f"   Running GPUMD...")
        result = subprocess.run(
            [gpumd_binary],
            cwd=gpumd_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Save GPUMD output
        output_log_path = os.path.join(gpumd_dir, 'gpumd_output.log')
        with open(output_log_path, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        # Check for errors
        if result.returncode != 0:
            print(f"   ERROR: GPUMD exited with code {result.returncode}")
            print(f"   See {output_log_path} for details")
            return None, {'error': f'GPUMD failed with exit code {result.returncode}'}

        # Parse output
        info = parse_gpumd_output(result.stdout)

        print(f"   GPUMD relaxation completed:")
        print(f"     Steps: {info['steps']}")
        print(f"     Converged: {info['converged']}")
        if info['final_fmax'] is not None:
            print(f"     Final fmax: {info['final_fmax']:.6f} eV/Å")
        if info['final_energy'] is not None:
            print(f"     Final energy: {info['final_energy']:.6f} eV")
        if info['final_pressure'] is not None:
            print(f"     Final pressure: {info['final_pressure']:.3f} GPa")

        # Read relaxed structure
        relaxed_xyz_path = os.path.join(gpumd_dir, 'relaxed.xyz')
        if not os.path.exists(relaxed_xyz_path):
            print(f"   ERROR: relaxed.xyz not found at {relaxed_xyz_path}")
            return None, {'error': 'relaxed.xyz not found'}

        relaxed_atoms = read(relaxed_xyz_path)
        print(f"   Read relaxed structure from relaxed.xyz")

        return relaxed_atoms, info

    except subprocess.TimeoutExpired:
        print(f"   ERROR: GPUMD timed out after 600 seconds")
        return None, {'error': 'GPUMD timeout'}
    except Exception as e:
        print(f"   ERROR: Exception during GPUMD relaxation: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


def save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir, suffix=""):
    """Helper function to save relaxed structure to CIF file."""
    base_name = os.path.splitext(os.path.basename(original_cif_path))[0]
    relaxed_cif_path = os.path.join(output_dir, f"{base_name}_relaxed{suffix}.cif")
    write(relaxed_cif_path, relaxed_atoms)
    print(f"Saved relaxed structure to: {relaxed_cif_path}")


def relax_structure(atoms, calculator, engine, fmax, output_dir, original_cif_path, save_trajectory=True, relaxation_patience=5, volume_expansion_threshold=2.5):
    """
    Performs structure relaxation using the specified engine.

    Args:
        atoms: ASE Atoms object to relax
        calculator: ASE calculator to use
        engine: Relaxation engine ('mace' or 'm3gnet')
        fmax: Maximum force tolerance
        output_dir: Directory to save outputs
        original_cif_path: Path to original CIF file
        save_trajectory: Whether to save relaxation trajectory
        relaxation_patience: Number of initial steps to wait before applying energy drop termination criteria
        volume_expansion_threshold: Volume expansion threshold for stopping criterion (default: 2.5)
    """
    print(f"\n» {engine.upper()} relaxation starting…")
    start_time = time.time()

    initial_atoms = atoms.copy()

    print("\n--- Analyzing Initial Structure Symmetry ---")
    analyze_symmetry(initial_atoms, output_dir, prefix="initial", auto_tune_symprec=True)
    print("------------------------------------------")

    atoms.set_calculator(calculator)

    # Get initial energy and stress
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
        if engine in ("mace", "nep", "calorine"):
            # Generic ASE-based relaxation using BFGS + UnitCellFilter.
            # Originally tuned for MACE, but equally applicable to other
            # ASE-compatible MLIPs such as calorine NEP.
            ucf = UnitCellFilter(atoms)
            relax_log_path = os.path.join(output_dir, "relax.log")
            engine_tag = engine.upper()
            print(f"   {engine_tag} relaxation log will be written to: {relax_log_path}")
            if save_trajectory:
                print(f"   {engine_tag} relaxation trajectory will be written to: {relax_traj_path}")

            # Initialize BFGS with the custom logger
            opt = BFGS(ucf, logfile=relax_log_path, trajectory=relax_traj_path if save_trajectory else None)

            # Create an instance of our custom logger with adjusted parameters
            # These parameters are crucial for force-based stagnation detection
            max_steps_generic = 1000

            energy_volume_logger = EnergyVolumeStopper(
                opt,
                energy_increase_threshold=0.5,   # Stop if energy increases > 0.5 eV/atom
                energy_decrease_threshold=-5.0,  # Stop if energy decreases < -5.0 eV/atom (e.g., -6.0, -7.0)
                volume_threshold=volume_expansion_threshold,  # Stop if volume > threshold × initial
                max_steps=max_steps_generic,
                min_iterations=relaxation_patience  # Use user-specified patience parameter
            )
            opt.attach(energy_volume_logger) # Attach the logger to the optimizer

            try:
                opt.run(fmax=fmax)
                relaxed_atoms = ucf.atoms.copy()  # Assign on successful run
            except StopIteration:
                print("   Optimization stopped by custom logger criteria.")
                relaxed_atoms = ucf.atoms.copy()  # Also assign when logger stops it
            except Exception as e:
                print(f"   ERROR: An unexpected error occurred during {engine_tag} optimization: {e}")
                import traceback
                traceback.print_exc()
                relaxed_atoms = None  # On failure, ensure it is None

            if relaxed_atoms is not None:  # Only copy if not set to None by an error
                relaxed_atoms = ucf.atoms.copy()

        elif engine == "m3gnet":
            if not HAVE_M3GNET:
                print("   ERROR: M3GNet not found – install with: pip install -e '.[m3gnet]'")
                sys.exit(1)
            relaxer = Relaxer()
            if save_trajectory:
                print(f"   M3GNet relaxation will generate a trajectory at: {relax_traj_path}")
            relax_results = relaxer.relax(atoms, fmax=fmax, verbose=True)

            relaxed_atoms = AseAtomsAdaptor().get_atoms(relax_results['final_structure'])

            if save_trajectory:
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
            else:
                print("   Trajectory saving is disabled for M3GNet relaxation.")

        elif engine == "esen":
            if not HAVE_ESEN:
                print("   ERROR: eSEN (fairchem-core) not found – activate eSEN environment or use --engine mace")
                sys.exit(1)

            ucf = UnitCellFilter(atoms)
            relax_log_path = os.path.join(output_dir, "relax.log")
            print(f"   eSEN relaxation log will be written to: {relax_log_path}")
            if save_trajectory:
                print(f"   eSEN relaxation trajectory will be written to: {relax_traj_path}")

            # Initialize BFGS with the custom logger
            opt = BFGS(ucf, logfile=relax_log_path, trajectory=relax_traj_path if save_trajectory else None)

            # Create an instance of our custom logger with adjusted parameters
            max_steps_for_esen = 1000

            energy_volume_logger = EnergyVolumeStopper(
                opt,
                energy_increase_threshold=0.5,   # Stop if energy increases > 0.5 eV/atom
                energy_decrease_threshold=-5.0,  # Stop if energy decreases < -5.0 eV/atom
                volume_threshold=volume_expansion_threshold,  # Stop if volume > threshold × initial
                max_steps=max_steps_for_esen,
                min_iterations=relaxation_patience  # Use user-specified patience parameter
            )
            opt.attach(energy_volume_logger)

            try:
                opt.run(fmax=fmax)
                relaxed_atoms = ucf.atoms.copy()
            except StopIteration:
                print("   Optimization stopped by custom logger criteria.")
                relaxed_atoms = ucf.atoms.copy()
            except Exception as e:
                print(f"   ERROR: An unexpected error occurred during eSEN optimization: {e}")
                import traceback
                traceback.print_exc()
                relaxed_atoms = None

            if relaxed_atoms is not None:
                relaxed_atoms = ucf.atoms.copy()

        elif engine == "uma":
            if not HAVE_UMA:
                print("   ERROR: UMA (fairchem-core) not found – activate UMA environment or use --engine mace")
                sys.exit(1)

            ucf = UnitCellFilter(atoms)
            relax_log_path = os.path.join(output_dir, "relax.log")
            print(f"   UMA relaxation log will be written to: {relax_log_path}")
            if save_trajectory:
                print(f"   UMA relaxation trajectory will be written to: {relax_traj_path}")

            # Initialize BFGS with the custom logger
            opt = BFGS(ucf, logfile=relax_log_path, trajectory=relax_traj_path if save_trajectory else None)

            # Create an instance of our custom logger with adjusted parameters
            max_steps_for_uma = 1000

            energy_volume_logger = EnergyVolumeStopper(
                opt,
                energy_increase_threshold=0.5,   # Stop if energy increases > 0.5 eV/atom
                energy_decrease_threshold=-5.0,  # Stop if energy decreases < -5.0 eV/atom
                volume_threshold=volume_expansion_threshold,  # Stop if volume > threshold × initial
                max_steps=max_steps_for_uma,
                min_iterations=relaxation_patience  # Use user-specified patience parameter
            )
            opt.attach(energy_volume_logger)

            try:
                opt.run(fmax=fmax)
                relaxed_atoms = ucf.atoms.copy()
            except StopIteration:
                print("   Optimization stopped by custom logger criteria.")
                relaxed_atoms = ucf.atoms.copy()
            except Exception as e:
                print(f"   ERROR: An unexpected error occurred during UMA optimization: {e}")
                import traceback
                traceback.print_exc()
                relaxed_atoms = None

            if relaxed_atoms is not None:
                relaxed_atoms = ucf.atoms.copy()

        elif engine == "gpumd":
            if not HAVE_GPUMD:
                print("   ERROR: GPUMD not found – ensure GPUMD is compiled or use --engine mace")
                sys.exit(1)

            # Import GPUMD binary path from utils
            from .utils import GPUMD_BINARY_PATH

            if GPUMD_BINARY_PATH is None:
                print("   ERROR: GPUMD binary path not found")
                sys.exit(1)

            # Get NEP potential path from calculator
            # The GPUMDCalculator stores it as potential_path
            nep_path = None
            if hasattr(calculator, 'potential_path'):
                nep_path = calculator.potential_path
            else:
                # Try alternative attribute names
                if hasattr(calculator, 'nep_path'):
                    nep_path = calculator.nep_path
                else:
                    print("   WARNING: Could not find NEP path in calculator")
                    # Use default NEP89 path as fallback
                    nep_path = "/globalscratch/ucl/modl/rgouvea/VibroML/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"
                    print(f"   Using default NEP path: {nep_path}")

            # Use GPUMD native minimizer
            max_steps_for_gpumd = 1000
            relaxed_atoms, relax_info = relax_with_gpumd_native(
                atoms=atoms,
                nep_path=nep_path,
                gpumd_binary=GPUMD_BINARY_PATH,
                fmax=fmax,
                max_steps=max_steps_for_gpumd,
                output_dir=output_dir
            )

            # Check for errors
            if relaxed_atoms is None:
                print(f"   ERROR: GPUMD native relaxation failed")
                if 'error' in relax_info:
                    print(f"   Error details: {relax_info['error']}")
            else:
                # Set calculator on relaxed atoms for subsequent force calculations
                relaxed_atoms.set_calculator(calculator)

    except Exception as e: # This outer try-except catches errors from M3GNet or general setup
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
            max_force = np.sqrt((forces**2).sum(axis=1)).max()
            if max_force <= 2*fmax: # 2*fmax to be conservative, there is some instability in the optimizer..
                converged = True
                print(f"   Relaxation converged! Max force ({max_force:.4f}) <= fmax ({2*fmax:.4f})")
            else:
                print(f"   WARNING: Relaxation did not converge! Max force ({max_force:.4f}) > fmax ({2*fmax:.4f})")
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


    # Assuming these are defined elsewhere or will be imported into this file
    # from your_module import load_structure, initialize_calculator, relax_structure, run_single_phonon_analysis, run_ga_soft_mode_optimization, run_traditional_soft_mode_optimization
    # Placeholder for structure_utils functions
    def print_final_structure_info(initial_atoms, relaxed_atoms, initial_stress, final_stress):
        print("\n--- Structure Relaxation Summary ---")
        print(f"Initial atoms: {len(initial_atoms)}")
        print(f"Relaxed atoms: {len(relaxed_atoms)}")
        if initial_stress is not None:
            print(f"Initial stress (GPa): {initial_stress.max():.4f}")
        if final_stress is not None:
            print(f"Final stress (GPa): {final_stress.max():.4f}")
        print("----------------------------------")

    print_final_structure_info(initial_atoms, relaxed_atoms, initial_stress, final_stress)
    save_relaxed_structure(relaxed_atoms, original_cif_path, engine, fmax, output_dir)

    if save_trajectory:
        relax_xyz_path = os.path.join(output_dir, "relax.xyz")
        # Check if trajectory file exists (won't exist for GPUMD native)
        if os.path.exists(relax_traj_path):
            try:
                frames = read(relax_traj_path, index=':')
                write(relax_xyz_path, frames)
                print(f"Converted relaxation trajectory to XYZ: {relax_xyz_path}")
            except Exception as e:
                print(f"Error converting relaxation trajectory to XYZ: {e}")
        else:
            # For GPUMD native, trajectory is not saved in ASE format
            if engine == "gpumd":
                print("Note: GPUMD native minimizer does not produce ASE trajectory file")
            else:
                print(f"Warning: Trajectory file not found at {relax_traj_path}")
    else:
        print("Trajectory conversion to XYZ skipped as trajectory saving was disabled.")

    print("\nStructure relaxed.")

    print("\n--- Analyzing Relaxed Structure Symmetry ---")
    analyze_symmetry(relaxed_atoms, output_dir, prefix="relaxed", auto_tune_symprec=True)
    print("------------------------------------------")

    return relaxed_atoms

def relax_structures_in_folder(folder_path: str, calculator: Calculator, engine: str, fmax: float, save_trajectory: bool = False, relaxation_patience: int = 5, volume_expansion_threshold: float = 2.5):
    """
    Relaxes all CIF structures found in a given folder and outputs a summary file.

    Args:
       folder_path (str): Path to the folder containing CIF files.
       calculator (ase.calculators.calculator.Calculator): The ASE calculator to use.
       engine (str): Name of the calculation engine.
       fmax (float): Maximum force tolerance for relaxation.
       save_trajectory (bool): Whether to save the relaxation trajectory.
       relaxation_patience (int): Number of initial steps to wait before applying energy drop termination criteria.
       volume_expansion_threshold (float): Volume expansion threshold for stopping criterion (default: 2.5).

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
             summary_f.write(f"Total Number of Atoms: {len(atoms)}\n")

             # Use GPUMD native minimizer for GPUMD engine, ASE optimizer for others
             if engine == "gpumd":
                 # GPUMD native relaxation path
                 from .utils import GPUMD_BINARY_PATH

                 if GPUMD_BINARY_PATH is None:
                     print(f"  ERROR: GPUMD binary path not found")
                     summary_f.write(f"Relaxation Status: FAILED (GPUMD binary not found)\n\n")
                     continue

                 # Get NEP potential path from calculator
                 nep_path = None
                 if hasattr(calculator, 'potential_path'):
                     nep_path = calculator.potential_path
                 else:
                     print(f"  WARNING: Could not find NEP path in calculator, using default")
                     nep_path = "/globalscratch/ucl/modl/rgouvea/VibroML/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"

                 print(f"  Using GPUMD native FIRE minimizer for {cif_file}")

                 # Create subdirectory for this structure's relaxation
                 structure_relax_dir = os.path.join(folder_path, cif_file.replace(".cif", "_gpumd_relax"))
                 os.makedirs(structure_relax_dir, exist_ok=True)

                 max_steps_for_gpumd = 1000
                 relaxed_atoms, relax_info = relax_with_gpumd_native(
                     atoms=atoms,
                     nep_path=nep_path,
                     gpumd_binary=GPUMD_BINARY_PATH,
                     fmax=fmax,
                     max_steps=max_steps_for_gpumd,
                     output_dir=structure_relax_dir
                 )

                 if relaxed_atoms is None:
                     print(f"  ERROR: GPUMD native relaxation failed for {cif_file}")
                     summary_f.write(f"Relaxation Status: FAILED (GPUMD relaxation error)\n")
                     if 'error' in relax_info:
                         summary_f.write(f"Error: {relax_info['error']}\n\n")
                     else:
                         summary_f.write(f"Error: Unknown GPUMD error\n\n")
                     continue

                 # Set calculator for subsequent force calculations
                 atoms = relaxed_atoms
                 atoms.set_calculator(calculator)
                 print(f"  Relaxation of {cif_file} completed with GPUMD native minimizer.")

             else:
                 # ASE optimizer path for MACE, M3GNet, eSEN, UMA
                 atoms.set_calculator(calculator)
                 log_filename = cif_file.replace(".cif", "_relaxation.log")
                 traj_filename = cif_file.replace(".cif", "_relaxation.traj")
                 ucf = UnitCellFilter(atoms)

                 # Conditionally pass trajectory argument
                 optimizer_kwargs = {"logfile": os.path.join(folder_path, log_filename)}
                 if save_trajectory:
                     optimizer_kwargs["trajectory"] = os.path.join(folder_path, traj_filename)
                     print(f"  Relaxation trajectory will be saved to: {os.path.join(folder_path, traj_filename)}")
                 else:
                     print("  Relaxation trajectory saving is disabled.")

                 optimizer = BFGS(ucf, **optimizer_kwargs)

                 # Adjusted parameters for force-centric stagnation
                 max_steps_for_mace = 1000
                 energy_volume_logger = EnergyVolumeStopper(
                                        optimizer,
                                        energy_increase_threshold=0.5,
                                        energy_decrease_threshold=-5.0,
                                        volume_threshold=volume_expansion_threshold,  # Stop if volume > threshold × initial
                                        max_steps=max_steps_for_mace,
                                        min_iterations=relaxation_patience  # Use user-specified patience parameter
                                    )
                 optimizer.attach(energy_volume_logger)

                 try:
                     optimizer.run(fmax=fmax)
                     print(f"  Relaxation of {cif_file} completed.")
                 except StopIteration:
                     print(f"  Relaxation of {cif_file} stopped by custom logger criteria.")
                 except Exception as e:
                     print(f"  ERROR: An unexpected error occurred during optimization of {cif_file}: {e}")
                     import traceback
                     traceback.print_exc()
                     # Mark as not converged if an error occurs during run
                     summary_f.write(f"Relaxation Status: FAILED (error during run)\n")
                     summary_f.write(f"Error: {e}\n\n")
                     continue # Skip to next cif_file
             # >>> END RELAXATION BLOCK <<<


             # Check for convergence
             converged = False
             final_energy = None
             final_energy_per_atom = None
             try:
                 forces = atoms.get_forces()
                 max_force = np.sqrt((forces**2).sum(axis=1)).max()
                 if max_force <= 2*fmax: # Conservative because there is some instability in the code
                     converged = True
                     final_energy = atoms.get_potential_energy()
                     final_energy_per_atom = final_energy / len(atoms)
                     print(f"  Convergence met. Max force: {max_force:.4f} <= {2*fmax:.4f}")
                     print(f"  Final energy: {final_energy:.4f} eV")
                     print(f"  Final energy per atom: {final_energy_per_atom:.6f} eV/atom")
                 else:
                     print(f"  WARNING: Relaxation of {cif_file} did not converge! Max force ({max_force:.4f}) > fmax ({2*fmax:.4f})")
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

                 if save_trajectory:
                     relaxed_xyz_filepath = original_filepath.replace(".cif", "_relaxed.xyz")
                     # Read from the .traj file if it was saved, otherwise just write the final atoms
                     if os.path.exists(os.path.join(folder_path, traj_filename)):
                         try:
                             frames = read(os.path.join(folder_path, traj_filename), index=':')
                             write(relaxed_xyz_filepath, frames)
                             print(f"  Relaxed trajectory converted to XYZ and saved to {relaxed_xyz_filepath}")
                             summary_f.write(f"Relaxed XYZ File: {relaxed_xyz_filepath}\n")
                         except Exception as e:
                             print(f"  Error converting relaxation trajectory to XYZ for {cif_file}: {e}")
                             summary_f.write(f"Relaxed XYZ File: Error converting trajectory to XYZ\n")
                     else:
                         # If trajectory wasn't explicitly saved, just write the final relaxed structure to XYZ
                         write(relaxed_xyz_filepath, atoms)
                         print(f"  Relaxed structure (final frame) saved to {relaxed_xyz_filepath}")
                         summary_f.write(f"Relaxed XYZ File (final frame only): {relaxed_xyz_filepath}\n")
                 else:
                     print("  Skipping XYZ conversion as trajectory saving was disabled.")
                     summary_f.write(f"Relaxed XYZ File: Skipped (trajectory saving disabled)\n")


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
                     summary_f.write(f"  Symmetry Precision (Auto-tuned): {best_dataset_for_relaxed.get('symprec_found', 'N/A')}\n")
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

          except Exception as e: # This outer try-except catches errors during setup or initial read
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
    if best_dataset and best_symprec is not None:
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

                    match = re.search(r"Final Energy per Atom: ([-+]?\d*\.\d+)", block_content)
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
                        entry["International_symbol"] = match.group(1) # Corrected from match(1) to match.group(1)
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