import os
import sys
import shutil
import tempfile
import subprocess
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import numpy as np
from .utils import (
    HAVE_MACE,
    get_mace_device,
    mace_mp,
    HAVE_ESEN,
    HAVE_UMA,
    FAIRCHEM_API_VERSION,
    HAVE_GPUMD,
    GPUMD_BINARY_PATH,
    HAVE_CALORINE,
    CPUNEP,
)
from ase.build import make_supercell
from ase.io import read, write
from ase.atoms import Atoms
from ase.cell import Cell
from ase.geometry.cell import cellpar_to_cell
from fractions import Fraction # For estimating supercell size
from ase.visualize import view
from ase.calculators.calculator import Calculator, all_changes

# Try importing M3GNet (optional)
try:
    from m3gnet.models import M3GNet, M3GNetCalculator, Potential
    HAVE_M3GNET = True
except ImportError:
    HAVE_M3GNET = False


class GPUMDCalculator(Calculator):
    """ASE-compatible calculator wrapper for GPUMD.

    Notes
    -----
    * We expose ``energy``, ``forces`` and ``stress`` as implemented properties so
      that ASE optimizers (e.g. :class:`ase.optimize.BFGS` and
      :class:`ase.constraints.UnitCellFilter`) can drive both atomic positions
      and cell degrees of freedom.
    * The actual recalculation logic is delegated to the base
      :class:`ase.calculators.calculator.Calculator` methods via
      :meth:`get_potential_energy`, :meth:`get_forces` and :meth:`get_stress`,
      which ensures that geometry changes between steps trigger fresh GPUMD
      evaluations (no stale caching across different Atoms objects).
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, gpumd_binary, potential_path):
        """
        Initialize GPUMD calculator.
        """
        Calculator.__init__(self)
        self.gpumd_binary = gpumd_binary
        self.potential_path = potential_path

        if not os.path.exists(gpumd_binary):
            raise FileNotFoundError(f"GPUMD binary not found at {gpumd_binary}")
        if not os.path.exists(potential_path):
            raise FileNotFoundError(f"GPUMD potential file not found at {potential_path}")

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """
        Calculate energy and forces using GPUMD.
        """
        # Standard ASE housekeeping
        Calculator.calculate(self, atoms, properties, system_changes)

        # GPUMD requires at least 2 atoms
        if len(self.atoms) < 2:
            raise ValueError(f"GPUMD requires at least 2 atoms, but structure has {len(self.atoms)} atom(s)")

        # Create temporary work directory
        work_dir = tempfile.mkdtemp(prefix="gpumd_calc_")

        try:
            # Write GPUMD input files
            self._write_gpumd_input(self.atoms, work_dir)

            # Run GPUMD
            result = subprocess.run(
                [self.gpumd_binary],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                # Save debug files before cleanup to a shared location
                debug_dir = os.path.expanduser("~/gpumd_calc_debug_last_error")
                os.makedirs(debug_dir, exist_ok=True)
                for fname in ["model.xyz", "run.in", "nep.txt"]:
                    src = os.path.join(work_dir, fname)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(debug_dir, fname))

                error_msg = f"GPUMD execution failed (Exit Code {result.returncode}):\n"
                error_msg += f"Debug files saved to: {debug_dir}\n"
                error_msg += f"--- STDOUT ---\n{result.stdout}\n"
                error_msg += f"--- STDERR ---\n{result.stderr}\n"
                raise RuntimeError(error_msg)

            # Parse GPUMD output
            energy, forces, stress = self._parse_gpumd_output(self.atoms, work_dir)

            # Store results for ASE
            self.results['energy'] = energy
            self.results['forces'] = forces
            if stress is not None:
                # ASE expects stress in eV/Å^3 in Voigt order (xx, yy, zz, yz, xz, xy)
                self.results['stress'] = stress

        finally:
            # Clean up temporary directory
            shutil.rmtree(work_dir, ignore_errors=True)

    def _write_gpumd_input(self, atoms, work_dir):
        """Write GPUMD input files (model.xyz and run.in)."""
        # 1. Filename MUST be model.xyz
        xyz_path = os.path.join(work_dir, "model.xyz")

        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        masses = atoms.get_masses()

        # Format lattice vector for GPUMD (row-major, 9 components)
        lattice_str = " ".join([f"{x:.10f}" for row in cell for x in row])

        with open(xyz_path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            # Use mass-based format (required for GPUMD to identify atom types correctly)
            f.write(f'pbc="T T T" ')
            f.write(f'Lattice="{lattice_str}" ')
            f.write(f'Properties=species:S:1:pos:R:3:mass:R:1\n')

            for symbol, pos, mass in zip(symbols, positions, masses):
                f.write(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} {mass:.10f}\n")

        # Copy potential file
        potential_dst = os.path.join(work_dir, "nep.txt")
        shutil.copy(self.potential_path, potential_dst)

        # 2. Write run.in for static calculation
        run_in_path = os.path.join(work_dir, "run.in")
        with open(run_in_path, 'w') as f:
            f.write("# GPUMD static calculation\n")
            f.write("potential   nep.txt\n")
            
            # Static settings: NVE ensemble + time_step 0 + run 1
            # This initializes the engine, computes forces, but does not move atoms.
            f.write("ensemble nve\n")
            f.write("time_step 0\n")
            
            f.write("dump_force 1\n")
            f.write("dump_thermo 1\n")
            f.write("run 1\n")

    def _parse_gpumd_output(self, atoms, work_dir):
        """Parse GPUMD output files (force.out, thermo.out).

        Returns
        -------
        energy : float
            Potential energy for the last MD step (eV).
        forces : (N, 3) ndarray
            Forces on atoms in eV/Å.
        stress : (6,) ndarray or None
            Stress tensor in Voigt order (xx, yy, zz, yz, xz, xy) in units
            of eV/Å^3, or ``None`` if it could not be parsed.
        """
        force_file = os.path.join(work_dir, "force.out")
        if not os.path.exists(force_file):
            raise FileNotFoundError(f"GPUMD force output not found at {force_file}")

        forces = []
        with open(force_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        forces.append([float(x) for x in parts[:3]])

        # GPUMD might output multiple frames if run > 0. We want the last one.
        # If length > len(atoms), take the last N atoms
        forces = np.array(forces)
        if len(forces) > len(atoms):
            forces = forces[-len(atoms):]
        
        if len(forces) != len(atoms):
            raise ValueError(f"Force count mismatch: expected {len(atoms)}, got {len(forces)}")

        thermo_file = os.path.join(work_dir, "thermo.out")
        energy = 0.0
        stress = None

        # According to the GPUMD documentation for thermo.out, the columns are::
        #
        #   column   1   2   3   4   5   6   7    8    9   10  11  12  13  14  15  16  17  18
        #   quantity T   K   U  Pxx Pyy Pzz Pyz  Pxz  Pxy  ax  ay  az  bx  by  bz  cx  cy  cz
        #
        # We use:
        #   * column 3 (index 2) for the potential energy U (eV)
        #   * columns 4–9 (indices 3–8) for the stress tensor components in GPa
        #     (Voigt order: xx, yy, zz, yz, xz, xy), which we convert to eV/Å^3.
        if os.path.exists(thermo_file):
            with open(thermo_file, 'r') as f:
                lines = f.readlines()
            # Parse the last valid line
            for line in reversed(lines):
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    energy = float(parts[2])  # U (potential energy), column 3
                except ValueError:
                    continue

                # Parse stress if all required columns are present
                if len(parts) >= 9:
                    try:
                        pxx = float(parts[3])  # GPa
                        pyy = float(parts[4])
                        pzz = float(parts[5])
                        pyz = float(parts[6])
                        pxz = float(parts[7])
                        pxy = float(parts[8])

                        # Convert from GPa to eV/Å^3
                        # 1 eV/Å^3 = 160.21766208 GPa
                        GPa_to_eV_per_A3 = 1.0 / 160.21766208

                        # CRITICAL: GPUMD thermo.out reports PRESSURE (Pxx, Pyy, Pzz, ...)
                        # ASE expects STRESS, where stress = -pressure
                        # We must negate the values to convert pressure → stress
                        stress_vals = -np.array(
                            [pxx, pyy, pzz, pyz, pxz, pxy],
                            dtype=float,
                        ) * GPa_to_eV_per_A3
                        stress = stress_vals
                    except ValueError:
                        # Keep energy but fall back to no stress
                        stress = None
                break

        return energy, forces, stress

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return potential energy.

        GPUMD does not distinguish between standard and force-consistent
        energies, so the ``force_consistent`` flag is accepted for API
        compatibility but has no effect. Delegating to the base class ensures
        that geometry changes trigger fresh calculations as needed.
        """

        return Calculator.get_potential_energy(self, atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Return forces on atoms in eV/Å.

        Delegates to :class:`ase.calculators.calculator.Calculator` so that
        ASE's internal caching and change-detection logic are respected.
        """

        return Calculator.get_forces(self, atoms)

    def get_stress(self, atoms=None):
        """Return stress tensor in Voigt order (xx, yy, zz, yz, xz, xy).

        The values are in eV/Å^3, obtained from the GPUMD ``thermo.out``
        pressures (which are reported in GPa). If stress could not be parsed
        from the output, this will return zeros.
        """

        try:
            # Prefer the standard Calculator implementation, which will
            # trigger a new calculation if required and read ``results['stress']``.
            stress = Calculator.get_stress(self, atoms)
        except Exception:
            # Fall back to zero stress if anything goes wrong; this preserves
            # backward compatibility with older behavior where stress was not
            # available at all.
            stress = np.zeros(6, dtype=float)
        return stress


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

def initialize_calculator(engine, model_name="medium-omat-0", checkpoint_path=None, nep_model_path=None, checkpoint_model_path=None):
    """
    Initializes and returns the appropriate calculator.
    
    Args:
        engine (str): Engine name ('mace', 'm3gnet', 'esen', 'uma', 'gpumd', 'nep', 'calorine')
        model_name (str): Model name for MACE calculator
        checkpoint_path (str): DEPRECATED. Use nep_model_path for NEP engines or checkpoint_model_path for eSEN/UMA
        nep_model_path (str): Path to NEP potential model file (for GPUMD, calorine engines)
        checkpoint_model_path (str): Path to model checkpoint file (for eSEN, UMA engines)
        
    Returns:
        calculator: ASE-compatible calculator object
    """
    import warnings
    
    # Handle backward compatibility: if checkpoint_path is provided, redirect appropriately
    if checkpoint_path is not None:
        if engine in ("gpumd", "nep", "calorine"):
            warnings.warn(
                "Using --checkpoint for NEP model files is deprecated. "
                "Please use --nep_model instead. "
                "The --checkpoint flag will be reserved for calculation restart functionality in future versions.",
                DeprecationWarning,
                stacklevel=2
            )
            if nep_model_path is None:
                nep_model_path = checkpoint_path
                print(f"⚠ Deprecation: --checkpoint used for NEP model. Redirecting to --nep_model.")
        elif engine in ("esen", "uma"):
            warnings.warn(
                "Using --checkpoint for model specification is deprecated. "
                "Please use --checkpoint_model instead. "
                "The --checkpoint flag will be reserved for calculation restart functionality in future versions.",
                DeprecationWarning,
                stacklevel=2
            )
            if checkpoint_model_path is None:
                checkpoint_model_path = checkpoint_path
                print(f"⚠ Deprecation: --checkpoint used for model specification. Redirecting to --checkpoint_model.")
    
    calculator = None
    if engine == "m3gnet":
        if not HAVE_M3GNET:
            sys.exit("M3GNet not found – `pip install m3gnet` or use --engine mace")
        print("Initializing M3GNet calculator...")
        potential = Potential(M3GNet.load())
        calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
    elif engine == "mace":
        if not HAVE_MACE:
            sys.exit("MACE not found – `pip install mace-torch` or use --engine m3gnet")
        device = get_mace_device()
        print(f"Initializing MACE calculator on device: {device}...")
        calculator = mace_mp(model=model_name, dispersion=False, default_dtype="float64", device=device, stress=True)
    elif engine == "esen":
        import torch
        if not HAVE_ESEN:
            sys.exit("fairchem-core not found")
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        if checkpoint_model_path is None:
            # Try multiple possible locations for the eSEN model
            module_dir = os.path.dirname(os.path.abspath(__file__))

            possible_paths = [
                # 1. Relative to module location (works for both dev install and site-packages)
                os.path.join(module_dir, "..", "..", "fairchem_models", "esen_30m_omat.pt"),
                # 2. Environment variable override (full path to model file)
                os.environ.get("VIBROML_ESEN_CHECKPOINT", ""),
                # 3. Environment variable for models directory
                os.path.join(os.environ.get("VIBROML_FAIRCHEM_MODELS", ""), "esen_30m_omat.pt"),
                # 4. Common installation locations
                "/globalscratch/ucl/modl/rgouvea/VibroML/fairchem_models/esen_30m_omat.pt",
                "/auto/globalscratch/users/r/g/rgouvea/VibroML/fairchem_models/esen_30m_omat.pt",
                # 5. Relative to current working directory
                os.path.join(os.getcwd(), "fairchem_models", "esen_30m_omat.pt"),
            ]

            checkpoint_model_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    checkpoint_model_path = os.path.abspath(path)
                    break

            if checkpoint_model_path is None:
                error_msg = "eSEN model not found. Tried:\n"
                for path in possible_paths:
                    if path:
                        error_msg += f"  - {os.path.abspath(path)}\n"
                error_msg += "\nYou can:\n"
                error_msg += "  1. Set VIBROML_ESEN_CHECKPOINT to the full path of esen_30m_omat.pt\n"
                error_msg += "  2. Set VIBROML_FAIRCHEM_MODELS to the directory containing the model files"
                sys.exit(error_msg)

        if not os.path.exists(checkpoint_model_path):
             sys.exit(f"eSEN model not found at {checkpoint_model_path}")
        print(f"Initializing eSEN calculator with model: {checkpoint_model_path}...")
        device_is_cpu = not torch.cuda.is_available()
        print(f"eSEN using CPU only: {device_is_cpu}")
        calculator = OCPCalculator(checkpoint_path=checkpoint_model_path, cpu=device_is_cpu)
    elif engine == "uma":
        print("Initializing UMA calculator...")
        import torch
        if not HAVE_UMA:
            sys.exit("fairchem-core (new API) not found")
        if FAIRCHEM_API_VERSION == "new":
            from fairchem.core import FAIRChemCalculator
            from fairchem.core.calculate.pretrained_mlip import load_predict_unit
            if checkpoint_model_path is None:
                # Try multiple possible locations for the UMA model
                module_dir = os.path.dirname(os.path.abspath(__file__))

                possible_paths = [
                    # 1. Relative to module location (works for both dev install and site-packages)
                    os.path.join(module_dir, "..", "..", "fairchem_models", "uma-m-1p1.pt"),
                    # 2. Environment variable override (full path to model file)
                    os.environ.get("VIBROML_UMA_CHECKPOINT", ""),
                    # 3. Environment variable for models directory
                    os.path.join(os.environ.get("VIBROML_FAIRCHEM_MODELS", ""), "uma-m-1p1.pt"),
                    # 4. Common installation locations
                    "/globalscratch/ucl/modl/rgouvea/VibroML/fairchem_models/uma-m-1p1.pt",
                    "/auto/globalscratch/users/r/g/rgouvea/VibroML/fairchem_models/uma-m-1p1.pt",
                    # 5. Relative to current working directory
                    os.path.join(os.getcwd(), "fairchem_models", "uma-m-1p1.pt"),
                ]

                checkpoint_model_path = None
                for path in possible_paths:
                    if path and os.path.exists(path):
                        checkpoint_model_path = os.path.abspath(path)
                        break

                if checkpoint_model_path is None:
                    error_msg = "UMA model not found. Tried:\n"
                    for path in possible_paths:
                        if path:
                            error_msg += f"  - {os.path.abspath(path)}\n"
                    error_msg += "\nYou can:\n"
                    error_msg += "  1. Set VIBROML_UMA_CHECKPOINT to the full path of uma-m-1p1.pt\n"
                    error_msg += "  2. Set VIBROML_FAIRCHEM_MODELS to the directory containing the model files"
                    sys.exit(error_msg)

            if not os.path.exists(checkpoint_model_path):
                sys.exit(f"UMA model not found at {checkpoint_model_path}")
            print(f"Initializing UMA calculator with model: {checkpoint_model_path}...")
            try:
                # FIX: Detect GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"UMA using device: {device}")
                
                predict_unit = load_predict_unit(checkpoint_model_path, device=device)
                calculator = FAIRChemCalculator(predict_unit=predict_unit, task_name="omc")
            except Exception as e:
                sys.exit(f"Failed to initialize UMA calculator: {e}")
        else:
            sys.exit("UMA requires fairchem-core 2.12.0+")
    elif engine in ("nep", "calorine"):
        # Calorine NEP engine via ASE-compatible CPUNEP calculator
        if not HAVE_CALORINE or CPUNEP is None:
            sys.exit("calorine not found – install it in this environment or use a different engine")

        if nep_model_path is None:
            sys.exit("For engine 'nep'/'calorine', you must provide --nep_model pointing to a NEP potential file")

        nep_model_path = os.path.abspath(nep_model_path)
        if not os.path.exists(nep_model_path):
            sys.exit(f"NEP potential not found at {nep_model_path}")

        print(f"Initializing calorine CPUNEP calculator with potential: {nep_model_path}...")
        try:
            calculator = CPUNEP(nep_model_path)
        except Exception as e:
            sys.exit(f"Failed to initialize calorine CPUNEP calculator: {e}")
    elif engine == "gpumd":
        if not HAVE_GPUMD:
            sys.exit("GPUMD binary not found")
        if nep_model_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            package_dir = os.path.dirname(module_dir)
            nep_model_path = os.path.join(package_dir, "..", "GPUMD", "examples", "nep_train", "nep.txt")
            nep_model_path = os.path.abspath(nep_model_path)
        if not os.path.exists(nep_model_path):
            sys.exit(f"GPUMD NEP potential not found at {nep_model_path}")
        print(f"Initializing GPUMD calculator with potential: {nep_model_path}...")
        try:
            calculator = GPUMDCalculator(GPUMD_BINARY_PATH, nep_model_path)
        except Exception as e:
            sys.exit(f"Failed to initialize GPUMD calculator: {e}")
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

def generate_supercell_variants(base_supercell, max_variants=5):
    """
    Generate variants of a supercell for optimization.
    
    Args:
        base_supercell (tuple): Base supercell dimensions (nx, ny, nz)
        max_variants (int): Maximum number of variants to generate
    
    Returns:
        list: List of supercell variants as tuples
    """
    variants = [base_supercell]  # Include the original
    
    nx, ny, nz = base_supercell
    
    # Generate variants by scaling individual dimensions
    scale_factors = [0.5, 1.5, 2.0]
    
    for scale in scale_factors:
        # Scale each dimension independently
        for i in range(3):
            new_dims = list(base_supercell)
            new_dims[i] = max(1, int(new_dims[i] * scale))
            variant = tuple(new_dims)
            
            if variant not in variants and len(variants) < max_variants:
                variants.append(variant)
    
    # Add some completely different variants
    additional_variants = [
        (max(1, nx-1), ny, nz),
        (nx, max(1, ny-1), nz),
        (nx, ny, max(1, nz-1)),
        (nx+1, ny, nz),
        (nx, ny+1, nz),
        (nx, ny, nz+1),
    ]
    
    for variant in additional_variants:
        if variant not in variants and len(variants) < max_variants:
            variants.append(variant)
    
    return variants[:max_variants]

def estimate_commensurate_supercell_size_custom(q_point_frac, base_supercell=(1, 1, 1), max_denominator=10):
    """
    Estimates commensurate supercell dimensions based on q-point and a base supercell.
    
    Args:
        q_point_frac (list): Q-point in fractional coordinates
        base_supercell (tuple): Base supercell to scale from
        max_denominator (int): Maximum denominator for fraction approximation
    
    Returns:
        tuple: Suggested supercell dimensions (N1, N2, N3)
    """
    from fractions import Fraction
    
    # Start with base supercell
    supercell_dims = list(base_supercell)
    
    if all(abs(q) < 1e-6 for q in q_point_frac):
        print("Q-point is Gamma (0,0,0). Using base supercell.")
        return tuple(supercell_dims)
    
    for i, q_comp in enumerate(q_point_frac):
        if abs(q_comp) < 1e-6:
            # Keep base dimension for zero component
            continue
        
        try:
            fraction = Fraction(q_comp).limit_denominator(max_denominator)
            required_multiple = fraction.denominator
            # Scale the base dimension to be commensurate
            supercell_dims[i] = base_supercell[i] * required_multiple
        except Exception as e:
            print(f"Error processing q-component {q_comp}: {e}. Using base dimension.")
    
    result = tuple(supercell_dims)
    print(f"Estimated commensurate supercell for q-point {q_point_frac} with base {base_supercell}: {result}")
    return result

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


def filter_supercells_by_atom_count(primitive_atoms, supercell_variants, max_atoms_per_supercell):
    """
    Select the best-fitting supercell variant based on maximum atom count constraint.

    Uses a "best fit" strategy: selects the largest supercell variant that stays within
    the atom limit. If all variants exceed the limit, falls back to the smallest variant.

    Args:
        primitive_atoms: ASE Atoms object for the primitive cell
        supercell_variants: List of supercell tuples (nx, ny, nz)
        max_atoms_per_supercell: Maximum number of atoms allowed (None = no limit)

    Returns:
        list: Single-element list containing the best-fitting supercell variant
    """
    if max_atoms_per_supercell is None:
        return supercell_variants

    num_atoms_primitive = len(primitive_atoms)

    # Calculate atom counts for all variants
    variants_with_counts = []
    for variant in supercell_variants:
        nx, ny, nz = variant
        num_atoms = num_atoms_primitive * nx * ny * nz
        variants_with_counts.append((variant, num_atoms))

    # Sort by atom count (descending) to find the largest valid variant
    variants_with_counts.sort(key=lambda x: x[1], reverse=True)

    # Find the largest variant that fits within the limit
    best_variant = None
    for variant, num_atoms in variants_with_counts:
        if num_atoms <= max_atoms_per_supercell:
            best_variant = variant
            print(f"   Selected best-fit supercell {variant} with {num_atoms} atoms (limit: {max_atoms_per_supercell})")
            break

    # If no variant fits, use the smallest one
    if best_variant is None:
        # variants_with_counts is sorted descending, so smallest is at the end
        smallest_variant, smallest_atoms = variants_with_counts[-1]
        best_variant = smallest_variant
        print(f"   WARNING: All supercell variants exceed atom limit of {max_atoms_per_supercell}.")
        print(f"   Using smallest variant {best_variant} with {smallest_atoms} atoms as fallback.")

    # Log skipped variants for transparency
    for variant, num_atoms in variants_with_counts:
        if variant != best_variant and num_atoms > max_atoms_per_supercell:
            print(f"   Skipped supercell {variant}: would have {num_atoms} atoms (exceeds limit)")

    return [best_variant]


def generate_displaced_supercells(primitive_atoms,
                                  softest_modes_info_list, # Changed to a list of mode infos
                                  scale_mode1,             # Scaling factor for mode 1 displacements
                                  ratio_mode2_to_mode1,    # Single ratio for mode 2
                                  supercell_variants,
                                  output_base_dir,
                                  iteration_idx,
                                  original_prefix,
                                  cell_transformation_vector, # 6-element vector
                                  use_phase_factor=True,   # NEW PARAMETER
                                  mutation_data=None,      # NEW PARAMETER for mode replacement info
                                  max_atoms_per_supercell=None):  # NEW PARAMETER for atom count constraint
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
      use_phase_factor (bool): If True, phase factors are applied. If False, phase factors are ignored (set to 1).
      mutation_data (dict): Optional mutation data for filename and logging.
      max_atoms_per_supercell (int): Maximum number of atoms allowed in supercells (None = no limit).

   Returns:
      list: A list of paths to the generated displaced CIF files.
   """
   print(f"\n--- Generating Displaced Supercells (Iteration {iteration_idx}) ---")
   print(f"   Use Phase Factor: {use_phase_factor}") # Log the new parameter

   # Filter supercell variants by atom count constraint
   supercell_variants = filter_supercells_by_atom_count(primitive_atoms, supercell_variants, max_atoms_per_supercell)
  
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
           phase_factor_1 = 1.0 # Default to 1 if use_phase_factor is False  
           if use_phase_factor:  
               dot_product_1 = np.dot(q_point_frac_1, cell_shift_primitive_units_i_sc)  
               phase_factor_1 = np.exp(1j * 2 * np.pi * dot_product_1)  
  
           # Calculate displacement for mode 1 (can be complex)  
           disp_mode1_vector_complex = normalized_displacements_1[prim_idx] * scale_mode1 * max_raw_disp_magnitude_1 * phase_factor_1  
  
           # Calculate displacement for mode 2, if available (can be complex)  
           disp_mode2_vector_complex = np.zeros(3, dtype=complex)  
           if normalized_displacements_2 is not None:  
               phase_factor_2 = 1.0 # Default to 1 if use_phase_factor is False  
               if use_phase_factor:  
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
      # pf_ for phase factor status
      # mr_ for mode replacement info (if applicable)
      # Using f-strings for precise formatting of floats in filename

      # Add mode replacement info to filename if available
      mode_replacement_str = ""
      if mutation_data and mutation_data.get('mode_replaced', False) and mutation_data.get('selected_mode'):
          selected_mode = mutation_data['selected_mode']
          # Format: mr_LABEL_bIDX_fFREQ (e.g., mr_M_b2_fm1p234 for M point, band 2, -1.234 THz)
          freq_str = f"{'m' if selected_mode['frequency'] < 0 else 'p'}{abs(selected_mode['frequency']):.3f}".replace('.', 'p')
          mode_replacement_str = f"_mr_{selected_mode['label']}_b{selected_mode['band_index']}_f{freq_str}"

      filename = (f"{original_prefix}_sc_{sc_n1}x{sc_n2}x{sc_n3}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}_pf_{str(use_phase_factor).lower()}{mode_replacement_str}.cif")
      filepath = os.path.join(output_base_dir, filename)
      write(filepath, displaced_atoms) # displaced_atoms already has the cell transformation applied
      generated_files.append(filepath)

      filename_xyz = (f"{original_prefix}_sc_{sc_n1}x{sc_n2}x{sc_n3}_d1_{scale_mode1:.3f}_r21_{ratio_mode2_to_mode1:.3f}_c_{cell_transform_str}_pf_{str(use_phase_factor).lower()}{mode_replacement_str}.xyz")
      filepath_xyz = os.path.join(output_base_dir, filename_xyz)
      write(filepath_xyz, displaced_atoms)

      # Enhanced logging for mode replacement
      if mutation_data and mutation_data.get('mode_replaced', False):
          selected_mode = mutation_data['selected_mode']
          print(f"  Generated {filename} and {filename_xyz}")
          print(f"    MODE REPLACEMENT: Second mode replaced with {selected_mode['label']} point, band {selected_mode['band_index']}, frequency {selected_mode['frequency']:.3f} THz")
      else:
          print(f"  Generated {filename} and {filename_xyz}")
  
   print("Finished generating displaced supercells with combined modes and cell transformations.")
   return generated_files


def generate_random_displaced_structures(primitive_atoms,
                                       displacement_bounds,
                                       supercell_variants,
                                       output_base_dir,
                                       iteration_idx,
                                       original_prefix,
                                       cell_transformation_vector=None,
                                       cell_perturbation=True,
                                       random_seed=None,
                                       max_atoms_per_supercell=None):
    """
    Generates structures with random atomic displacements and optional cell perturbations.

    Args:
        primitive_atoms (ase.atoms.Atoms): The primitive cell structure.
        displacement_bounds (list): [min_displacement, max_displacement] in Angstroms.
        supercell_variants (list): List of tuples defining supercell sizes, e.g., [(2,2,2), (3,3,3)].
        output_base_dir (str): The main output directory.
        iteration_idx (int): Current iteration index for naming.
        original_prefix (str): Prefix for output filenames.
        cell_transformation_vector (tuple): 6-element vector for cell parameter changes.
        cell_perturbation (bool): Whether to apply random cell perturbations.
        random_seed (int): Random seed for reproducibility.
        max_atoms_per_supercell (int): Maximum number of atoms allowed in supercells (None = no limit).

    Returns:
        list: A list of paths to the generated random CIF files.
    """
    import numpy as np
    import random
    from ase.io import write
    from ase.geometry.cell import cellpar_to_cell

    print(f"\n--- Generating Random Displaced Structures (Iteration {iteration_idx}) ---")

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        print(f"   Random seed set to: {random_seed}")

    generated_files = []
    min_disp, max_disp = displacement_bounds

    print(f"   Displacement bounds: [{min_disp:.3f}, {max_disp:.3f}] Å")
    print(f"   Cell perturbation: {cell_perturbation}")
    print(f"   Supercell variants: {supercell_variants}")

    # Filter supercell variants by atom count constraint
    supercell_variants = filter_supercells_by_atom_count(primitive_atoms, supercell_variants, max_atoms_per_supercell)

    for supercell_variant in supercell_variants:
        sc_n1, sc_n2, sc_n3 = supercell_variant

        # Create supercell
        supercell_atoms = primitive_atoms.repeat((sc_n1, sc_n2, sc_n3))

        # Apply cell transformation if provided
        if cell_transformation_vector is not None:
            try:
                # Get original cell parameters
                original_cell_params = supercell_atoms.cell.cellpar()

                # Apply transformations: [a_scale, b_scale, c_scale, alpha_change, beta_change, gamma_change]
                new_cell_params = original_cell_params.copy()
                new_cell_params[0] *= (1.0 + cell_transformation_vector[0])  # a
                new_cell_params[1] *= (1.0 + cell_transformation_vector[1])  # b
                new_cell_params[2] *= (1.0 + cell_transformation_vector[2])  # c
                new_cell_params[3] += cell_transformation_vector[3]  # alpha
                new_cell_params[4] += cell_transformation_vector[4]  # beta
                new_cell_params[5] += cell_transformation_vector[5]  # gamma

                # Create new cell matrix
                new_cell_matrix = cellpar_to_cell(new_cell_params)
                supercell_atoms.set_cell(new_cell_matrix, scale_atoms=True)

            except (AssertionError, Exception) as e:
                print(f"Warning: Cell transformation failed for supercell {supercell_variant}: {e}")
                print("Continuing with original cell parameters.")

        # Apply additional random cell perturbations if enabled
        if cell_perturbation:
            try:
                current_cell_params = supercell_atoms.cell.cellpar()

                # Small random perturbations to cell parameters (±2% for lengths, ±2° for angles)
                cell_scale_perturbations = np.random.uniform(-0.02, 0.02, 3)
                angle_perturbations = np.random.uniform(-2.0, 2.0, 3)

                perturbed_cell_params = current_cell_params.copy()
                perturbed_cell_params[0] *= (1.0 + cell_scale_perturbations[0])  # a
                perturbed_cell_params[1] *= (1.0 + cell_scale_perturbations[1])  # b
                perturbed_cell_params[2] *= (1.0 + cell_scale_perturbations[2])  # c
                perturbed_cell_params[3] += angle_perturbations[0]  # alpha
                perturbed_cell_params[4] += angle_perturbations[1]  # beta
                perturbed_cell_params[5] += angle_perturbations[2]  # gamma

                perturbed_cell_matrix = cellpar_to_cell(perturbed_cell_params)
                supercell_atoms.set_cell(perturbed_cell_matrix, scale_atoms=True)

            except (AssertionError, Exception) as e:
                print(f"Warning: Random cell perturbation failed for supercell {supercell_variant}: {e}")
                print("Continuing without additional cell perturbation.")

        # Generate random atomic displacements
        num_atoms = len(supercell_atoms)

        # Random displacement magnitudes for each atom
        displacement_magnitudes = np.random.uniform(min_disp, max_disp, num_atoms)

        # Random displacement directions (unit vectors)
        displacement_directions = np.random.randn(num_atoms, 3)
        displacement_directions = displacement_directions / np.linalg.norm(displacement_directions, axis=1, keepdims=True)

        # Calculate total displacements
        total_displacements = displacement_directions * displacement_magnitudes[:, np.newaxis]

        # Apply displacements to atomic positions
        displaced_positions = supercell_atoms.get_positions() + total_displacements
        supercell_atoms.set_positions(displaced_positions)

        # Wrap atoms back into the unit cell
        supercell_atoms.wrap(pbc=True)

        # Generate filename using GA-style naming convention
        cell_transform_str = "none"
        if cell_transformation_vector is not None:
            cell_transform_str = "_".join([f"{x:.3f}" for x in cell_transformation_vector])

        # Use GA-style naming: similar to displaced supercells but with random prefix
        filename = (f"{original_prefix}_random_sc_{sc_n1}x{sc_n2}x{sc_n3}_"
                   f"disp_{min_disp:.3f}to{max_disp:.3f}_"
                   f"c_{cell_transform_str}.cif")

        filepath = os.path.join(output_base_dir, filename)
        write(filepath, supercell_atoms)
        generated_files.append(filepath)

        # Also save as XYZ for convenience
        filename_xyz = filename.replace('.cif', '.xyz')
        filepath_xyz = os.path.join(output_base_dir, filename_xyz)
        write(filepath_xyz, supercell_atoms)

        print(f"   Generated random structure: {os.path.basename(filepath)}")

    print(f"Generated {len(generated_files)} random displaced structure files.")
    return generated_files

