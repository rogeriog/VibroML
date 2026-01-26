import math
import os
import shutil
import logging
import sys
import torch
import json
import numpy as np
import argparse

# Try importing TensorFlow (optional, used by M3GNet)
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)
    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False

# Try importing MACE
try:
    from mace.calculators import mace_mp
    HAVE_MACE = True
except ImportError:
    HAVE_MACE = False
    mace_mp = None  # Placeholder when MACE is not available

# Try importing fairchem (for eSEN/UMA calculators)
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

# Try importing calorine (optional, for NEP calculators)
try:
    from calorine.calculators import CPUNEP, GPUNEP
    HAVE_CALORINE = True
except ImportError:
    HAVE_CALORINE = False
    CPUNEP = None
    GPUNEP = None

# Check for GPUMD binary
HAVE_GPUMD = False
GPUMD_BINARY_PATH = None

# Try to find GPUMD binary in common locations
gpumd_search_paths = [
    "/auto/globalscratch/users/r/g/rgouvea/VibroML/GPUMD/src/gpumd",
    os.path.expanduser("~/VibroML/GPUMD/src/gpumd"),
    os.path.expanduser("~/GPUMD/src/gpumd"),
    "/opt/gpumd/bin/gpumd",
    "/usr/local/bin/gpumd",
]

for path in gpumd_search_paths:
    if os.path.exists(path) and os.access(path, os.X_OK):
        GPUMD_BINARY_PATH = path
        HAVE_GPUMD = True
        break

def load_default_settings(file_path="default_settings.json"):  
    """  
    Loads default settings from a JSON file.  
    """  
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    full_path = os.path.join(package_root, file_path)  

    if not os.path.exists(full_path):  
        print(f"Error: Default settings file not found at {full_path}")  
        return {}  
      
    try:  
        with open(full_path, 'r') as f:  
            settings = json.load(f)  
        print(f"Default settings loaded from {full_path}")  
        return settings  
    except json.JSONDecodeError as e:  
        print(f"Error decoding default settings JSON from {full_path}: {e}")  
        return {}  
    except Exception as e:  
        print(f"An unexpected error occurred while loading default settings from {full_path}: {e}")  
        return {}
    
def parse_supercell_dimensions(supercell_input):
    """
    Parse supercell dimensions from various input formats.
    """
    if isinstance(supercell_input, (list, tuple)):
        if len(supercell_input) == 1:
            return (supercell_input[0], supercell_input[0], supercell_input[0])
        elif len(supercell_input) == 3:
            return tuple(supercell_input)
        else:
            raise ValueError(f"Supercell list/tuple must have 1 or 3 elements, got {len(supercell_input)}")
    elif isinstance(supercell_input, int):
        return (supercell_input, supercell_input, supercell_input)
    elif isinstance(supercell_input, str):
        parts = [part.strip() for part in supercell_input.split(',')]
        if len(parts) == 1:
            try:
                n = int(parts[0])
                return (n, n, n)
            except ValueError:
                raise ValueError(f"Invalid supercell dimension: '{parts[0]}' is not an integer")
        elif len(parts) == 3:
            try:
                dimensions = tuple(int(part) for part in parts)
            except ValueError:
                raise ValueError(f"Invalid supercell dimensions: all values must be integers")
            return dimensions
        else:
            raise ValueError(f"Supercell string must have 1 or 3 comma-separated values, got {len(parts)}")
    else:
        raise ValueError(f"Unsupported supercell input type: {type(supercell_input)}")

def parse_cli_screen_supercell_ns(cli_input_list):  
    """  
    Parses screen_supercell_ns from a list of strings from the command line.  
    """  
    parsed_list = []  
    for item_str in cli_input_list:  
        try:  
            if ',' in item_str:  
                dims = tuple(int(x.strip()) for x in item_str.split(','))  
                if len(dims) != 3:  
                    raise ValueError(f"Supercell dimension must have 3 elements: {item_str}")  
                parsed_list.append(dims)  
            else:  
                n = int(item_str)  
                parsed_list.append((n, n, n))  
        except (ValueError, TypeError) as e:  
            raise ValueError(f"Invalid format for screen_supercell_ns argument '{item_str}': {e}")  
    return parsed_list

def parse_screen_supercell_ns(screen_supercell_ns_input):
    """
    Parse screen_supercell_ns from various input formats.
    """
    if not isinstance(screen_supercell_ns_input, list):
        raise ValueError(f"screen_supercell_ns must be a list, got {type(screen_supercell_ns_input)}")
    result = []
    for i, item in enumerate(screen_supercell_ns_input):
        if isinstance(item, int):
            result.append((item, item, item))
        elif isinstance(item, (list, tuple)):
            if len(item) != 3:
                raise ValueError(f"Supercell dimensions at index {i} must have 3 elements, got {len(item)}")
            try:
                dimensions = tuple(int(x) for x in item)
                for j, dim in enumerate(dimensions):
                    if dim <= 0:
                        raise ValueError(f"Supercell dimension {j+1} at index {i} must be positive, got {dim}")
                result.append(dimensions)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid supercell dimensions at index {i}: {e}")
        else:
            raise ValueError(f"Invalid supercell format at index {i}: expected int or list/tuple, got {type(item)}")
    return result

def custom_round(number, interval):
   """Rounds a number down to the nearest multiple of an interval."""
   return math.floor(number / interval) * interval

def clean_phonon_cache(phonon_cache_dir='phonon'):
   """Checks for and deletes the phonon cache directory if it exists."""
   if os.path.exists(phonon_cache_dir):
      cache_files = [f for f in os.listdir(phonon_cache_dir) if f.startswith('cache') and f.endswith('.json')]
      if cache_files:
         try:
               shutil.rmtree(phonon_cache_dir)
               print(f"Deleted existing phonon cache directory: {phonon_cache_dir}")
         except OSError as e:
               print(f"Error deleting phonon cache directory {phonon_cache_dir}: {e}")

def get_mace_device():
   """Determines the appropriate device for MACE calculation (cuda or cpu)."""
   if HAVE_MACE:
      if torch.cuda.is_available():
         device = "cuda"
         print("CUDA is available. Using GPU for MACE calculation.")
      else:
         device = "cpu"
         print("CUDA is not available. Falling back to CPU for MACE calculation.")
      return device
   else:
      return None # MACE not available
    
def get_arg_parser_and_settings():
    """
    Loads default settings and initializes the ArgumentParser.
    """
    default_settings = load_default_settings()
    settings = {
        "default_engine": default_settings.get("default_engine", "mace"),
        "default_model_name": default_settings.get("default_model_name", "medium-omat-0"),
        "default_fmax": default_settings.get("default_fmax", 0.001),
        "default_delta": default_settings.get("default_delta", 0.03),
        "default_supercell_n": default_settings.get("default_supercell_n", 3),
        "screen_supercell_ns": default_settings.get("screen_supercell_ns", [[2,2,2], [3,3,3], [4,4,4]]),
        "screen_deltas": default_settings.get("screen_deltas", [0.05, 0.03, 0.01]),
        "screen_fmax_values": default_settings.get("screen_fmax_values", [0.001, 0.0005, 0.0001]),
        "phonon_path_npoints": default_settings.get("phonon_path_npoints", 100),
        "phonon_dos_grid": default_settings.get("phonon_dos_grid", [40, 40, 40]),
        "default_units": default_settings.get("default_units", "THz"),
        "default_traj_kT": default_settings.get("default_traj_kT", 1.0),
        "negative_phonon_threshold_thz": default_settings.get("negative_phonon_threshold_thz", -0.1),
        "soft_mode_max_iterations": default_settings.get("soft_mode_max_iterations", 3),
        "soft_mode_displacement_scales": default_settings.get("soft_mode_displacement_scales",  [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        "mode2_ratio_scales": default_settings.get("mode2_ratio_scales", [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]),
        "soft_mode_num_top_structures_to_analyze": default_settings.get("soft_mode_num_top_structures_to_analyze", 3),
        "cell_scale_factors": default_settings.get("cell_scale_factors", [-0.05, 0.0, 0.05, 0.10]),
        "num_modes_to_return": default_settings.get("num_modes_to_return", 2),
        "ga_population_size": default_settings.get("ga_population_size", 50),
        "ga_mutation_rate": default_settings.get("ga_mutation_rate", 0.1),
        "ga_generations": default_settings.get("ga_generations", 3),
        "num_new_points_per_iteration": default_settings.get("num_new_points_per_iteration", 30),
        "default_method": default_settings.get("default_method", "phonon_only"),
        "random_displacement_bounds": default_settings.get("random_displacement_bounds", [0.1, 2.0]),
        "random_cell_perturbation": default_settings.get("random_cell_perturbation", True),
        "random_seed": default_settings.get("random_seed", None),
        "decomposition_threshold": default_settings.get("decomposition_threshold", 0.5),
        "md_temperature": default_settings.get("md_temperature", 300.0),
        "md_pressure": default_settings.get("md_pressure", 0.0),
        "md_time": default_settings.get("md_time", 10.0),
        "md_supercell_size": default_settings.get("md_supercell_size", "2x2x2"),
        "md_equilibration_fraction": default_settings.get("md_equilibration_fraction", 0.2),
        "volume_expansion_threshold": default_settings.get("volume_expansion_threshold", 2.5),
        "max_atoms_per_supercell": default_settings.get("max_atoms_per_supercell", None)
    }

    parser = argparse.ArgumentParser(description="Calculate phonon band structure and DOS for crystal structures, with optional relaxation and soft mode analysis.")
    parser.add_argument("--cif", type=str, required=True, help="Path to the CIF file.")
    parser.add_argument("--no-relax", action="store_true", help="Skip relaxation of the structure before calculation.")
    parser.add_argument("--engine", type=str, default=settings["default_engine"], help=f"Calculation engine (default: {settings['default_engine']}).")
    parser.add_argument("--units", type=str, default=settings["default_units"], choices=["THz", "cm-1", "eV"], help=f"Units for frequency (default: {settings['default_units']}).")
    parser.add_argument("--model_name", type=str, default=settings["default_model_name"], help=f"Model name for the calculator (default: {settings['default_model_name']}).")
    parser.add_argument("--nep_model", type=str, default=None, help="Path to NEP potential model file (for GPUMD, calorine engines). For neural network potential (NEP) models only.")
    parser.add_argument("--checkpoint_model", type=str, default=None, help="Path to model checkpoint file (for eSEN, UMA engines). Specifies the trained model to load.")
    parser.add_argument("--checkpoint", type=str, default=None, help="DEPRECATED: Use --nep_model for NEP engines or --checkpoint_model for eSEN/UMA engines. Reserved for calculation state restart functionality.")
    parser.add_argument("--supercell_n", type=int, default=settings["default_supercell_n"], help=f"Size of the supercell (N, N, N). Deprecated, use --supercell.")
    parser.add_argument("--supercell", type=str, default=None, help="Supercell dimensions. 'N' for cubic (N,N,N) or 'Nx,Ny,Nz'.")
    parser.add_argument("--delta", type=float, default=settings["default_delta"], help=f"Displacement distance (default: {settings['default_delta']}).")
    parser.add_argument("--fmax", type=float, default=settings["default_fmax"], help=f"Max force tolerance for relaxation (default: {settings['default_fmax']} eV/Å).")
    parser.add_argument("--relaxation-patience", type=int, default=5, help="Number of initial relaxation steps to wait before applying energy drop termination criteria (default: 5). Increase this value (e.g., 30) if structures fail during initial relaxation due to large energy changes.")
    parser.add_argument("--volume-expansion-threshold", type=float, default=settings["volume_expansion_threshold"], help=f"Volume expansion threshold for relaxation stopping criterion (default: {settings['volume_expansion_threshold']}). Relaxation stops if volume expands > threshold × initial volume.")
    parser.add_argument("--max-atoms-per-supercell", type=int, default=settings["max_atoms_per_supercell"], help=f"Maximum number of atoms allowed in generated supercells (default: None = no limit). When set, supercells exceeding this limit will be skipped.")
    parser.add_argument("--auto", action="store_true", help="Automatically run parameter sweep and soft mode optimization.")
    parser.add_argument("--method", type=str, default=settings["default_method"], choices=["ga", "traditional", "traditional_all", "opt_random", "neb", "ci_neb", "md_stability", "phonon_only"], help=f"Method for soft mode optimization (default: {settings['default_method']}).")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional custom prefix to add before the structure name in output folder names.")
    
    # [ADDED THIS LINE]
    parser.add_argument("--resume_dir", type=str, default=None, help="Directory to resume an interrupted calculation from.")

    # --- CORRECTED ARGUMENTS: default=None for nargs='+' arguments ---
    parser.add_argument("--screen_supercell_ns", type=str, nargs='+', default=None, help=f"List of supercell sizes for sweep. E.g., 2 '3,3,3'. (Default from settings)")
    parser.add_argument("--screen_deltas", type=float, nargs='+', default=None, help=f"List of delta values for sweep. (Default from settings: {settings['screen_deltas']})")
    parser.add_argument("--screen_fmax_values", type=float, nargs='+', default=None, help=f"List of fmax values for sweep. (Default from settings: {settings['screen_fmax_values']})")
    parser.add_argument("--soft_mode_displacement_scales", type=float, nargs='+', default=None, help=f"List of displacement scales for soft mode generation. (Default from settings)")
    parser.add_argument("--mode2_ratio_scales", type=float, nargs='+', default=None, help=f"List of ratios for the second soft mode's displacement. (Default from settings)")
    parser.add_argument("--cell_scale_factors", type=float, nargs='+', default=None, help=f"List of cell scale factors for soft mode optimization. (Default from settings)")

    parser.add_argument("--phonon_path_npoints", type=int, default=settings["phonon_path_npoints"], help=f"Number of points along the phonon path (default: {settings['phonon_path_npoints']}).")
    parser.add_argument("--phonon_dos_grid", type=int, nargs=3, default=settings["phonon_dos_grid"], help=f"Grid for DOS calculation (default: {settings['phonon_dos_grid']}).")
    parser.add_argument("--traj_kT", type=float, default=settings["default_traj_kT"], help=f"Temperature for trajectory generation (default: {settings['default_traj_kT']} eV).")
    parser.add_argument("--negative_phonon_threshold_thz", type=float, default=settings["negative_phonon_threshold_thz"], help=f"Threshold to trigger soft mode optimization (default: {settings['negative_phonon_threshold_thz']} THz).")
    parser.add_argument("--soft_mode_max_iterations", type=int, default=settings["soft_mode_max_iterations"], help=f"Maximum iterations for soft mode optimization (default: {settings['soft_mode_max_iterations']}).")
    parser.add_argument("--soft_mode_num_top_structures_to_analyze", type=int, default=settings["soft_mode_num_top_structures_to_analyze"], help=f"Number of top structures to analyze in final step (default: {settings['soft_mode_num_top_structures_to_analyze']}).")
    parser.add_argument("--num_modes_to_return", type=int, default=settings["num_modes_to_return"], help=f"Number of softest modes to return (default: {settings['num_modes_to_return']}).")
    parser.add_argument("--ga_population_size", type=int, default=settings["ga_population_size"], help=f"Population size for GA (default: {settings['ga_population_size']}).")
    parser.add_argument("--ga_mutation_rate", type=float, default=settings["ga_mutation_rate"], help=f"Mutation rate for GA (default: {settings['ga_mutation_rate']}).")
    parser.add_argument("--ga_generations", type=int, default=settings["ga_generations"], help=f"Number of GA generations per main iteration (default: {settings['ga_generations']}).")
    parser.add_argument("--num_new_points_per_iteration", type=int, default=settings["num_new_points_per_iteration"], help=f"New structures per GA iteration (default: {settings['num_new_points_per_iteration']}).")
    parser.add_argument("--q", type=str, default=None, help="Q-point for generating a displaced supercell (e.g., '0.5,0,0').")
    parser.add_argument("--band_idx", type=int, default=None, help="Index of the phonon mode for displacement.")
    parser.add_argument("--displacement", type=float, default=1.0, help="Displacement magnitude in Angstroms (default: 1.0).")
    parser.add_argument("--band_yaml_path", type=str, default=None, help="Path to existing band.yaml file containing eigenmode data. When provided with --q and --band_idx, uses pre-calculated eigenmode instead of computing new phonons.")
    parser.add_argument('--ga_disp_scale_bounds', type=str, default="0.0,10.0",  
                        help='Comma-separated min,max bounds for displacement scales in GA. Default: "0.0,10.0"')  
    parser.add_argument('--ga_ratio_bounds', type=str, default="-1.5,1.5",  
                            help='Comma-separated min,max bounds for mode2_ratio_scales in GA. Default: "-1.5,1.5"')  
    parser.add_argument('--ga_cell_scale_bounds', type=str, default="-0.5,0.5",  
                            help='Comma-separated min,max bounds for cell_scale_factors in GA. Default: "-0.5,0.5"')  
    parser.add_argument('--ga_cell_angle_bounds', type=str, default="-45.0,45.0",
                        help='Comma-separated min,max bounds for cell_angle_factors in GA. Default: "-45.0,45.0"')
    parser.add_argument('--decomposition_threshold', type=float, default=settings["decomposition_threshold"],
                        help=f'Energy threshold (in eV) below reference energy for flagging structures as decomposed. '
                             f'Structures with E_relax < E_ref - decomposition_threshold are flagged as DECOMPOSED. '
                             f'Default: {settings["decomposition_threshold"]} eV')
    parser.add_argument("--constrained-supercell-growth", action="store_true", 
                    help="Enable incremental supercell variants: (1,1,1) -> +(2,1,1) -> +(1,2,1) -> +(1,1,2) iteratively.")
    parser.add_argument("--maximum-indexes-on-same-points", type=int, default=None,
                    help="Limit the number of modes per high-symmetry point in traditional_all method.")

    # NEB-specific arguments
    parser.add_argument('--final_cif', type=str, default=None,
                        help='Path to the final CIF structure for NEB/CI-NEB methods. Required when using NEB or CI-NEB methods.')
    parser.add_argument('--neb_num_images', type=int, default=10,
                        help='Number of intermediate images for NEB/CI-NEB (default: 10). Total path will have num_images+2 structures (including initial and final).')
    parser.add_argument('--neb_spring_constant', type=float, default=5.0,
                        help='Spring constant for NEB method (default: 5.0 eV/Å²).')
    parser.add_argument('--neb_max_iterations', type=int, default=1000,
                        help='Maximum number of NEB optimization iterations (default: 1000).')
    parser.add_argument('--neb_force_tolerance', type=float, default=0.01,
                        help='Force tolerance for NEB convergence (default: 0.01 eV/Å).')
    parser.add_argument('--neb_climbing_start_iteration', type=int, default=50,
                        help='Iteration to start climbing image in CI-NEB (default: 50). Only applies to CI-NEB method.')

    # Optional YAML file saving
    parser.add_argument('--save-yaml', action='store_true', default=False,
                        help='Save YAML files during phonon analysis. By default, YAML files are NOT saved to reduce file size overhead.')

    # Optional phonon calculations for NEB methods
    parser.add_argument('--with-phonon', action='store_true', default=False,
                        help='Include phonon calculations in NEB methods. By default, NEB methods only perform NEB optimization without phonon analysis.')
    parser.add_argument('--compute-pdos', action='store_true', default=False,
                        help='Compute Projected Density of States (PDOS). This is computationally expensive and disabled by default.')
    # MD stability-specific arguments
    parser.add_argument('--temp', type=float, default=settings["md_temperature"],
                        help=f'Simulation temperature in Kelvin for MD stability analysis (default: {settings["md_temperature"]} K).')
    parser.add_argument('--pressure', type=float, default=settings["md_pressure"],
                        help=f'Simulation pressure in GPa for MD stability analysis (default: {settings["md_pressure"]} GPa).')
    parser.add_argument('--time', type=float, default=settings["md_time"],
                        help=f'Total simulation time in picoseconds for MD stability analysis (default: {settings["md_time"]} ps).')
    parser.add_argument('--supercell-size', type=str, default=settings["md_supercell_size"],
                        help=f'Supercell dimensions as "NxNxN" format for MD stability analysis (default: {settings["md_supercell_size"]}).')
    parser.add_argument('--equilibration-fraction', type=float, default=settings["md_equilibration_fraction"],
                        help=f'Fraction of total time for equilibration in MD stability analysis (default: {settings["md_equilibration_fraction"]}).')

    # MD stability assessment thresholds (generous defaults for MLIP fluctuations)
    parser.add_argument('--volume-threshold', type=float, default=6.0,
                        help='Volume fluctuation threshold as percentage of average volume for stability assessment (default: 4.0%%. Generous threshold accounts for MLIP fluctuations).')
    parser.add_argument('--rmsd-threshold', type=float, default=1.0,
                        help='RMSD threshold in Angstroms for structural stability assessment (default: 1.0 Å. Generous threshold accounts for MLIP fluctuations).')
    parser.add_argument('--rdf-threshold', type=float, default=0.5,
                        help='RDF correlation threshold for structural integrity assessment (default: 0.5. Lower threshold accounts for MLIP fluctuations).')

    return parser, settings

def save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_dims, delta, fmax, output_dir):  
   """Saves raw band structure and DOS data to text files."""  
      
   # Create a filename-friendly string for the supercell dimensions  
   if isinstance(supercell_dims, (list, tuple)):  
      supercell_str_filename = f"{supercell_dims[0]}x{supercell_dims[1]}x{supercell_dims[2]}"  
   else:  
      supercell_str_filename = str(supercell_dims)  
   
   # Reshape bs_energies to 2D before saving  
   # Assuming bs_energies shape is (num_k_points, num_bands, num_spin_channels)  
   # We want to flatten the last dimension into the second, resulting in (num_k_points, num_bands * num_spin_channels)  
   bs_energies_2d = bs_energies.reshape(bs_energies.shape[0], -1)  
   np.savetxt(os.path.join(output_dir, f"band_structure_energies_N{supercell_str_filename}_D{delta}_F{fmax}.txt"), bs_energies_2d)  
   
   np.savetxt(os.path.join(output_dir, f"dos_energies_N{supercell_str_filename}_D{delta}_F{fmax}.txt"), dos_energies)  
   np.savetxt(os.path.join(output_dir, f"k_point_distances_N{supercell_str_filename}_D{delta}_F{fmax}.txt"), all_k_point_distances)  
   
   with open(os.path.join(output_dir, f"special_k_points_N{supercell_str_filename}_D{delta}_F{fmax}.txt"), 'w') as f:  
      f.write("Special K-point Distances:\n")  
      for dist in special_k_point_distances:  
         f.write(f"{dist}\n")  
      f.write("\nSpecial K-point Labels:\n")  
      for label in special_k_point_labels:  
         f.write(f"{label}\n")  
   
   print("Raw band structure and DOS data saved.")