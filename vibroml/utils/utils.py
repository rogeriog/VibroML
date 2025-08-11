import math
import os
import shutil
import logging
import sys
import torch
import json
import numpy as np
import argparse

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# Try importing MACE
try:
   from mace.calculators import mace_mp
   HAVE_MACE = True
except ImportError:
   HAVE_MACE = False

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
        "num_new_points_per_iteration": default_settings.get("num_new_points_per_iteration", 30),
        "default_method": default_settings.get("default_method", "ga")
    }

    parser = argparse.ArgumentParser(description="Calculate phonon band structure and DOS for crystal structures, with optional relaxation and soft mode analysis.")
    parser.add_argument("--cif", type=str, required=True, help="Path to the CIF file.")
    parser.add_argument("--no-relax", action="store_true", help="Skip relaxation of the structure before calculation.")
    parser.add_argument("--engine", type=str, default=settings["default_engine"], help=f"Calculation engine (default: {settings['default_engine']}).")
    parser.add_argument("--units", type=str, default=settings["default_units"], choices=["THz", "cm-1", "eV"], help=f"Units for frequency (default: {settings['default_units']}).")
    parser.add_argument("--model_name", type=str, default=settings["default_model_name"], help=f"Model name for the calculator (default: {settings['default_model_name']}).")
    parser.add_argument("--supercell_n", type=int, default=settings["default_supercell_n"], help=f"Size of the supercell (N, N, N). Deprecated, use --supercell.")
    parser.add_argument("--supercell", type=str, default=None, help="Supercell dimensions. 'N' for cubic (N,N,N) or 'Nx,Ny,Nz'.")
    parser.add_argument("--delta", type=float, default=settings["default_delta"], help=f"Displacement distance (default: {settings['default_delta']}).")
    parser.add_argument("--fmax", type=float, default=settings["default_fmax"], help=f"Max force tolerance for relaxation (default: {settings['default_fmax']} eV/Å).")
    parser.add_argument("--auto", action="store_true", help="Automatically run parameter sweep and soft mode optimization.")
    parser.add_argument("--method", type=str, default=settings["default_method"], choices=["ga", "traditional"], help=f"Method for soft mode optimization (default: {settings['default_method']}).")

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