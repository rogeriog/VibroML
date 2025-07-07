import math
import os
import shutil
import logging
import sys
import torch
import json # NEW IMPORT
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
    # Construct the path relative to the package's root  
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

   Returns:
      tuple: A tuple containing:
         - argparse.ArgumentParser: The configured argument parser.
         - dict: A dictionary of the default settings used.
   """
   default_settings = load_default_settings()

   # Define hardcoded fallbacks if JSON loading fails or values are missing
   # These will be used if the key is not found in default_settings
   settings = {
      "default_engine": default_settings.get("default_engine", "mace"),
      "default_model_name": default_settings.get("default_model_name", "medium-omat-0"),
      "default_fmax": default_settings.get("default_fmax", 0.001),
      "default_delta": default_settings.get("default_delta", 0.03),
      "default_supercell_n": default_settings.get("default_supercell_n", 3),
      "screen_supercell_ns": default_settings.get("screen_supercell_ns", [2, 3, 4]),
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
      "num_new_points_per_iteration": default_settings.get("num_new_points_per_per_iteration", 30),
      "default_method": default_settings.get("default_method", "ga")
   }

   parser = argparse.ArgumentParser(description="Calculate phonon band structure and DOS for crystal structures, with optional relaxation and soft mode analysis.")
   parser.add_argument("--cif", type=str, required=True, help="Path to the CIF file.")
   parser.add_argument("--no-relax", action="store_true", help="Skip relaxation of the structure before calculation.")
   parser.add_argument("--engine", type=str, default=settings["default_engine"], help=f"Calculation engine (default: {settings['default_engine']}).")
   parser.add_argument("--units", type=str, default=settings["default_units"], choices=["THz", "cm-1", "eV"],
                     help=f"Units for frequency (default: {settings['default_units']}). Choose from THz, cm-1, eV.")
   parser.add_argument("--model_name", type=str, default=settings["default_model_name"],
                     help=f"Model name for the calculator (default: {settings['default_model_name']}).")
   parser.add_argument("--supercell_n", type=int, default=settings["default_supercell_n"],
                     help=f"Size of the supercell (N, N, N) for phonon calculation (default: {settings['default_supercell_n']}).")
   parser.add_argument("--delta", type=float, default=settings["default_delta"],
                     help=f"Displacement distance for finite difference phonon calculation (default: {settings['default_delta']}).")
   parser.add_argument("--fmax", type=float, default=settings["default_fmax"],
                     help=f"Maximum force tolerance for structure relaxation (default: {settings['default_fmax']} eV/Å).")
   parser.add_argument("--auto", action="store_true", help="Automatically test multiple settings (parameter sweep) to minimize negative imaginary phonons. If a soft mode persists, it will trigger the iterative soft mode workflow.")
   parser.add_argument("--run-soft-mode-after-single", action="store_true", help="If a soft mode is detected in a single phonon calculation, automatically run the iterative soft mode displacement and relaxation workflow.")
   parser.add_argument("--displace-primitive", action="store_true", help="Also perform soft mode displacements and relaxation directly on the primitive cell, in a separate directory.")
   parser.add_argument("--method", type=str, default=settings["default_method"], choices=["ga", "traditional"],
                     help=f"Method for soft mode optimization (default: {settings['default_method']}). Choose from 'ga' (Genetic Algorithm) or 'traditional' (Grid Search).")
   parser.add_argument("--screen_supercell_ns", type=int, nargs='+', default=settings["screen_supercell_ns"],
                     help=f"List of supercell N values for parameter sweep (default: {settings['screen_supercell_ns']}).")
   parser.add_argument("--screen_deltas", type=float, nargs='+', default=settings["screen_deltas"],
                     help=f"List of delta values for parameter sweep (default: {settings['screen_deltas']}).")
   parser.add_argument("--screen_fmax_values", type=float, nargs='+', default=settings["screen_fmax_values"],
                     help=f"List of fmax values for parameter sweep (default: {settings['screen_fmax_values']}).")
   parser.add_argument("--phonon_path_npoints", type=int, default=settings["phonon_path_npoints"],
                     help=f"Number of points along the phonon path (default: {settings['phonon_path_npoints']}).")
   parser.add_argument("--phonon_dos_grid", type=int, nargs=3, default=settings["phonon_dos_grid"],
                     help=f"Grid for DOS calculation (e.g., 40 40 40) (default: {settings['phonon_dos_grid']}).")
   parser.add_argument("--traj_kT", type=float, default=settings["default_traj_kT"],
                     help=f"Temperature for trajectory generation (default: {settings['default_traj_kT']} eV).")
   parser.add_argument("--negative_phonon_threshold_thz", type=float, default=settings["negative_phonon_threshold_thz"],
                     help=f"Threshold for negative phonon frequency in THz to trigger soft mode optimization (default: {settings['negative_phonon_threshold_thz']} THz).")
   parser.add_argument("--soft_mode_max_iterations", type=int, default=settings["soft_mode_max_iterations"],
                     help=f"Maximum iterations for the soft mode optimization (default: {settings['soft_mode_max_iterations']}).")
   parser.add_argument("--soft_mode_displacement_scales", type=float, nargs='+', default=settings["soft_mode_displacement_scales"],
                     help=f"List of displacement scales for soft mode generation (default: {settings['soft_mode_displacement_scales']}).")
   parser.add_argument("--mode2_ratio_scales", type=float, nargs='+', default=settings["mode2_ratio_scales"],  
                     help=f"List of ratios for the second soft mode's displacement relative to the first (default: {settings['mode2_ratio_scales']}).")
   parser.add_argument("--soft_mode_num_top_structures_to_analyze", type=int, default=settings["soft_mode_num_top_structures_to_analyze"],
                     help=f"Number of top structures to analyze in final soft mode optimization step (default: {settings['soft_mode_num_top_structures_to_analyze']}).")
   parser.add_argument("--cell_scale_factors", type=float, nargs='+', default=settings["cell_scale_factors"],
                     help=f"List of cell scale factors for soft mode optimization (default: {settings['cell_scale_factors']}).")
   parser.add_argument("--num_modes_to_return", type=int, default=settings["num_modes_to_return"],
                     help=f"Number of softest modes to return for analysis (default: {settings['num_modes_to_return']}).")
   parser.add_argument("--ga_population_size", type=int, default=settings["ga_population_size"],
                     help=f"Population size for the Genetic Algorithm (default: {settings['ga_population_size']}).")
   parser.add_argument("--ga_mutation_rate", type=float, default=settings["ga_mutation_rate"],
                     help=f"Mutation rate for the Genetic Algorithm (default: {settings['ga_mutation_rate']}).")
   parser.add_argument("--num_new_points_per_iteration", type=int, default=settings["num_new_points_per_iteration"],
                     help=f"Number of new structures to generate per GA iteration (default: {settings['num_new_points_per_iteration']}).")
   parser.add_argument("--q", type=str,default=None,
                     help="Q-point in fractional coordinates (e.g., '0.5,0,0') for generating a displaced supercell.")
   parser.add_argument("--band_idx", type=int,default=None,
                     help="Index of the phonon mode (0-indexed) for generating a displaced supercell.")
   parser.add_argument("--displacement", type=float,
                     help="Displacement magnitude in Angstroms for generating a displaced supercell.")

   return parser, settings

def save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_n, delta, fmax, output_dir):
   """Saves raw band structure and DOS data to text files."""
   # Reshape bs_energies to 2D before saving
   # Assuming bs_energies shape is (num_k_points, num_bands, num_spin_channels)
   # We want to flatten the last dimension into the second, resulting in (num_k_points, num_bands * num_spin_channels)
   bs_energies_2d = bs_energies.reshape(bs_energies.shape[0], -1)
   np.savetxt(os.path.join(output_dir, f"band_structure_energies_N{supercell_n}_D{delta}_F{fmax}.txt"), bs_energies_2d)

   np.savetxt(os.path.join(output_dir, f"dos_energies_N{supercell_n}_D{delta}_F{fmax}.txt"), dos_energies)
   np.savetxt(os.path.join(output_dir, f"k_point_distances_N{supercell_n}_D{delta}_F{fmax}.txt"), all_k_point_distances)

   with open(os.path.join(output_dir, f"special_k_points_N{supercell_n}_D{delta}_F{fmax}.txt"), 'w') as f:
      f.write("Special K-point Distances:\n")
      for dist in special_k_point_distances:
         f.write(f"{dist}\n")
      f.write("\nSpecial K-point Labels:\n")
      for label in special_k_point_labels:
         f.write(f"{label}\n")

   print("Raw band structure and DOS data saved.")