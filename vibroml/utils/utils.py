import math
import os
import shutil
import logging
import sys
import torch
import json # NEW IMPORT

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
   