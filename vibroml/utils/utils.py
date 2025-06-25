import math
import os
import shutil
import logging
import sys
import torch

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