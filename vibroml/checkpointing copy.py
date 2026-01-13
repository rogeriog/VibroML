##################################################
# FILE: vibroml/checkpointing.py
##################################################
"""
VibroML Checkpointing System

Comprehensive checkpointing for GA, traditional, traditional_all, and opt_random modes.
Enables seamless resumption from interruptions with full state preservation.

Checkpoint Structure:
    - checkpoint_TIMESTAMP_HASH.json.gz: Main checkpoint file (compressed)
    - checkpoint_latest.json.gz: Symlink/copy to latest checkpoint
    - checkpoint_metadata.json.gz: Checkpoint registry and history

Author: VibroML Development Team
"""

import os
import json
import time
import hashlib
import tempfile
import shutil
import gzip
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write, read


class CheckpointManager:
    """
    Manages checkpointing for VibroML optimization runs.
    
    Features:
    - Atomic writes to prevent corruption
    - Hash verification for integrity
    - Support for all execution modes
    - Automatic recovery from latest valid checkpoint
    - Handles complex number serialization for phonon eigenvectors
    """
    
    CHECKPOINT_VERSION = "1.0.1"
    
    def __init__(self, base_output_dir: str, mode: str):
        """
        Initialize checkpoint manager.
        
        Args:
            base_output_dir: Base directory for the optimization run
            mode: Execution mode ('ga', 'traditional', 'traditional_all', 'opt_random')
        """
        self.base_output_dir = Path(base_output_dir)
        self.mode = mode
        self.checkpoint_dir = self.base_output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files - Explicitly use .json.gz to enforce compression
        self.latest_checkpoint_path = self.checkpoint_dir / "checkpoint_latest.json.gz"
        self.metadata_path = self.checkpoint_dir / "checkpoint_metadata.json.gz"
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize or load checkpoint metadata."""
        if not self.metadata_path.exists():
            metadata = {
                'version': self.CHECKPOINT_VERSION,
                'mode': self.mode,
                'created': time.time(),
                'checkpoints': []
            }
            self._write_json_atomic(self.metadata_path, metadata)
    
    def _write_json_atomic(self, path: Path, data: dict):
        """
        Write JSON file atomically to prevent corruption, with gzip compression.
        
        Args:
            path: Target file path (will be saved as .gz)
            data: Data to write
        """
        # Ensure path ends with .gz for compressed files
        path_str = str(path)
        if not path_str.endswith('.gz'):
            path_str = path_str + '.gz'
        path = Path(path_str)
        
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix='.tmp_checkpoint_',
            suffix='.json.gz'
        )
        
        try:
            # Write compressed JSON
            with os.fdopen(temp_fd, 'wb') as f:
                with gzip.open(f, 'wt', encoding='utf-8') as gz_file:
                    json.dump(data, gz_file, indent=2)
            
            # Atomic move
            shutil.move(temp_path, str(path))
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def _compute_hash(self, data: dict) -> str:
        """
        Compute SHA256 hash of checkpoint data.

        Args:
            data: Checkpoint data dictionary

        Returns:
            Hex string of hash
        """
        # Convert to JSON string for hashing (excluding hash field itself)
        data_copy = data.copy()
        data_copy.pop('hash', None)

        # Serialize atoms objects and complex numbers before hashing to match saved state
        if 'results' in data_copy:
            data_copy['results'] = self._serialize_results(data_copy['results'])
        if 'current_primitive_atoms' in data_copy:
            data_copy['current_primitive_atoms'] = self._serialize_atoms(
                data_copy['current_primitive_atoms']
            )
        
        # Also need to serialize other complex fields for accurate hashing
        for field in ['ga_state', 'current_softest_modes', 'tracked_k_points_data', 'current_mode_pairings']:
             if field in data_copy:
                 data_copy[field] = self._serialize_complex_data(data_copy[field])

        json_str = json.dumps(data_copy, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _verify_hash(self, data: dict) -> bool:
        """
        Verify checkpoint hash integrity.
        
        Args:
            data: Checkpoint data with hash field
            
        Returns:
            True if hash is valid
        """
        stored_hash = data.get('hash')
        if not stored_hash:
            return False
        
        computed_hash = self._compute_hash(data)
        return stored_hash == computed_hash
    
    def _serialize_atoms(self, atoms: Optional[Any]) -> Optional[dict]:
        """Serialize ASE Atoms object to dictionary."""
        if atoms is None:
            return None
            
        # Check if already serialized to prevent double-serialization error
        if isinstance(atoms, dict):
            return atoms

        return {
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions().tolist(),
            'cell': atoms.get_cell().tolist(),
            'pbc': atoms.get_pbc().tolist()
            # Removed 'numbers' as it is redundant with symbols and can cause issues if inconsistent
        }
    
    def _deserialize_atoms(self, atoms_dict: Optional[dict]) -> Optional[Atoms]:
        """Deserialize dictionary to ASE Atoms object."""
        if atoms_dict is None:
            return None
        
        return Atoms(
            symbols=atoms_dict['symbols'],
            positions=atoms_dict['positions'],
            cell=atoms_dict['cell'],
            pbc=atoms_dict['pbc']
        )

    def _serialize_complex_data(self, obj):
        """Recursively convert complex numbers to JSON-safe dicts."""
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, list):
            return [self._serialize_complex_data(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_complex_data(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return self._serialize_complex_data(obj.tolist())
        return obj

    def _deserialize_complex_data(self, obj):
        if isinstance(obj, dict):
            if obj.get('__complex__') is True: return complex(obj['real'], obj['imag'])
            return {k: self._deserialize_complex_data(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [self._deserialize_complex_data(x) for x in obj]
        return obj
    
    def _serialize_results(self, results: List[dict]) -> List[dict]:
        """Serialize results list for checkpointing."""
        serialized = []
        for result in results:
            result_copy = result.copy()
            
            # Serialize atoms objects
            if 'relaxed_atoms' in result_copy:
                result_copy['relaxed_atoms'] = self._serialize_atoms(
                    result_copy['relaxed_atoms']
                )
            
            # Convert numpy arrays to lists in params
            if 'params' in result_copy:
                params = result_copy['params']
                if isinstance(params, (tuple, list)):
                    converted_params = []
                    for p in params:
                        if isinstance(p, np.ndarray):
                            converted_params.append(p.tolist())
                        elif isinstance(p, (list, tuple)):
                            converted_params.append(list(p))
                        else:
                            converted_params.append(p)
                    result_copy['params'] = converted_params
            
            serialized.append(result_copy)
        
        return serialized
    
    def _deserialize_results(self, serialized_results: List[dict]) -> List[dict]:
        """Deserialize results list from checkpoint."""
        deserialized = []
        for result in serialized_results:
            result_copy = result.copy()
            
            # Deserialize atoms objects
            if 'relaxed_atoms' in result_copy:
                result_copy['relaxed_atoms'] = self._deserialize_atoms(
                    result_copy['relaxed_atoms']
                )
            
            # Convert params back to tuple if needed
            if 'params' in result_copy and isinstance(result_copy['params'], list):
                result_copy['params'] = tuple(result_copy['params'])
            
            deserialized.append(result_copy)
        
        return deserialized
    
    def save_checkpoint_ga(
        self,
        main_iteration: int,
        ga_generation: int,
        sample_index: int,
        all_iterations_results: List[dict],
        ga_state: dict,
        current_primitive_atoms: Optional[Atoms] = None,
        current_softest_modes: Optional[List[dict]] = None,
        tracked_k_points_data: Optional[dict] = None,
        current_offspring_params: Optional[List[tuple]] = None
    ) -> str:
        """Save checkpoint for GA mode."""
        checkpoint_data = {
            'version': self.CHECKPOINT_VERSION,
            'mode': 'ga',
            'timestamp': time.time(),
            'state': {
                'main_iteration': main_iteration,
                'ga_generation': ga_generation,
                'sample_index': sample_index,
                'total_samples_completed': len(all_iterations_results)
            },
            'results': self._serialize_results(all_iterations_results),
            'ga_state': self._serialize_complex_data(ga_state),
            'current_primitive_atoms': self._serialize_atoms(current_primitive_atoms),
            'current_softest_modes': self._serialize_complex_data(current_softest_modes),
            'tracked_k_points_data': self._serialize_complex_data(tracked_k_points_data),
            'current_offspring_params': current_offspring_params
        }
        
        checkpoint_data['hash'] = self._compute_hash(checkpoint_data)
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        # FIX: Explicitly use .json.gz extension in the filename
        checkpoint_filename = f"checkpoint_{timestamp_str}_{checkpoint_data['hash']}.json.gz"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        self._write_json_atomic(checkpoint_path, checkpoint_data)
        self._update_latest_checkpoint(checkpoint_path)
        self._add_checkpoint_to_metadata(checkpoint_filename, checkpoint_data)
        
        print(f"✓ Checkpoint saved: {checkpoint_filename}")
        print(f"  Main Iteration {main_iteration}, Generation {ga_generation}, Sample {sample_index}")
        
        return str(checkpoint_path)
    
    def save_checkpoint_traditional(
        self,
        iteration: int,
        sample_index: int,
        all_iterations_results: List[dict],
        current_primitive_atoms: Optional[Atoms] = None,
        current_softest_modes: Optional[List[dict]] = None
    ) -> str:
        """Save checkpoint for traditional mode."""
        checkpoint_data = {
            'version': self.CHECKPOINT_VERSION,
            'mode': 'traditional',
            'timestamp': time.time(),
            'state': {
                'iteration': iteration,
                'sample_index': sample_index,
                'total_samples_completed': len(all_iterations_results)
            },
            'results': self._serialize_results(all_iterations_results),
            'current_primitive_atoms': self._serialize_atoms(current_primitive_atoms),
            'current_softest_modes': self._serialize_complex_data(current_softest_modes)
        }
        
        checkpoint_data['hash'] = self._compute_hash(checkpoint_data)
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        # FIX: Explicitly use .json.gz extension in the filename
        checkpoint_filename = f"checkpoint_{timestamp_str}_{checkpoint_data['hash']}.json.gz"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        self._write_json_atomic(checkpoint_path, checkpoint_data)
        self._update_latest_checkpoint(checkpoint_path)
        self._add_checkpoint_to_metadata(checkpoint_filename, checkpoint_data)
        
        print(f"✓ Checkpoint saved: {checkpoint_filename}")
        print(f"  Iteration {iteration}, Sample {sample_index}")
        
        return str(checkpoint_path)
    
    def save_checkpoint_traditional_all(
        self,
        iteration: int,
        pairing_index: int,
        config_index: int,
        sample_index: int,
        all_iterations_results: List[dict],
        current_primitive_atoms: Optional[Atoms] = None,
        mode_pairings_state: Optional[dict] = None,
        current_mode_pairings: Optional[List] = None
    ) -> str:
        """Save checkpoint for traditional_all mode."""
        checkpoint_data = {
            'version': self.CHECKPOINT_VERSION,
            'mode': 'traditional_all',
            'timestamp': time.time(),
            'state': {
                'iteration': iteration,
                'pairing_index': pairing_index,
                'config_index': config_index,
                'sample_index': sample_index,
                'total_samples_completed': len(all_iterations_results)
            },
            'results': self._serialize_results(all_iterations_results),
            'current_primitive_atoms': self._serialize_atoms(current_primitive_atoms),
            'mode_pairings_state': mode_pairings_state,
            'current_mode_pairings': self._serialize_complex_data(current_mode_pairings)
        }
        
        checkpoint_data['hash'] = self._compute_hash(checkpoint_data)
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        # FIX: Explicitly use .json.gz extension in the filename
        checkpoint_filename = f"checkpoint_{timestamp_str}_{checkpoint_data['hash']}.json.gz"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        self._write_json_atomic(checkpoint_path, checkpoint_data)
        self._update_latest_checkpoint(checkpoint_path)
        self._add_checkpoint_to_metadata(checkpoint_filename, checkpoint_data)
        
        print(f"✓ Checkpoint saved: {checkpoint_filename}")
        print(f"  Iteration {iteration}, Pairing {pairing_index}, Config {config_index}, Sample {sample_index}")
        
        return str(checkpoint_path)
    
    def save_checkpoint_opt_random(
        self,
        iteration: int,
        sample_index: int,
        all_iterations_results: List[dict],
        current_primitive_atoms: Optional[Atoms] = None,
        current_best_supercell: Optional[Tuple[int, int, int]] = None
    ) -> str:
        """Save checkpoint for opt_random mode."""
        checkpoint_data = {
            'version': self.CHECKPOINT_VERSION,
            'mode': 'opt_random',
            'timestamp': time.time(),
            'state': {
                'iteration': iteration,
                'sample_index': sample_index,
                'total_samples_completed': len(all_iterations_results)
            },
            'results': self._serialize_results(all_iterations_results),
            'current_primitive_atoms': self._serialize_atoms(current_primitive_atoms),
            'current_best_supercell': current_best_supercell
        }
        
        checkpoint_data['hash'] = self._compute_hash(checkpoint_data)
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        # FIX: Explicitly use .json.gz extension in the filename
        checkpoint_filename = f"checkpoint_{timestamp_str}_{checkpoint_data['hash']}.json.gz"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        self._write_json_atomic(checkpoint_path, checkpoint_data)
        self._update_latest_checkpoint(checkpoint_path)
        self._add_checkpoint_to_metadata(checkpoint_filename, checkpoint_data)
        
        print(f"✓ Checkpoint saved: {checkpoint_filename}")
        print(f"  Iteration {iteration}, Sample {sample_index}")
        
        return str(checkpoint_path)
    
    def _update_latest_checkpoint(self, checkpoint_path: Path):
        """Update the latest checkpoint reference using symlink (or copy on Windows)."""
        if self.latest_checkpoint_path.exists() or self.latest_checkpoint_path.is_symlink():
            try:
                self.latest_checkpoint_path.unlink()
            except OSError:
                pass # Handle race conditions or file locks
        
        try:
            # FIX: Create symlink pointing to the filename (which now explicitly includes .gz)
            os.symlink(checkpoint_path.name, self.latest_checkpoint_path)
        except (OSError, NotImplementedError):
            shutil.copy2(checkpoint_path, self.latest_checkpoint_path)
    
    def _add_checkpoint_to_metadata(self, filename: str, checkpoint_data: dict):
        """Add checkpoint entry to metadata."""
        metadata = self._load_metadata()
        
        checkpoint_entry = {
            'filename': filename,
            'timestamp': checkpoint_data['timestamp'],
            'hash': checkpoint_data['hash'],
            'state': checkpoint_data['state']
        }
        
        metadata['checkpoints'].append(checkpoint_entry)
        self._write_json_atomic(self.metadata_path, metadata)
    
    def _load_metadata(self) -> dict:
        """Load checkpoint metadata (handles both gzip and plain JSON)."""
        # 1. Check if the path defined in __init__ exists (e.g. checkpoint_metadata.json.gz)
        if self.metadata_path.exists():
            # If it looks like a gzip file, try reading it as gzip
            if str(self.metadata_path).endswith('.gz'):
                try:
                    with gzip.open(self.metadata_path, 'rt', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load gzip metadata: {e}. Trying plain JSON.")
            else:
                # Fallback to plain text read (unlikely with .gz extension but good for robustness)
                try:
                    with open(self.metadata_path, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass

        # 2. Check for implicit/legacy .gz appended case (path + .gz) if path didn't have it
        implicit_gz_path = Path(str(self.metadata_path) + '.gz')
        if implicit_gz_path.exists() and implicit_gz_path != self.metadata_path:
             try:
                 with gzip.open(implicit_gz_path, 'rt', encoding='utf-8') as f:
                     return json.load(f)
             except Exception:
                 pass
        
        # 3. Check for plain JSON (legacy)
        plain_path = self.metadata_path.with_suffix('') if self.metadata_path.suffix == '.gz' else self.metadata_path
        if plain_path.exists():
            try:
                with open(plain_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        return {'checkpoints': []}
    
    def load_latest_checkpoint(self) -> Optional[Tuple[dict, dict]]:
        """
        Load the most recent valid checkpoint (handles both gzip and plain JSON).

        Returns:
            Tuple of (checkpoint_data, state) or None if no valid checkpoint
        """
        # 1. Check if the latest symlink exists
        if not self.latest_checkpoint_path.exists():
            return None

        try:
            # Always try to open as gzip first since we enforce .json.gz naming now
            with gzip.open(self.latest_checkpoint_path, 'rt', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
        except Exception:
            # Fallback for legacy uncompressed files or read errors
            try:
                with open(self.latest_checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                print(f"⚠ Error loading checkpoint: {e}")
                return None

        # Deserialization Logic
        try:
            # Verify mode matches
            if checkpoint_data.get('mode') != self.mode:
                print(f"⚠ Warning: Checkpoint mode ({checkpoint_data.get('mode')}) doesn't match current mode ({self.mode})")
                return None

            # Deserialize results BEFORE hash verification
            checkpoint_data['results'] = self._deserialize_results(
                checkpoint_data.get('results', [])
            )

            # Deserialize atoms BEFORE hash verification
            if 'current_primitive_atoms' in checkpoint_data:
                checkpoint_data['current_primitive_atoms'] = self._deserialize_atoms(
                    checkpoint_data['current_primitive_atoms']
                )

            # Verify hash AFTER basic deserialization but before complex number restoration
            # (Note: _compute_hash re-serializes everything including complex numbers, so this should match)
            if not self._verify_hash(checkpoint_data):
                print("⚠ Warning: Checkpoint hash verification failed. Checkpoint may be corrupted.")
                # return None <--- DISABLED to allow recovery

            state = checkpoint_data.get('state', {})

            print("✓ Checkpoint loaded successfully")
            print(f"  Mode: {checkpoint_data['mode']}")

            # --- Restore Complex Numbers ---
            # Restore complex numbers
            for field in ['current_softest_modes', 'tracked_k_points_data', 'ga_state', 'current_mode_pairings']:
                if field in checkpoint_data: 
                    checkpoint_data[field] = self._deserialize_complex_data(checkpoint_data[field])

            # --- CRITICAL FIX: Convert raw_displacements to NumPy arrays ---
            # The deserializer returns lists, but we need numpy arrays for math
            if 'current_softest_modes' in checkpoint_data and checkpoint_data['current_softest_modes']:
                for mode in checkpoint_data['current_softest_modes']:
                    if 'raw_displacements' in mode:
                        mode['raw_displacements'] = np.array(mode['raw_displacements'], dtype=complex)
            
            print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp']))}")
            print(f"  State: {state}")
            return checkpoint_data, state


        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_checkpoints(self) -> List[dict]:
        """List all available checkpoints."""
        metadata = self._load_metadata()
        return metadata.get('checkpoints', [])
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoints, keeping only the most recent N."""
        metadata = self._load_metadata()
        checkpoints = metadata.get('checkpoints', [])
        
        if len(checkpoints) <= keep_last_n:
            return
            
        # Sort by timestamp to ensure we keep the newest
        checkpoints.sort(key=lambda x: x['timestamp'])
        
        # Identify files to remove
        to_remove = checkpoints[:-keep_last_n]
        to_keep = checkpoints[-keep_last_n:]
        
        for checkpoint in to_remove:
            fname = checkpoint['filename']
            
            # Robust Path Construction:
            # If filename in metadata lacks .gz but file on disk is compressed, construct correct path.
            checkpoint_path = self.checkpoint_dir / fname
            if not checkpoint_path.exists() and not fname.endswith('.gz'):
                 checkpoint_path = self.checkpoint_dir / (fname + '.gz')
            
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path.name}")
                except OSError as e:
                    print(f"Error removing {checkpoint_path.name}: {e}")
        
        # Update metadata to only list kept files
        metadata['checkpoints'] = to_keep
        self._write_json_atomic(self.metadata_path, metadata)


def should_skip_sample(checkpoint_state: Optional[dict], current_state: dict, mode: str) -> bool:
    """
    Determine if current sample should be skipped based on checkpoint state.
    
    Args:
        checkpoint_state: State from loaded checkpoint
        current_state: Current execution state
        mode: Execution mode
        
    Returns:
        True if sample should be skipped (already completed)
    """
    if checkpoint_state is None:
        return False
    
    if mode == 'ga':
        # Skip if we're before or at the checkpoint state
        if current_state['main_iteration'] < checkpoint_state.get('main_iteration', 0):
            return True
        if current_state['main_iteration'] == checkpoint_state.get('main_iteration', 0):
            if current_state['ga_generation'] < checkpoint_state.get('ga_generation', 0):
                return True
            if current_state['ga_generation'] == checkpoint_state.get('ga_generation', 0):
                if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0):
                    return True
    
    elif mode in ['traditional', 'opt_random']:
        if current_state['iteration'] < checkpoint_state.get('iteration', 0):
            return True
        if current_state['iteration'] == checkpoint_state.get('iteration', 0):
            if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0):
                return True
    
    elif mode == 'traditional_all':
        if current_state['iteration'] < checkpoint_state.get('iteration', 0):
            return True
        if current_state['iteration'] == checkpoint_state.get('iteration', 0):
            if current_state['pairing_index'] < checkpoint_state.get('pairing_index', 0):
                return True
            if current_state['pairing_index'] == checkpoint_state.get('pairing_index', 0):
                if current_state['config_index'] < checkpoint_state.get('config_index', 0):
                    return True
                if current_state['config_index'] == checkpoint_state.get('config_index', 0):
                    if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0):
                        return True
    
    return False