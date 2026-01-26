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

class VibroJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and complex numbers."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.complex64, np.complex128, complex)):
            return {'__complex__': True, 'real': float(obj.real), 'imag': float(obj.imag)}
        if isinstance(obj, (np.int64, np.int32, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super().default(obj)

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
    
    CHECKPOINT_VERSION = "1.0.2"
    
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
        
        # Checkpoint files
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
        """Write JSON file atomically to prevent corruption, with gzip compression."""
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
            with os.fdopen(temp_fd, 'wb') as f:
                with gzip.open(f, 'wt', encoding='utf-8') as gz_file:
                    json.dump(data, gz_file, indent=2, cls=VibroJSONEncoder)
            
            # Atomic move
            shutil.move(temp_path, str(path))
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def _compute_hash(self, data: dict) -> str:
        """Compute SHA256 hash of checkpoint data."""
        data_copy = data.copy()
        data_copy.pop('hash', None)

        # Serialize complex objects before hashing
        if 'results' in data_copy:
            data_copy['results'] = self._serialize_results(data_copy['results'])
        if 'current_primitive_atoms' in data_copy:
            data_copy['current_primitive_atoms'] = self._serialize_atoms(data_copy['current_primitive_atoms'])
        
        for field in ['ga_state', 'current_softest_modes', 'tracked_k_points_data', 'current_mode_pairings']:
             if field in data_copy:
                 data_copy[field] = self._serialize_complex_data(data_copy[field])

        json_str = json.dumps(data_copy, sort_keys=True, default=str, cls=VibroJSONEncoder)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _verify_hash(self, data: dict) -> bool:
        """Verify checkpoint hash integrity."""
        stored_hash = data.get('hash')
        if not stored_hash:
            return False
        
        computed_hash = self._compute_hash(data)
        return stored_hash == computed_hash
    
    def _serialize_atoms(self, atoms: Optional[Any]) -> Optional[dict]:
        """Serialize ASE Atoms object to dictionary."""
        if atoms is None: return None
        if isinstance(atoms, dict): return atoms

        return {
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions().tolist(),
            'cell': atoms.get_cell().tolist(),
            'pbc': atoms.get_pbc().tolist()
        }
    
    def _deserialize_atoms(self, atoms_dict: Optional[dict]) -> Optional[Atoms]:
        """Deserialize dictionary to ASE Atoms object."""
        if atoms_dict is None: return None
        
        return Atoms(
            symbols=atoms_dict['symbols'],
            positions=atoms_dict['positions'],
            cell=atoms_dict['cell'],
            pbc=atoms_dict['pbc']
        )

    def _serialize_complex_data(self, obj):
        """Recursively convert complex numbers to JSON-safe dicts."""
        if isinstance(obj, (complex, np.complex64, np.complex128)):
            return {'__complex__': True, 'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, list):
            return [self._serialize_complex_data(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_complex_data(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return self._serialize_complex_data(obj.tolist())
        # Handle numpy scalars
        elif isinstance(obj, (np.int64, np.int32, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return obj

    def _deserialize_complex_data(self, obj):
        """Recursively convert JSON-safe dicts back to complex numbers."""
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
            if 'relaxed_atoms' in result_copy:
                result_copy['relaxed_atoms'] = self._serialize_atoms(result_copy['relaxed_atoms'])
            if 'params' in result_copy:
                params = result_copy['params']
                if isinstance(params, (tuple, list)):
                    converted_params = []
                    for p in params:
                        if isinstance(p, np.ndarray): converted_params.append(p.tolist())
                        elif isinstance(p, (list, tuple)): converted_params.append(list(p))
                        # Handle numpy scalars in params tuple
                        elif isinstance(p, (np.int64, np.int32, np.int8)): converted_params.append(int(p))
                        elif isinstance(p, (np.float64, np.float32)): converted_params.append(float(p))
                        else: converted_params.append(p)
                    result_copy['params'] = converted_params
            serialized.append(result_copy)
        return serialized
    
    def _deserialize_results(self, serialized_results: List[dict]) -> List[dict]:
        """Deserialize results list from checkpoint."""
        deserialized = []
        for result in serialized_results:
            result_copy = result.copy()
            if 'relaxed_atoms' in result_copy:
                result_copy['relaxed_atoms'] = self._deserialize_atoms(result_copy['relaxed_atoms'])
            if 'params' in result_copy and isinstance(result_copy['params'], list):
                result_copy['params'] = tuple(result_copy['params'])
            deserialized.append(result_copy)
        return deserialized
    
    # --- SAVE METHODS ---

    def save_checkpoint_ga(self, main_iteration, ga_generation, sample_index, all_iterations_results, ga_state, current_primitive_atoms=None, current_softest_modes=None, tracked_k_points_data=None, current_offspring_params=None):
        """Save checkpoint for GA mode."""
        self._save_generic_checkpoint('ga', {
            'main_iteration': main_iteration,
            'ga_generation': ga_generation,
            'sample_index': sample_index,
            'total_samples_completed': len(all_iterations_results)
        }, all_iterations_results, current_primitive_atoms, current_softest_modes, 
        extra_data={
            'ga_state': self._serialize_complex_data(ga_state),
            'tracked_k_points_data': self._serialize_complex_data(tracked_k_points_data),
            'current_offspring_params': current_offspring_params
        })

    def save_checkpoint_traditional(self, iteration, sample_index, all_iterations_results, current_primitive_atoms=None, current_softest_modes=None):
        """Save checkpoint for traditional mode."""
        self._save_generic_checkpoint('traditional', {
            'iteration': iteration,
            'sample_index': sample_index,
            'total_samples_completed': len(all_iterations_results)
        }, all_iterations_results, current_primitive_atoms, current_softest_modes)

    def save_checkpoint_traditional_all(self, iteration, pairing_index, config_index, sample_index, all_iterations_results, current_primitive_atoms=None, current_softest_modes=None):
        """Save checkpoint for traditional_all mode."""
        self._save_generic_checkpoint('traditional_all', {
            'iteration': iteration,
            'pairing_index': pairing_index,
            'config_index': config_index,
            'sample_index': sample_index,
            'total_samples_completed': len(all_iterations_results)
        }, all_iterations_results, current_primitive_atoms, current_softest_modes)

    def save_checkpoint_opt_random(self, iteration, sample_index, all_iterations_results, current_primitive_atoms=None, current_best_supercell=None):
        """Save checkpoint for opt_random mode."""
        # Ensure current_best_supercell is JSON serializable if it's a numpy array or tuple of numpy ints
        if current_best_supercell is not None:
            if isinstance(current_best_supercell, np.ndarray):
                current_best_supercell = current_best_supercell.tolist()
            elif isinstance(current_best_supercell, (list, tuple)):
                current_best_supercell = [int(x) if isinstance(x, (np.int64, np.int32)) else x for x in current_best_supercell]

        self._save_generic_checkpoint('opt_random', {
            'iteration': iteration,
            'sample_index': sample_index,
            'total_samples_completed': len(all_iterations_results)
        }, all_iterations_results, current_primitive_atoms, extra_data={
            'current_best_supercell': current_best_supercell
        })

    def _save_generic_checkpoint(self, mode, state_dict, results, atoms, soft_modes=None, extra_data=None):
        """Internal helper to save checkpoint."""
        checkpoint_data = {
            'version': self.CHECKPOINT_VERSION,
            'mode': mode,
            'timestamp': time.time(),
            'state': state_dict,
            'results': self._serialize_results(results),
            'current_primitive_atoms': self._serialize_atoms(atoms),
        }
        
        if soft_modes is not None:
            checkpoint_data['current_softest_modes'] = self._serialize_complex_data(soft_modes)
            
        if extra_data:
            # Pre-process extra data to handle complex/numpy types
            processed_extra_data = {k: self._serialize_complex_data(v) for k, v in extra_data.items()}
            checkpoint_data.update(processed_extra_data)
            
        checkpoint_data['hash'] = self._compute_hash(checkpoint_data)
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_filename = f"checkpoint_{timestamp_str}_{checkpoint_data['hash']}.json.gz"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        self._write_json_atomic(checkpoint_path, checkpoint_data)
        self._update_latest_checkpoint(checkpoint_path)
        self._add_checkpoint_to_metadata(checkpoint_filename, checkpoint_data)
        
        # Friendly print
        state_str = ", ".join([f"{k} {v}" for k, v in state_dict.items() if k != 'total_samples_completed'])
        print(f"✓ Checkpoint saved: {state_str}")

    def _update_latest_checkpoint(self, checkpoint_path: Path):
        """Update the latest checkpoint reference."""
        if self.latest_checkpoint_path.exists() or self.latest_checkpoint_path.is_symlink():
            try: self.latest_checkpoint_path.unlink()
            except OSError: pass
        
        try:
            os.symlink(checkpoint_path.name, self.latest_checkpoint_path)
        except (OSError, NotImplementedError):
            shutil.copy2(checkpoint_path, self.latest_checkpoint_path)
    
    def _add_checkpoint_to_metadata(self, filename: str, checkpoint_data: dict):
        """Add checkpoint entry to metadata."""
        metadata = self._load_metadata()
        metadata['checkpoints'].append({
            'filename': filename,
            'timestamp': checkpoint_data['timestamp'],
            'hash': checkpoint_data['hash'],
            'state': checkpoint_data['state']
        })
        self._write_json_atomic(self.metadata_path, metadata)
    
    def _load_metadata(self) -> dict:
        """Load checkpoint metadata."""
        if self.metadata_path.exists():
            try:
                with gzip.open(self.metadata_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                # Fallback for uncompressed (if any)
                try:
                    with open(self.metadata_path.with_suffix(''), 'r') as f:
                        return json.load(f)
                except Exception: pass
        return {'checkpoints': []}
    
    def load_latest_checkpoint(self) -> Optional[Tuple[dict, dict]]:
        """Load the most recent valid checkpoint."""
        if not self.latest_checkpoint_path.exists():
            return None

        try:
            with gzip.open(self.latest_checkpoint_path, 'rt', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
        except Exception:
            try:
                # Fallback for uncompressed legacy files
                with open(self.latest_checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                print(f"⚠ Error loading checkpoint: {e}")
                return None

        # Verify mode
        if checkpoint_data.get('mode') != self.mode:
            print(f"⚠ Warning: Checkpoint mode ({checkpoint_data.get('mode')}) != current ({self.mode})")
            return None

        try:
            # Deserialize
            checkpoint_data['results'] = self._deserialize_results(checkpoint_data.get('results', []))
            
            if 'current_primitive_atoms' in checkpoint_data:
                checkpoint_data['current_primitive_atoms'] = self._deserialize_atoms(checkpoint_data['current_primitive_atoms'])

            # Restore complex numbers
            for field in ['current_softest_modes', 'tracked_k_points_data', 'ga_state', 'current_mode_pairings']:
                if field in checkpoint_data: 
                    checkpoint_data[field] = self._deserialize_complex_data(checkpoint_data[field])

            # Explicit NumPy conversion for raw_displacements
            if 'current_softest_modes' in checkpoint_data and checkpoint_data['current_softest_modes']:
                for mode in checkpoint_data['current_softest_modes']:
                    if 'raw_displacements' in mode:
                        mode['raw_displacements'] = np.array(mode['raw_displacements'], dtype=complex)
            
            return checkpoint_data, checkpoint_data.get('state', {})

        except Exception as e:
            print(f"⚠ Error deserializing checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return None

    def list_checkpoints(self) -> List[dict]:
        """List all available checkpoints."""
        return self._load_metadata().get('checkpoints', [])
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoints."""
        metadata = self._load_metadata()
        checkpoints = metadata.get('checkpoints', [])
        
        if len(checkpoints) <= keep_last_n: return
            
        checkpoints.sort(key=lambda x: x['timestamp'])
        to_remove = checkpoints[:-keep_last_n]
        to_keep = checkpoints[-keep_last_n:]
        
        for checkpoint in to_remove:
            fname = checkpoint['filename']
            # Try compressed then uncompressed
            paths = [self.checkpoint_dir / fname, self.checkpoint_dir / (fname + '.gz')]
            for p in paths:
                if p.exists():
                    try: p.unlink(); print(f"Removed old checkpoint: {p.name}")
                    except OSError: pass
        
        metadata['checkpoints'] = to_keep
        self._write_json_atomic(self.metadata_path, metadata)
    
    def save_checkpoint_final_analysis(self, structure_type, index, all_results, state_info, current_primitive_atoms=None):
        """
        Saves checkpoint during the final phonon analysis phase.
        
        Args:
            structure_type: 'top' or 'unique'
            index: current structure index being processed
            all_results: all_iterations_results list
            state_info: dictionary containing current iteration/generation for context
            current_primitive_atoms: (Optional) The primitive atoms object
        """
        state_dict = state_info.copy()
        state_dict.update({
            'phase': 'final_analysis',
            'final_struct_type': structure_type,
            'final_struct_index': index,
            'total_samples_completed': len(all_results)
        })
        
        self._save_generic_checkpoint(
            self.mode, 
            state_dict, 
            all_results, 
            current_primitive_atoms
        )

def should_skip_final_analysis(checkpoint_state, current_type, current_index):
    """
    Determine if final analysis for a specific structure should be skipped.
    
    Args:
        checkpoint_state (dict): The loaded checkpoint state
        current_type (str): 'top' or 'unique'
        current_index (int): The index of the structure being processed
        
    Returns:
        bool: True if this step should be skipped
    """
    if not checkpoint_state or checkpoint_state.get('phase') != 'final_analysis':
        return False
    
    saved_type = checkpoint_state.get('final_struct_type')
    saved_index = checkpoint_state.get('final_struct_index', 0)

    # If we are doing 'top' but saved state is already at 'unique', skip 'top'
    if current_type == 'top' and saved_type == 'unique':
        return True
    
    # If types match, check index
    if current_type == saved_type:
        if current_index <= saved_index:
            return True
            
    return False
def should_skip_sample(checkpoint_state: Optional[dict], current_state: dict, mode: str) -> bool:
    """Determine if current sample should be skipped based on checkpoint state."""
    if checkpoint_state is None:
        return False
    
    if mode == 'ga':
        if current_state['main_iteration'] < checkpoint_state.get('main_iteration', 0): return True
        if current_state['main_iteration'] == checkpoint_state.get('main_iteration', 0):
            if current_state['ga_generation'] < checkpoint_state.get('ga_generation', 0): return True
            if current_state['ga_generation'] == checkpoint_state.get('ga_generation', 0):
                if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0): return True
    
    elif mode in ['traditional', 'opt_random']:
        if current_state['iteration'] < checkpoint_state.get('iteration', 0): return True
        if current_state['iteration'] == checkpoint_state.get('iteration', 0):
            if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0): return True
    
    elif mode == 'traditional_all':
        if current_state['iteration'] < checkpoint_state.get('iteration', 0): return True
        if current_state['iteration'] == checkpoint_state.get('iteration', 0):
            if current_state['pairing_index'] < checkpoint_state.get('pairing_index', 0): return True
            if current_state['pairing_index'] == checkpoint_state.get('pairing_index', 0):
                if current_state['config_index'] < checkpoint_state.get('config_index', 0): return True
                if current_state['config_index'] == checkpoint_state.get('config_index', 0):
                    if current_state['sample_index'] <= checkpoint_state.get('sample_index', 0): return True
    
    return False