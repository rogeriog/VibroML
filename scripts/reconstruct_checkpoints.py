#!/usr/bin/env python3
"""
Retroactive Checkpoint Reconstruction Script for VibroML

Scans existing interrupted run directories and generates valid checkpoint files
that enable seamless resumption from the point of interruption.

Usage:
    python scripts/reconstruct_checkpoints.py <run_directory> [--mode {ga,traditional,traditional_all,opt_random}]

Example:
    python scripts/reconstruct_checkpoints.py examples/LiF_simplecubic/LiFsimplecubic_UMA_GA_phonon_output_20251201-005549 --mode ga

Author: VibroML Development Team
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Add parent directory to path to import vibroml
sys.path.insert(0, str(Path(__file__).parent.parent))

from vibroml.checkpointing import CheckpointManager
from ase.io import read


class CheckpointReconstructor:
    """Reconstruct checkpoints from interrupted runs."""
    
    def __init__(self, run_dir: str, mode: Optional[str] = None):
        """
        Initialize reconstructor.
        
        Args:
            run_dir: Path to interrupted run directory
            mode: Execution mode (auto-detected if None)
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")
        
        self.mode = mode or self._detect_mode()
        print(f"Detected mode: {self.mode}")
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(str(self.run_dir), self.mode)
    
    def _detect_mode(self) -> str:
        """Auto-detect execution mode from directory structure."""
        # Check for mode-specific directories
        if (self.run_dir / "main_iter_1_gen_1").exists():
            return "ga"
        elif any(self.run_dir.glob("iter_*/pairing_*")):
            return "traditional_all"
        elif (self.run_dir / "iter_1").exists():
            return "traditional"
        elif (self.run_dir / "main_iter_1").exists() and not (self.run_dir / "main_iter_1_gen_1").exists():
            return "opt_random"
        
        # Try to detect from initial_settings.json
        settings_file = self.run_dir / "initial_settings.json"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return settings.get('method', 'unknown')
        
        raise ValueError("Could not auto-detect mode. Please specify --mode explicitly.")
    
    def reconstruct(self) -> List[str]:
        """
        Reconstruct all checkpoints from the interrupted run.
        
        Returns:
            List of reconstructed checkpoint file paths
        """
        if self.mode == "ga":
            return self._reconstruct_ga()
        elif self.mode == "traditional":
            return self._reconstruct_traditional()
        elif self.mode == "traditional_all":
            return self._reconstruct_traditional_all()
        elif self.mode == "opt_random":
            return self._reconstruct_opt_random()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _parse_relaxation_summary(self, summary_path: Path) -> List[dict]:
        """
        Parse relaxation_summary_generation.txt or similar files.

        Args:
            summary_path: Path to summary file

        Returns:
            List of result dictionaries
        """
        results = []

        if not summary_path.exists():
            return results

        with open(summary_path, 'r') as f:
            lines = f.readlines()

        # Find the header line to determine column positions
        header_line = None
        data_start = 0

        for i, line in enumerate(lines):
            if 'Num Atoms' in line and 'Energy per Atom' in line:
                header_line = line
                # Skip separator line
                data_start = i + 2
                break

        if not header_line:
            return results

        # Determine column layout based on header
        is_ga_mode = 'GA Gen' in header_line or 'Main Iter' in header_line

        # Parse data lines
        for line in lines[data_start:]:
            if line.strip() and not line.startswith('---') and not line.startswith('==='):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        result = {
                            'num_atoms': parts[0],
                            'international_symbol': parts[1],
                            'crystal_system': parts[2],
                            'energy_per_atom': float(parts[3]) if parts[3] != 'FAIL' else None
                        }

                        # Extract iteration and sample info based on mode
                        if is_ga_mode and len(parts) >= 7:
                            # GA mode: Main Iter, GA Gen, Sample
                            result['iteration'] = parts[4]
                            result['ga_generation'] = parts[5]
                            result['sample'] = parts[6]
                        elif len(parts) >= 6:
                            # Traditional modes: Iteration, Sample
                            result['iteration'] = parts[4]
                            result['sample'] = parts[5]

                        if result['energy_per_atom'] is not None:
                            results.append(result)
                    except (ValueError, IndexError):
                        continue

        return results
    
    def _find_relaxed_structure(self, sample_dir: Path) -> Optional[Path]:
        """
        Find the relaxed structure file in a sample directory.
        
        Args:
            sample_dir: Sample directory path
            
        Returns:
            Path to relaxed structure or None
        """
        # Look for common relaxed structure files
        for pattern in ['*_relaxed.cif', '*_relaxed.xyz', 'relaxed_*']:
            matches = list(sample_dir.glob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _reconstruct_ga(self) -> List[str]:
        """Reconstruct checkpoints for GA mode."""
        print("\nReconstructing GA checkpoints...")

        checkpoint_files = []
        all_iterations_results = []

        # Find all main iterations and generations
        main_iter_dirs = sorted(self.run_dir.glob("main_iter_*"))
        print(f"Found {len(main_iter_dirs)} main iteration directories: {[d.name for d in main_iter_dirs]}")

        for main_iter_dir in main_iter_dirs:
            # Extract main iteration number
            match = re.search(r'main_iter_(\d+)_gen_(\d+)', main_iter_dir.name)
            if not match:
                continue
            main_iteration = int(match.group(1))
            ga_generation = int(match.group(2))

            print(f"  Processing {main_iter_dir.name}: main_iter={main_iteration}, gen={ga_generation}")

            # Parse relaxation summary if exists
            summary_file = main_iter_dir / "relaxation_summary_generation.txt"
            summary_results = self._parse_relaxation_summary(summary_file)
            print(f"    Found {len(summary_results)} results in summary")

            # Find all sample directories
            sample_dirs = sorted(main_iter_dir.glob("sample_*"), key=lambda x: int(x.name.split('_')[1]))
            print(f"    Found {len(sample_dirs)} sample directories")

            if not sample_dirs:
                # Might be opt_random style with direct samples
                continue
            
            for sample_dir in sample_dirs:
                    # Extract sample number
                    match = re.search(r'sample_(\d+)', sample_dir.name)
                    if not match:
                        continue
                    sample_index = int(match.group(1))

                    # Check if this sample completed
                    relaxed_structure = self._find_relaxed_structure(sample_dir)

                    if relaxed_structure:
                        print(f"  ✓ Found relaxed structure for sample {sample_index}: {relaxed_structure.name}")
                        # Try to find energy from summary or calculate
                        energy_per_atom = None

                        # Look for matching result in summary
                        matching_result = None
                        for result in summary_results:
                            try:
                                result_sample = result.get('sample', -1)
                                if int(result_sample) == sample_index:
                                    matching_result = result
                                    energy_per_atom = result['energy_per_atom']
                                    break
                            except (ValueError, TypeError):
                                continue

                        if energy_per_atom is None:
                            print(f"  ⚠ No energy found for sample {sample_index} in summary (found {len(summary_results)} results)")
                            continue
                        
                        if energy_per_atom is not None:
                            # Load the structure
                            try:
                                atoms = read(str(relaxed_structure))
                                
                                result_dict = {
                                    'energy_per_atom': energy_per_atom,
                                    'relaxed_atoms': atoms,
                                    'main_iteration': main_iteration,
                                    'ga_generation': ga_generation,
                                    'sample': sample_index,
                                    'num_atoms': matching_result.get('num_atoms', len(atoms)),
                                    'international_symbol': matching_result.get('international_symbol', 'N/A'),
                                    'crystal_system': matching_result.get('crystal_system', 'N/A'),
                                    'params': (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1, 1, 1), True)  # Placeholder
                                }
                                
                                all_iterations_results.append(result_dict)
                                
                                # Create checkpoint after each sample
                                checkpoint_file = self.checkpoint_manager.save_checkpoint_ga(
                                    main_iteration=main_iteration,
                                    ga_generation=ga_generation,
                                    sample_index=sample_index,
                                    all_iterations_results=all_iterations_results,
                                    ga_state={'population': [], 'generation': ga_generation},
                                    current_primitive_atoms=atoms,
                                    current_softest_modes=[],
                                    tracked_k_points_data={'soft_modes': [], 'highest_freq_modes': [], 'lowest_freq_modes': []}
                                )
                                checkpoint_files.append(checkpoint_file)
                                
                            except Exception as e:
                                print(f"  ⚠ Warning: Could not load structure from {relaxed_structure}: {e}")
        
        print(f"\n✓ Reconstructed {len(checkpoint_files)} GA checkpoints")
        print(f"  Total samples: {len(all_iterations_results)}")
        
        return checkpoint_files
    
    def _reconstruct_traditional(self) -> List[str]:
        """Reconstruct checkpoints for traditional mode."""
        print("\nReconstructing traditional checkpoints...")
        
        checkpoint_files = []
        all_iterations_results = []
        
        # Find all iteration directories
        iter_dirs = sorted(self.run_dir.glob("iter_*"), key=lambda x: int(x.name.split('_')[1]))
        
        for iter_dir in iter_dirs:
            # Extract iteration number
            match = re.search(r'iter_(\d+)', iter_dir.name)
            if not match:
                continue
            iteration = int(match.group(1))
            
            # Parse relaxation summary
            summary_file = iter_dir / "relaxation_summary_iter.txt"
            summary_results = self._parse_relaxation_summary(summary_file)
            
            # Find all sample directories
            sample_dirs = sorted(iter_dir.glob("sample_*"), key=lambda x: int(x.name.split('_')[1]))
            
            for sample_dir in sample_dirs:
                match = re.search(r'sample_(\d+)', sample_dir.name)
                if not match:
                    continue
                sample_index = int(match.group(1))
                
                relaxed_structure = self._find_relaxed_structure(sample_dir)
                
                if relaxed_structure:
                    energy_per_atom = None
                    matching_result = None
                    
                    for result in summary_results:
                        try:
                            if int(result.get('sample', -1)) == sample_index:
                                matching_result = result
                                energy_per_atom = result['energy_per_atom']
                                break
                        except (ValueError, TypeError):
                            continue
                    
                    if energy_per_atom is not None:
                        try:
                            atoms = read(str(relaxed_structure))
                            
                            result_dict = {
                                'energy_per_atom': energy_per_atom,
                                'relaxed_atoms': atoms,
                                'iteration': iteration,
                                'sample': sample_index,
                                'num_atoms': matching_result.get('num_atoms', len(atoms)),
                                'international_symbol': matching_result.get('international_symbol', 'N/A'),
                                'crystal_system': matching_result.get('crystal_system', 'N/A'),
                                'params': (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                            }
                            
                            all_iterations_results.append(result_dict)
                            
                            checkpoint_file = self.checkpoint_manager.save_checkpoint_traditional(
                                iteration=iteration,
                                sample_index=sample_index,
                                all_iterations_results=all_iterations_results,
                                current_primitive_atoms=atoms,
                                current_softest_modes=[]
                            )
                            checkpoint_files.append(checkpoint_file)
                            
                        except Exception as e:
                            print(f"  ⚠ Warning: Could not load structure from {relaxed_structure}: {e}")
        
        print(f"\n✓ Reconstructed {len(checkpoint_files)} traditional checkpoints")
        print(f"  Total samples: {len(all_iterations_results)}")
        
        return checkpoint_files
    
    def _reconstruct_traditional_all(self) -> List[str]:
        """Reconstruct checkpoints for traditional_all mode."""
        print("\nReconstructing traditional_all checkpoints...")
        
        checkpoint_files = []
        all_iterations_results = []
        
        # Find all iteration directories
        iter_dirs = sorted(self.run_dir.glob("iter_*"), key=lambda x: int(x.name.split('_')[1]))
        
        for iter_dir in iter_dirs:
            match = re.search(r'iter_(\d+)', iter_dir.name)
            if not match:
                continue
            iteration = int(match.group(1))
            
            # Find all pairing directories
            pairing_dirs = sorted(iter_dir.glob("pairing_*"))
            
            for pairing_idx, pairing_dir in enumerate(pairing_dirs):
                # Determine if this is original (0) or swapped (1) configuration
                config_index = 0 if 'swapped' not in pairing_dir.name else 1
                
                # Find all sample directories
                sample_dirs = sorted(pairing_dir.glob("sample_*"), key=lambda x: int(x.name.split('_')[1]))
                
                for sample_dir in sample_dirs:
                    match = re.search(r'sample_(\d+)', sample_dir.name)
                    if not match:
                        continue
                    sample_index = int(match.group(1))
                    
                    relaxed_structure = self._find_relaxed_structure(sample_dir)
                    
                    if relaxed_structure:
                        try:
                            atoms = read(str(relaxed_structure))
                            
                            # Estimate energy if possible
                            energy_per_atom = -5.0  # Placeholder
                            
                            result_dict = {
                                'energy_per_atom': energy_per_atom,
                                'relaxed_atoms': atoms,
                                'iteration': iteration,
                                'pairing': pairing_dir.name,
                                'config_index': config_index,
                                'sample_id': sample_index,
                                'params': (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                            }
                            
                            all_iterations_results.append(result_dict)
                            
                            checkpoint_file = self.checkpoint_manager.save_checkpoint_traditional_all(
                                iteration=iteration,
                                pairing_index=pairing_idx,
                                config_index=config_index,
                                sample_index=sample_index,
                                all_iterations_results=all_iterations_results,
                                current_primitive_atoms=atoms,
                                mode_pairings_state={}
                            )
                            checkpoint_files.append(checkpoint_file)
                            
                        except Exception as e:
                            print(f"  ⚠ Warning: Could not load structure from {relaxed_structure}: {e}")
        
        print(f"\n✓ Reconstructed {len(checkpoint_files)} traditional_all checkpoints")
        print(f"  Total samples: {len(all_iterations_results)}")
        
        return checkpoint_files
    
    def _reconstruct_opt_random(self) -> List[str]:
        """Reconstruct checkpoints for opt_random mode."""
        print("\nReconstructing opt_random checkpoints...")
        
        checkpoint_files = []
        all_iterations_results = []
        
        # Find all main iteration directories
        iter_dirs = sorted(self.run_dir.glob("main_iter_*"), key=lambda x: int(x.name.split('_')[2]))
        
        for iter_dir in iter_dirs:
            match = re.search(r'main_iter_(\d+)', iter_dir.name)
            if not match:
                continue
            iteration = int(match.group(1))
            
            # Parse iteration summary if exists
            summary_file = iter_dir / f"iteration_{iteration}_summary.txt"
            summary_results = self._parse_relaxation_summary(summary_file)
            
            # Find all sample directories
            sample_dirs = sorted(iter_dir.glob("sample_*"), key=lambda x: int(x.name.split('_')[1]))
            
            for sample_dir in sample_dirs:
                match = re.search(r'sample_(\d+)', sample_dir.name)
                if not match:
                    continue
                sample_index = int(match.group(1))
                
                relaxed_structure = self._find_relaxed_structure(sample_dir)
                
                if relaxed_structure:
                    energy_per_atom = None
                    matching_result = None
                    
                    for result in summary_results:
                        try:
                            if int(result.get('sample', -1)) == sample_index:
                                matching_result = result
                                energy_per_atom = result['energy_per_atom']
                                break
                        except (ValueError, TypeError):
                            continue
                    
                    if energy_per_atom is not None:
                        try:
                            atoms = read(str(relaxed_structure))
                            
                            result_dict = {
                                'energy_per_atom': energy_per_atom,
                                'relaxed_atoms': atoms,
                                'iteration': iteration,
                                'sample': sample_index,
                                'selected_supercell': (2, 2, 2),  # Placeholder
                                'params': {
                                    'displacement_bounds': [0.1, 2.0],
                                    'cell_transformation_vector': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                    'cell_perturbation': True
                                }
                            }
                            
                            all_iterations_results.append(result_dict)
                            
                            checkpoint_file = self.checkpoint_manager.save_checkpoint_opt_random(
                                iteration=iteration,
                                sample_index=sample_index,
                                all_iterations_results=all_iterations_results,
                                current_primitive_atoms=atoms,
                                current_best_supercell=(2, 2, 2)
                            )
                            checkpoint_files.append(checkpoint_file)
                            
                        except Exception as e:
                            print(f"  ⚠ Warning: Could not load structure from {relaxed_structure}: {e}")
        
        print(f"\n✓ Reconstructed {len(checkpoint_files)} opt_random checkpoints")
        print(f"  Total samples: {len(all_iterations_results)}")
        
        return checkpoint_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reconstruct checkpoints from interrupted VibroML runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect mode and reconstruct
  python scripts/reconstruct_checkpoints.py path/to/run_dir

  # Specify mode explicitly
  python scripts/reconstruct_checkpoints.py path/to/run_dir --mode ga
  
  # With cleanup of old checkpoints
  python scripts/reconstruct_checkpoints.py path/to/run_dir --cleanup --keep 5
        """
    )
    
    parser.add_argument('run_dir', help='Path to interrupted run directory')
    parser.add_argument('--mode', choices=['ga', 'traditional', 'traditional_all', 'opt_random'],
                       help='Execution mode (auto-detected if not specified)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old checkpoints after reconstruction')
    parser.add_argument('--keep', type=int, default=10,
                       help='Number of checkpoints to keep when cleaning up (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VibroML Checkpoint Reconstruction")
    print("=" * 80)
    print(f"\nRun directory: {args.run_dir}")
    
    try:
        reconstructor = CheckpointReconstructor(args.run_dir, args.mode)
        checkpoint_files = reconstructor.reconstruct()
        
        if checkpoint_files:
            print(f"\n{'=' * 80}")
            print("✓ Reconstruction complete!")
            print(f"{'=' * 80}")
            print(f"\nCreated {len(checkpoint_files)} checkpoint(s)")
            print(f"Latest checkpoint: {Path(checkpoint_files[-1]).name}")
            print(f"\nYou can now resume the run by restarting it normally.")
            print("The code will automatically detect and load the checkpoint.")
            
            if args.cleanup:
                print(f"\nCleaning up old checkpoints (keeping last {args.keep})...")
                reconstructor.checkpoint_manager.cleanup_old_checkpoints(args.keep)
        else:
            print("\n⚠ No checkpoints were reconstructed.")
            print("Possible reasons:")
            print("  - Run directory has no completed samples")
            print("  - Incorrect mode specified")
            print("  - Missing or corrupted data files")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()