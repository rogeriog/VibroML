#!/usr/bin/env python3
"""
Debug script to test relaxation summary parsing.
"""

import sys
from pathlib import Path

# Add parent directory to path to import vibroml
sys.path.insert(0, str(Path(__file__).parent))

from scripts.reconstruct_checkpoints import CheckpointReconstructor

def debug_parsing():
    run_dir = "examples/LiF_simplecubic/LiFsimplecubic_UMA_GA_phonon_output_20251201-005549"
    reconstructor = CheckpointReconstructor(run_dir, "ga")

    summary_file = Path(run_dir) / "main_iter_1_gen_1" / "relaxation_summary_generation.txt"
    results = reconstructor._parse_relaxation_summary(summary_file)

    print(f"Parsed {len(results)} results from {summary_file}")
    if results:
        print("First result:", results[0])
        print("Sample values found:", [r.get('sample') for r in results[:5]])

if __name__ == "__main__":
    debug_parsing()