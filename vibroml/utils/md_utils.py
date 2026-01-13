"""
Molecular Dynamics utilities for VibroML AIMD stability analysis.

This module provides functionality for:
- NPT ensemble molecular dynamics simulations
- Trajectory analysis and stability metrics calculation
- RMSD and RDF analysis for crystal stability assessment
- Integration with VibroML's MLIP calculator framework
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from ase import Atoms
from ase.io import write, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase import units
from ase.build import make_supercell
from ase.md.logger import MDLogger
import json


class StressMDLogger:
    """
    Enhanced MD logger that includes total stress (hydrostatic pressure) logging.

    Custom MD logger that logs standard MD properties plus detailed stress tensor
    information including total stress (hydrostatic pressure) calculations.
    """

    def __init__(self, dyn, atoms, logfile, header=True, stress=True, peratom=True, mode="a"):
        """
        Initialize enhanced MD logger with stress tracking.

        Args:
            dyn: MD dynamics object
            atoms: Atoms object
            logfile: Path to log file
            header: Whether to write header
            stress: Whether to log stress data
            peratom: Whether to log per-atom quantities
            mode: File mode ('w' for write, 'a' for append)
        """
        self.dyn = dyn
        self.atoms = atoms
        self.logfile = logfile
        self.stress_logging = stress
        self.mode = mode

        # Open log file
        self.fd = open(logfile, mode)

        # Write header if requested
        if header:
            self._write_header()

    def _write_header(self):
        """Write header with all logged quantities."""
        header_line = "# Time[ps]  Etot[eV]  Epot[eV]  Ekin[eV]  T[K]  Volume[A^3]"
        if self.stress_logging:
            header_line += "  Stress_xx[GPa]  Stress_yy[GPa]  Stress_zz[GPa]  Stress_yz[GPa]  Stress_xz[GPa]  Stress_xy[GPa]  Total_Stress[GPa]"
        header_line += "\n"

        self.fd.write(header_line)
        self.fd.flush()

    def __call__(self):
        """Log MD properties including enhanced stress data."""
        # Get standard properties
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        volume = self.atoms.get_volume()

        # Calculate time in picoseconds
        time_ps = self.dyn.get_time() / (1000.0)  # Convert from fs to ps

        # Prepare log line
        log_line = f"{time_ps:12.6f}  {epot + ekin:12.6f}  {epot:12.6f}  {ekin:12.6f}  {temp:8.2f}  {volume:12.3f}"

        # Add stress data if available
        if self.stress_logging:
            try:
                stress = self.atoms.get_stress(voigt=True)  # Get stress in Voigt notation [xx, yy, zz, yz, xz, xy]
                # Convert from eV/Å³ to GPa
                stress_gpa = stress / units.GPa

                # Calculate total stress (hydrostatic pressure) as average of diagonal components
                total_stress_gpa = np.mean(stress_gpa[:3])  # Average of xx, yy, zz components

                # Add individual stress components
                for component in stress_gpa:
                    log_line += f"  {component:12.6f}"

                # Add total stress
                log_line += f"  {total_stress_gpa:12.6f}"

            except Exception as e:
                # If stress calculation fails, fill with zeros
                log_line += "  " + "  ".join(["0.000000"] * 7)  # 6 components + 1 total

        log_line += "\n"

        # Write to file
        self.fd.write(log_line)
        self.fd.flush()

    def close(self):
        """Close the log file."""
        if hasattr(self, 'fd') and self.fd is not None:
            self.fd.close()


def parse_supercell_size(supercell_size_str: str) -> Tuple[int, int, int]:
    """
    Parse supercell size string into tuple of integers.
    
    Args:
        supercell_size_str: String in format "NxNxN" (e.g., "2x2x2")
    
    Returns:
        Tuple of three integers (nx, ny, nz)
    
    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = supercell_size_str.lower().split('x')
        if len(parts) != 3:
            raise ValueError("Must have exactly 3 dimensions")
        dims = [int(x) for x in parts]
        if any(dim < 1 for dim in dims):
            raise ValueError("All dimensions must be positive")
        return tuple(dims)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid supercell size format '{supercell_size_str}'. Expected format: 'NxNxN'") from e


def prepare_md_supercell(primitive_atoms: Atoms, supercell_size: str) -> Atoms:
    """
    Prepare supercell for MD simulation.
    
    Args:
        primitive_atoms: Primitive cell structure
        supercell_size: Supercell size string (e.g., "2x2x2")
    
    Returns:
        Supercell atoms object ready for MD
    """
    nx, ny, nz = parse_supercell_size(supercell_size)
    supercell_matrix = np.diag([nx, ny, nz])
    
    print(f"Creating {nx}x{ny}x{nz} supercell for MD simulation")
    supercell_atoms = make_supercell(primitive_atoms, supercell_matrix)
    
    # Validate supercell has sufficient atoms for meaningful MD
    num_atoms = len(supercell_atoms)
    if num_atoms < 50:
        print(f"Warning: Supercell has only {num_atoms} atoms. Consider using a larger supercell for more reliable MD statistics.")
    elif num_atoms < 20:
        raise ValueError(f"Supercell too small ({num_atoms} atoms). Minimum 20 atoms required for meaningful MD simulation.")
    
    print(f"Supercell created with {num_atoms} atoms")
    return supercell_atoms


def setup_nvt_ensemble(atoms: Atoms, temperature: float, calculator,
                      timestep: float = 1.0):
    """
    Set up NVT ensemble for temperature equilibration phase.

    Args:
        atoms: Atoms object for simulation
        temperature: Target temperature in Kelvin
        calculator: ASE calculator (MLIP)
        timestep: MD timestep in fs (default: 1.0 fs)

    Returns:
        NVT dynamics object
    """
    from ase.md.nvtberendsen import NVTBerendsen

    print(f"Setting up NVT ensemble for temperature equilibration:")
    print(f"  Target temperature: {temperature} K")
    print(f"  Timestep: {timestep} fs")
    print(f"  Starting from 0K (zero velocities)")

    # Attach calculator
    atoms.calc = calculator

    # Initialize with zero velocities (0K start)
    atoms.set_velocities(np.zeros_like(atoms.get_positions()))

    # Set up NVT ensemble with Berendsen thermostat
    # Start with low temperature that will be ramped up
    dynamics = NVTBerendsen(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=1.0,  # Start very low, will be ramped
        taut=50.0 * units.fs  # Thermostat time constant
    )

    return dynamics


def setup_npt_ensemble(atoms: Atoms, temperature: float, pressure: float,
                      calculator, timestep: float = 1.0) -> NPT:
    """
    Set up NPT ensemble for pressure/volume equilibration and production phases.

    Args:
        atoms: Atoms object for simulation
        temperature: Target temperature in Kelvin
        pressure: Target pressure in GPa
        calculator: ASE calculator (MLIP)
        timestep: MD timestep in fs (default: 1.0 fs)

    Returns:
        NPT dynamics object
    """
    print(f"Setting up NPT ensemble for pressure/volume equilibration:")
    print(f"  Temperature: {temperature} K")
    print(f"  Pressure: {pressure} GPa")
    print(f"  Timestep: {timestep} fs")

    # Attach calculator (atoms should already have proper velocities from NVT phase)
    atoms.calc = calculator

    # Convert pressure from GPa to ASE units (eV/Å³)
    pressure_ase = pressure * units.GPa

    # Set up NPT dynamics with Langevin thermostat and Parrinello-Rahman barostat
    # Using reasonable coupling parameters
    dynamics = NPT(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        externalstress=pressure_ase,
        ttime=50 * units.fs,  # Thermostat time constant
        pfactor=75 * units.fs**2  # Barostat time constant
    )

    return dynamics


def temperature_ramp_callback(dynamics, target_temp: float, current_step: int, ramp_steps: int):
    """
    Callback function to gradually ramp temperature during NVT equilibration.

    Args:
        dynamics: MD dynamics object
        target_temp: Target temperature in Kelvin
        current_step: Current simulation step
        ramp_steps: Total steps for temperature ramping
    """
    if current_step <= ramp_steps:
        # Linear temperature ramp from 1K to target_temp
        current_temp = 1.0 + (target_temp - 1.0) * (current_step / ramp_steps)
        dynamics.set_temperature(temperature_K=current_temp)

        # Print progress every 10% of ramp
        if current_step % max(1, ramp_steps // 10) == 0:
            progress = 100.0 * current_step / ramp_steps
            print(f"  Temperature ramp progress: {progress:.1f}% (T = {current_temp:.1f} K)")


def monitor_equilibration_convergence(trajectory_file: str, start_step: int,
                                    window_size: int = 100) -> dict:
    """
    Monitor equilibration convergence by analyzing temperature, pressure, volume, and energy.

    Args:
        trajectory_file: Path to trajectory file
        start_step: Step to start monitoring from
        window_size: Number of steps to average over for convergence check

    Returns:
        dict: Convergence metrics and status
    """
    try:
        from ase.io import read

        # Read trajectory
        trajectory = read(trajectory_file, index=f'{start_step}:')

        if len(trajectory) < window_size:
            return {'converged': False, 'reason': 'Insufficient data for convergence check'}

        # Extract properties
        temperatures = []
        volumes = []
        energies = []

        for atoms in trajectory:
            # Temperature from kinetic energy
            if hasattr(atoms, 'get_kinetic_energy'):
                ke = atoms.get_kinetic_energy()
                temp = 2.0 * ke / (3.0 * len(atoms) * units.kB)
                temperatures.append(temp)

            # Volume
            volumes.append(atoms.get_volume())

            # Potential energy per atom
            if hasattr(atoms, 'get_potential_energy'):
                energies.append(atoms.get_potential_energy() / len(atoms))

        # Check convergence using running averages and standard deviations
        convergence_metrics = {}

        if temperatures:
            temp_array = np.array(temperatures[-window_size:])
            temp_mean = np.mean(temp_array)
            temp_std = np.std(temp_array)
            temp_cv = temp_std / temp_mean if temp_mean > 0 else float('inf')

            convergence_metrics['temperature'] = {
                'mean': temp_mean,
                'std': temp_std,
                'cv': temp_cv,
                'converged': temp_cv < 0.05  # 5% coefficient of variation
            }

        if volumes:
            vol_array = np.array(volumes[-window_size:])
            vol_mean = np.mean(vol_array)
            vol_std = np.std(vol_array)
            vol_cv = vol_std / vol_mean if vol_mean > 0 else float('inf')

            convergence_metrics['volume'] = {
                'mean': vol_mean,
                'std': vol_std,
                'cv': vol_cv,
                'converged': vol_cv < 0.02  # 2% coefficient of variation
            }

        if energies:
            energy_array = np.array(energies[-window_size:])
            energy_mean = np.mean(energy_array)
            energy_std = np.std(energy_array)
            energy_cv = abs(energy_std / energy_mean) if energy_mean != 0 else float('inf')

            convergence_metrics['energy'] = {
                'mean': energy_mean,
                'std': energy_std,
                'cv': energy_cv,
                'converged': energy_cv < 0.001  # 0.1% coefficient of variation
            }

        # Overall convergence
        all_converged = all(
            metrics.get('converged', False)
            for metrics in convergence_metrics.values()
        )

        return {
            'converged': all_converged,
            'metrics': convergence_metrics,
            'reason': 'All properties converged' if all_converged else 'Some properties not converged'
        }

    except Exception as e:
        return {'converged': False, 'reason': f'Error in convergence check: {e}'}


def calculate_rmsd(positions1: np.ndarray, positions2: np.ndarray,
                  cell1: np.ndarray, cell2: np.ndarray) -> float:
    """
    Calculate root-mean-square deviation between two configurations.
    Accounts for periodic boundary conditions and cell deformation.
    
    Args:
        positions1: Initial atomic positions (N, 3)
        positions2: Final atomic positions (N, 3)
        cell1: Initial cell matrix (3, 3)
        cell2: Final cell matrix (3, 3)
    
    Returns:
        RMSD in Angstroms
    """
    # Convert to fractional coordinates to handle cell deformation
    frac_pos1 = np.linalg.solve(cell1.T, positions1.T).T
    frac_pos2 = np.linalg.solve(cell2.T, positions2.T).T
    
    # Handle periodic boundary conditions
    diff_frac = frac_pos2 - frac_pos1
    diff_frac = diff_frac - np.round(diff_frac)  # Wrap to [-0.5, 0.5]
    
    # Convert back to Cartesian coordinates using average cell
    avg_cell = (cell1 + cell2) / 2
    diff_cart = np.dot(diff_frac, avg_cell)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum(diff_cart**2, axis=1)))
    return rmsd


def calculate_rdf(atoms: Atoms, r_max: float = 10.0, n_bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate radial distribution function for the given configuration.
    
    Args:
        atoms: Atoms object
        r_max: Maximum distance for RDF calculation (Angstroms)
        n_bins: Number of bins for RDF histogram
    
    Returns:
        Tuple of (distances, g(r)) arrays
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    n_atoms = len(positions)
    
    # Create distance bins
    dr = r_max / n_bins
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = r_bins[:-1] + dr / 2
    
    # Initialize histogram
    hist = np.zeros(n_bins)
    
    # Calculate all pairwise distances with periodic boundary conditions
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Vector between atoms
            vec = positions[j] - positions[i]
            
            # Apply minimum image convention
            frac_vec = np.linalg.solve(cell.T, vec)
            frac_vec = frac_vec - np.round(frac_vec)
            vec_pbc = np.dot(frac_vec, cell)
            
            # Distance
            dist = np.linalg.norm(vec_pbc)
            
            # Add to histogram
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < n_bins:
                    hist[bin_idx] += 2  # Count both i-j and j-i pairs
    
    # Normalize to get g(r)
    volume = atoms.get_volume()
    density = n_atoms / volume
    
    # Normalization factor for each shell
    for i in range(n_bins):
        r = r_centers[i]
        if r > 0:
            shell_volume = 4 * np.pi * r**2 * dr
            expected_pairs = density * shell_volume * n_atoms
            if expected_pairs > 0:
                hist[i] /= expected_pairs
    
    return r_centers, hist


def analyze_trajectory_stability(trajectory_file: str, initial_atoms: Atoms,
                               timestep_fs: float = 1.0, sampling_interval: int = 1,
                               production_steps: int = 0) -> Dict[str, Any]:
    """
    Analyze production-only MD trajectory for stability metrics.

    Simplified version that works with production-only trajectory files,
    eliminating the need for equilibration frame skipping.

    Args:
        trajectory_file: Path to production trajectory file (contains only production frames)
        initial_atoms: Initial structure for comparison
        timestep_fs: MD timestep in femtoseconds (default: 1.0)
        sampling_interval: Trajectory sampling interval (frames saved every N steps)
        production_steps: Number of production steps for validation

    Returns:
        Dictionary with stability analysis results
    """
    print(f"Analyzing production trajectory stability from {trajectory_file}")

    # Read production-only trajectory
    try:
        production_trajectory = read(trajectory_file, index=':')
    except Exception as e:
        raise RuntimeError(f"Failed to read trajectory file {trajectory_file}: {e}")

    total_production_frames = len(production_trajectory)
    print(f"Production trajectory frames: {total_production_frames}")

    # Validate expected frame count
    expected_frames = production_steps // sampling_interval if production_steps > 0 else total_production_frames
    print(f"Expected production frames: {expected_frames} (sampling interval: {sampling_interval})")

    if abs(total_production_frames - expected_frames) > 10:  # Allow some tolerance
        print(f"Warning: Frame count mismatch. Expected ~{expected_frames}, got {total_production_frames}")

    print(f"Production analysis will use all {total_production_frames} frames")

    # Create time arrays for production phase (in femtoseconds)
    production_time_fs = np.arange(len(production_trajectory)) * timestep_fs * sampling_interval
    print(f"Production time range: 0 to {production_time_fs[-1]:.1f} fs ({len(production_trajectory)} frames)")

    # Calculate RMSD evolution, volume evolution, and stress evolution
    initial_positions = initial_atoms.get_positions()
    initial_cell = initial_atoms.get_cell()

    rmsd_values = []
    cell_volumes = []
    stress_tensors = []

    for frame in production_trajectory:
        # RMSD calculation
        rmsd = calculate_rmsd(initial_positions, frame.get_positions(),
                            initial_cell, frame.get_cell())
        rmsd_values.append(rmsd)

        # Cell volume
        cell_volumes.append(frame.get_volume())

        # Stress calculation (if available)
        if hasattr(frame, 'calc') and frame.calc is not None:
            try:
                stress = frame.get_stress(voigt=True)  # Get stress in Voigt notation [xx, yy, zz, yz, xz, xy]
                stress_tensors.append(stress)
            except:
                stress_tensors.append(None)
        else:
            stress_tensors.append(None)

    rmsd_values = np.array(rmsd_values)
    cell_volumes = np.array(cell_volumes)

    # Process stress data
    valid_stress_data = []
    for stress in stress_tensors:
        if stress is not None:
            valid_stress_data.append(stress)

    has_stress_data = len(valid_stress_data) > 0
    if has_stress_data:
        stress_array = np.array(valid_stress_data)  # Shape: (n_frames, 6) for Voigt notation
        # Calculate total stress (trace of stress tensor)
        total_stress = np.mean(stress_array[:, :3], axis=1)  # Average of xx, yy, zz components
        print(f"Stress data available: {len(valid_stress_data)} frames")
    
    # Calculate stability metrics
    initial_volume = initial_atoms.get_volume()

    # RMSD analysis
    rmsd_mean = np.mean(rmsd_values)
    rmsd_std = np.std(rmsd_values)
    rmsd_final_half = rmsd_values[len(rmsd_values)//2:]
    rmsd_trend = np.polyfit(range(len(rmsd_final_half)), rmsd_final_half, 1)[0]  # Linear slope

    # Volume analysis - use first 500 fs of production for average calculation
    # This accounts for MLIP fluctuations by using early production phase as reference
    reference_time_fs = 500.0  # Use first 500 fs of production
    reference_frame_count = int(reference_time_fs / (timestep_fs * sampling_interval))
    volume_reference_frames = min(reference_frame_count, len(cell_volumes))

    # Ensure we have at least some frames for reference
    volume_reference_frames = max(1, volume_reference_frames)

    reference_volumes = cell_volumes[:volume_reference_frames]
    average_volume = np.mean(reference_volumes)
    actual_reference_time = volume_reference_frames * timestep_fs * sampling_interval

    print(f"Volume analysis using first {actual_reference_time:.1f} fs of production ({volume_reference_frames} frames)")
    print(f"  Reference volume average: {average_volume:.2f} Å³")

    # Calculate volume fluctuations relative to the production average (not initial)
    volume_fluctuations = (cell_volumes - average_volume) / average_volume * 100
    volume_max_change = np.max(np.abs(volume_fluctuations))
    volume_mean_change = np.mean(volume_fluctuations)
    volume_std_change = np.std(volume_fluctuations)

    # Also calculate traditional volume change from initial for comparison
    volume_changes_from_initial = (cell_volumes - initial_volume) / initial_volume * 100
    
    # RDF analysis (initial vs averaged final structures)
    initial_r, initial_rdf = calculate_rdf(initial_atoms)

    # Use last 20 structures for more statistically robust RDF analysis
    n_final_structures = min(20, len(production_trajectory))
    final_structures = production_trajectory[-n_final_structures:]

    print(f"Computing RDF correlation using last {n_final_structures} structures for statistical robustness")

    # Calculate RDF for each of the final structures
    final_rdfs = []
    for structure in final_structures:
        _, rdf = calculate_rdf(structure)
        final_rdfs.append(rdf)

    # Average the RDF profiles to get representative final-state RDF
    final_rdfs = np.array(final_rdfs)
    averaged_final_rdf = np.mean(final_rdfs, axis=0)

    # Calculate RDF correlation using the averaged final RDF
    rdf_correlation = np.corrcoef(initial_rdf, averaged_final_rdf)[0, 1]

    # Also calculate standard deviation of final RDFs to assess equilibration
    final_rdf_std = np.std(final_rdfs, axis=0)
    final_rdf_variability = np.mean(final_rdf_std)  # Average variability across all r values
    
    results = {
        'rmsd_mean': rmsd_mean,
        'rmsd_std': rmsd_std,
        'rmsd_trend': rmsd_trend,
        'rmsd_values': rmsd_values,
        'volume_mean_change': volume_mean_change,  # Mean fluctuation from production average
        'volume_std_change': volume_std_change,    # Std dev of fluctuations
        'volume_max_change': volume_max_change,    # Max fluctuation from production average
        'volume_changes': volume_fluctuations,     # All fluctuations from production average
        'volume_changes_from_initial': volume_changes_from_initial,  # Traditional metric
        'average_volume': average_volume,          # Production phase average volume
        'volume_reference_frames': volume_reference_frames,  # Frames used for reference
        'volume_reference_time_fs': actual_reference_time,   # Time used for reference (fs)
        'rdf_correlation': rdf_correlation,
        'initial_rdf': (initial_r, initial_rdf),
        'final_rdf': (initial_r, averaged_final_rdf),  # Use same r-grid as initial
        'final_rdf_variability': final_rdf_variability,  # Measure of RDF equilibration
        'n_final_structures_rdf': n_final_structures,    # Number of structures used for RDF averaging
        'production_frames': len(production_trajectory),
        'production_time_fs': production_time_fs,  # Time array for production phase
        'timestep_fs': timestep_fs,                # MD timestep
        'production_interval': sampling_interval,  # Sampling interval for production
        'has_stress_data': has_stress_data,        # Whether stress data is available
        'stress_components': stress_array if has_stress_data else None,  # Full stress tensor components
        'total_stress': total_stress if has_stress_data else None        # Total stress (average of diagonal)
    }
    
    return results


def analyze_energy_trajectory(trajectory_file: str) -> Dict[str, Any]:
    """
    Analyze energy evolution during production MD simulation.

    Simplified version that works with production-only trajectory files.

    Args:
        trajectory_file: Path to production trajectory file

    Returns:
        Dictionary with energy analysis results
    """
    print("Analyzing production energy trajectory")

    try:
        production_trajectory = read(trajectory_file, index=':')
    except Exception as e:
        raise RuntimeError(f"Failed to read trajectory file {trajectory_file}: {e}")

    # Extract energies (if available)
    energies = []
    for frame in production_trajectory:
        if hasattr(frame, 'calc') and frame.calc is not None:
            try:
                energy = frame.get_potential_energy()
                energies.append(energy)
            except:
                energies.append(None)
        else:
            energies.append(None)

    # Filter out None values and convert to per-atom energies
    valid_energies = []
    valid_indices = []
    n_atoms = len(production_trajectory[0])

    for i, energy in enumerate(energies):
        if energy is not None:
            valid_energies.append(energy / n_atoms)  # Per-atom energy
            valid_indices.append(i)

    if not valid_energies:
        print("Warning: No energy data found in trajectory")
        return {'has_energy_data': False}

    valid_energies = np.array(valid_energies)

    # Since we're working with production-only trajectory, use all available energy data
    production_energies = valid_energies

    if len(production_energies) == 0:
        print("Warning: No production energy data available")
        return {'has_energy_data': False}

    # Energy statistics
    energy_mean = np.mean(production_energies)
    energy_std = np.std(production_energies)
    energy_drift = np.polyfit(range(len(production_energies)), production_energies, 1)[0]

    results = {
        'has_energy_data': True,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'energy_drift': energy_drift,
        'energy_values': production_energies
    }

    return results


def determine_stability(analysis_results: Dict[str, Any],
                       rmsd_threshold: float = 1.0,
                       volume_threshold: float = 6.0,
                       rdf_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Determine crystal stability based on analysis results.

    Uses generous thresholds that account for inherent fluctuations in
    Machine Learning Interatomic Potentials (MLIPs).

    Args:
        analysis_results: Results from analyze_trajectory_stability
        rmsd_threshold: RMSD threshold for stability (Angstroms, default: 1.0)
        volume_threshold: Volume fluctuation threshold for stability (%, default: 4.0)
        rdf_threshold: RDF correlation threshold for stability (default: 0.5)

    Returns:
        Dictionary with stability verdict and reasoning
    """
    print("\n--- Determining Crystal Stability ---")

    # Extract key metrics
    rmsd_mean = analysis_results['rmsd_mean']
    rmsd_trend = analysis_results['rmsd_trend']
    volume_max_change = analysis_results['volume_max_change']
    rdf_correlation = analysis_results['rdf_correlation']

    # Stability criteria
    criteria = {
        'rmsd_stable': rmsd_mean < rmsd_threshold and rmsd_trend < 0.01,  # Low RMSD and no growth
        'volume_stable': volume_max_change < volume_threshold,
        'rdf_stable': rdf_correlation > rdf_threshold
    }

    # Count passed criteria
    passed_criteria = sum(criteria.values())

    # Determine overall stability
    if passed_criteria >= 3:
        verdict = "STABLE"
        confidence = "HIGH"
    elif passed_criteria == 2:
        verdict = "STABLE"
        confidence = "MODERATE"
    elif passed_criteria == 1:
        verdict = "UNSTABLE"
        confidence = "LOW"
    else:
        verdict = "UNSTABLE"
        confidence = "HIGH"

    # Generate reasoning
    reasoning = []
    reasoning.append(f"RMSD: {rmsd_mean:.3f} Å (threshold: {rmsd_threshold} Å) - {'PASS' if criteria['rmsd_stable'] else 'FAIL'}")
    reasoning.append(f"Volume change: {volume_max_change:.1f}% (threshold: {volume_threshold}%) - {'PASS' if criteria['volume_stable'] else 'FAIL'}")
    reasoning.append(f"RDF correlation: {rdf_correlation:.3f} (threshold: {rdf_threshold}) - {'PASS' if criteria['rdf_stable'] else 'FAIL'}")

    print(f"Stability verdict: {verdict} (confidence: {confidence})")
    for reason in reasoning:
        print(f"  {reason}")

    results = {
        'verdict': verdict,
        'confidence': confidence,
        'criteria': criteria,
        'reasoning': reasoning,
        'metrics': {
            'rmsd_mean': rmsd_mean,
            'rmsd_trend': rmsd_trend,
            'volume_max_change': volume_max_change,
            'rdf_correlation': rdf_correlation
        }
    }

    return results


def save_trajectory_analysis_plots(analysis_results: Dict[str, Any],
                                 energy_results: Dict[str, Any],
                                 output_dir: str, prefix: str) -> List[str]:
    """
    Generate and save trajectory analysis plots.

    Args:
        analysis_results: Results from analyze_trajectory_stability
        energy_results: Results from analyze_energy_trajectory
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of generated plot filenames
    """
    print("Generating trajectory analysis plots")

    plot_files = []

    # Set up matplotlib
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 150

    # 1. RMSD vs time plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rmsd_values = analysis_results['rmsd_values']
    production_time_fs = analysis_results['production_time_fs']

    ax.plot(production_time_fs, rmsd_values, 'b-', linewidth=1.5, alpha=0.7)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Stability threshold (1.0 Å)')

    ax.set_xlabel('Production Time (fs)')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title(f'Root Mean Square Deviation Evolution\nMean RMSD: {analysis_results["rmsd_mean"]:.3f} Å')
    ax.legend()
    ax.grid(True, alpha=0.3)

    rmsd_plot_file = os.path.join(output_dir, f"{prefix}_rmsd_analysis.png")
    plt.savefig(rmsd_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(rmsd_plot_file)

    # 2. Volume change vs time plot
    fig, ax = plt.subplots(figsize=(10, 6))
    volume_changes = analysis_results['volume_changes']

    ax.plot(production_time_fs, volume_changes, 'g-', linewidth=1.5, alpha=0.7)
    ax.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='Stability threshold (±6%)')
    ax.axhline(y=-6.0, color='r', linestyle='--', alpha=0.7)

    ax.set_xlabel('Production Time (fs)')
    ax.set_ylabel('Volume Fluctuation (%)')
    ax.set_title(f'Cell Volume Evolution\nMax fluctuation: {analysis_results["volume_max_change"]:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    volume_plot_file = os.path.join(output_dir, f"{prefix}_volume_evolution.png")
    plt.savefig(volume_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(volume_plot_file)

    # 3. RDF comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    initial_r, initial_rdf = analysis_results['initial_rdf']
    final_r, final_rdf = analysis_results['final_rdf']

    ax.plot(initial_r, initial_rdf, 'b-', linewidth=2, label='Initial structure', alpha=0.8)
    ax.plot(final_r, final_rdf, 'r-', linewidth=2, label=f'Final averaged ({analysis_results["n_final_structures_rdf"]} structures)', alpha=0.8)

    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title(f'Radial Distribution Function Comparison\nCorrelation: {analysis_results["rdf_correlation"]:.3f} | RDF Variability: {analysis_results["final_rdf_variability"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)

    rdf_plot_file = os.path.join(output_dir, f"{prefix}_rdf_comparison.png")
    plt.savefig(rdf_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(rdf_plot_file)

    # 4. Energy evolution plot (if available)
    if energy_results.get('has_energy_data', False):
        fig, ax = plt.subplots(figsize=(10, 6))
        energy_values = energy_results['energy_values']
        # Create time array for energy data (may be different sampling than trajectory)
        energy_time_fs = np.arange(len(energy_values)) * analysis_results['timestep_fs'] * analysis_results['production_interval']

        ax.plot(energy_time_fs, energy_values, 'purple', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Production Time (fs)')
        ax.set_ylabel('Energy per Atom (eV/atom)')
        ax.set_title(f'Energy Evolution\nMean: {energy_results["energy_mean"]:.4f} eV/atom, '
                    f'Drift: {energy_results["energy_drift"]:.2e} eV/atom/fs')
        ax.grid(True, alpha=0.3)

        energy_plot_file = os.path.join(output_dir, f"{prefix}_energy_evolution.png")
        plt.savefig(energy_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(energy_plot_file)

    # 5. Stress evolution plot (if available)
    if analysis_results.get('has_stress_data', False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        stress_components = analysis_results['stress_components']
        total_stress = analysis_results['total_stress']
        production_time_fs = analysis_results['production_time_fs']

        # Adjust time array to match stress data length (in case of different sampling)
        stress_time_fs = production_time_fs[:len(stress_components)]

        # Plot individual stress components
        stress_labels = ['σ_xx', 'σ_yy', 'σ_zz', 'σ_yz', 'σ_xz', 'σ_xy']
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        for i in range(6):
            ax1.plot(stress_time_fs, stress_components[:, i],
                    color=colors[i], linewidth=1.5, alpha=0.7, label=stress_labels[i])

        ax1.set_xlabel('Production Time (fs)')
        ax1.set_ylabel('Stress Components (eV/Å³)')
        ax1.set_title('Individual Stress Components Evolution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot total stress (average of diagonal components)
        ax2.plot(stress_time_fs, total_stress, 'black', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Production Time (fs)')
        ax2.set_ylabel('Total Stress (eV/Å³)')
        ax2.set_title(f'Total Stress Evolution\nMean: {np.mean(total_stress):.4f} eV/Å³')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        stress_plot_file = os.path.join(output_dir, f"{prefix}_stress_evolution.png")
        plt.savefig(stress_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(stress_plot_file)

    print(f"Generated {len(plot_files)} analysis plots")
    return plot_files
