# vibroml/utils/neb_utils.py

import numpy as np
import os
import time
from ase.atoms import Atoms
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import make_supercell, sort
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import logging

from .plotting_utils import plot_neb_profile

def check_chemical_compatibility(atoms1: Atoms, atoms2: Atoms) -> bool:
    """
    Check if two structures have the same chemical composition (same elements in same ratios).

    Args:
        atoms1: First structure
        atoms2: Second structure

    Returns:
        True if structures have compatible chemical compositions
    """
    # Get element counts for both structures
    symbols1 = Counter(atoms1.get_chemical_symbols())
    symbols2 = Counter(atoms2.get_chemical_symbols())

    # Check if they have the same elements
    if set(symbols1.keys()) != set(symbols2.keys()):
        return False

    # Check if ratios are the same (normalize by GCD)
    from math import gcd
    from functools import reduce

    # Get ratios for structure 1
    counts1 = list(symbols1.values())
    gcd1 = reduce(gcd, counts1)
    ratios1 = [c // gcd1 for c in counts1]

    # Get ratios for structure 2
    counts2 = list(symbols2.values())
    gcd2 = reduce(gcd, counts2)
    ratios2 = [c // gcd2 for c in counts2]

    # Sort ratios to compare (since order might be different)
    elements = sorted(symbols1.keys())
    ratios1_sorted = [symbols1[elem] // gcd1 for elem in elements]
    ratios2_sorted = [symbols2[elem] // gcd2 for elem in elements]

    return ratios1_sorted == ratios2_sorted


def find_supercell_multipliers(atoms1: Atoms, atoms2: Atoms) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find supercell multipliers to make two structures have the same number of atoms.

    Args:
        atoms1: First structure
        atoms2: Second structure

    Returns:
        Tuple of (multipliers1, multipliers2) where each is a 3x3 transformation matrix
    """
    # Get element counts
    symbols1 = Counter(atoms1.get_chemical_symbols())
    symbols2 = Counter(atoms2.get_chemical_symbols())

    # Find the least common multiple for each element
    from math import gcd

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    target_counts = {}
    for element in set(symbols1.keys()) | set(symbols2.keys()):
        count1 = symbols1.get(element, 0)
        count2 = symbols2.get(element, 0)
        if count1 == 0 or count2 == 0:
            raise ValueError(f"Element {element} not present in both structures")
        target_counts[element] = lcm(count1, count2)

    # Calculate multipliers needed for each structure
    mult1 = max(target_counts[elem] // symbols1[elem] for elem in symbols1.keys())
    mult2 = max(target_counts[elem] // symbols2[elem] for elem in symbols2.keys())

    # Create diagonal transformation matrices (isotropic scaling)
    # We'll try to make cubic supercells when possible
    mult1_per_dim = int(round(mult1 ** (1/3)))
    mult2_per_dim = int(round(mult2 ** (1/3)))

    # If cubic doesn't work exactly, use the full multiplier in one dimension
    if mult1_per_dim ** 3 != mult1:
        # Try 2D scaling
        mult1_2d = int(round(mult1 ** 0.5))
        if mult1_2d ** 2 == mult1:
            transform1 = np.diag([mult1_2d, mult1_2d, 1])
        else:
            transform1 = np.diag([mult1, 1, 1])
    else:
        transform1 = np.diag([mult1_per_dim, mult1_per_dim, mult1_per_dim])

    if mult2_per_dim ** 3 != mult2:
        # Try 2D scaling
        mult2_2d = int(round(mult2 ** 0.5))
        if mult2_2d ** 2 == mult2:
            transform2 = np.diag([mult2_2d, mult2_2d, 1])
        else:
            transform2 = np.diag([mult2, 1, 1])
    else:
        transform2 = np.diag([mult2_per_dim, mult2_per_dim, mult2_per_dim])

    return transform1, transform2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from scipy.optimize import linear_sum_assignment
from ase.data import atomic_masses, atomic_numbers

from scipy.optimize import linear_sum_assignment
from ase.data import atomic_masses, atomic_numbers

def pbc_center_of_mass(fractional_positions: np.ndarray, masses: np.ndarray = None) -> np.ndarray:
    """
    Calculates the Center of Mass respecting Periodic Boundary Conditions (Circular Mean).
    Standard np.mean fails if atoms straddle the boundary (e.g., 0.99 and 0.01).
    """
    print("Calculating PBC-aware center of mass...")
    if masses is None:
        masses = np.ones(len(fractional_positions))
    
    total_mass = np.sum(masses)
    
    # Map fractional coordinates [0, 1) to angular space [0, 2pi)
    theta = fractional_positions * 2 * np.pi
    
    # Calculate mass-weighted vector sums in complex plane
    # (equivalent to averaging sin and cos components)
    xi = np.cos(theta) * masses[:, None]
    zeta = np.sin(theta) * masses[:, None]
    
    mean_xi = np.sum(xi, axis=0) / total_mass
    mean_zeta = np.sum(zeta, axis=0) / total_mass
    
    # Calculate mean angle
    mean_theta = np.arctan2(mean_zeta, mean_xi)
    
    # Map back to fractional space [0, 1)
    com_frac = mean_theta / (2 * np.pi)
    
    # Wrap results to [0, 1)
    com_frac = np.mod(com_frac, 1.0)
    
    return com_frac

def reorder_atoms_to_match(reference_atoms: Atoms, target_atoms: Atoms) -> Atoms:
    """
    Reorders target_atoms to minimize displacement relative to reference_atoms.
    Robust against Global Shifts (using Circular Mean) and Lattice Distortion.
    """
    if len(reference_atoms) != len(target_atoms):
        raise ValueError("Structures must have the same number of atoms to reorder.")

    # 1. Identify Heaviest Species for robust alignment (Scaffold atoms like Pb, I)
    symbols = reference_atoms.get_chemical_symbols()
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    max_mass = np.max(masses)
    # Use atoms that are at least 50% of the max mass (catches Pb+I, or just Pb)
    heavy_indices = np.where(masses > 0.5 * max_mass)[0]
    
    if len(heavy_indices) == 0:
        heavy_indices = np.arange(len(masses)) # Fallback to all atoms
    
    # 2. Calculate Global Shift using Periodic Center of Mass
    scaled_ref = reference_atoms.get_scaled_positions()
    scaled_tgt = target_atoms.get_scaled_positions()
    
    com_ref = pbc_center_of_mass(scaled_ref[heavy_indices], masses[heavy_indices])
    com_tgt = pbc_center_of_mass(scaled_tgt[heavy_indices], masses[heavy_indices])
    
    # Calculate shift needed to align Target to Reference
    global_shift = com_ref - com_tgt
    global_shift -= np.round(global_shift) # Wrap to simplest shift [-0.5, 0.5]
    
    print(f"  Detected global origin shift (Circular Mean): {global_shift}")

    # 3. Apply Global Shift to Target (Temporary for mapping)
    scaled_tgt_aligned = scaled_tgt + global_shift
    
    # 4. Setup Matching Metric using Average Cell
    # Using Average Cell reduces bias from large lattice distortions
    ref_cell = reference_atoms.get_cell()
    tgt_cell = target_atoms.get_cell()
    avg_cell = 0.5 * (ref_cell + tgt_cell)

    new_indices = np.zeros(len(target_atoms), dtype=int)
    unique_species = set(symbols)

    for species in unique_species:
        # Get indices for this species
        ref_idxs = [i for i, s in enumerate(symbols) if s == species]
        tgt_idxs = [i for i, s in enumerate(target_atoms.get_chemical_symbols()) if s == species]
        
        if len(ref_idxs) != len(tgt_idxs):
            raise ValueError(f"Species count mismatch for {species}")
            
        p_ref = scaled_ref[ref_idxs]
        p_tgt = scaled_tgt_aligned[tgt_idxs] # Use aligned positions!
        
        # Calculate difference vector in fractional space
        diff_frac = p_tgt[:, None, :] - p_ref[None, :, :]
        diff_frac -= np.round(diff_frac) # MIC wrap
        
        # Convert to Cartesian using Average Cell metric for physical distance
        diff_cart = np.dot(diff_frac, avg_cell)
        
        # Cost matrix: Squared Euclidean distances
        cost_matrix = np.sum(diff_cart**2, axis=2)
        
        # Hungarian Algorithm
        # row_ind corresponds to index in tgt_idxs, col_ind to ref_idxs
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Map: ref_global_index -> tgt_global_index
        # We need to fill new_indices such that new_indices[ref_idx] = tgt_idx
        mapping = dict(zip(col_ind, row_ind))
        
        for r_sub in range(len(ref_idxs)):
            t_sub = mapping[r_sub]
            new_indices[ref_idxs[r_sub]] = tgt_idxs[t_sub]

    # 5. Create final reordered atoms
    reordered_atoms = target_atoms[new_indices]
    
    # 6. Apply the Global Shift Permanently
    # This moves the final structure to be physically "on top" of the initial one
    final_positions_frac = reordered_atoms.get_scaled_positions() + global_shift
    reordered_atoms.set_scaled_positions(final_positions_frac)
    
    return reordered_atoms

def make_structures_compatible(initial_atoms: Atoms, final_atoms: Atoms) -> Tuple[Atoms, Atoms]:
    """
    Make two structures compatible for NEB by creating supercells AND matching atom indices.
    """
    print(f"Checking structure compatibility...")
    
    # 1. Check composition and create Supercells if needed
    if not check_chemical_compatibility(initial_atoms, final_atoms):
        raise ValueError(f"Structures have incompatible chemical compositions.")

    try:
        transform1, transform2 = find_supercell_multipliers(initial_atoms, final_atoms)
        initial_supercell = make_supercell(initial_atoms, transform1)
        final_supercell = make_supercell(final_atoms, transform2)
    except Exception as e:
        raise ValueError(f"Failed to create compatible supercells: {str(e)}")

    print("✓ Successfully created compatible supercells.")

    # 2. Sort the INITIAL structure deterministically (e.g. by tag/Z)
    # This gives us a stable starting point
    sorted_initial = sort(initial_supercell)
    
    # 3. Reorder the FINAL structure to match the sorted Initial structure spatially
    # This effectively "maps" the atoms so they don't cross paths
    print("  Reordering final structure atoms to match initial structure (Hungarian Algorithm)...")
    matched_final = reorder_atoms_to_match(sorted_initial, final_supercell)
    
    return sorted_initial, matched_final
def linear_interpolate_structures(initial_atoms: Atoms, final_atoms: Atoms, num_images: int) -> List[Atoms]:
    """
    Create intermediate images by linear interpolation of FRACTIONAL coordinates.
    This prevents atom collisions during large cell shape changes (e.g., cubic -> hexagonal).
    """
    print(f"Creating {num_images} intermediate images via fractional interpolation...")

    # 1. Make structures compatible and mapped
    compatible_initial, compatible_final = make_structures_compatible(initial_atoms, final_atoms)

    images = [compatible_initial.copy()]
    
    # Get cells
    initial_cell = compatible_initial.get_cell()
    final_cell = compatible_final.get_cell()
    
    # Get scaled (fractional) positions
    p1 = compatible_initial.get_scaled_positions()
    p2 = compatible_final.get_scaled_positions()
    
    # Handle PBC wrapping for fractional coordinates
    # We want the shortest path in fractional space (e.g. 0.9 -> 0.1 should be +0.2, not -0.8)
    diff = p2 - p1
    diff -= np.round(diff) # Wrap to [-0.5, 0.5]
    p2_unwrapped = p1 + diff # p2 relative to p1
    
    for i in range(1, num_images + 1):
        alpha = i / (num_images + 1)
        
        # Interpolate Cell
        current_cell = (1 - alpha) * initial_cell + alpha * final_cell
        
        # Interpolate Fractional Positions
        current_frac = (1 - alpha) * p1 + alpha * p2_unwrapped
        
        # Create image
        img = compatible_initial.copy()
        img.set_cell(current_cell, scale_atoms=False) # Set new box dimensions
        img.set_scaled_positions(current_frac)        # Set atoms relative to new box
        
        images.append(img)

    # Final image
    # We create a fresh copy from compatible_final to ensure exact end-state geometry
    final_img = compatible_final.copy()
    images.append(final_img)

    return images
def get_mic_vector(pos1: np.ndarray, pos2: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """
    Calculate the vector pointing from pos1 to pos2 adhering to MIC.
    """
    diff = pos2 - pos1
    
    # If no PBC or cell volume is zero, return direct difference
    if not np.any(pbc) or np.abs(np.linalg.det(cell)) < 1e-8:
        return diff

    # Convert to fractional coordinates
    # We use np.linalg.solve for speed instead of explicit inverse
    diff_frac = np.linalg.solve(cell.T, diff.T).T
    
    # Apply MIC to PBC directions: wrap fractional coordinate diffs to [-0.5, 0.5]
    if np.all(pbc):
        diff_frac -= np.round(diff_frac)
    else:
        for i, is_periodic in enumerate(pbc):
            if is_periodic:
                diff_frac[:, i] -= np.round(diff_frac[:, i])
                
    # Convert back to Cartesian
    return np.dot(diff_frac, cell)

def calculate_tangent(prev_image: Atoms, current_image: Atoms, next_image: Atoms) -> np.ndarray:
    """
    Calculate the tangent vector for the current image using the improved tangent method
    with Minimum Image Convention support.
    """
    # Get standard data
    prev_pos = prev_image.get_positions()
    curr_pos = current_image.get_positions()
    next_pos = next_image.get_positions()
    
    cell = current_image.get_cell()
    pbc = current_image.get_pbc()
    
    # --- FIX START ---
    # Calculate vectors to neighboring images using MIC
    # Vectors point FROM prev/curr TO curr/next
    vec_to_next = get_mic_vector(curr_pos, next_pos, cell, pbc).flatten()
    vec_to_prev = get_mic_vector(prev_pos, curr_pos, cell, pbc).flatten()
    # --- FIX END ---
    
    # Use the bisector method for tangent estimation
    # Note: vec_to_prev here is actually (Current - Prev), so it points forward relative to path
    tangent = vec_to_next + vec_to_prev
    
    # Normalize tangent
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-10:
        # Fallback if tangent is zero
        tangent = vec_to_next - vec_to_prev
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            return np.zeros_like(tangent)
    
    return tangent / tangent_norm

def calculate_spring_force(prev_image: Atoms, current_image: Atoms, next_image: Atoms, 
                           tangent: np.ndarray, spring_constant: float) -> np.ndarray:
    """
    Calculate the spring force for the current image using MIC.
    """
    prev_pos = prev_image.get_positions()
    curr_pos = current_image.get_positions()
    next_pos = next_image.get_positions()
    
    cell = current_image.get_cell()
    pbc = current_image.get_pbc()
    
    # --- FIX START ---
    # Calculate distances using MIC vectors
    vec_next = get_mic_vector(curr_pos, next_pos, cell, pbc)
    vec_prev = get_mic_vector(prev_pos, curr_pos, cell, pbc) # Points Prev -> Curr
    
    dist_to_next = np.linalg.norm(vec_next) # Norm of matrix (uses frobenius, effectively dist between all atoms)
    dist_to_prev = np.linalg.norm(vec_prev)
    # --- FIX END ---
    
    # Spring force magnitude (difference in distances)
    spring_force_magnitude = spring_constant * (dist_to_next - dist_to_prev)
    
    # Apply force along tangent direction
    spring_force = spring_force_magnitude * tangent
    
    return spring_force

def calculate_force_statistics(images: List[Atoms], calculator) -> Tuple[List[float], List[float]]:
    """
    Calculate force statistics for all images.

    Args:
        images: List of all images in the path
        calculator: ASE calculator for force calculations

    Returns:
        Tuple of (total_forces_list, max_forces_list) where:
        - total_forces_list: Sum of all atomic forces for each image
        - max_forces_list: Maximum force magnitude for each image
    """
    total_forces = []
    max_forces = []

    for i, image in enumerate(images):
        image.set_calculator(calculator)
        try:
            forces = image.get_forces()  # Shape: (N_atoms, 3)

            # Calculate total force (sum of all atomic forces)
            total_force = np.sum(np.linalg.norm(forces, axis=1))
            total_forces.append(total_force)

            # Calculate maximum force magnitude
            max_force = np.max(np.linalg.norm(forces, axis=1))
            max_forces.append(max_force)

        except Exception as e:
            print(f"  Error calculating forces for image {i}: {e}")
            total_forces.append(0.0)
            max_forces.append(0.0)

    return total_forces, max_forces


def calculate_neb_forces(images: List[Atoms], calculator, spring_constant: float,
                        climbing_image_idx: Optional[int] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Calculate NEB forces for all intermediate images.
    
    Args:
        images: List of all images in the path
        calculator: ASE calculator for force calculations
        spring_constant: Spring constant for NEB
        climbing_image_idx: Index of climbing image for CI-NEB (None for standard NEB)
    
    Returns:
        Tuple of (neb_forces_list, energies_list)
    """
    print("Calculating NEB forces...")
    
    neb_forces = []
    energies = []
    
    # Calculate energies and true forces for all images
    true_forces_list = []
    for i, image in enumerate(images):
        image.set_calculator(calculator)
        try:
            energy = image.get_potential_energy()
            true_force = image.get_forces()
            
            energies.append(energy)
            true_forces_list.append(true_force.flatten())
            print(f"  Image {i}: Energy = {energy:.6f} eV")
        except Exception as e:
            print(f"  Error calculating energy/forces for image {i}: {e}")
            raise
    
    # Calculate NEB forces for intermediate images
    for i in range(1, len(images) - 1):  # Skip first and last (fixed endpoints)
        # Calculate tangent
        tangent = calculate_tangent(images[i-1], images[i], images[i+1])
        
        # Get true force
        true_force = true_forces_list[i]
        
        if climbing_image_idx is not None and i == climbing_image_idx:
            # CI-NEB force calculation for climbing image
            parallel_component = np.dot(true_force, tangent) * tangent
            neb_force = true_force - 2 * parallel_component
            print(f"  Image {i} (CLIMBING): Applied CI-NEB force modification")
        else:
            # Standard NEB force calculation
            # Project true force perpendicular to tangent
            parallel_component = np.dot(true_force, tangent) * tangent
            perpendicular_force = true_force - parallel_component
            
            # Calculate spring force
            spring_force = calculate_spring_force(images[i-1], images[i], images[i+1], 
                                                tangent, spring_constant)
            
            # Combine perpendicular true force with parallel spring force
            parallel_spring_component = np.dot(spring_force, tangent) * tangent
            neb_force = perpendicular_force + parallel_spring_component
        
        neb_forces.append(neb_force)
    
    return neb_forces, energies


def save_neb_images(images: List[Atoms], output_dir: str, prefix: str, iteration: int = 0):
    """
    Save all NEB images to individual files and write the full path animation.
    
    Args:
        images: List of ASE Atoms objects [initial, img1, ..., imgN, final]
        output_dir: Output directory
        prefix: Filename prefix
        iteration: Current iteration number
    """
    # 1. Save individual files (Optional, but good for inspection)
    images_dir = os.path.join(output_dir, f"images_iter_{iteration:04d}")
    os.makedirs(images_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        cif_filename = f"{prefix}_image_{i:02d}.cif"
        write(os.path.join(images_dir, cif_filename), image)

    # 2. SAVE THE FULL ANIMATION
    animation_base = f"{prefix}_animation_iter_{iteration:04d}"
    xyz_anim_path = os.path.join(output_dir, f"{animation_base}.xyz")
    traj_anim_path = os.path.join(output_dir, f"{animation_base}.traj")
    
    # FIX: Delete existing files to prevent appending (especially for .traj)
    if os.path.exists(xyz_anim_path):
        try:
            os.remove(xyz_anim_path)
        except OSError:
            pass
            
    if os.path.exists(traj_anim_path):
        try:
            os.remove(traj_anim_path)
        except OSError:
            pass
    
    # Save fresh files
    write(xyz_anim_path, images)
    write(traj_anim_path, images)
    
    print(f"Saved NEB animation to: {xyz_anim_path}")


def check_neb_convergence(neb_forces: List[np.ndarray], force_tolerance: float) -> Tuple[bool, float]:
    """
    Check if NEB calculation has converged.

    Args:
        neb_forces: List of NEB force vectors
        force_tolerance: Force tolerance for convergence

    Returns:
        Tuple of (converged, max_force)
    """
    if not neb_forces:
        return False, float('inf')

    # Calculate maximum force magnitude across all intermediate images
    max_force = 0.0
    for force in neb_forces:
        force_magnitude = np.linalg.norm(force)
        max_force = max(max_force, force_magnitude)

    converged = max_force < force_tolerance
    return converged, max_force


def update_image_positions(images: List[Atoms], neb_forces: List[np.ndarray], step_size: float = 0.01):
    """
    Update positions of intermediate images using steepest descent.

    Args:
        images: List of all images (endpoints remain fixed)
        neb_forces: List of NEB forces for intermediate images
        step_size: Step size for position updates
    """
    # Update only intermediate images (skip first and last)
    for i, force in enumerate(neb_forces):
        image_idx = i + 1  # Offset by 1 since we skip the first image

        # Reshape force back to (N_atoms, 3) format
        force_reshaped = force.reshape(-1, 3)

        # Update positions using steepest descent
        current_positions = images[image_idx].get_positions()
        new_positions = current_positions + step_size * force_reshaped
        images[image_idx].set_positions(new_positions)


def find_highest_energy_image(energies: List[float]) -> int:
    """
    Find the index of the highest energy intermediate image.

    Args:
        energies: List of energies for all images

    Returns:
        Index of highest energy intermediate image (excluding endpoints)
    """
    # Only consider intermediate images (exclude first and last)
    intermediate_energies = energies[1:-1]
    if not intermediate_energies:
        return None

    # Find index of maximum energy among intermediate images
    max_idx = np.argmax(intermediate_energies)

    # Convert back to full image list index
    return max_idx + 1


def run_neb_optimization(initial_atoms: Atoms, final_atoms: Atoms, calculator,
                        num_images: int, spring_constant: float, max_iterations: int,
                        force_tolerance: float, output_dir: str, prefix: str,
                        climbing_start_iteration: Optional[int] = None) -> Dict[str, Any]:
    """
    Run NEB or CI-NEB optimization.

    Args:
        initial_atoms: Initial structure
        final_atoms: Final structure
        calculator: ASE calculator
        num_images: Number of intermediate images
        spring_constant: Spring constant for NEB
        max_iterations: Maximum number of iterations
        force_tolerance: Force tolerance for convergence
        output_dir: Output directory
        prefix: Filename prefix
        climbing_start_iteration: Iteration to start climbing image (None for standard NEB)

    Returns:
        Dictionary with optimization results
    """
    print(f"\n--- Starting {'CI-NEB' if climbing_start_iteration else 'NEB'} Optimization ---")
    print(f"Number of intermediate images: {num_images}")
    print(f"Spring constant: {spring_constant} eV/Å²")
    print(f"Force tolerance: {force_tolerance} eV/Å")
    print(f"Maximum iterations: {max_iterations}")
    if climbing_start_iteration:
        print(f"Climbing image starts at iteration: {climbing_start_iteration}")

    start_time = time.time()

    # Create initial path by linear interpolation
    images = linear_interpolate_structures(initial_atoms, final_atoms, num_images)

    # Save initial path
    save_neb_images(images, output_dir, prefix, iteration=0)

    # Optimization loop
    climbing_image_idx = None
    convergence_history = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- NEB Iteration {iteration} ---")

        # Determine if we should activate climbing image
        if (climbing_start_iteration is not None and
            iteration >= climbing_start_iteration and
            climbing_image_idx is None):
            # Calculate energies to find highest energy image
            temp_forces, energies = calculate_neb_forces(images, calculator, spring_constant)
            climbing_image_idx = find_highest_energy_image(energies)
            if climbing_image_idx is not None:
                print(f"Activating climbing image at index {climbing_image_idx}")

        # Calculate NEB forces
        neb_forces, energies = calculate_neb_forces(images, calculator, spring_constant,
                                                   climbing_image_idx)

        # Check convergence
        converged, max_force = check_neb_convergence(neb_forces, force_tolerance)
        convergence_history.append({'iteration': iteration, 'max_force': max_force, 'energies': energies.copy()})

        print(f"Max force: {max_force:.6f} eV/Å (tolerance: {force_tolerance:.6f})")

        if converged:
            print(f"NEB converged after {iteration} iterations!")
            break

        # Update positions
        update_image_positions(images, neb_forces)

        # Save images every 10 iterations or at the end
        # if iteration % 10 == 0 or iteration == max_iterations:
        #     save_neb_images(images, output_dir, prefix, iteration)

    else:
        print(f"NEB did not converge after {max_iterations} iterations")

    # Save final path
    save_neb_images(images, output_dir, prefix, iteration=max_iterations)

    end_time = time.time()
    optimization_time = end_time - start_time

    print(f"NEB optimization completed in {optimization_time:.2f} seconds")

    # Return results
    results = {
        'converged': converged,
        'final_max_force': max_force,
        'iterations': iteration,
        'optimization_time': optimization_time,
        'images': images,
        'convergence_history': convergence_history,
        'climbing_image_idx': climbing_image_idx,
        'final_energies': energies
    }

    return results


def generate_enhanced_neb_summary(results: Dict[str, Any], method_name: str, args, final_cif_path: str,
                                 neb_params: Dict[str, Any], output_paths: List[str]) -> None:
    """
    Generate enhanced NEB summary files with energy and force information. 

    Args:
        results: NEB optimization results dictionary
        method_name: Name of the method ("NEB" or "CI-NEB")
        args: Command line arguments
        final_cif_path: Path to final CIF structure
        neb_params: Dictionary of NEB parameters
        output_paths: List of paths where to save the summary
    """
    import time

    # Calculate force statistics for final images
    final_images = results['images']
    calculator = final_images[0].calc if hasattr(final_images[0], 'calc') else None

    total_forces = []
    max_forces = []
    if calculator is not None:
        try:
            total_forces, max_forces = calculate_force_statistics(final_images, calculator)
        except Exception as e:
            print(f"Warning: Could not calculate force statistics: {e}")
            total_forces = [0.0] * len(final_images)
            max_forces = [0.0] * len(final_images)
    else:
        total_forces = [0.0] * len(final_images)
        max_forces = [0.0] * len(final_images)

    try:
        # Get number of atoms from the first image
        num_atoms = len(final_images[0])
        
        # Extract data for plotting
        final_energies = results['final_energies']
        image_indices = list(range(len(final_energies)))
        
        # Ensure max_forces list matches energy list length (fill 0.0 if calc missing)
        if len(max_forces) < len(final_energies):
            max_forces.extend([0.0] * (len(final_energies) - len(max_forces)))

        # Determine output directory (use the main directory)
        plot_output_dir = os.path.dirname(output_paths[1])
        
        # Generate the plot
        plot_neb_profile(
            images=image_indices,
            energies=final_energies,
            max_forces=max_forces,
            num_atoms=num_atoms,
            output_dir=plot_output_dir,
            prefix="neb_final"
        )
    except Exception as e:
        print(f"Warning: Could not generate NEB plot: {e}")

    # Generate summary content
    summary_content = []
    summary_content.append(f"--- {method_name} Optimization Summary ---")
    summary_content.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_content.append("")
    summary_content.append(f"Method: {method_name}")
    summary_content.append(f"Initial structure: {args.cif}")
    summary_content.append(f"Final structure: {final_cif_path}")
    summary_content.append(f"Number of intermediate images: {neb_params['num_images']}")
    summary_content.append(f"Spring constant: {neb_params['spring_constant']} eV/Å²")
    summary_content.append(f"Force tolerance: {neb_params['force_tolerance']} eV/Å")
    summary_content.append(f"Maximum iterations: {neb_params['max_iterations']}")

    if method_name == "CI-NEB":
        summary_content.append(f"Climbing image start iteration: {neb_params.get('climbing_start_iteration', 'N/A')}")

    summary_content.append("")
    summary_content.append("Results:")
    summary_content.append(f"  Converged: {results['converged']}")
    summary_content.append(f"  Final max force: {results['final_max_force']:.6f} eV/Å")
    summary_content.append(f"  Iterations completed: {results['iterations']}")
    summary_content.append(f"  Optimization time: {results['optimization_time']:.2f} seconds")
    summary_content.append("")

    # Enhanced energy and force profile
    if method_name == "CI-NEB":
        summary_content.append("Energy and Force Profile Along Path:")
        summary_content.append("-" * 90)
        summary_content.append(f"{'Image':<8} {'Energy (eV)':<15} {'Rel. Energy (eV)':<18} {'Total Force':<15} {'Max Force':<15} {'Type':<12}")
        summary_content.append("-" * 90)
    else:
        summary_content.append("Energy and Force Profile Along Path:")
        summary_content.append("-" * 75)
        summary_content.append(f"{'Image':<8} {'Energy (eV)':<15} {'Rel. Energy (eV)':<18} {'Total Force':<15} {'Max Force':<15}")
        summary_content.append("-" * 75)

    final_energies = results['final_energies']
    min_energy = min(final_energies)
    climbing_idx = results.get('climbing_image_idx', None)

    for i, energy in enumerate(final_energies):
        rel_energy = energy - min_energy
        total_force = total_forces[i] if i < len(total_forces) else 0.0
        max_force = max_forces[i] if i < len(max_forces) else 0.0

        if method_name == "CI-NEB":
            if i == 0:
                img_type = "Initial"
            elif i == len(final_energies) - 1:
                img_type = "Final"
            elif i == climbing_idx:
                img_type = "Climbing"
            else:
                img_type = "Intermediate"
            summary_content.append(f"{i:<8} {energy:<15.6f} {rel_energy:<18.6f} {total_force:<15.6f} {max_force:<15.6f} {img_type:<12}")
        else:
            summary_content.append(f"{i:<8} {energy:<15.6f} {rel_energy:<18.6f} {total_force:<15.6f} {max_force:<15.6f}")

    if method_name == "CI-NEB":
        summary_content.append("-" * 90)
    else:
        summary_content.append("-" * 75)

    # Write to all specified output paths
    for output_path in output_paths:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_content))
        print(f"Enhanced {method_name} summary saved to: {output_path}")


    return results
