# vibroml/utils/neb_utils.py

import numpy as np
import os
import time
from ase.atoms import Atoms
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import make_supercell
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import logging


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


def make_structures_compatible(initial_atoms: Atoms, final_atoms: Atoms) -> Tuple[Atoms, Atoms]:
    """
    Make two structures compatible for NEB by creating supercells if needed.

    Args:
        initial_atoms: Initial structure
        final_atoms: Final structure

    Returns:
        Tuple of (compatible_initial, compatible_final) with same number of atoms
    """
    print(f"Checking structure compatibility...")
    print(f"Initial structure: {len(initial_atoms)} atoms ({Counter(initial_atoms.get_chemical_symbols())})")
    print(f"Final structure: {len(final_atoms)} atoms ({Counter(final_atoms.get_chemical_symbols())})")

    # If already compatible, return as-is
    if len(initial_atoms) == len(final_atoms):
        # Still check chemical compatibility
        if not check_chemical_compatibility(initial_atoms, final_atoms):
            raise ValueError("Structures have different chemical compositions and cannot be made compatible")
        print("✓ Structures already have the same number of atoms")
        return initial_atoms.copy(), final_atoms.copy()

    # Check chemical compatibility
    if not check_chemical_compatibility(initial_atoms, final_atoms):
        raise ValueError(
            f"Structures have incompatible chemical compositions:\n"
            f"Initial: {Counter(initial_atoms.get_chemical_symbols())}\n"
            f"Final: {Counter(final_atoms.get_chemical_symbols())}\n"
            f"Cannot create supercells to match atom counts."
        )

    print("✓ Structures have compatible chemical compositions")

    # Find supercell multipliers
    try:
        transform1, transform2 = find_supercell_multipliers(initial_atoms, final_atoms)

        print(f"Creating supercells to match atom counts...")
        print(f"Initial structure supercell transformation: {np.diag(transform1)}")
        print(f"Final structure supercell transformation: {np.diag(transform2)}")

        # Create supercells
        initial_supercell = make_supercell(initial_atoms, transform1)
        final_supercell = make_supercell(final_atoms, transform2)

        print(f"After supercell creation:")
        print(f"Initial supercell: {len(initial_supercell)} atoms ({Counter(initial_supercell.get_chemical_symbols())})")
        print(f"Final supercell: {len(final_supercell)} atoms ({Counter(final_supercell.get_chemical_symbols())})")

        # Verify they now have the same number of atoms
        if len(initial_supercell) != len(final_supercell):
            raise ValueError(
                f"Failed to create compatible supercells. "
                f"Got {len(initial_supercell)} and {len(final_supercell)} atoms"
            )

        print("✓ Successfully created compatible supercells")
        return initial_supercell, final_supercell

    except Exception as e:
        raise ValueError(f"Failed to create compatible supercells: {str(e)}")


def linear_interpolate_structures(initial_atoms: Atoms, final_atoms: Atoms, num_images: int) -> List[Atoms]:
    """
    Create intermediate images by linear interpolation between initial and final structures.
    Automatically handles structure compatibility by creating supercells if needed.

    Args:
        initial_atoms: Initial structure (ASE Atoms object)
        final_atoms: Final structure (ASE Atoms object)
        num_images: Number of intermediate images (excluding initial and final)

    Returns:
        List of ASE Atoms objects representing the complete path [initial, image1, ..., imageN, final]
        Note: If supercells were created, the returned structures will have more atoms than the originals
    """
    print(f"Creating {num_images} intermediate images by linear interpolation...")

    # Make structures compatible (handles supercell creation if needed)
    try:
        compatible_initial, compatible_final = make_structures_compatible(initial_atoms, final_atoms)
    except ValueError as e:
        print(f"Error: {e}")
        raise

    images = []

    # Add initial structure
    images.append(compatible_initial.copy())

    # Create intermediate images
    initial_positions = compatible_initial.get_positions()
    final_positions = compatible_final.get_positions()
    
    for i in range(1, num_images + 1):
        # Linear interpolation parameter
        alpha = i / (num_images + 1)
        
        # Interpolate positions
        interpolated_positions = (1 - alpha) * initial_positions + alpha * final_positions
        
        # Create new atoms object
        image = compatible_initial.copy()
        image.set_positions(interpolated_positions)

        # Interpolate cell parameters if they differ
        if not np.allclose(compatible_initial.cell, compatible_final.cell, atol=1e-6):
            initial_cell = compatible_initial.cell.array
            final_cell = compatible_final.cell.array
            interpolated_cell = (1 - alpha) * initial_cell + alpha * final_cell
            image.set_cell(interpolated_cell, scale_atoms=False)

        images.append(image)

    # Add final structure
    images.append(compatible_final.copy())
    
    print(f"Created {len(images)} total images (including initial and final)")
    return images


def calculate_tangent(prev_image: Atoms, current_image: Atoms, next_image: Atoms) -> np.ndarray:
    """
    Calculate the tangent vector for the current image using the improved tangent method.
    
    Args:
        prev_image: Previous image in the path
        current_image: Current image
        next_image: Next image in the path
    
    Returns:
        Normalized tangent vector (flattened positions)
    """
    # Get positions as flattened arrays
    prev_pos = prev_image.get_positions().flatten()
    curr_pos = current_image.get_positions().flatten()
    next_pos = next_image.get_positions().flatten()
    
    # Calculate vectors to neighboring images
    vec_to_next = next_pos - curr_pos
    vec_to_prev = curr_pos - prev_pos
    
    # Use the bisector method for tangent estimation
    tangent = vec_to_next + vec_to_prev
    
    # Normalize tangent
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-10:
        # If tangent is nearly zero, use simple difference
        tangent = vec_to_next - vec_to_prev
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            # If still zero, return zero tangent
            return np.zeros_like(tangent)
    
    return tangent / tangent_norm


def calculate_spring_force(prev_image: Atoms, current_image: Atoms, next_image: Atoms, 
                          tangent: np.ndarray, spring_constant: float) -> np.ndarray:
    """
    Calculate the spring force for the current image.
    
    Args:
        prev_image: Previous image in the path
        current_image: Current image
        next_image: Next image in the path
        tangent: Normalized tangent vector
        spring_constant: Spring constant (eV/Å²)
    
    Returns:
        Spring force vector (flattened)
    """
    # Get positions as flattened arrays
    prev_pos = prev_image.get_positions().flatten()
    curr_pos = current_image.get_positions().flatten()
    next_pos = next_image.get_positions().flatten()
    
    # Calculate distances to neighboring images
    dist_to_next = np.linalg.norm(next_pos - curr_pos)
    dist_to_prev = np.linalg.norm(curr_pos - prev_pos)
    
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
    Save all NEB images to files.
    
    Args:
        images: List of ASE Atoms objects
        output_dir: Output directory
        prefix: Filename prefix
        iteration: Current iteration number
    """
    images_dir = os.path.join(output_dir, f"images_iter_{iteration:04d}")
    os.makedirs(images_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        # Save as CIF
        cif_filename = f"{prefix}_image_{i:02d}.cif"
        cif_path = os.path.join(images_dir, cif_filename)
        write(cif_path, image)
        
        # Save as XYZ for visualization
        xyz_filename = f"{prefix}_image_{i:02d}.xyz"
        xyz_path = os.path.join(images_dir, xyz_filename)
        write(xyz_path, image)
    
    print(f"Saved {len(images)} images to {images_dir}")


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
        if iteration % 10 == 0 or iteration == max_iterations:
            save_neb_images(images, output_dir, prefix, iteration)

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
