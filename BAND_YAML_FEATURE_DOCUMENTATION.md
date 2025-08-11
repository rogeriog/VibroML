# Band.yaml Input Feature Documentation

## Overview

This feature allows users to load eigenmode data from existing band.yaml files and use it to generate displaced supercells without performing new phonon calculations. This is particularly useful for:

1. **Reusing existing phonon calculations**: Use previously computed eigenmode data to generate displaced structures
2. **Exploring specific modes**: Generate displaced supercells for specific q-points and band indices from existing calculations
3. **Faster workflow**: Skip the computationally expensive phonon calculation step when eigenmode data is already available

## New Command-Line Arguments

### `--band_yaml_path`
- **Type**: String
- **Required**: No (optional)
- **Description**: Path to an existing band.yaml file containing eigenmode data
- **Usage**: Must be used together with `--q` and `--band_idx`

## Usage Examples

### Basic Usage
```bash
vibroml --cif structure.cif \
        --band_yaml_path path/to/existing_band.yaml \
        --q 0.0,0.0,0.0 \
        --band_idx 0 \
        --displacement 0.1 \
        --engine mace \
        --no-relax
```

### With Unit Conversion
```bash
vibroml --cif structure.cif \
        --band_yaml_path path/to/existing_band.yaml \
        --q 0.0,0.25,0.0 \
        --band_idx 2 \
        --displacement 0.15 \
        --units cm-1 \
        --engine mace \
        --no-relax
```

## How It Works

1. **Input Validation**: The system checks that all three arguments (`--band_yaml_path`, `--q`, `--band_idx`) are provided together
2. **Eigenmode Loading**: The specified eigenmode is loaded from the band.yaml file
3. **Unit Conversion**: Frequencies are automatically converted from THz (band.yaml format) to the requested units
4. **Supercell Generation**: A commensurate supercell is estimated and generated with the specified displacement
5. **Visualization**: Animation trajectories are created to visualize the phonon mode

## Output Files

When using the band.yaml input feature, the following files are generated:

### Summary File
- **Filename**: `{prefix}_preloaded_eigenmode_summary.txt`
- **Content**: Summary of the loaded eigenmode including q-point, frequency, and displacement information

### Displaced Supercell
- **Directory**: `preloaded_mode_q_{qx}_{qy}_{qz}_mode_{band_idx}_disp_{displacement}/`
- **Files**:
  - `{prefix}_preloaded_q_{qx}_{qy}_{qz}_mode_{band_idx}_disp_{displacement}_sc_{nx}x{ny}x{nz}.cif`
  - `{prefix}_preloaded_q_{qx}_{qy}_{qz}_mode_{band_idx}_disp_{displacement}_sc_{nx}x{ny}x{nz}.xyz`

### Animation Files
- **Trajectory**: `{prefix}_preloaded_q_{qx}_{qy}_{qz}_mode_{band_idx}_disp_{displacement}_sc_{nx}x{ny}x{nz}_mode_animation.traj`
- **XYZ Animation**: `{prefix}_preloaded_q_{qx}_{qy}_{qz}_mode_{band_idx}_disp_{displacement}_sc_{nx}x{ny}x{nz}_mode_animation.xyz`

## Band.yaml Format Requirements

The band.yaml file must follow the Phonopy standard format with the following structure:

```yaml
natom: <number_of_atoms>
lattice:
  - [a1x, a1y, a1z]
  - [a2x, a2y, a2z]
  - [a3x, a3y, a3z]
points:
  - q-point: [qx, qy, qz]
    distance: <distance_along_path>
    bands:
      - frequency: <frequency_in_THz>
        eigenvector:
          - [[real, imag], [real, imag], [real, imag]]  # atom 1: x, y, z
          - [[real, imag], [real, imag], [real, imag]]  # atom 2: x, y, z
          # ... for each atom
```

## Error Handling

The system provides clear error messages for common issues:

- **Missing arguments**: "Error: --band_yaml_path requires both --q and --band_idx to be specified."
- **File not found**: "Error reading band.yaml file {path}: [error details]"
- **Q-point not found**: "Error: Q-point [qx, qy, qz] not found in {band_yaml_path}"
- **Invalid band index**: "Error: Band index {idx} is out of bounds. Available bands: 0 to {max_idx-1}"
- **Invalid file format**: "Error: Invalid band.yaml format in {path}"

## Integration with Existing Features

This feature integrates seamlessly with existing VibroML functionality:

- **Unit conversion**: Automatic conversion between THz, cm⁻¹, and eV
- **Supercell estimation**: Uses the same commensurate supercell estimation as regular phonon calculations
- **Visualization**: Generates the same animation and structure files as normal mode analysis
- **Output organization**: Follows the same directory structure and naming conventions

## Performance Benefits

Using preloaded eigenmode data provides significant performance benefits:

- **No phonon calculation**: Skips the computationally expensive phonon calculation step
- **Fast processing**: Typical processing time is under 3 seconds vs. minutes for full phonon calculations
- **Resource efficiency**: Requires minimal computational resources compared to full DFT-based phonon calculations

## Compatibility

This feature is compatible with:
- All supported structure formats (CIF, POSCAR, etc.)
- All calculation engines (MACE, M3GNet, etc.)
- All unit systems (THz, cm⁻¹, eV)
- Existing band.yaml files generated by VibroML or Phonopy
