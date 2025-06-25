### VibroML

[![PyPI Version](https://img.shields.io/pypi/v/vibroml.svg)](https://pypi.org/project/vibroml/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

### Overview  
VibroML is a Python toolkit for efficient, ML-driven vibrational analysis of crystalline materials. Leveraging Machine-Learned Interatomic Potentials (MLIPs), VibroML can:  
1. Compute phonon band structures  
2. Screen for imaginary (negative) modes  
3. Automatically displace atoms along unstable eigenmodes and re-optimize  
4. Validate dynamic stability via MLIP-powered AIMD trajectories  

This end-to-end workflow helps you discover lower-symmetry, dynamically stable phases with minimal manual intervention.

### Features  
- **Phonon-Only Mode:** Fast phonon band‐structure prediction & imaginary‐mode reporting  
- **Auto-Screen Mode:** Detect → displace → re-calculate → stabilize pipelines  
- **AIMD-Check Mode:** Run short MLIP-based molecular dynamics for dynamic stability metrics  
- **Modular API:** Mix & match phonon analysis, stability screening, supercell builders, and AIMD runners  
- **Command-Line Interface:** Simple CLI for scripted high-throughput screening  

### Installation  

```bash
# Via pip
pip install vibroml

# From source
git clone https://github.com/your-org/vibroml.git
cd vibroml
pip install -e .
```

### Quickstart  

#### 1. Phonon-Only Mode  
```python
from vibroml import PhononCalculator

# Load your structure (ASE, pymatgen, etc.)
structure = ...  

# Compute phonon bands
calc = PhononCalculator(potential="my_mlip.model")
bands = calc.run(structure)

# Inspect imaginary modes
print(bands.imaginary_modes_summary())
calc.plot(bands, show=True)
```

#### 2. Auto-Screen Mode  
```python
from vibroml import AutoScreen

screen = AutoScreen(potential="my_mlip.model", supercell_size=(2,2,2))
result = screen.run(structure)

# result.stable_structure → new optimized cell
# result.history        → details of each screening iteration
```

#### 3. AIMD-Check Mode  
```python
from vibroml import AIMDChecker

aimd = AIMDChecker(potential="my_mlip.model", temperature=300, steps=5000, timestep=1.0)
trajectory, stats = aimd.run(result.stable_structure)

print(f"RMSD: {stats.rmsd:.3f} Å, Drift: {stats.energy_drift:.2f} eV/ps")
```

### Command-Line Interface  

```bash
# Phonon-only
vibroml phonon --input POSCAR --model mlip.tar.gz --output bands.json

# Auto-screen
vibroml screen --input POSCAR --model mlip.tar.gz --supercell 2x2x2 --outdir screened/

# AIMD-check
vibroml aimd --input screened/CONTCAR --model mlip.tar.gz --temp 300 --steps 5000
```

Use `vibroml --help` or `vibroml <command> --help` for detailed options.

### Package Structure  
```text
vibroml/
├── core/         # settings, logging, I/O
├── phonon/       # band‐structure, stability checks, viz
├── screening/    # imaginary‐mode detection & relaxation
├── aimd/         # MLIP‐based MD runners & analyzers
├── utils/        # supercell builders, converters
└── cli.py        # entry point for command-line interface
```

### Contributing  
1. Fork the repo & create a feature branch  
2. Write tests & ensure existing tests pass  
3. Submit a pull request with a clear description  

See CONTRIBUTING.md for details.

### License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for full text.

### Citation  
If you use VibroML in your work, please cite:  
```
@software{vibroml2025,
  author = {Rogerio A. Gouvea},
  title  = {{VibroML}: Machine-Learned Vibrational Analysis & Stability Toolkit},
  year   = {2025},
  url    = {https://github.com/rogeriog/vibroml}
}
```