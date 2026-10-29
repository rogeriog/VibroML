# VibroML example structures

Input CIF files used by the Quickstart commands in the main README and by the
benchmark workflows described in the manuscript. All files are unrelaxed
starting structures, so every remediation mode (GA, TRAD, ALL, RAND) has
something to repair.

| File | Composition | Parent structure | Notes |
|---|---|---|---|
| `LiF_cubic.cif` | LiF | Pm-3m simple cubic | Canonical unstable input; the GA recovers the Fm-3m ground state. Fastest demo (2 atoms). |
| `CsPbI3_cubic.cif` | CsPbI3 | Pm-3m perovskite | Halide perovskite aristotype; soft tilting modes. |
| `HfO2_cubic.cif` | HfO2 | Fm-3m fluorite | Fluorite aristotype; the GA recovers the P21/c ground state and lower-symmetry phases. |
| `Bi2Sn2O7_cubic.cif` | Bi2Sn2O7 | Fd-3m pyrochlore (88 atoms) | Largest benchmark cell; tests scalability. |
| `Cs2KInI6_cubic.cif` | Cs2KInI6 | Fm-3m elpasolite (double perovskite) | Frustrated perovskite used in the ProtoCSP alloying case. |
| `CuN_b2.cif` | CuN | B2 (CsCl-type) | Metallic intermetallic benchmark. |
| `ZnCu_b2.cif` | ZnCu | B2 (CsCl-type) | Metallic intermetallic benchmark. |

## Quick commands

```bash
# Phonon evaluation only
vibroml --cif examples/LiF_cubic.cif --engine mace --no-relax

# Automated remediation (genetic algorithm)
vibroml --cif examples/LiF_cubic.cif --engine mace --auto --method ga

# Other remediation strategies
vibroml --cif examples/LiF_cubic.cif --engine mace --auto --method traditional      # single soft-mode following
vibroml --cif examples/LiF_cubic.cif --engine mace --auto --method traditional_all  # coupled mode following
vibroml --cif examples/LiF_cubic.cif --engine mace --auto --method opt_random       # stochastic search
```

Full-scale run outputs for these structures (relaxed polymorphs, phonon
dispersions, MD assessments, NEB pathways) are hosted in the companion
repository <https://github.com/rogeriog/VibroML_SupportData>.
