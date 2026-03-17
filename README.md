# PyRosetta Utils

A Python utility library for protein structure modeling, energy analysis, and in silico deep mutational scanning (DMS) using [PyRosetta](https://www.pyrosetta.org/).

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Section 1 — Pose Utilities](#section-1--pose-utilities)
  - [Section 2 — Relaxation and Packing](#section-2--relaxation-and-packing)
  - [Section 3 — Mutation and Repacking](#section-3--mutation-and-repacking)
  - [Section 4 — Energy Calculation](#section-4--energy-calculation)
  - [Section 5 — Binding Free Energy](#section-5--binding-free-energy)
  - [Section 6 — Interface Descriptors](#section-6--interface-descriptors)
  - [Section 7 — Full Descriptor Pipeline](#section-7--full-descriptor-pipeline)
  - [Section 8 — In Silico DMS](#section-8--in-silico-dms)
  - [Section 9 — Sequence Modeling](#section-9--sequence-modeling)
  - [Section 10 — I/O Utilities](#section-10--io-utilities)
- [Workflows](#workflows)
  - [1. Compute Interface Descriptors](#1-compute-interface-descriptors)
  - [2. Run a Deep Mutational Scan](#2-run-a-deep-mutational-scan)
  - [3. Model a Target Sequence onto a Backbone](#3-model-a-target-sequence-onto-a-backbone)
  - [4. Calculate Binding Free Energy](#4-calculate-binding-free-energy)
- [Output Files](#output-files)
- [Notes on Parallelism](#notes-on-parallelism)
- [License](#license)

---

## Overview

`utils_pyrosetta.py` is a single-module library that wraps PyRosetta's lower-level API into clean, well-documented functions organised into 10 thematic sections. It is intended for computational biologists who need reproducible, scriptable access to common Rosetta workflows without writing boilerplate for every project.

Key capabilities:

| Capability | Key functions |
|---|---|
| PDB ↔ Pose index mapping | `PDB_pose_dictionairy`, `residues_list` |
| Structure relaxation | `pack_relax`, `minimize`, `fast_relax` |
| Point mutation + repacking | `mutate_repack` |
| Per-residue energy decomposition | `Energy_contribution`, `Get_energy_per_term` |
| Binding free energy (ΔG) | `dG_binding` |
| Interface descriptors | `Interaction_energy_metric`, `Contact_molecular_surface`, `Interface_analyzer_mover`, `Get_interface_selector` |
| Full descriptor pipeline | `Get_Interface_descriptors` |
| In silico DMS (ΔΔG) | `Run_DMS_Parallel`, `_dms_worker`, `fast_relax` |
| Sequence modeling | `Model_structure`, `model_sequence`, `Compare_sequences` |
| JD2 format conversion | `jd2_format` |

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyRosetta | ≥ 4 (academic or commercial license required) |
| pandas | ≥ 1.5 |

> **PyRosetta license:** PyRosetta requires a free academic license or a commercial license. See [https://www.pyrosetta.org/downloads](https://www.pyrosetta.org/downloads) for instructions.

---

## Installation

1. **Install PyRosetta** following the official instructions for your platform:
   ```bash
   pip install pyrosetta-installer
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/rosetta_utils.git
   cd rosetta_utils
   ```

3. **Install Python dependencies:**
   ```bash
   pip install pandas
   ```

4. **Import the module in your scripts:**
   ```python
   from rosetta_utils import (
       Get_descriptors, Run_DMS_Parallel, Model_structure, dG_binding
   )
   ```

---

## File Structure

```
rosetta_utils/
├── utils_pyrosetta.py      # Main utility module (all functions)
├── README.md             # This file
└── examples/             # (optional) example scripts and notebooks
```

---

## Quick Start

```python
import pyrosetta
from rosetta_utils import Get_descriptors, Run_DMS_Parallel, Model_structure

# --- 1. Compute interface descriptors for a protein complex ---
pose, df_descriptors = Get_Interface_descriptors(
    pdb="complex.pdb",
    partner1="A",
    partner2="D",
)
df_descriptors.to_csv("descriptors.csv", index=False)

# --- 2. Run an in silico DMS scan on positions 10–15 ---
df_dms = Run_DMS_Parallel(
    pdb="complex.pdb",
    positions_list=list(range(10, 16)),
    n_cpu=4,
    save_structures=False,
    output_dir="./DMS_output",
    fast_relax_repeats=0,
)

# --- 3. Model a new sequence onto an existing backbone ---
new_pose = Model_structure(
    pdb="template.pdb",
    sequence="ACDEFGHIKLMNPQRSTVWY...",
    output_name="./output/variant_01",
    relax=True,
)
```

---

## API Reference

### Section 1 — Pose Utilities

#### `PDB_pose_dictionairy(pose) → pd.DataFrame`

Builds a lookup table mapping every residue between PDB numbering (as written in the file) and Rosetta's internal Pose numbering (always 1-based, sequential across all chains).

**Returns** a DataFrame with columns `Chain`, `IndexPDB`, `IndexPose`.

```python
df_map = PDB_pose_dictionairy(pose)
# Chain  IndexPDB  IndexPose
#   A         1          1
#   A         2          2
#   B         1        152
```

---

#### `residues_list(df, chain) → list[int]`

Filters the mapping DataFrame to return only the Pose indices for a specific chain.

```python
chain_a_residues = residues_list(df_map, "A")
# [1, 2, 3, ..., 151]
```

---

### Section 2 — Relaxation and Packing

#### `pack_relax(pose, scorefxn) → None`

Runs a **Cartesian FastRelax** protocol with all backbone and side-chain degrees of freedom free. Uses L-BFGS Armijo non-monotone minimisation. Modifies the pose **in place**.

Suitable for: preparing structures before energy evaluation, relieving steric clashes after mutations or translations.

```python
pack_relax(pose, scorefxn)
```

---

#### `minimize(pose, scorefxn, minimizer_type) → None`

Energy minimisation using `MinMover` (torsional space). Two modes:

| `minimizer_type` | Degrees of freedom |
|---|---|
| `'minmover1'` | Side-chain chi only (backbone fixed) |
| `'minmover2'` | Backbone φ/ψ + side-chain chi |

```python
minimize(pose, scorefxn, "minmover1")  # fast side-chain only
minimize(pose, scorefxn, "minmover2")  # full torsional minimization
```

---

#### `fast_relax(pose, scorefxn, repeats=1) → None`

Same Cartesian FastRelax as `pack_relax` but with a configurable number of repeat cycles. Use higher values for improved geometry at the cost of runtime.

```python
fast_relax(pose, scorefxn, repeats=3)
```

---

### Section 3 — Mutation and Repacking

#### `mutate_repack(starting_pose, posi, amino, scorefxn) → pyrosetta.Pose`

Introduces a single point mutation at Pose position `posi` and repacks the surrounding neighbourhood. The input pose is **never modified** — the function returns a clone.

The TaskFactory logic:
- Target residue → restricted to the single specified amino acid
- Neighbourhood residues → repack only (no sequence change)
- Residues outside neighbourhood → frozen
- Disulfide bonds → preserved

```python
mutated_pose = mutate_repack(
    starting_pose=pose,
    posi=42,       # Pose index
    amino="A",     # Mutate to Alanine
    scorefxn=scorefxn,
)
```

---

### Section 4 — Energy Calculation

#### `Energy_contribution(pose, by_term=True) → pd.DataFrame`

Computes per-residue energy contributions using a `ref2015_cart` score function with hydrogen-bond pair decomposition enabled.

| `by_term` | Output shape |
|---|---|
| `True` | Wide DataFrame: one row per residue, one column per active energy term + metadata |
| `False` | Transposed single-row DataFrame with total energy per residue |

Metadata columns (always present when `by_term=True`): `Residue_Index_Pose`, `Residue_Index_PDB`, `Residue_Name`, `Residue_Name1`, `Chain`.

All values are **weighted** (raw value × score function weight).

```python
df_energy = Energy_contribution(pose, by_term=True)
# Residue_Index_Pose  Residue_Index_PDB  Residue_Name  ...  fa_atr  fa_rep  hbond_sc
#         1                   1              ALA        ...  -1.23   0.12    -0.05
```

---

#### `Get_energy_per_term(pose, scorefxn) → dict[str, float]`

Returns the total weighted energy for the entire pose, broken down by score term. Requires the pose to have been scored beforehand.

```python
terms = Get_energy_per_term(pose, scorefxn)
# {'fa_atr': -342.1, 'fa_rep': 18.4, 'hbond_sc': -22.7, ...}
```

---

### Section 5 — Binding Free Energy

#### `dG_binding(pose, partners, scorefxn) → float`

Estimates ΔG_bind by:
1. Cloning the pose and translating one partner 100 Å away.
2. Relaxing the separated state with `pack_relax`.
3. Computing: **ΔG = E(bound) − E(unbound)**

A negative value indicates a stabilising interaction.

```python
dg = dG_binding(pose, partners="A_B", scorefxn=scorefxn)
print(f"ΔG_bind = {dg:.2f} REU")
```

> The `partners` string follows Rosetta docking notation: `'A_B'` separates chain A from chain B; `'AB_C'` separates chains A+B from chain C.

---

### Section 6 — Interface Descriptors

#### `Interaction_energy_metric(pose, scorefxn, partner1, partner2) → float`

Computes cross-partner pairwise interaction energy using Rosetta's `InteractionEnergyMetric`. Only residue pairs where one belongs to `partner1` and the other to `partner2` are included.

```python
ie = Interaction_energy_metric(pose, scorefxn, partner1="A", partner2="B")
```

---

#### `Contact_molecular_surface(pose, partner1, partner2) → float`

Estimates the buried contact surface area at the interface via Rosetta's `ContactMolecularSurfaceFilter` (distance weight = 0.5).

```python
cms = Contact_molecular_surface(pose, partner1="A", partner2="B")
```

---

#### `Interface_analyzer_mover(pose, partner1, partner2) → dict[str, float]`

Applies `InterfaceAnalyzerMover` and returns all metrics as a flat dictionary with the prefix `ifa_`. Includes ΔG_separated, ΔG_cross, buried SASA, packing statistics, and more.

```python
ifa_data = Interface_analyzer_mover(pose, partner1="A", partner2="B")
# {'ifa_dG_separated': -12.3, 'ifa_packstat': 0.67, ...}
```

---

#### `Get_interface_selector(pose, partner1, partner2) → ResidueIndexSelector`

Returns a `ResidueIndexSelector` pre-loaded with all interface residue Pose indices, ready to be used in downstream TaskFactory or MoveMap operations.

```python
iface_sel = Get_interface_selector(pose, partner1="A", partner2="B")
```

---

### Section 7 — Full Descriptor Pipeline

#### `Get_descriptors(pdb, ions, outdir, basename, partner1, partner2) → tuple[Pose, pd.DataFrame]`

End-to-end pipeline that loads a structure, minimises it, and computes a full set of interface descriptors in one call.

**Steps performed internally:**
1. Initialise PyRosetta with `beta_nov16` corrections (+ `-auto_setup_metals` if `ions` is non-empty)
2. Load PDB and create `beta_nov16` score function
3. Remap chain letters to match JD2 renumbering
4. `minmover1` (side-chain minimisation)
5. `minmover2` (full torsional minimisation)
6. Compute: per-term energies, interaction energy, CMS, InterfaceAnalyzerMover metrics
7. Return a single-row DataFrame with all descriptors

```python
pose, df = Get_descriptors(
    pdb="complex.pdb",
    ions=["ZN"],          # non-empty → enables -auto_setup_metals
    outdir="./output",
    basename="my_protein",
    partner1="A",
    partner2="B",
)
```

**Output DataFrame columns:** all `beta_nov16` per-term energies + `ifa_*` metrics + `cms` + `interaction_energy` + `total_score`.

> Note: columns `ifa_dG_separated/dSASAx100` and `ifa_dG_cross/dSASAx100` are automatically renamed to `ifa_dG_separated_dSASAx100` and `ifa_dG_cross_dSASAx100`.

---

### Section 8 — In Silico DMS

#### `Run_DMS_Parallel(pdb, positions_list, n_cpu, save_structures=False, output_dir="./DMS_output", fast_relax_repeats=0) → pd.DataFrame`

Runs a full in silico Deep Mutational Scan across multiple residue positions in parallel. For each position, all 20 canonical amino acids are tested and **ΔΔG per energy term** is computed:

```
ΔΔG_term = Σ_residues [ E_term(mutant) − E_term(WT) ]
```

Uses `multiprocessing.Pool` for parallelism — each worker is a fully independent process that reinitialises PyRosetta internally.

```python
df_dms = Run_DMS_Parallel(
    pdb="my_protein.pdb",
    positions_list=[10, 11, 12, 45, 46],
    n_cpu=8,
    save_structures=True,          # dump PDB for every mutant
    output_dir="./DMS_results",
    fast_relax_repeats=1,          # apply 1 FastRelax round per mutant
)
```

**Output DataFrame columns:**

| Column | Description |
|---|---|
| `Position_Pose` | Rosetta Pose index of the scanned residue |
| `Position_PDB` | PDB residue number of the scanned residue |
| `Chain` | Chain identifier |
| `WT` | Wild-type amino acid (one-letter) |
| `Mutation` | Mutant amino acid (one-letter) |
| `Label` | Human-readable label, e.g. `A42G` |
| `ddG_<term>` | ΔΔG contribution for each active energy term |

**Output files:**

```
DMS_output/
├── csv/
│   ├── DMS_pos10.csv
│   ├── DMS_pos11.csv
│   └── ...
├── structures/          # only if save_structures=True
│   ├── 10_A10G.pdb
│   └── ...
└── DMS_report.csv       # consolidated report (all positions)
```

> **`fast_relax_repeats`:** Set to `0` (default) for speed. Use `1` for a light geometry correction after each mutation. Values > 1 are rarely needed and scale runtime linearly.

---

#### `fast_relax(pose, scorefxn, repeats=1) → None`

Standalone Cartesian FastRelax with configurable repeat count. Can be used independently outside of the DMS context.

---

### Section 9 — Sequence Modeling

#### `Model_structure(pdb, sequence, output_name, relax=True) → pyrosetta.Pose`

High-level function that models a target amino acid sequence onto an existing backbone. Only positions that differ between the template and target are mutated.

```python
new_pose = Model_structure(
    pdb="template.pdb",
    sequence="MKTIIALSYIFCLVFA...",   # full target sequence (same length as template)
    output_name="./output/variant_A", # written to variant_A.pdb
    relax=True,
)
```

**Internal pipeline:**
1. `read_pose` → load structure + score function
2. `Get_residues_from_pose` → extract current sequence and Pose indices
3. `Compare_sequences` → identify positions that differ
4. `model_sequence` → apply mutations sequentially + optional FastRelax

---

#### `model_sequence(pose, mutations, scorefxn, relax=True) → pyrosetta.Pose`

Applies a dictionary of mutations to a pose and optionally relaxes the result. The input pose is not modified.

```python
mutations = {42: "A", 87: "K", 103: "W"}   # {pose_index: target_aa}
new_pose = model_sequence(pose, mutations, scorefxn, relax=True)
```

---

#### `Compare_sequences(before_seq, after_seq, indexes) → dict[int, str]`

Positional diff between two sequences of equal length. Returns a `{pose_index: target_aa}` dict for all differing positions and prints a summary.

```python
mutations = Compare_sequences(
    before_seq="MKTII",
    after_seq="MKTIA",
    indexes=[1, 2, 3, 4, 5],
)
# New mutation: I5A
# → {5: 'A'}
```

---

#### `Get_residues_from_pose(pose) → tuple[str, list[int]]`

Extracts the one-letter sequence and the corresponding Pose index list from a pose.

```python
seq, idx = Get_residues_from_pose(pose)
```

---

#### `read_pose(pdb) → tuple[pyrosetta.Pose, pyrosetta.ScoreFunction]`

Convenience loader: initialises PyRosetta, reads a PDB, creates a `ref2015_cart` score function, and returns both.

```python
pose, scorefxn = read_pose("my_protein.pdb")
```

---

### Section 10 — I/O Utilities

#### `jd2_format(pdbfile, basename, outdir) → None`

Converts a PDB file to Rosetta's JD2-compatible format with `beta_nov16` corrections and PDB renumbering. Output is written to `<outdir>/<basename>_jd2_0001.pdb`.

```python
jd2_format("raw_structure.pdb", basename="my_protein", outdir="./jd2_ready")
```

---

## Workflows

### 1. Compute Interface Descriptors

```python
from rosetta_utils import Get_descriptors

pose, df = Get_descriptors(
    pdb="antibody_antigen.pdb",
    ions=[],
    outdir="./results",
    basename="ab_ag",
    partner1="HL",   # heavy + light chains
    partner2="A",    # antigen
)

print(df[["total_score", "ifa_dG_separated", "ifa_packstat", "cms", "interaction_energy"]])
df.to_csv("./results/descriptors.csv", index=False)
```

---

### 2. Run a Deep Mutational Scan

```python
import multiprocessing
from rosetta_utils import Run_DMS_Parallel

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)   # required on macOS/Windows

    df = Run_DMS_Parallel(
        pdb="target.pdb",
        positions_list=list(range(50, 65)),   # scan positions 50–64
        n_cpu=8,
        save_structures=False,
        output_dir="./DMS_run",
        fast_relax_repeats=0,
    )

    # Inspect the most destabilising single mutations
    print(df.nlargest(10, "ddG_total_energy")[["Label", "ddG_total_energy"]])
```

---

### 3. Model a Target Sequence onto a Backbone

```python
from rosetta_utils import Model_structure

new_pose = Model_structure(
    pdb="parent.pdb",
    sequence=open("target.fasta").readlines()[1].strip(),
    output_name="./output/designed_variant",
    relax=True,
)
```

---

### 4. Calculate Binding Free Energy

```python
import pyrosetta
from rosetta_utils import read_pose, dG_binding

pose, scorefxn = read_pose("complex.pdb")
dg = dG_binding(pose, partners="A_B", scorefxn=scorefxn)
print(f"ΔG_bind = {dg:.2f} REU")
```

---

## Output Files

| File / Directory | Generated by | Content |
|---|---|---|
| `<output_dir>/csv/DMS_pos<N>.csv` | `Run_DMS_Parallel` | ΔΔG per energy term for all 20 mutations at position N |
| `<output_dir>/DMS_report.csv` | `Run_DMS_Parallel` | Consolidated report across all scanned positions |
| `<output_dir>/structures/*.pdb` | `Run_DMS_Parallel` (if `save_structures=True`) | Mutant PDB structures |
| `<output_name>.pdb` | `Model_structure` | Modeled variant structure |
| `<outdir>/<basename>_jd2_0001.pdb` | `jd2_format` | JD2-compatible PDB |

---

## Notes on Parallelism

- `Run_DMS_Parallel` uses `multiprocessing.Pool` (one process per position).
- Each worker reinitialises PyRosetta internally (`pyrosetta.init(options="-mute all")`).
- On **macOS** and **Windows**, call `multiprocessing.set_start_method("spawn", force=True)` in your `if __name__ == "__main__":` block before calling `Run_DMS_Parallel`.
- PyRosetta's internal state is **not** shared across fork boundaries — this design is intentional and required for stability.

---

## License

This project is provided for academic and research use. PyRosetta itself requires a separate license — see [https://www.pyrosetta.org](https://www.pyrosetta.org).
