# FieldTools

FieldTools.py calculates electric fields from MD trajectories. The script requires standard MD **trajectory** and **parameter** files as input,
from Amber (`.parm7`/`.prmtop` with `.nc`/`.mdcrd`) or GROMACS (`.tpr`/`.top` with `.xtc`/`.trr`).
Furthermore a **target** file needs to be provided that specifies the positions at which the field will be calculated.

To test FieldTools, click on: <a target="_blank" href="https://colab.research.google.com/github/pjmartel/FieldTools/blob/main/FieldTools.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

> [!NOTE]
> Fields are defined in the **target** file and are either calculated at an atom or along a bond.
> Atoms are written as `:<residue>@<atom>`, where `<residue>` is a residue number (`:40@OG`) or a unique residue name (`:LIG@C1`).
> Use one atom for the field at that atom, or two atoms for the field along the bond between them.
> Several targets can be calculated in parallel, by adding additional lines to the **target** file.

> [!WARNING]
> By default, FieldTools calculates the fields from the exact location of all atoms in the system without considering periodicity.
> Trajectories must thus be imaged, or run FieldTools with `-pbc True` to make all residues whole and place each residue at its periodic image closest to the target.

### Installation
FieldTools is installed from a local copy of this repository (it is not published on PyPI).
It needs Python 3.10 or newer; numpy and MDAnalysis are installed automatically.

    git clone https://github.com/pjmartel/FieldTools
    cd FieldTools

With [uv](https://docs.astral.sh/uv/):

    uv venv                        # creates the environment in .venv
    source .venv/bin/activate
    uv pip install -e .

or, to only use the command-line tools, each in an isolated environment: `uv tool install .`

With venv and pip:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

This installs the commands `fieldtools` and `qmchargestools`. With `-e` (editable), changes to the code in the
repository take effect without reinstalling; leave it out to install a fixed copy. Optional extras:
`pip install -e ".[test]"` (pytest and test data) and `".[notebook]"` (matplotlib).

- For GROMACS 2026 `.tpr` files, install with the development version of MDAnalysis:
  `pip install -e ".[mdanalysis-dev]"` (or `uv pip install -e ".[mdanalysis-dev]"`). It is built from source,
  which takes a few minutes and needs git and a C compiler.
- pytraj (an alternative trajectory reader, Amber files only) is not installed automatically; it is part of AmberTools.
- The scripts in `utils/` still work without installing (`python utils/FieldTools.py ...`, given numpy and MDAnalysis).

### Usage
    fieldtools -nc <trajectory file>               (or -traj)
               -parm <parameter file>              (or -top)
               -target <target file>
               -out <output file>
               [-solvent <non-protein residues>]   default: auto
               [-exclude_atoms <exclusion file>]
               [-TIP4P <True|False>]
               [-pbc <True|False>]
               [-energy_out <energy file>]
               [-vector_out <vector file>]
               [-backend <auto|mdanalysis|pytraj>]
               [-gmx_include <GROMACS force field directory>]
               [-filter_distance <atom1> <atom2> <cutoff>]
               [-filter_angle <atom1> <atom2> <atom3> <cutoff>]
               [-verbose <True|False>]

Run `fieldtools -h` for a description of all options, and `fieldtools --version` for the version.
`utils/FieldTools_pytraj.py` is equivalent to `fieldtools -backend pytraj`.

**Solvent.** With `-solvent auto` (default), common water and ion residue names found in the system
(e.g. `WAT`, `HOH`, `SOL`, `Na+`, `Cl-`, `NA`, `CL`, `K`) are treated as solvent.
Give an explicit comma-separated list to include other molecules, such as ligands or lipids.

**Frame filters.** Restrict the calculation to frames with a given geometry, e.g. reactive conformations:

    -filter_distance :IAA@O1 :SAM@CE 3.2            # O1-CE distance <= 3.2 A
    -filter_angle :IAA@O1 :SAM@CE :SAM@SD 160       # O1-CE-SD angle >= 160 degrees (vertex: middle atom)

A single value is a maximum distance or a minimum angle; `MIN:MAX` gives a range (`2.5:3.2`).
Both options can be repeated, and a frame must pass all filters. With `-pbc True`, distances and angles use the
nearest periodic images. The output then only contains the kept frames, and `<out>_frames.dat` lists their
frame numbers with the measured distances and angles.
Each run also prints the range of every filter's values over the whole trajectory and how many frames pass
each filter on its own, which helps to choose cutoffs and to see which filter removes the frames.

**Charges.** At the start of every run, FieldTools prints the total charge of the system (with a warning if it is
not an integer), of each segment (chains and molecules in GROMACS topologies), of the protein, of each hetero
molecule (residues that are neither amino acids nor solvent, e.g. ligands and cofactors) and of each solvent species.

### GROMACS
    fieldtools -top topol.tpr -traj traj.xtc -target target.dat -pbc True -out field.pkl

- Use a `.tpr` (recommended) or a `.top` file as topology; both contain the charges, a `.gro` or `.pdb` does not.
  Files `#include`d by a `.top` are searched next to it and in the GROMACS force field directory
  (`$GMXLIB`, `$GMXDATA/top`, or the installation of `gmx` found in the `PATH`); set it with `-gmx_include` otherwise.
- Each MDAnalysis release reads `.tpr` files up to a certain GROMACS version (MDAnalysis 2.10 reads up to GROMACS 2025).
  For newer files (e.g. GROMACS 2026, "tpx version 138"), install FieldTools with the development version of
  MDAnalysis (`pip install -e ".[mdanalysis-dev]"`),
  or write a self-contained topology with `gmx grompp ... -pp processed.top` and use that instead.
- GROMACS trajectories usually contain molecules broken over the periodic boundaries. Use `-pbc True`,
  or process the trajectory first with `gmx trjconv -pbc mol -center`.
- Residues are numbered sequentially from 1 over the whole system (as read by MDAnalysis), which can differ from
  the numbering in your structure files. `-verbose True` prints the residue and atom names of every target.
- 4-point water models (`OW HW1 HW2 MW`) are recognized by `-TIP4P True`.
- `.xtc` files store coordinates with a precision of 0.01 Å, which changes fields from nearby atoms by up to about 1%.
  Use `.trr` files (or a higher `compressed-x-precision`) when this matters.
- `data/KPC.top` and `data/KPC.xtc` are the Amber example system converted to GROMACS format.

**Exclusions.** Without `-exclude_atoms`, all atoms of the residue of the first target atom are excluded from the field.
Otherwise, the exclusion file contains one line per target (or a single line used for all targets) listing atoms
(`:40@CA`), several atoms of a residue (`:40@CA,CB,HA`) or whole residues (`:264`).
The target atom of a point target is always excluded.

**Output.** The fields (MV/cm) are saved as a pickled dictionary:

    Fields[target][component] = [field_frame_1, field_frame_2, ...]

    component : Total, Protein, Solvent, each solvent species (-solvent),
                and each protein residue (RESNAME_RESNUMBER)

- Bond targets (`:47@OG :47@HG`): field at the bond midpoint projected onto the unit vector from the first to the second atom.
  The components add up to the Total.
- Point targets (`:47@OG`): magnitude of each component's field vector |E|.
  Magnitudes do not add up; use `-vector_out` to obtain the field vectors (arrays of shape `(n_frames, 3)` per target and component).
- `-energy_out` saves the Coulomb interaction energy (kJ/mol) between the target atom(s) and the environment, with the same structure.

### Tests
    pip install -e ".[test]"
    python -m pytest

`MDAnalysisTests` provides the GROMACS `.tpr` test system; without it, those tests are skipped.

`data/KPC_field.pkl` holds bond and point fields of the example system computed with the original FieldTools
(which used 5140 MV/cm per atomic unit instead of 5142.2, and the signed sum −q/r² for point targets);
the tests use its bond fields as a regression reference.

> [!NOTE]
> FieldTools can also calculate QM/MM point charges for a more refined field analysis (still experimental!) with `qmchargestools`. <br />
> Contact [adrian.bunzel@bsse.ethz.ch](mailto:adrian.bunzel@bsse.ethz.ch) for early access.

### Citation
Please cite the following paper when using FieldTools:
**H. Jabeen et al., bioRxiv 2023**.

> [!NOTE]
> Compared to the published version of FieldTools, the FieldTools.py script provided here reads trajectories with MDAnalysis by default instead of pytraj, because pytraj cannot be readily installed in Google Colab.

### Contact
For questions, help, or to report any bugs, please feel free to reach out to [adrian.bunzel@bsse.ethz.ch](mailto:adrian.bunzel@bsse.ethz.ch).
