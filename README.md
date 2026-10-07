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
> Atoms are written Amber-style as `:<residue>@<atom>` (`:40@OG`, `:LIG@C1`) or PyMOL-style as `[<chain>/]<residue>/<atom>` (`40/OG`, `LIG/C1`, `A/40/OG`),
> where `<residue>` is a residue number or a unique residue name (see [Atom specifiers](#atom-specifiers)).
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

This installs the commands `fieldtools`, `fieldplot`, `fieldplotly` and `qmchargestools`. With `-e` (editable), changes to the code in the
repository take effect without reinstalling; leave it out to install a fixed copy. Optional extras:
`pip install -e ".[test]"` (pytest and test data).

- For GROMACS 2026 `.tpr` files, install with the development version of MDAnalysis:
  `pip install -e ".[mdanalysis-dev]"` (or `uv pip install -e ".[mdanalysis-dev]"`). It is built from source,
  which takes a few minutes and needs git and a C compiler.
- pytraj (an alternative trajectory reader, Amber files only) is not installed automatically; it is part of AmberTools.
- The scripts in `utils/` still work without installing (`python utils/FieldTools.py ...`, given numpy and MDAnalysis).

### Usage
    fieldtools -nc <trajectory file>               (or -traj)
               -parm <parameter file>              (or -top)
               -target <target file>
               -out <output file>                  (or --prefix)
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
               [--prefix <prefix>]
    fieldtools --parameter-file <parameter file> [options]

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

### Plots
`fieldplot` draws a bar plot of the field contribution of every residue from the fields file written by `fieldtools`:
the bar is the mean over the trajectory frames, the error bar the standard deviation.

    fieldplot field.pkl                                              # all targets -> field_residues.png
    fieldplot field.pkl -target "40/C7_40/O71" -highlight 43,136,205-207 -out bond.pdf
    fieldplot field.pkl -vectors vectors.pkl -residues 1-261 -csv residues.csv
    fieldplot field.pkl -webagg -port 8988                           # interactive, at http://127.0.0.1:8988

- `-highlight` colors the bars of the given residue numbers differently (with a light band, so that residues with
  small values can still be found) and labels them, e.g. `-highlight 43,136,205-207`. `-highlight_label` and
  `-other_label` change the legend labels (e.g. `-highlight_label Mutants`). `-residues` limits the plot to a range
  of residues.
- Bond targets show the projected field (signed). Point targets show the field magnitude, which is not additive;
  with `-vectors` (the file from `fieldtools -vector_out`), they show each residue's field projected onto the
  direction of the total field instead, which is signed and adds up to the total.
- `-out` saves the plot (`.png`, `.pdf`, `.svg`, ...); `-csv` writes the plotted means and standard deviations.
- `-webagg` serves the interactive plot (zoom, pan, and a tooltip with residue, mean and SD on each bar) on a web server
  without opening a browser; stop it with Ctrl+C. It prints links with the machine's name and IP address
  (e.g. `http://marvin.example.org:8988`) that can be opened in a browser on any machine that can reach this one.
  By default it listens on all network interfaces (`0.0.0.0`), so anyone who can reach the port can see the plot;
  use `-host 127.0.0.1` to allow this computer only (and, from elsewhere, an SSH tunnel:
  `ssh -L 8988:localhost:8988 user@server`, then open `http://localhost:8988`). A firewall may need to allow the port.

#### Interactive plots with Plotly
`fieldplotly` draws the same per-residue plot with [Plotly](https://plotly.com/python/) and takes the same options
(`-target`, `-highlight`, `-highlight_label`, `-other_label`, `-residues`, `-vectors`, `-csv`). The result is a standalone web page with zoom and pan,
a range slider along the residues, panels that zoom together, hover details on every bar, and a legend that hides or
shows the highlighted and other residues:

    fieldplotly field.pkl -highlight 43,136,205-207                 # -> field_residues.html
    fieldplotly field.pkl -vectors vectors.pkl -out point.html
    fieldplotly field.pkl -serve -port 8988                         # serve it, and print the links to open

- The `.html` file embeds the Plotly JavaScript, so it can be opened offline or sent to others (about 5 MB);
  `-cdn` makes a much smaller file that loads Plotly from the internet when opened.
- `-serve` (alias `-webagg`) serves the page like `fieldplot -webagg`, with the same `-port` and `-host` options and links.
- `-out` with `.png`, `.pdf` or `.svg` writes a static image; this needs the optional `kaleido` package
  (`pip install kaleido`, which downloads a headless Chrome the first time). `fieldplot` writes static images
  without extra packages.

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

### Parameter files
Instead of a long command line, all options can be kept in a parameter file in [TOML](https://toml.io) format,
which also makes runs easy to reproduce:

```toml
# MnMT4 methyl transfer, reactive frames only
[input]
top = "MnMT4.tpr"
traj = "md.xtc"

[targets]
targets = ["IAA/O1 SAM/SD", "IAA/O2 SAM/SD"]   # one or two atoms per target
exclude = ["IAA SAM"]                           # one entry per target, or one for all targets

[system]
pbc = true

[filters]
distance = ["IAA/O1 SAM/CE 3.2"]
angle = ["IAA/O1 SAM/CE SAM/SD 160"]

[output]
out = "field.pkl"
vector_out = "vectors.pkl"
```

    fieldtools --parameter-file run.toml
    fieldtools --parameter-file run.toml -traj md2.xtc -out field2.pkl     # override some options
    fieldtools --parameter-file run.toml -traj rep2.xtc --prefix rep2      # rep2_field.pkl, rep2_vectors.pkl, ...

- Options on the command line override the parameter file; a filter on the command line replaces the file's
  filters of that kind.
- `--prefix NAME` (or `prefix` in `[output]`) adds `NAME_` to the names of all output files, keeping their folders
  (`results/field.pkl` becomes `results/NAME_field.pkl`); without `out`, the fields go to `NAME_field.pkl`.
  The command-line prefix overrides the one in the file. Everything printed on the screen, including errors, is also
  written to `NAME.log` in the folder of the field file. Every run starts by printing the FieldTools version and the
  exact command, so the log records how its results were produced.
- Relative paths are relative to the folder of the parameter file, so it can be run from anywhere.
- Targets and exclusions can be listed in the file (`targets`, `exclude`) or given as files (`target_file`, `exclude_file`).
- Unknown sections or keys are reported as errors, so typos are not silently ignored.
- `fieldtools --write-parameter-file run.toml` writes a commented template with all options; added to a complete
  command line, it writes that command's options to the file instead of running it.

#### Sections and keys
Section and key names are case-insensitive. Options that are left out take their default
(or the value given on the command line).

| Section | Key | Type | Default | Command line | Meaning |
|---|---|---|---|---|---|
| `[input]` | `top` | path | **required** | `-top`, `-parm` | topology with charges (`.parm7`, `.tpr`, `.top`); aliases `topology`, `parm` |
| | `traj` | path | **required** | `-traj`, `-nc` | trajectory (`.nc`, `.xtc`, `.trr`, ...); aliases `trajectory`, `nc` |
| | `backend` | text | `"auto"` | `-backend` | `"auto"`, `"mdanalysis"` or `"pytraj"` |
| | `gmx_include` | path | found automatically | `-gmx_include` | folder with the GROMACS force fields `#include`d by a `.top` |
| `[targets]` | `targets` | list | **required**¹ | | one target per entry: one atom (point) or two atoms (bond), e.g. `["IAA/O1 SAM/SD", "40/O71"]` |
| | `exclude` | list | residue of the first target atom | | excluded atoms: one entry per target, or one entry for all targets, e.g. `["IAA SAM"]` |
| | `target_file` | path | | `-target` | a target file instead of `targets`¹ |
| | `exclude_file` | path | | `-exclude_atoms` | an exclusion file instead of `exclude`¹ |
| `[system]` | `solvent` | text or list | `"auto"` | `-solvent` | solvent residue names (`"SOL,NA,CL"` or `["SOL", "NA", "CL"]`), or `"auto"` |
| | `tip4p` | true/false | `false` | `-TIP4P` | 4-point water model |
| | `pbc` | true/false | `false` | `-pbc` | make residues whole and use the periodic image closest to the target |
| `[filters]` | `distance` | list | none | `-filter_distance` | `"ATOM1 ATOM2 CUTOFF"` entries: distance <= CUTOFF (Å), or `MIN:MAX` |
| | `angle` | list | none | `-filter_angle` | `"ATOM1 ATOM2 ATOM3 CUTOFF"` entries: angle >= CUTOFF (degrees, vertex = middle atom), or `MIN:MAX` |
| `[output]` | `out` | path | **required**² | `-out` | fields (`.pkl`) |
| | `energy_out` | path | not written | `-energy_out` | Coulomb interaction energies (`.pkl`) |
| | `vector_out` | path | not written | `-vector_out` | field vectors (`.pkl`) |
| | `prefix` | text | none | `--prefix` | added with `_` to the names of all output files (`out` then defaults to `field.pkl`); the screen output is also written to `<prefix>.log` |
| `[qm]` | `use_qm_charges` | true/false | `false` | `-use_qm_charges` | replace charges with per-frame QM charges (experimental) |
| | `qm_charges` | path | | `-qm_charges` | QM charges file from `qmchargestools` |
| | `qm_dict` | path | | `-qm_dict` | QM dict file from `qmchargestools` |
| `[run]` | `verbose` | true/false | `false` | `-verbose` | display additional information |

¹ Use either `targets` or `target_file`, and either `exclude` or `exclude_file`; giving both of a pair is an error.
² Not required with a prefix: the fields are then written to `<prefix>_field.pkl`.

- Paths and text are written in quotes; true/false without quotes. Lists can span several lines, one entry per line.
- A frame must pass all filters. With filters, the kept frames are also written to `<out>_frames.dat`.

### Atom specifiers
Two equivalent syntaxes can be mixed freely in target and exclusion files:

| Amber style | PyMOL style | Selects |
|---|---|---|
| `:40@OG` | `40/OG` | atom OG of residue 40 |
| `:SAM@SD` | `SAM/SD` | atom SD of the (only) residue named SAM |
| `:40@CA,CB` | `40/CA,CB` | several atoms of residue 40 |
| `:264` | `264/` or `264` | all atoms of residue 264 |
| | `A/40/OG` | atom OG of residue 40 in chain A |

- Residue numbers are those of the topology: residues are numbered sequentially from 1 over the whole system
  (chains, ligands, ions and water), which can differ from the numbering of the original PDB file.
  `-verbose True` prints the residue and atom names of every target.
- A residue name must match exactly one residue in a target; in an exclusion line it selects all residues with that name.
- Chains are only available for GROMACS topologies, where `<chain>` is a molecule name as read by MDAnalysis
  (e.g. `seg_0_Protein_chain_A`), the name without the `seg_0_` prefix (`Protein_chain_A`), or its last part (`A`).
  Amber topologies do not store chains.

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
