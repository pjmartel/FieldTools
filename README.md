# FieldTools

FieldTools.py calculates electric fields from MD trajectories. The script requires standard MD **trajectory** and **parameter** files as input.
Furthermore a **target** file needs to be provided that specifies the positions at which the field will be calculated.

To test FieldTools, click on: <a target="_blank" href="https://colab.research.google.com/github/bunzela/FieldTools/blob/main/FieldTools.ipynb">
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
> Trajectories must thus be imaged, or run FieldTools with `-pbc True` to use the periodic image of every atom closest to the target.

### Requirements
- Python 3
- numpy
- MDAnalysis (`pip install mdanalysis`) or pytraj

Install with `pip install -r requirements.txt`.

### Usage
    python utils/FieldTools.py -nc <trajectory file>
                               -parm <parameter file>
                               -target <target file>
                               -out <output file>
                               [-solvent <non-protein residues>]   default: WAT,Na+,Cl-
                               [-exclude_atoms <exclusion file>]
                               [-TIP4P <True|False>]
                               [-pbc <True|False>]
                               [-energy_out <energy file>]
                               [-vector_out <vector file>]
                               [-backend <auto|mdanalysis|pytraj>]
                               [-verbose <True|False>]

Run `python utils/FieldTools.py -h` for a description of all options.
`utils/FieldTools_pytraj.py` is equivalent to `utils/FieldTools.py -backend pytraj`.

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
    pip install pytest
    python -m pytest tests

`data/KPC_field.pkl` holds bond and point fields of the example system computed with the original FieldTools
(which used 5140 MV/cm per atomic unit instead of 5142.2, and the signed sum −q/r² for point targets);
the tests use its bond fields as a regression reference.

> [!NOTE]
> FieldTools can also calculate QM/MM point charges for a more refined field analysis (still experimental!) with `utils/QMChargesTools.py`. <br />
> Contact [adrian.bunzel@bsse.ethz.ch](mailto:adrian.bunzel@bsse.ethz.ch) for early access.

### Citation
Please cite the following paper when using FieldTools:
**H. Jabeen et al., bioRxiv 2023**.

> [!NOTE]
> Compared to the published version of FieldTools, the FieldTools.py script provided here reads trajectories with MDAnalysis by default instead of pytraj, because pytraj cannot be readily installed in Google Colab.

### Contact
For questions, help, or to report any bugs, please feel free to reach out to [adrian.bunzel@bsse.ethz.ch](mailto:adrian.bunzel@bsse.ethz.ch).
