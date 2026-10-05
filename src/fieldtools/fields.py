#!/usr/bin/env python3
"""FieldTools: electric fields from MD trajectories.

Calculates the electric field exerted by all atoms of a system (point charges from
the parameter file, or optionally QM charges) at selected atoms or projected onto
selected bonds, decomposed into protein residues and solvent species.

Run `fieldtools -h` for usage.
"""

import argparse
import datetime
import os
import pickle
import re
import shutil
import sys
import warnings

import numpy as np

from . import __version__
from .parameters import load_parameter_file, write_parameter_file

### Physical constants (CODATA 2018)
BOHR = 0.529177210903                       # Angstrom
AU_FIELD_MV_CM = 5142.20674763              # 1 a.u. of electric field in MV/cm
HARTREE_KJ_MOL = 2625.4996394799            # 1 Hartree in kJ/mol
FIELD_CONST = AU_FIELD_MV_CM * BOHR**2      # q [e] / r^2 [A^2]   -> MV/cm
ENERGY_CONST = HARTREE_KJ_MOL * BOHR        # q*Q [e^2] / r [A]  -> kJ/mol

### Residue names treated as solvent with -solvent auto (Amber, GROMACS and CHARMM conventions)
WATER_NAMES = ["WAT", "HOH", "SOL", "TIP3", "TIP4", "TIP5", "SPC", "T3P", "T4P", "T5P"]
ION_NAMES = ["Na+", "Cl-", "K+", "Li+", "Rb+", "Cs+", "Mg2+", "Ca2+", "Zn2+", "Br-", "F-", "I-",
             "NA", "CL", "K", "LI", "RB", "CS", "MG", "CA", "ZN", "BR", "F", "I",
             "NA+", "CL-", "MG2", "CA2", "SOD", "CLA", "POT", "CAL", "CES", "LIT"]
### Massless charged sites of 4-point water models (Amber EPW, GROMACS MW)
EXTRA_POINT_NAMES = ["EPW", "EP", "MW"]

TARGET_HELP = """
Atom specifiers: two equivalent syntaxes
  Amber:  :<residue>@<atom>        e.g. :47@OG   :LIG@C1   :47@CA,CB   :264
  PyMOL:  [<chain>/]<residue>/<atom>   e.g. 47/OG   LIG/C1   47/CA,CB   264/   A/47/OG
  <residue> is a residue number (as numbered in the topology) or a unique residue
  name; <atom> is one atom name or a comma-separated list; leaving out the atom
  selects the whole residue. <chain> is the GROMACS molecule name of a .tpr/.top,
  or its last part (A matches seg_0_Protein_chain_A); Amber topologies have no chains.

Target file (-target):
  One target per line, written as one or two atom specifiers.
    :47@OG          field at atom OG of residue 47; the output is the magnitude
                    of the field vector |E| (the vectors are written with -vector_out)
    :47@OG :47@HG   field at the bond midpoint, projected onto the unit vector
                    pointing from the first to the second atom (signed, MV/cm)

Exclusion file (-exclude_atoms):
  One line per target (same order as the target file), or a single line that is
  applied to all targets. Each line lists atom specifiers whose charges are
  ignored for that target: :40@CA (one atom), :40@CA,CB,HA (several atoms of a
  residue) or :264 (a whole residue). Without -exclude_atoms, all atoms of the
  residue of the first target atom are excluded. The target atom of a point
  target is always excluded.

Output (-out): pickled dictionary
  Fields[target][component] = [field_frame_1, field_frame_2, ...]   (MV/cm)
  Components: Total, Protein, Solvent, each solvent species (as given with
  -solvent) and each protein residue (RESNAME_RESNUMBER).
  For point targets, each component is the magnitude of that component's field
  vector, so magnitudes of the components do not add up to the Total.

Parameter file (--parameter-file params.toml):
  All options can be stored in a TOML file, in sections, e.g.
    [input]
    top = "topol.tpr"
    traj = "traj.xtc"
    [targets]
    targets = ["IAA/O1 SAM/SD", "IAA/O2 SAM/SD"]
    exclude = ["IAA SAM"]
    [system]
    pbc = true
    [filters]
    distance = ["IAA/O1 SAM/CE 3.2"]
    [output]
    out = "field.pkl"
  Options on the command line override the file. Relative paths are relative to the
  folder of the parameter file. fieldtools --write-parameter-file params.toml writes a
  commented template, or the options of the rest of the command line.

Frame filters (-filter_distance, -filter_angle):
  Fields are only calculated for frames that pass all filters, e.g. reactive geometries:
    -filter_distance :IAA@O1 :SAM@CE 3.2            distance <= 3.2 A
    -filter_angle :IAA@O1 :SAM@CE :SAM@SD 160       angle O1-CE-SD >= 160 degrees
  A single value is a maximum distance or a minimum angle; use MIN:MAX for a range
  (e.g. 2.5:3.2). Both options can be given several times. The kept frames and the
  measured values are written to <out>_frames.dat.

Trajectories must be imaged unless -pbc True is used. GROMACS trajectories usually
contain molecules broken over the periodic boundaries: use -pbc True, or process them
first with gmx trjconv -pbc mol -center.

GROMACS: use a .tpr (recommended) or a .top file with -parm. Files #included by a
.top are searched next to it and in -gmx_include. Residues are numbered sequentially
from 1 over the whole system, as read by MDAnalysis (check with -verbose True).
"""


#####################################################################################
### Input
#####################################################################################

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")


def optional_path(value):
    # Allows wrappers (e.g. the notebook) to pass "False" or "" for unset files
    return None if value in ("", "False", "None") else value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fieldtools",
        description="Calculate electric fields from MD trajectories.",
        epilog=TARGET_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"FieldTools {__version__}")
    parser.add_argument("--parameter-file", dest="parameter_file", metavar="FILE",
                        help="read the options from a parameter file (TOML, see below); "
                             "options given on the command line override it")
    parser.add_argument("--write-parameter-file", dest="write_parameter_file", metavar="FILE",
                        help="write the options of this command line to a parameter file and exit")
    parser.add_argument("-nc", "-traj", dest="nc",
                        help="trajectory file (Amber .nc/.mdcrd, GROMACS .xtc/.trr, ...) [required]")
    parser.add_argument("-parm", "-top", dest="parm",
                        help="topology file with charges (Amber .parm7/.prmtop, GROMACS .tpr/.top) [required]")
    parser.add_argument("-target", help="target file (see below) [required]")
    parser.add_argument("-out", help="output file for the fields (.pkl) [required]")
    parser.add_argument("-solvent", default="auto",
                        help="comma separated list of non-protein residue names, or auto to use the "
                             "common water and ion residue names found in the system [default: auto]")
    parser.add_argument("-gmx_include", type=optional_path, default=None,
                        help="directory with the files #included by a GROMACS .top (force field "
                             "directories) [default: $GMXLIB, $GMXDATA/top, or the GROMACS installation of gmx]")
    parser.add_argument("-exclude_atoms", type=optional_path, default=None,
                        help="file with atoms excluded from the field calculation (see below)")
    parser.add_argument("-TIP4P", type=str2bool, default=False,
                        help="system uses a 4-point water model (WAT with EPW) [default: False]")
    parser.add_argument("-pbc", type=str2bool, default=False,
                        help="make residues whole and place every residue at its periodic image "
                             "closest to the target, so the trajectory does not have to be imaged "
                             "[default: False]")
    parser.add_argument("-energy_out", type=optional_path, default=None,
                        help="also write the Coulomb interaction energy (kJ/mol) between the "
                             "target atom(s) and the environment to this file (.pkl)")
    parser.add_argument("-vector_out", type=optional_path, default=None,
                        help="also write the field vectors (MV/cm) of every target and component "
                             "to this file (.pkl), as arrays of shape (n_frames, 3)")
    parser.add_argument("-filter_distance", nargs=3, action="append", default=[],
                        metavar=("ATOM1", "ATOM2", "CUTOFF"),
                        help="only use frames where the ATOM1-ATOM2 distance is <= CUTOFF (A) "
                             "or within MIN:MAX; can be repeated")
    parser.add_argument("-filter_angle", nargs=4, action="append", default=[],
                        metavar=("ATOM1", "ATOM2", "ATOM3", "CUTOFF"),
                        help="only use frames where the ATOM1-ATOM2-ATOM3 angle (vertex ATOM2) is "
                             ">= CUTOFF (degrees) or within MIN:MAX; can be repeated")
    parser.add_argument("-backend", choices=["auto", "mdanalysis", "pytraj"], default="auto",
                        help="library used to read the trajectory [default: auto]")
    parser.add_argument("-verbose", type=str2bool, default=False,
                        help="display additional information [default: False]")
    parser.add_argument("-use_qm_charges", type=str2bool, default=False,
                        help="replace charges with per-frame QM charges from QMChargesTools [default: False]")
    parser.add_argument("-qm_charges", type=optional_path, default=None,
                        help="QM charges file (per-frame partial charges made with QMChargesTools)")
    parser.add_argument("-qm_dict", type=optional_path, default=None,
                        help="QM dict file (atom names of the QM region, made with QMChargesTools)")
    parser.add_argument("-qm_mask", type=optional_path, default=None,
                        help="QM region mask file (accepted for compatibility, not used)")
    return parser


def parse_arguments(argv):
    """Options from the command line and the parameter file (the command line takes precedence)."""
    parser = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--parameter-file", dest="parameter_file")
    file_values = {}
    if "-h" not in argv and "--help" not in argv:
        parameter_file = pre_parser.parse_known_args(argv)[0].parameter_file
        if parameter_file:
            file_values = load_parameter_file(parameter_file)
    # Filters are lists: file values only apply if the command line gives none of that kind
    file_filters = {k: file_values.pop(k) for k in ("filter_distance", "filter_angle") if k in file_values}
    parser.set_defaults(**file_values)
    args = parser.parse_args(argv)
    for dest, value in file_filters.items():
        if not getattr(args, dest):
            setattr(args, dest, value)

    args.solvent = None if args.solvent == "auto" else [i for i in args.solvent.split(",") if i]
    if args.write_parameter_file:
        return args
    missing = [flag for dest, flag in (("parm", "-top"), ("nc", "-traj"), ("target", "-target"), ("out", "-out"))
               if not getattr(args, dest)]
    if missing:
        parser.error(f"missing {', '.join(missing)} (give it on the command line or in the parameter file)")
    if args.use_qm_charges and (args.qm_charges is None or args.qm_dict is None):
        parser.error("-use_qm_charges True requires -qm_charges and -qm_dict")
    return args


def read_lines(path):
    # Non-empty lines, with comments (#) removed
    with open(path) as f:
        lines = [line.split("#")[0].strip() for line in f]
    return [line for line in lines if line]


#####################################################################################
### Trajectory loading
#####################################################################################

class System:
    """Topology arrays plus a generator over (positions, box) of each frame."""

    def __init__(self, names, resnames, resids, charges, n_frames, frames, segids=None):
        self.names = np.asarray(names, dtype=str)
        self.resnames = np.asarray(resnames, dtype=str)
        self.resids = np.asarray(resids, dtype=str)
        self.charges = np.asarray(charges, dtype=np.float64)
        self.segids = None if segids is None else np.asarray(segids, dtype=str)
        self.n_atoms = len(self.names)
        self.n_frames = n_frames
        self._frames = frames

    def frames(self):
        return self._frames()


def is_gromacs_top(parm):
    """GROMACS .top files share their extension with Amber topologies (which start with %VERSION)."""
    extension = os.path.splitext(parm)[1].lower()
    if extension == ".itp":
        return True
    if extension != ".top":
        return False
    with open(parm) as f:
        for line in f:
            if line.strip():
                return not line.startswith(("%VERSION", "%FLAG"))
    return False


def gromacs_include_dir():
    """Directory with the GROMACS force fields, for #include statements in .top files."""
    candidates = os.environ.get("GMXLIB", "").split(os.pathsep)
    if os.environ.get("GMXDATA"):
        candidates.append(os.path.join(os.environ["GMXDATA"], "top"))
    for gmx in ("gmx", "gmx_mpi", "gmx_d"):
        executable = shutil.which(gmx)
        if executable:
            prefix = os.path.dirname(os.path.dirname(os.path.realpath(executable)))
            candidates.append(os.path.join(prefix, "share", "gromacs", "top"))
    candidates.append("/usr/local/gromacs/share/gromacs/top")
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def load_mdanalysis(parm, nc, gmx_include=None):
    import MDAnalysis as mda
    kwargs = {}
    if is_gromacs_top(parm):
        kwargs["topology_format"] = "ITP"
        include_dir = gmx_include or gromacs_include_dir()
        if include_dir:
            kwargs["include_dir"] = include_dir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        try:
            u = mda.Universe(parm, nc, **kwargs)
        except (IOError, ValueError, TypeError) as error:
            hint = ""
            if "topology_format" in kwargs and "Could not find" in str(error):
                hint = "\nSet the directory with the GROMACS force fields with -gmx_include."
            if "tpx version" in str(error):
                hint = (f"\nThis .tpr was written by a GROMACS version that MDAnalysis {mda.__version__} cannot read. "
                        "Either install the development version of MDAnalysis, from the FieldTools folder:\n"
                        "  pip install -e \".[mdanalysis-dev]\"\n"
                        "or directly:\n"
                        "  pip install \"git+https://github.com/MDAnalysis/mdanalysis.git@develop#subdirectory=package\"\n"
                        "or use a self-contained topology instead of the .tpr:\n"
                        "  gmx grompp -f md.mdp -c conf.gro -p topol.top -pp processed.top")
            sys.exit(f"Error! Could not read topology {parm} with trajectory {nc}:\n{error}{hint}")
    if not hasattr(u.atoms, "charges"):
        sys.exit(f"Error! {parm} contains no partial charges. Use a topology file such as .parm7, .tpr or .top.")
    atoms = u.atoms

    # MDAnalysis releases up to 2.10 return coordinates read from a .tpr file in nm instead of A
    version = tuple(int(v) for v in re.findall(r"\d+", mda.__version__)[:2])
    reads_tpr_coordinates = type(u.trajectory).__name__ == "TPRReader"   # MDAnalysis >= 2.10
    scale = 10.0 if reads_tpr_coordinates and version < (2, 11) else 1.0
    if scale != 1.0:
        print(f"Note: converting the coordinates of {nc} from nm to A (not done by MDAnalysis {mda.__version__}).")

    def frames():
        for ts in u.trajectory:
            box = None if ts.dimensions is None else np.array(ts.dimensions, dtype=np.float64)
            yield scale * ts.positions.astype(np.float64), box

    return System(atoms.names, atoms.resnames, atoms.resids.astype(str), atoms.charges,
                  len(u.trajectory), frames, atoms.segids)


def load_pytraj(parm, nc):
    import pytraj as pt
    if parm.lower().endswith(".tpr"):
        sys.exit("Error! pytraj cannot read GROMACS .tpr files; use -backend mdanalysis.")
    traj = pt.iterload(nc, parm)
    top = traj.topology
    residues = list(top.residues)
    atoms = list(top.atoms)

    def frames():
        for frame in traj:
            box = np.array(frame.box.values, dtype=np.float64)
            yield np.array(frame.xyz, dtype=np.float64), (box if box[:3].any() else None)

    return System([str(a.name).strip() for a in atoms],
                  [str(residues[a.resid].name).strip() for a in atoms],
                  [str(a.resid + 1) for a in atoms],
                  top.charge, traj.n_frames, frames)


def is_installed(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


def load_system(parm, nc, backend="auto", gmx_include=None):
    # Fall back to pytraj only if MDAnalysis is not installed, so that other errors are not hidden
    if backend == "mdanalysis" or (backend == "auto" and is_installed("MDAnalysis")):
        return load_mdanalysis(parm, nc, gmx_include)
    if not is_installed("pytraj"):
        sys.exit("Error! Neither MDAnalysis nor pytraj is installed (pip install mdanalysis).")
    return load_pytraj(parm, nc)


#####################################################################################
### Atom selection
#####################################################################################

SPECIFIER_HELP = "Use :<residue>@<atom> (e.g. :40@OG) or [<chain>/]<residue>/<atom> (e.g. 40/OG, A/40/OG)"


def parse_specifier(spec):
    """Split an atom specifier into (chain, residue, atom names).

    Amber style:  :<residue>[@<atom>[,<atom>...]]
    PyMOL style:  [<chain>/]<residue>[/<atom>[,<atom>...]]  (read from the right; a trailing
                  slash or a missing atom field selects the whole residue)
    """
    if spec.startswith(":"):
        residue, _, atoms = spec[1:].partition("@")
        chain = None
    else:
        fields = spec.lstrip("/").split("/")
        if len(fields) > 3:
            sys.exit(f"Error! Invalid atom specifier '{spec}'. {SPECIFIER_HELP}")
        if len(fields) == 3:
            chain, residue, atoms = fields
        else:
            chain, residue, atoms = None, fields[0], (fields[1] if len(fields) == 2 else "")
        chain = chain or None
    if residue == "" or "@" in residue or ":" in residue:
        sys.exit(f"Error! Invalid atom specifier '{spec}'. {SPECIFIER_HELP}")
    return chain, residue, [a for a in atoms.split(",") if a]


def chain_mask(chain, system):
    """Atoms of a chain: its GROMACS molecule (segment) name, or the name's last part after '_'."""
    segids = system.segids
    if segids is None or set(np.unique(segids)) <= {"SYSTEM", ""}:
        sys.exit(f"Error! Chain '{chain}' given, but this topology has no chain information "
                 "(Amber topologies do not store chains). Select residues by number instead.")
    names = np.array([re.sub(r"^seg_\d+_", "", seg) for seg in segids])
    mask = (segids == chain) | (names == chain) | np.char.endswith(names, "_" + chain)
    if not mask.any():
        available = ", ".join(dict.fromkeys(names))
        sys.exit(f"Error! No chain or molecule '{chain}' in the topology. Available: {available}")
    return mask


def select_atoms(spec, system):
    """Indices of the atoms matching an atom specifier (see parse_specifier)."""
    chain, residue, atom_names = parse_specifier(spec)
    selection = (system.resids == residue) if residue.isdigit() else (system.resnames == residue)
    if chain is not None:
        selection &= chain_mask(chain, system)
    if atom_names:
        selection &= np.isin(system.names, atom_names)
    indices = np.flatnonzero(selection)
    if len(indices) == 0:
        sys.exit(f"Error! No atom in the system matches '{spec}'.")
    return indices


def select_one_atom(spec, system):
    indices = select_atoms(spec, system)
    if len(indices) != 1:
        found = ", ".join(f":{system.resids[i]}@{system.names[i]} ({system.resnames[i]})" for i in indices[:10])
        sys.exit(f"Error! Target '{spec}' must match exactly one atom, but matches {len(indices)}: {found}")
    return int(indices[0])


def load_targets(target_file, system):
    """Targets from a target file, or from a list of lines (parameter file)."""
    targets = []
    lines = [line.strip() for line in target_file if line.strip()] if isinstance(target_file, list) \
        else read_lines(target_file)
    for line in lines:
        specs = line.split()
        if len(specs) > 2:
            sys.exit(f"Error! Target '{line}' has {len(specs)} atoms; use one (atom) or two (bond).")
        targets.append({
            "name": "_".join(specs),
            "kind": "point" if len(specs) == 1 else "bond",
            "atoms": [select_one_atom(spec, system) for spec in specs],
        })
    if not targets:
        sys.exit("Error! No targets defined.")
    return targets


def load_exclusions(exclude_file, targets, system):
    """Boolean mask (n_atoms) of excluded atoms for each target."""
    if isinstance(exclude_file, list):   # from a parameter file
        lines = [line.strip() for line in exclude_file if line.strip()]
    else:
        lines = read_lines(exclude_file) if exclude_file else []
    if lines in ([], ["False"]):
        lines = None
    elif len(lines) == 1:
        lines = lines * len(targets)
    elif len(lines) != len(targets):
        source = "The exclusion list" if isinstance(exclude_file, list) else exclude_file
        sys.exit(f"Error! {source} has {len(lines)} lines but there are {len(targets)} targets. "
                 "Use one line per target or a single line for all targets.")

    for i, target in enumerate(targets):
        excluded = np.zeros(system.n_atoms, dtype=bool)
        if lines is None:
            # Default: exclude the full residue of the first target atom
            excluded[system.resids == system.resids[target["atoms"][0]]] = True
        else:
            for spec in lines[i].split():
                excluded[select_atoms(spec, system)] = True
        if target["kind"] == "point":
            excluded[target["atoms"][0]] = True   # its own charge sits at r = 0
        target["excluded"] = excluded


def tip4p_oxygens(system):
    """Mask of the (uncharged) oxygens of 4-point waters; the extra point follows O, H1, H2."""
    extra_points = np.flatnonzero(np.isin(system.names, EXTRA_POINT_NAMES) & np.isin(system.resnames, WATER_NAMES))
    oxygens = extra_points - 3
    if len(extra_points) == 0 or np.any(oxygens < 0) or np.any(system.resids[oxygens] != system.resids[extra_points]):
        sys.exit("Error! -TIP4P True, but no water residues with atom order O, H1, H2, EPW (or OW, HW1, HW2, MW) were found.")
    mask = np.zeros(system.n_atoms, dtype=bool)
    mask[oxygens] = True
    return mask


#####################################################################################
### Charges
#####################################################################################

PROTEIN_CAPS = ["ACE", "NME", "NHE", "NH2"]


def is_amino_acid(system, first_atoms):
    """Per atom: True if its residue has backbone atoms N, CA and C, or is a terminal cap."""
    backbone = np.ones(system.n_atoms, dtype=bool)
    for name in ("N", "CA", "C"):
        has_atom = np.zeros(system.n_atoms, dtype=bool)
        has_atom[first_atoms[system.names == name]] = True   # flag on the residue's first atom
        backbone &= has_atom[first_atoms]
    return backbone | np.isin(system.resnames, PROTEIN_CAPS)


def charge_report(system, solvent, first_atoms):
    """Total charges of the system, its segments, the protein, hetero residues and solvent."""
    q = system.charges
    report = {"Total": float(q.sum()), "Segments": {}, "Groups": {}}
    if system.segids is not None and len(np.unique(system.segids)) > 1:
        for segid in dict.fromkeys(system.segids):
            report["Segments"][str(segid)] = float(q[system.segids == segid].sum())

    is_solvent = np.isin(system.resnames, solvent)
    is_protein = is_amino_acid(system, first_atoms) & ~is_solvent
    residue_starts = np.unique(first_atoms)
    if is_protein.any():
        report["Groups"]["Protein"] = (float(q[is_protein].sum()), int(is_protein[residue_starts].sum()), "")
    hetero = ~(is_protein | is_solvent)
    for resname in dict.fromkeys(system.resnames[hetero]):
        atoms = hetero & (system.resnames == resname)
        resids = system.resids[residue_starts[atoms[residue_starts]]]
        label = ("resid " + ",".join(resids)) if len(resids) <= 5 else f"resid {resids[0]}-{resids[-1]}"
        report["Groups"][str(resname)] = (float(q[atoms].sum()), len(resids), label)
    for resname in solvent:
        atoms = system.resnames == resname
        if atoms.any():
            report["Groups"][resname] = (float(q[atoms].sum()), int(atoms[residue_starts].sum()), "solvent")
    return report


def print_charge_report(report):
    print("\nCharges (e):")
    print(f"  {'System':<28} {report['Total']:+9.3f}")
    if abs(report["Total"] - round(report["Total"])) > 0.01:
        print("  Warning! The total charge of the system is not an integer.")
    for segid, charge in report["Segments"].items():
        print(f"  segment {segid:<20} {charge:+9.3f}")
    for name, (charge, n_residues, label) in report["Groups"].items():
        residues = f"{n_residues} residue{'s' if n_residues != 1 else ''}"
        print(f"  {name:<28} {charge:+9.3f}   {residues}{'  ' + label if label else ''}")
    print()


#####################################################################################
### Frame filters
#####################################################################################

def parse_range(text, kind):
    """'3.2' is a maximum distance or a minimum angle; 'MIN:MAX' is a range."""
    try:
        if ":" in text:
            low, high = (float(v) for v in text.split(":"))
        elif kind == "distance":
            low, high = 0.0, float(text)
        else:
            low, high = float(text), 180.0
    except ValueError:
        sys.exit(f"Error! Invalid {kind} cutoff '{text}'. Use a number or MIN:MAX, e.g. 3.2 or 2.5:3.2")
    if low > high:
        sys.exit(f"Error! Invalid {kind} range '{text}': MIN is larger than MAX.")
    return low, high


def load_filters(args, system):
    filters = []
    for kind, specs in [("distance", f) for f in args.filter_distance] + [("angle", f) for f in args.filter_angle]:
        *atoms, cutoff = specs
        low, high = parse_range(cutoff, kind)
        filters.append({"kind": kind, "label": "-".join(atoms), "low": low, "high": high,
                        "atoms": [select_one_atom(spec, system) for spec in atoms]})
    return filters


def measure(geometry_filter, positions, box):
    """Distance (A) or angle (degrees) of a filter in one frame."""
    def bond(i, j):
        vector = positions[j] - positions[i]
        return vector if box is None else minimum_image(vector[None, :], box)[0]

    atoms = geometry_filter["atoms"]
    if geometry_filter["kind"] == "distance":
        return float(np.linalg.norm(bond(atoms[0], atoms[1])))
    a, b = bond(atoms[1], atoms[0]), bond(atoms[1], atoms[2])
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def filter_statistics(filters, all_values):
    """Range of each filter's values over all frames and how many frames pass it on its own."""
    if not filters:
        return []
    values = np.array(all_values, dtype=np.float64).reshape(-1, len(filters))
    stats = []
    for i, f in enumerate(filters):
        column = values[:, i]
        stats.append({"kind": f["kind"], "label": f["label"], "low": f["low"], "high": f["high"],
                      "min": float(column.min()) if len(column) else None,
                      "max": float(column.max()) if len(column) else None,
                      "passed": int(((column >= f["low"]) & (column <= f["high"])).sum())})
    return stats


def print_filter_statistics(stats, n_frames):
    print(f"\nFilter values over all {n_frames} frames:")
    for s in stats:
        unit = "A" if s["kind"] == "distance" else "deg"
        if s["kind"] == "distance" and s["low"] == 0.0:
            condition = f"<= {s['high']:g}"
        elif s["kind"] == "angle" and s["high"] == 180.0:
            condition = f">= {s['low']:g}"
        else:
            condition = f"{s['low']:g} to {s['high']:g}"
        measured = "no frames" if s["min"] is None else f"{s['min']:8.3f} to {s['max']:8.3f} {unit:<3}"
        print(f"  {s['kind']:<8} {s['label']:<32} {measured}   (filter {condition}: passed by {s['passed']} frames)")


#####################################################################################
### Field components
#####################################################################################

def detect_solvent(system):
    """Common water and ion residue names present in the system, in topology order."""
    known = set(WATER_NAMES) | set(ION_NAMES)
    _, first = np.unique(system.resnames, return_index=True)
    return [str(name) for name in system.resnames[np.sort(first)] if name in known]


def field_components(system, solvent):
    """Component names and, per atom, the index of its residue component."""
    is_solvent = np.isin(system.resnames, solvent)
    components = ["Total", "Protein", "Solvent"] + list(solvent)
    index = {name: i for i, name in enumerate(components)}
    atom_component = np.empty(system.n_atoms, dtype=np.int64)
    for i in range(system.n_atoms):
        key = system.resnames[i] if is_solvent[i] else f"{system.resnames[i]}_{system.resids[i]}"
        if key not in index:
            index[key] = len(components)
            components.append(key)
        atom_component[i] = index[key]
    return components, atom_component, is_solvent


def decompose(values, atom_component, is_solvent, n_components):
    """Sum per-atom values (n, 3) into components (n_components, 3)."""
    result = np.empty((n_components, values.shape[1]))
    for j in range(values.shape[1]):
        result[:, j] = np.bincount(atom_component, weights=values[:, j], minlength=n_components)
    result[0] = values.sum(axis=0)
    result[1] = values[~is_solvent].sum(axis=0)
    result[2] = values[is_solvent].sum(axis=0)
    return result


#####################################################################################
### Physics
#####################################################################################

def box_matrix(dimensions):
    """Box vectors (rows) from [a, b, c, alpha, beta, gamma]."""
    a, b, c = dimensions[:3]
    alpha, beta, gamma = np.radians(dimensions[3:6])
    bx, by = b * np.cos(gamma), b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    return np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])


NEIGHBOUR_CELLS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.float64)


def minimum_image(vectors, box):
    """Shortest periodic image of each displacement vector."""
    inverse = np.linalg.inv(box)
    fractional = vectors @ inverse
    fractional -= np.round(fractional)
    wrapped = fractional @ box
    if not np.any(box[np.tril_indices(3, -1)]):
        return wrapped   # Rectangular box: wrapping gives the minimum image
    # Triclinic box (e.g. truncated octahedron, rhombic dodecahedron): every lattice vector is
    # at least as long as the smallest distance between opposite faces, so a wrapped vector no
    # longer than half of it is already the shortest image. Only longer ones can be shorter in
    # a neighbouring cell.
    half_width = 0.5 / np.linalg.norm(inverse, axis=0).max()
    far = np.flatnonzero(np.einsum("ij,ij->i", wrapped, wrapped) > half_width**2)
    if len(far):
        candidates = wrapped[far, None, :] + (NEIGHBOUR_CELLS @ box)[None, :, :]
        shortest = np.argmin(np.einsum("ijk,ijk->ij", candidates, candidates), axis=1)
        wrapped[far] = candidates[np.arange(len(far)), shortest]
    return wrapped


def residue_first_atoms(system):
    """Index of the first atom of its residue, for every atom."""
    new_residue = np.r_[True, (system.resids[1:] != system.resids[:-1]) |
                              (system.resnames[1:] != system.resnames[:-1])]
    starts = np.flatnonzero(new_residue)
    return np.repeat(starts, np.diff(np.r_[starts, system.n_atoms]))


class Residues:
    """Residue membership of the atoms, for moving whole residues across the periodic boundaries."""

    def __init__(self, first_atoms):
        self.first_atoms = first_atoms                          # per atom: first atom of its residue
        self.starts = np.unique(first_atoms)                    # first atom of each residue
        self.index = np.searchsorted(self.starts, first_atoms)  # per atom: index of its residue


def make_whole(positions, residues, box):
    """Join residues broken over the periodic boundaries."""
    first = positions[residues.first_atoms]
    return first + minimum_image(positions - first, box)


def displacements(point, positions, box=None, residues=None):
    """Vectors pointing from every atom to `point`.

    With a box, every (whole) residue is moved as a unit to the periodic image
    whose first atom is closest to `point`, so no molecule is split.
    """
    vectors = point - positions
    if box is None:
        return vectors
    reference = vectors[residues.starts]
    shift = minimum_image(reference, box) - reference
    return vectors + shift[residues.index]


def field_vectors(vectors, charges):
    """Field (MV/cm) at the end of each vector from a charge at its start: E = k*q*r_vec/r^3."""
    r = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    return FIELD_CONST * (charges / r**3)[:, None] * vectors


def coulomb_energies(vectors, target_charge, charges):
    """Coulomb energy (kJ/mol) of a target charge with each charge at distance |vector|: U = k*q*Q/r."""
    r = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    return ENERGY_CONST * target_charge * charges / r


#####################################################################################
### QM charges
#####################################################################################

def load_qm_charges(qm_charges_file, qm_dict_file, system):
    """Per-frame QM charges and the atom index of each QM atom."""
    frames, current = {}, None
    with open(qm_charges_file) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Frame":
                current = int(parts[1])
                frames[current] = []
            elif current is not None:
                frames[current].append(float(parts[2]))

    keys = {}
    for i in range(system.n_atoms):
        keys.setdefault(f"{system.names[i]}_{system.resnames[i]}_{system.resids[i]}", i)
    qm_index = []
    for key in read_lines(qm_dict_file):
        if key not in keys:
            sys.exit(f"Error! QM atom '{key}' from {qm_dict_file} not found in the topology.")
        qm_index.append(keys[key])
    return frames, np.array(qm_index, dtype=np.int64)


def qm_frame_charges(charges, qm_frames, qm_index, frame_i):
    frame_number = frame_i + 1
    if frame_number not in qm_frames:
        sys.exit(f"Error! No QM charges for frame {frame_number}.")
    qm = qm_frames[frame_number]
    if len(qm) < len(qm_index):
        sys.exit(f"Error! Frame {frame_number} has {len(qm)} QM charges, but the QM dict has {len(qm_index)} atoms.")
    charges = charges.copy()
    charges[qm_index] = qm[:len(qm_index)]   # Link atom charges (after the QM atoms) are ignored
    return charges


#####################################################################################
### Main
#####################################################################################

def calculate_fields(args):
    system = load_system(args.parm, args.nc, args.backend, args.gmx_include)
    if args.solvent is None:
        args.solvent = detect_solvent(system)
        print("-solvent        Detected solvent     : ", ",".join(args.solvent) or "none")
    first_atoms = residue_first_atoms(system)
    residues = Residues(first_atoms)
    charges_summary = charge_report(system, args.solvent, first_atoms)
    print_charge_report(charges_summary)
    targets = load_targets(args.target, system)
    load_exclusions(args.exclude_atoms, targets, system)
    filters = load_filters(args, system)
    components, atom_component, is_solvent = field_components(system, args.solvent)
    n_components = len(components)

    ignored = tip4p_oxygens(system) if args.TIP4P else np.zeros(system.n_atoms, dtype=bool)
    if args.use_qm_charges:
        qm_frames, qm_index = load_qm_charges(args.qm_charges, args.qm_dict, system)

    if args.verbose:
        print(f"Atoms: {system.n_atoms}  Frames: {system.n_frames}  Components: {n_components}")
        for target in targets:
            atoms = ", ".join(f"{system.resnames[i]} {system.resids[i]} {system.names[i]} (index {i})"
                              for i in target["atoms"])
            print(f"Target {target['name']} ({target['kind']}): {atoms}; "
                  f"{int(target['excluded'].sum())} atoms excluded")

        for f in filters:
            print(f"Filter {f['kind']} {f['label']}: {f['low']:g} to {f['high']:g}, atoms {f['atoms']}")

    fields = {t["name"]: [] for t in targets}
    vectors = {t["name"]: [] for t in targets}
    energies = {t["name"]: [] for t in targets}
    kept_frames, filter_values = [], []
    all_values = []   # every frame's filter values, to report their ranges

    for frame_i, (positions, box) in enumerate(system.frames()):
        if args.pbc:
            if box is None:
                sys.exit("Error! -pbc True, but the trajectory has no box information.")
            box = box_matrix(box)
            positions = make_whole(positions, residues, box)
        else:
            box = None

        values = [measure(f, positions, box) for f in filters]
        all_values.append(values)
        if not all(f["low"] <= v <= f["high"] for f, v in zip(filters, values)):
            continue
        kept_frames.append(frame_i + 1)
        filter_values.append(values)

        charges = system.charges
        if args.use_qm_charges:
            charges = qm_frame_charges(charges, qm_frames, qm_index, frame_i)

        for target in targets:
            name, atoms = target["name"], target["atoms"]
            included = ~(target["excluded"] | ignored)
            q = charges[included]
            comp, solv = atom_component[included], is_solvent[included]

            if target["kind"] == "point":
                point = positions[atoms[0]]
            else:
                bond = positions[atoms[1]] - positions[atoms[0]]
                if box is not None:
                    bond = minimum_image(bond[None, :], box)[0]
                point = positions[atoms[0]] + bond / 2
                unit = bond / np.linalg.norm(bond)

            r_vectors = displacements(point, positions, box, residues)[included]
            field = decompose(field_vectors(r_vectors, q), comp, solv, n_components)
            vectors[name].append(field)
            fields[name].append(np.linalg.norm(field, axis=1) if target["kind"] == "point" else field @ unit)

            if args.energy_out:
                energy = np.zeros(len(q))
                for atom in atoms:
                    # The target atoms do not interact with themselves
                    other = np.flatnonzero(included) != atom
                    r_atom = displacements(positions[atom], positions, box, residues)[included]
                    energy[other] += coulomb_energies(r_atom[other], charges[atom], q[other])
                energies[name].append(decompose(energy[:, None], comp, solv, n_components)[:, 0])
            else:
                energies[name].append(np.zeros(n_components))

            if args.verbose:
                print(f"Frame {frame_i + 1} {name}: Total {fields[name][-1][0]:.2f} MV/cm")

    filter_stats = filter_statistics(filters, all_values)
    if filters:
        print_filter_statistics(filter_stats, len(all_values))
        print(f"Frames passing the filters: {len(kept_frames)} of {system.n_frames}")
        if not kept_frames:
            print("Warning! No frame passes the filters; the output contains no data.")

    def stack(data, shape):
        return {name: np.array(data[name]).reshape((-1,) + shape) for name in data}

    fields, energies = stack(fields, (n_components,)), stack(energies, (n_components,))
    vectors = stack(vectors, (n_components, 3))
    as_dict = lambda data: {name: {c: data[name][:, i].tolist() for i, c in enumerate(components)}
                            for name in data}
    result = {"fields": as_dict(fields), "energies": as_dict(energies),
              "vectors": {name: {c: vectors[name][:, i] for i, c in enumerate(components)}
                          for name in vectors},
              "frames": kept_frames, "filters": filters, "filter_values": filter_values,
              "filter_stats": filter_stats,
              "charges": charges_summary}
    return result


def write_frames(path, result):
    """Kept frame numbers and the measured filter values."""
    with open(path, "w") as f:
        f.write("# frame " + " ".join(f"{g['kind']}:{g['label']}" for g in result["filters"]) + "\n")
        for frame, values in zip(result["frames"], result["filter_values"]):
            f.write(f"{frame:7d} " + " ".join(f"{v:10.3f}" for v in values) + "\n")


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    if args.write_parameter_file:
        write_parameter_file(args.write_parameter_file, args)
        print("Parameter file written to : ", args.write_parameter_file)
        return None

    show = lambda value: " | ".join(value) if isinstance(value, list) else value
    if args.parameter_file:
        print("--parameter-file                     : ", args.parameter_file)
    print("-nc             Trajectory file      : ", args.nc)
    print("-parm           Parameter file       : ", args.parm)
    print("-target         Field target         : ", show(args.target))
    print("-out            Output file          : ", args.out)
    print("-solvent        Non-protein residues : ", ",".join(args.solvent) if args.solvent else "auto")
    print("-exclude_atoms  Atoms excluded       : ",
          show(args.exclude_atoms) or "Full residue of the first atom of each target")
    for f in args.filter_distance:
        print("-filter_distance: ", " ".join(f))
    for f in args.filter_angle:
        print("-filter_angle   : ", " ".join(f))
    for flag in ("gmx_include", "TIP4P", "pbc", "energy_out", "vector_out", "use_qm_charges", "qm_charges", "qm_dict"):
        if getattr(args, flag):
            print(f"-{flag:<15}: ", getattr(args, flag))
    print("\nField calculation RUNNING : ", datetime.datetime.now())

    result = calculate_fields(args)

    with open(args.out, "wb") as f:
        pickle.dump(result["fields"], f)
    if args.energy_out:
        with open(args.energy_out, "wb") as f:
            pickle.dump(result["energies"], f)
    if args.vector_out:
        with open(args.vector_out, "wb") as f:
            pickle.dump(result["vectors"], f)
    if result["filters"]:
        frames_file = os.path.splitext(args.out)[0] + "_frames.dat"
        write_frames(frames_file, result)
        print("Kept frames written to : ", frames_file)
    print("\nField calculation DONE : ", datetime.datetime.now())
    return result


def cli():
    """Entry point of the fieldtools command."""
    main()


def python_main(args):
    """Run from Python with a sys.argv-like list (args[0] is the program name)."""
    return main(list(args[1:]))


if __name__ == "__main__":
    main()
