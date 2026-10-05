"""Parameter files for fieldtools (TOML format).

A parameter file holds the options of a run in sections. Command-line options override
the values in the file. Relative paths are relative to the folder of the parameter file.
Run `fieldtools --write-parameter-file params.toml` for a commented template.
"""

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:   # Python < 3.11
    import tomli as tomllib

PATH, BOOL, TEXT, LIST = "path", "bool", "text", "list"

# section -> {key: (argparse destination, kind, description)}
SCHEMA = {
    "input": {
        "top": ("parm", PATH, "topology file with charges (.parm7, .tpr, .top)"),
        "traj": ("nc", PATH, "trajectory file (.nc, .xtc, .trr, ...)"),
        "backend": ("backend", TEXT, "auto, mdanalysis or pytraj"),
        "gmx_include": ("gmx_include", PATH, "folder with the GROMACS force fields #included by a .top"),
    },
    "targets": {
        "targets": ("target", LIST, "one target per entry: one atom (point) or two atoms (bond)"),
        "exclude": ("exclude_atoms", LIST, "excluded atoms: one entry per target, or one entry for all"),
        "target_file": ("target", PATH, "instead of targets: a target file"),
        "exclude_file": ("exclude_atoms", PATH, "instead of exclude: an exclusion file"),
    },
    "system": {
        "solvent": ("solvent", TEXT, "solvent residue names (\"SOL,NA,CL\" or a list), or auto"),
        "tip4p": ("TIP4P", BOOL, "4-point water model"),
        "pbc": ("pbc", BOOL, "make residues whole and use the periodic image closest to the target"),
    },
    "filters": {
        "distance": ("filter_distance", LIST, "\"ATOM1 ATOM2 CUTOFF\" entries: distance <= CUTOFF (A) or MIN:MAX"),
        "angle": ("filter_angle", LIST, "\"ATOM1 ATOM2 ATOM3 CUTOFF\" entries: angle >= CUTOFF (deg) or MIN:MAX"),
    },
    "output": {
        "out": ("out", PATH, "fields (.pkl)"),
        "energy_out": ("energy_out", PATH, "Coulomb energies (.pkl)"),
        "vector_out": ("vector_out", PATH, "field vectors (.pkl)"),
        "prefix": ("prefix", TEXT, "added with _ to the names of all output files (out defaults to field.pkl); also writes <prefix>.log"),
    },
    "qm": {
        "use_qm_charges": ("use_qm_charges", BOOL, "replace charges with per-frame QM charges"),
        "qm_charges": ("qm_charges", PATH, "QM charges file from qmchargestools"),
        "qm_dict": ("qm_dict", PATH, "QM dict file from qmchargestools"),
    },
    "run": {
        "verbose": ("verbose", BOOL, "display additional information"),
    },
}
ALIASES = {("input", "topology"): "top", ("input", "parm"): "top", ("input", "trajectory"): "traj",
           ("input", "nc"): "traj"}
FILTER_TOKENS = {"filter_distance": 3, "filter_angle": 4}


def error(message):
    sys.exit(f"Error in parameter file! {message}")


def load_parameter_file(path):
    """Values of a parameter file as {argparse destination: value}."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        sys.exit(f"Error! Parameter file {path} not found.")
    except tomllib.TOMLDecodeError as decode_error:
        error(f"{path} is not valid TOML: {decode_error}")

    folder = os.path.dirname(os.path.abspath(path))
    values, origin = {}, {}
    for section, entries in data.items():
        if section.lower() not in SCHEMA:
            error(f"unknown section [{section}]. Valid sections: {', '.join(f'[{s}]' for s in SCHEMA)}")
        if not isinstance(entries, dict):
            error(f"'{section}' must be a section written as [{section}].")
        section = section.lower()
        for key, value in entries.items():
            name = ALIASES.get((section, key.lower()), key.lower())
            if name not in SCHEMA[section]:
                error(f"unknown key '{key}' in [{section}]. Valid keys: {', '.join(SCHEMA[section])}")
            dest, kind, _ = SCHEMA[section][name]
            if dest in values:
                error(f"'{name}' in [{section}] sets the same option as '{origin[dest]}'; use only one of them.")
            values[dest], origin[dest] = convert(value, kind, f"[{section}] {name}", folder), name

    if "solvent" in values and isinstance(values["solvent"], list):
        values["solvent"] = ",".join(values["solvent"])
    for dest, n_tokens in FILTER_TOKENS.items():
        if dest in values:
            values[dest] = [entry.split() for entry in values[dest]]
            for tokens in values[dest]:
                if len(tokens) != n_tokens:
                    error(f"filter '{' '.join(tokens)}' must have {n_tokens} parts: "
                          f"{n_tokens - 1} atoms and a cutoff.")
    return values


def convert(value, kind, name, folder):
    if kind == PATH:
        if value is False or value in ("", "False", "None"):
            return None
        if not isinstance(value, str):
            error(f"{name} must be a file name in quotes.")
        return value if os.path.isabs(value) else os.path.normpath(os.path.join(folder, value))
    if kind == BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        error(f"{name} must be true or false.")
    if kind == LIST:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            error(f"{name} must be a string or a list of strings.")
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    if not isinstance(value, str):
        error(f"{name} must be text in quotes.")
    return value


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[\n" + "".join(f"    {toml_value(v)},\n" for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_parameter_file(path, args):
    """Write the options of a run (args from argparse) as a commented parameter file."""
    lines = ["# FieldTools parameter file. Run with: fieldtools --parameter-file " + os.path.basename(path),
             "# Command-line options override these values. Relative paths are relative to this file.",
             "# Lines starting with # are comments; remove the # to use an option.", ""]
    for section, entries in SCHEMA.items():
        lines.append(f"[{section}]")
        for name, (dest, kind, description) in entries.items():
            value = getattr(args, dest, None)
            if dest == "solvent" and value is None:
                value = "auto"
            elif dest == "solvent":
                value = ",".join(value)
            if dest in FILTER_TOKENS:
                value = [" ".join(tokens) for tokens in value] if value else None
            if name.endswith("_file") or name in ("targets", "exclude"):
                is_list = isinstance(value, list)
                if (name in ("targets", "exclude")) != is_list:
                    value = None   # written under the other key of the pair
            if kind == PATH and isinstance(value, str):
                value = portable_path(value, path)
            lines.append(f"# {description}")
            if value is None or value == []:
                lines.append(f"# {name} = {example(name, kind)}")
            else:
                lines.append(f"{name} = {toml_value(value)}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def portable_path(value, parameter_file):
    """Relative to the parameter file if the file is inside its folder, absolute otherwise."""
    absolute = os.path.abspath(value)
    relative = os.path.relpath(absolute, os.path.dirname(os.path.abspath(parameter_file)))
    return absolute if relative.startswith("..") else relative


def example(name, kind):
    examples = {"top": '"topol.tpr"', "traj": '"traj.xtc"', "gmx_include": '"/usr/share/gromacs/top"',
                "targets": '["40/C7 40/O71", "40/O71"]', "exclude": '["40 264"]',
                "target_file": '"target.dat"', "exclude_file": '"exclude.dat"',
                "distance": '["IAA/O1 SAM/CE 3.2"]', "angle": '["IAA/O1 SAM/CE SAM/SD 160"]',
                "out": '"field.pkl"', "energy_out": '"energy.pkl"', "vector_out": '"vectors.pkl"',
                "prefix": '"run1"', "qm_charges": '"qm_charges.dat"', "qm_dict": '"qm.dict"'}
    return examples.get(name, "true" if kind == BOOL else '""')
