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
        # further groups: [[filters.or]] tables with distance/angle keys (see load_or_groups)
    },
    "output": {
        "out": ("out", PATH, "fields (.pkl)"),
        "energy_out": ("energy_out", PATH, "Coulomb energies (.pkl)"),
        "vector_out": ("vector_out", PATH, "field vectors (.pkl)"),
        "frames_all": ("frames_all", BOOL, "list all frames in <out>_frames.dat (kept column 1/0), not only kept ones"),
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
            if section == "filters" and key.lower() == "or":   # [[filters.or]] tables: alternative filter groups
                values["filter_or"] = load_or_groups(value)
                continue
            name = ALIASES.get((section, key.lower()), key.lower())
            if name not in SCHEMA[section]:
                error(f"unknown key '{key}' in [{section}]. Valid keys: {', '.join(SCHEMA[section])}")
            dest, kind, _ = SCHEMA[section][name]
            if dest in values:
                error(f"'{name}' in [{section}] sets the same option as '{origin[dest]}'; use only one of them.")
            values[dest], origin[dest] = convert(value, kind, f"[{section}] {name}", folder), name

    if "solvent" in values and isinstance(values["solvent"], list):
        values["solvent"] = ",".join(values["solvent"])
    # Filter groups: [filters] is the first group, each [[filters.or]] table another one
    first = [(KIND[dest], tokens) for dest in FILTER_TOKENS for tokens in split_filters(values.pop(dest, []), dest)]
    groups = ([first] if first else []) + values.pop("filter_or", [])
    if groups:
        values["filter_groups"] = groups
    return values


KIND = {"filter_distance": "distance", "filter_angle": "angle"}


def split_filters(entries, dest):
    """'ATOM1 ATOM2 CUTOFF' strings -> token lists, checking the number of parts."""
    n_tokens = FILTER_TOKENS[dest]
    tokens = [entry.split() for entry in entries]
    for t in tokens:
        if len(t) != n_tokens:
            error(f"filter '{' '.join(t)}' must have {n_tokens} parts: {n_tokens - 1} atoms and a cutoff.")
    return tokens


def load_or_groups(tables):
    """[[filters.or]] tables -> list of filter groups [(kind, tokens), ...]."""
    if isinstance(tables, dict):
        tables = [tables]
    if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
        error("alternative filter groups must be written as [[filters.or]] tables.")
    groups = []
    for table in tables:
        group = []
        for key, value in table.items():
            name = key.lower()
            if name not in ("distance", "angle"):
                error(f"unknown key '{key}' in [[filters.or]]. Valid keys: distance, angle")
            dest = SCHEMA["filters"][name][0]
            group += [(name, t) for t in split_filters(convert(value, LIST, f"[[filters.or]] {name}", ""), dest)]
        if not group:
            error("a [[filters.or]] table has no distance or angle filters.")
        groups.append(group)
    return groups


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
            if dest in FILTER_TOKENS:   # the first filter group
                groups = getattr(args, "filter_groups", None) or [[]]
                value = [" ".join(t) for kind, t in groups[0] if kind == KIND[dest]] or None
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
        if section == "filters":   # further filter groups, combined with OR
            groups = getattr(args, "filter_groups", None) or []
            if len(groups) < 2:
                lines += ["# Alternative filter group, combined with OR (a frame is used if it passes all filters",
                          "# of [filters] or all filters of a [[filters.or]] table; repeat the table for more groups):",
                          "# [[filters.or]]", '# distance = ["IAA/O2 SAM/CE 3.2"]',
                          '# angle = ["IAA/O2 SAM/CE SAM/SD 160"]', ""]
            for i, group in enumerate(groups[1:], start=2):
                lines.append(f"# alternative filter group {i} (a frame is used if it passes all filters of any group)")
                lines.append("[[filters.or]]")
                for kind in ("distance", "angle"):
                    entries = [" ".join(t) for k, t in group if k == kind]
                    if entries:
                        lines.append(f"{kind} = {toml_value(entries)}")
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
