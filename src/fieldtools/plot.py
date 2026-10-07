"""fieldplot: plots of FieldTools results.

Bar plot of the field contribution of every residue (mean over the trajectory, with the
standard deviation as error bars), from the .pkl files written by fieldtools.

Run `fieldplot -h` for usage.
"""

import argparse
import csv
import os
import pickle
import re
import sys

import numpy as np

from . import __version__

SUMMARY = ("Total", "Protein", "Solvent")

# Colors of the validated reference palette (categorical slots 1 and 2, text and surface tokens)
COLOR_BAR = "#2a78d6"
COLOR_HIGHLIGHT = "#eb6834"
HIGHLIGHT_BAND = "#fbe1d6"   # light tint of the highlight color, behind highlighted residues
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
LABELS = ("Highlighted residues", "Other residues")   # legend labels: highlighted, other


#####################################################################################
### Data
#####################################################################################

def parse_residue_list(text):
    """'40,70,230-237' -> {40, 70, 230, ..., 237}"""
    residues = set()
    for part in (p.strip() for p in text.split(",") if p.strip()):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            sys.exit(f"Error! Invalid residue list '{text}'. Use numbers and ranges, e.g. 40,70,230-237")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first:
            sys.exit(f"Error! Invalid residue range '{part}'.")
        residues.update(range(first, last + 1))
    return residues


def residue_components(target_fields):
    """(residue number, residue name, component key) of every residue component, in output order."""
    rows = []
    for key in target_fields:
        match = re.fullmatch(r"(.+)_(\d+)", key)
        if match and key not in SUMMARY:
            rows.append((int(match[2]), match[1], key))
    return rows


def frame_values(fields, vectors, target, component):
    """Per-frame values of a component.

    Bond targets: the projected field. Point targets: the field magnitude, or, with the
    vectors, the field projected onto the direction of the total field in each frame,
    which is signed and additive over residues.
    """
    if vectors is not None and is_point_target(target):
        total = np.asarray(vectors[target]["Total"], dtype=float)
        norm = np.linalg.norm(total, axis=1, keepdims=True)
        unit = np.divide(total, norm, out=np.zeros_like(total), where=norm > 0)
        return np.einsum("ij,ij->i", np.asarray(vectors[target][component], dtype=float), unit)
    return np.asarray(fields[target][component], dtype=float)


def display_name(target):
    """'40/C7_40/O71' -> '40/C7 → 40/O71' (the arrow points in the positive direction)."""
    return target.replace("_", " → ") if not is_point_target(target) else target


def is_point_target(target):
    return "_" not in target


def statistics(values):
    """Mean and standard deviation over frames."""
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def residue_table(fields, vectors, target, residues=None, highlight=()):
    rows = []
    for resid, resname, key in residue_components(fields[target]):
        if residues is not None and resid not in residues:
            continue
        mean, sd = statistics(frame_values(fields, vectors, target, key))
        rows.append({"resid": resid, "resname": resname, "mean": mean, "sd": sd,
                     "highlighted": resid in highlight})
    return rows


def axis_label(target, vectors):
    if not is_point_target(target):
        return "Field along the target bond (MV/cm)"
    if vectors is not None:
        return "Field along the total field (MV/cm)"
    return "Field magnitude |E| (MV/cm)"


#####################################################################################
### Plot
#####################################################################################

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def plot_target(ax, fields, vectors, target, rows, n_frames, labels=LABELS):
    from matplotlib.patches import Patch

    style_axes(ax)
    x = np.array([r["resid"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    sds = np.array([r["sd"] for r in rows])
    colors = [COLOR_HIGHLIGHT if r["highlighted"] else COLOR_BAR for r in rows]
    bars = ax.bar(x, means, width=0.8, color=colors, linewidth=0, zorder=2,
                  yerr=sds, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 0.7, "capsize": 0, "zorder": 3})
    ax.axhline(0, color=TEXT_PRIMARY, linewidth=0.8, zorder=4)
    for row in rows:   # a band, so that highlighted residues with small values can be found
        if row["highlighted"]:
            ax.axvspan(row["resid"] - 0.5, row["resid"] + 0.5, color=HIGHLIGHT_BAND, linewidth=0, zorder=1)
    if len(x):
        ax.set_xlim(x.min() - 1, x.max() + 1)

    # Direct labels for the highlighted residues, beyond the end of the error bar. Labels of
    # neighbouring highlighted residues on the same side are spread sideways so they do not overlap.
    span = np.nanmax(np.abs(means) + sds) if len(x) else 1.0
    labelled = [r for r in rows if r["highlighted"]]
    for row, shift in label_shifts(rows):
        up = row["mean"] >= 0
        end = row["mean"] + (1 if up else -1) * row["sd"]
        ax.annotate(f"{row['resname']} {row['resid']}", (row["resid"], end),
                    xytext=(shift * 9, 3 if up else -3), textcoords="offset points",
                    ha="center", va="bottom" if up else "top",
                    rotation=90, fontsize=7, color=TEXT_PRIMARY, zorder=5)
    room = 1.45 if labelled else 1.15   # headroom for the labels
    ax.set_ylim(-room * span if (means < 0).any() else -0.05 * span, room * span if (means > 0).any() else 0.05 * span)

    summary = []
    for component in SUMMARY:
        if component in fields[target]:
            mean, sd = statistics(frame_values(fields, vectors, target, component))
            summary.append(f"{component} {mean:.1f} ± {sd:.1f}")
    ax.set_title(display_name(target), loc="left", fontsize=11, color=TEXT_PRIMARY, pad=18)
    ax.text(0, 1.02, f"Mean ± SD over {n_frames} frames  ·  " + "  ·  ".join(summary) + " MV/cm",
            transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Residue number", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(axis_label(target, vectors), fontsize=9, color=TEXT_SECONDARY)
    if any(r["highlighted"] for r in rows):
        ax.legend(handles=[Patch(color=COLOR_HIGHLIGHT, label=labels[0]),
                           Patch(color=COLOR_BAR, label=labels[1])],
                  loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    return bars


def label_shifts(rows):
    """(row, sideways shift in label widths) for each highlighted residue.

    Labels of highlighted residues within 3 residues of each other, on the same side of zero,
    are spread sideways around their bars so they do not overlap.
    """
    groups = []
    for row in (r for r in rows if r["highlighted"]):
        up = row["mean"] >= 0
        if groups and groups[-1][-1]["resid"] >= row["resid"] - 3 and (groups[-1][-1]["mean"] >= 0) == up:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [(row, i - (len(group) - 1) / 2) for group in groups for i, row in enumerate(group)]


def add_hover(fig, plotted):
    """Tooltip with residue, mean and SD when the mouse is over a bar (interactive backends)."""
    tooltips = {}
    for ax, rows, _ in plotted:
        tooltip = ax.annotate("", (0, 0), xytext=(8, 8), textcoords="offset points", fontsize=8,
                              color=TEXT_PRIMARY, zorder=10,
                              bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": GRID})
        tooltip.set_visible(False)
        tooltips[ax] = tooltip

    def on_move(event):
        changed = False
        for ax, rows, bars in plotted:
            tooltip = tooltips[ax]
            hit = None
            if event.inaxes is ax and event.xdata is not None:
                for row, bar in zip(rows, bars):
                    if bar.contains(event)[0] or abs(event.xdata - row["resid"]) < 0.5 and \
                            min(0, row["mean"]) - row["sd"] <= event.ydata <= max(0, row["mean"]) + row["sd"]:
                        hit = row
                        break
            if hit:
                tooltip.xy = (hit["resid"], hit["mean"])
                tooltip.set_text(f"{hit['resname']} {hit['resid']}\nmean {hit['mean']:.2f} MV/cm\n"
                                 f"SD {hit['sd']:.2f} MV/cm")
            if tooltip.get_visible() != bool(hit) or hit:
                tooltip.set_visible(bool(hit))
                changed = True
        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def make_figure(fields, vectors, targets, residues, highlight, labels=LABELS):
    import matplotlib.pyplot as plt

    tables = {t: residue_table(fields, vectors, t, residues, highlight) for t in targets}
    n_residues = max((len(rows) for rows in tables.values()), default=1)
    width = min(max(8.0, 2.5 + 0.045 * n_residues), 30.0)
    fig, axes = plt.subplots(len(targets), 1, figsize=(width, 4.2 * len(targets)), squeeze=False,
                             facecolor=SURFACE, layout="constrained")
    plotted = []
    for ax, target in zip(axes[:, 0], targets):
        n_frames = len(fields[target]["Total"])
        bars = plot_target(ax, fields, vectors, target, tables[target], n_frames, labels)
        plotted.append((ax, tables[target], bars))
    add_hover(fig, plotted)
    return fig, tables


def write_csv(path, tables):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "resid", "resname", "mean_MV_cm", "sd_MV_cm", "highlighted"])
        for target, rows in tables.items():
            for r in rows:
                writer.writerow([target, r["resid"], r["resname"], f"{r['mean']:.4f}", f"{r['sd']:.4f}",
                                 int(r["highlighted"])])


#####################################################################################
### Main
#####################################################################################

def build_parser():
    parser = argparse.ArgumentParser(
        prog="fieldplot", allow_abbrev=False,
        description="Bar plot of the field contribution of every residue: mean over the trajectory, "
                    "with the standard deviation as error bars.",
        epilog="Examples:\n"
               "  fieldplot field.pkl -highlight 70,130,234-237 -out residues.png\n"
               "  fieldplot field.pkl -vectors vectors.pkl -target 40/O71 -out point.pdf\n"
               "  fieldplot field.pkl -webagg -port 8988      (then open http://127.0.0.1:8988)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"FieldTools {__version__}")
    parser.add_argument("fields", help="fields file (.pkl) written by fieldtools")
    parser.add_argument("-target", action="append", default=[],
                        help="target to plot, as named in the fields file (repeat for several) [default: all]")
    parser.add_argument("-highlight", default="",
                        help="residue numbers whose bars are colored differently, e.g. 70,130,234-237")
    parser.add_argument("-highlight_label", default="Highlighted residues",
                        help="legend label of the highlighted residues, e.g. \"Mutants\" "
                             "[default: Highlighted residues]")
    parser.add_argument("-other_label", default="Other residues",
                        help="legend label of the other residues [default: Other residues]")
    parser.add_argument("-residues", default=None,
                        help="only plot these residue numbers, e.g. 1-305")
    parser.add_argument("-vectors", default=None,
                        help="vectors file (.pkl) from fieldtools -vector_out: point targets are then "
                             "plotted as projections onto the total field (signed and additive) "
                             "instead of magnitudes")
    parser.add_argument("-out", default=None,
                        help="image file (.png, .pdf, .svg, ...) [default: <fields>_residues.png, "
                             "unless -webagg is used]")
    parser.add_argument("-dpi", type=int, default=200, help="resolution of raster images [default: 200]")
    parser.add_argument("-csv", default=None, help="also write the plotted values to this CSV file")
    parser.add_argument("-webagg", action="store_true",
                        help="serve the interactive plot on a web server (WebAgg) instead of only "
                             "saving it; no browser is opened. Stop with Ctrl+C")
    parser.add_argument("-port", type=int, default=8988, help="port of the web server [default: 8988]")
    parser.add_argument("-host", default="0.0.0.0",
                        help="address the web server listens on [default: 0.0.0.0, reachable from other "
                             "machines on the network; 127.0.0.1 for this computer only]")
    return parser


def check_port(host, port):
    """Stop with a clear message if the web server cannot listen on host:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test:
        try:
            test.bind((host, port))
        except OSError as error:
            sys.exit(f"Error! Cannot serve on {host}:{port} ({error.strerror}). Choose another port with -port.")


def local_ip():
    """IP address of this machine on the network (no data is sent)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return None


def server_links(host, port):
    """Links to open the plot: with the machine's name and IP address when listening on all interfaces."""
    import socket
    if host not in ("0.0.0.0", "", "::"):
        return [f"http://{host}:{port}"]
    fqdn, name = socket.getfqdn().rstrip("."), socket.gethostname()
    links = [f"http://{fqdn if '.' in fqdn else name}:{port}"]
    ip = local_ip()
    if ip and not ip.startswith("127."):
        links.append(f"http://{ip}:{port}")
    return links


class DropLines:
    """Stream that hides lines starting with a given text (matplotlib's own, unusable 0.0.0.0 link)."""

    def __init__(self, stream, prefix):
        self.stream, self.prefix, self.dropped = stream, prefix, False

    def write(self, text):
        if text.startswith(self.prefix):
            self.dropped = not text.endswith("\n")   # print() sends the newline separately
        elif self.dropped and text == "\n":
            self.dropped = False
        else:
            self.dropped = False
            self.stream.write(text)

    def flush(self):
        self.stream.flush()


def load_pickle(path, kind):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        sys.exit(f"Error! {kind} file {path} not found.")


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    fields = load_pickle(args.fields, "Fields")
    vectors = load_pickle(args.vectors, "Vectors") if args.vectors else None

    targets = args.target or list(fields)
    for target in targets:
        if target not in fields:
            sys.exit(f"Error! Target '{target}' is not in {args.fields}. Available: {', '.join(fields)}")
        if vectors is not None and target not in vectors:
            sys.exit(f"Error! Target '{target}' is not in the vectors file {args.vectors}.")
        if len(fields[target]["Total"]) == 0:
            sys.exit(f"Error! Target '{target}' has no frames (did any frame pass the filters?).")
    residues = parse_residue_list(args.residues) if args.residues else None
    highlight = parse_residue_list(args.highlight) if args.highlight else set()

    import matplotlib
    if args.webagg:
        matplotlib.use("WebAgg")
        matplotlib.rcParams["webagg.open_in_browser"] = False
        matplotlib.rcParams["webagg.port"] = args.port
        matplotlib.rcParams["webagg.address"] = args.host
        matplotlib.rcParams["webagg.port_retries"] = 1   # only the requested port
        check_port(args.host, args.port)
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, tables = make_figure(fields, vectors, targets, residues, highlight,
                              (args.highlight_label, args.other_label))
    for target, rows in tables.items():
        missing = sorted(highlight - {r["resid"] for r in rows})
        if missing:
            print(f"Note: highlighted residues not in the plot of {target}: {', '.join(map(str, missing))}")

    out = args.out or (None if args.webagg else os.path.splitext(args.fields)[0] + "_residues.png")
    if out:
        fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
        print("Plot written to : ", out)
    if args.csv:
        write_csv(args.csv, tables)
        print("Values written to : ", args.csv)
    if args.webagg:
        print("\nPlot served at (open in a browser on any machine that can reach this one; Ctrl+C to stop):")
        for link in server_links(args.host, args.port):
            print("   ", link)
        sys.stdout.flush()
        stdout = sys.stdout
        sys.stdout = DropLines(stdout, "To view figure, visit")
        try:
            plt.show()
        finally:
            sys.stdout = stdout
    return fig, tables


def cli():
    """Entry point of the fieldplot command."""
    main()


if __name__ == "__main__":
    cli()
