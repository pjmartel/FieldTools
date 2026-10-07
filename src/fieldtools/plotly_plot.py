"""fieldplotly: interactive Plotly plots of FieldTools results.

The same per-residue bar plot as fieldplot (mean over the trajectory, standard deviation as
error bars), drawn with Plotly: a standalone HTML page with zoom, pan, a range slider along the
residues and hover details on every bar. It can also be served on a web port.

Run `fieldplotly -h` for usage.
"""

import argparse
import os
import sys

from . import __version__
from .plot import (COLOR_BAR, COLOR_HIGHLIGHT, GRID, HIGHLIGHT_BAND, SUMMARY, SURFACE, TEXT_PRIMARY,
                   TEXT_SECONDARY, axis_label, check_port, display_name, frame_values, label_shifts, load_pickle,
                   parse_residue_list, residue_table, server_links, statistics, write_csv)

FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def subtitle(fields, vectors, target):
    n_frames = len(fields[target]["Total"])
    parts = []
    for component in SUMMARY:
        if component in fields[target]:
            mean, sd = statistics(frame_values(fields, vectors, target, component))
            parts.append(f"{component} {mean:.1f} ± {sd:.1f}")
    return f"Mean ± SD over {n_frames} frames · " + " · ".join(parts) + " MV/cm"


def make_figure(fields, vectors, targets, residues=None, highlight=()):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tables = {t: residue_table(fields, vectors, t, residues, highlight) for t in targets}
    titles = [f"<b>{display_name(t)}</b><br><span style='font-size:11px;color:{TEXT_SECONDARY}'>"
              f"{subtitle(fields, vectors, t)}</span>" for t in targets]
    fig = make_subplots(rows=len(targets), cols=1, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=min(0.12, 0.35 / max(len(targets), 1)))

    for row_index, target in enumerate(targets, start=1):
        rows = tables[target]
        for highlighted, name, color in ((False, "Other residues", COLOR_BAR),
                                         (True, "Highlighted residues", COLOR_HIGHLIGHT)):
            subset = [r for r in rows if r["highlighted"] == highlighted]
            if not subset:
                continue
            fig.add_trace(go.Bar(
                x=[r["resid"] for r in subset], y=[r["mean"] for r in subset],
                error_y={"type": "data", "array": [r["sd"] for r in subset], "color": TEXT_SECONDARY,
                         "thickness": 1, "width": 0},
                customdata=[[r["resname"], r["sd"]] for r in subset],
                hovertemplate="<b>%{customdata[0]} %{x}</b><br>mean %{y:.2f} MV/cm<br>"
                              "SD %{customdata[1]:.2f} MV/cm<extra></extra>",
                marker={"color": color, "line": {"width": 0}}, name=name,
                legendgroup=name, showlegend=row_index == 1 and any(r["highlighted"] for r in rows),
            ), row=row_index, col=1)

        for r in rows:
            if r["highlighted"]:
                fig.add_vrect(x0=r["resid"] - 0.5, x1=r["resid"] + 0.5, fillcolor=HIGHLIGHT_BAND, opacity=1,
                              line_width=0, layer="below", row=row_index, col=1)
        for r, shift in label_shifts(rows):
            up = r["mean"] >= 0
            fig.add_annotation(x=r["resid"], y=r["mean"] + (r["sd"] if up else -r["sd"]),
                               text=f"{r['resname']} {r['resid']}", showarrow=False, textangle=-90,
                               yanchor="bottom" if up else "top", yshift=3 if up else -3, xshift=shift * 13,
                               font={"size": 10, "color": TEXT_PRIMARY}, row=row_index, col=1)

        fig.update_yaxes(title_text=axis_label(target, vectors), zeroline=True, zerolinecolor=TEXT_PRIMARY,
                         zerolinewidth=1, gridcolor=GRID, row=row_index, col=1)

    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_xaxes(title_text="Residue number", rangeslider={"visible": True, "thickness": 0.06},
                     row=len(targets), col=1)
    fig.update_layout(
        template="simple_white", barmode="overlay", bargap=0.2,
        height=360 * len(targets) + 140, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": FONT, "size": 12, "color": TEXT_SECONDARY},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
                "font": {"color": TEXT_SECONDARY}},
        hoverlabel={"bgcolor": "white", "bordercolor": GRID, "font": {"color": TEXT_PRIMARY}},
        margin={"l": 70, "r": 30, "t": 90, "b": 50}, title=None,
    )
    for annotation in fig.layout.annotations[:len(targets)]:   # subplot titles: left-aligned
        annotation.update(x=0, xanchor="left", align="left", font={"size": 14, "color": TEXT_PRIMARY})
    return fig, tables


def html_page(fig, cdn=False):
    return fig.to_html(include_plotlyjs="cdn" if cdn else True, full_html=True,
                       config={"displaylogo": False, "responsive": True,
                               "toImageButtonOptions": {"format": "svg", "filename": "fieldtools_residues"}})


def serve(page, host, port):
    """Serve the HTML page on host:port until Ctrl+C."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    body = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):   # no line per request
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print("\nPlot served at (open in a browser on any machine that can reach this one; Ctrl+C to stop):")
    for link in server_links(host, port):
        print("   ", link)
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fieldplotly", allow_abbrev=False,
        description="Interactive (Plotly) bar plot of the field contribution of every residue: mean over the "
                    "trajectory, with the standard deviation as error bars.",
        epilog="Examples:\n"
               "  fieldplotly field.pkl -highlight 70,130,234-237              -> field_residues.html\n"
               "  fieldplotly field.pkl -vectors vectors.pkl -out point.html\n"
               "  fieldplotly field.pkl -serve -port 8988                       (prints the links to open)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"FieldTools {__version__}")
    parser.add_argument("fields", help="fields file (.pkl) written by fieldtools")
    parser.add_argument("-target", action="append", default=[],
                        help="target to plot, as named in the fields file (repeat for several) [default: all]")
    parser.add_argument("-highlight", default="",
                        help="residue numbers whose bars are colored differently, e.g. 70,130,234-237")
    parser.add_argument("-residues", default=None, help="only plot these residue numbers, e.g. 1-305")
    parser.add_argument("-vectors", default=None,
                        help="vectors file (.pkl) from fieldtools -vector_out: point targets are then plotted "
                             "as projections onto the total field (signed and additive) instead of magnitudes")
    parser.add_argument("-out", default=None,
                        help="output file: .html (interactive), or .png/.pdf/.svg (needs the kaleido package) "
                             "[default: <fields>_residues.html, unless -serve is used]")
    parser.add_argument("-cdn", action="store_true",
                        help="load the Plotly JavaScript from the internet instead of embedding it "
                             "(much smaller .html files that need internet access to display)")
    parser.add_argument("-csv", default=None, help="also write the plotted values to this CSV file")
    parser.add_argument("-serve", "-webagg", dest="serve", action="store_true",
                        help="serve the interactive plot on a web server instead of only saving it; "
                             "no browser is opened. Stop with Ctrl+C")
    parser.add_argument("-port", type=int, default=8988, help="port of the web server [default: 8988]")
    parser.add_argument("-host", default="0.0.0.0",
                        help="address the web server listens on [default: 0.0.0.0, reachable from other "
                             "machines on the network; 127.0.0.1 for this computer only]")
    return parser


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        import plotly  # noqa: F401
    except ModuleNotFoundError:
        sys.exit("Error! fieldplotly needs Plotly: pip install plotly")
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
    if args.serve:
        check_port(args.host, args.port)

    fig, tables = make_figure(fields, vectors, targets, residues, highlight)
    for target, rows in tables.items():
        missing = sorted(highlight - {r["resid"] for r in rows})
        if missing:
            print(f"Note: highlighted residues not in the plot of {target}: {', '.join(map(str, missing))}")

    page = html_page(fig, args.cdn)
    out = args.out or (None if args.serve else os.path.splitext(args.fields)[0] + "_residues.html")
    if out:
        if out.lower().endswith((".html", ".htm")):
            with open(out, "w", encoding="utf-8") as f:
                f.write(page)
        else:
            try:
                fig.write_image(out, width=1600, height=fig.layout.height, scale=2)
            except (ValueError, ImportError, RuntimeError) as error:
                sys.exit(f"Error! Could not write {out}: static images need the kaleido package "
                         f"(pip install kaleido). Use an .html file instead, or fieldplot. ({error})")
        print("Plot written to : ", out)
    if args.csv:
        write_csv(args.csv, tables)
        print("Values written to : ", args.csv)
    if args.serve:
        serve(page, args.host, args.port)
    return fig, tables


def cli():
    """Entry point of the fieldplotly command."""
    main()


if __name__ == "__main__":
    cli()
