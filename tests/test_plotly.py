"""Tests for fieldplotly."""

import os
import socket
import threading
import urllib.request

import pytest

pytest.importorskip("MDAnalysis")
pytest.importorskip("plotly")

from fieldtools import fields as ft  # noqa: E402
from fieldtools import plotly_plot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EXCLUDE = "40/HB2 40/HB1 40/CB 40/OG 40/C7 40/O71 40/C25 40/H6 40/CA 40/HA 264/"
BOND, POINT = "40/C7_40/O71", "40/O71"


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plotly")
    (folder / "t.dat").write_text("40/C7 40/O71\n40/O71\n")
    (folder / "e.dat").write_text(EXCLUDE + "\n")
    ft.main(["-top", os.path.join(DATA, "KPC.parm7"), "-traj", os.path.join(DATA, "KPC.nc"),
             "-target", str(folder / "t.dat"), "-exclude_atoms", str(folder / "e.dat"), "-TIP4P", "True",
             "-out", str(folder / "field.pkl"), "-vector_out", str(folder / "vectors.pkl")])
    return folder


def test_html_with_highlights(folder):
    fig, tables = plotly_plot.main([str(folder / "field.pkl"), "-highlight", "43,136,205-207",
                                    "-csv", str(folder / "values.csv")])
    html = (folder / "field_residues.html").read_text()
    assert "Plotly.newPlot" in html and len(html) > 1_000_000          # Plotly embedded: works offline
    names = [(trace.name, len(trace.x)) for trace in fig.data]
    assert names == [("Other residues", 256), ("Highlighted residues", 5)] * 2   # two targets
    assert fig.data[1].error_y.array is not None
    labels = sorted(a.text for a in fig.layout.annotations if a.textangle == -90)
    assert labels == sorted(["LYS 43", "GLU 136", "THR 205", "GLY 206", "THR 207"] * 2)
    assert os.path.exists(folder / "values.csv")


def test_same_values_as_fieldplot(folder):
    from fieldtools import plot
    import matplotlib
    matplotlib.use("Agg")
    args = [str(folder / "field.pkl"), "-vectors", str(folder / "vectors.pkl"), "-target", POINT]
    _, plotly_tables = plotly_plot.main(args + ["-out", str(folder / "p.html")])
    _, mpl_tables = plot.main(args + ["-out", str(folder / "p.png")])
    assert plotly_tables == mpl_tables


def test_cdn_page_is_small(folder):
    plotly_plot.main([str(folder / "field.pkl"), "-target", BOND, "-cdn", "-out", str(folder / "cdn.html")])
    html = (folder / "cdn.html").read_text()
    assert "cdn.plot.ly" in html and len(html) < 500_000


def test_server_serves_only_the_plot(folder):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    fields = plotly_plot.load_pickle(str(folder / "field.pkl"), "Fields")
    fig, _ = plotly_plot.make_figure(fields, None, [BOND])
    page = plotly_plot.html_page(fig, cdn=True)
    thread = threading.Thread(target=plotly_plot.serve, args=(page, "127.0.0.1", port), daemon=True)
    thread.start()
    for _ in range(50):
        try:
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read().decode()
            break
        except OSError:
            threading.Event().wait(0.1)
    assert "Plotly.newPlot" in body
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/etc/passwd", timeout=2)


def test_errors(folder):
    with pytest.raises(SystemExit) as error:
        plotly_plot.main([str(folder / "field.pkl"), "-target", "nope"])
    assert "is not in" in str(error.value)


def test_legend_labels(folder):
    fig, _ = plotly_plot.main([str(folder / "field.pkl"), "-target", BOND, "-highlight", "136",
                               "-highlight_label", "Mutants", "-other_label", "Other positions",
                               "-out", str(folder / "l.html")])
    assert [trace.name for trace in fig.data] == ["Other positions", "Mutants"]
