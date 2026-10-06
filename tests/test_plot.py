"""Tests for fieldplot."""

import csv
import os

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")
pytest.importorskip("matplotlib")

from fieldtools import fields as ft  # noqa: E402
from fieldtools import plot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EXCLUDE = "40/HB2 40/HB1 40/CB 40/OG 40/C7 40/O71 40/C25 40/H6 40/CA 40/HA 264/"
BOND, POINT = "40/C7_40/O71", "40/O71"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plot")
    (folder / "t.dat").write_text("40/C7 40/O71\n40/O71\n")
    (folder / "e.dat").write_text(EXCLUDE + "\n")
    result = ft.main(["-top", os.path.join(DATA, "KPC.parm7"), "-traj", os.path.join(DATA, "KPC.nc"),
                      "-target", str(folder / "t.dat"), "-exclude_atoms", str(folder / "e.dat"),
                      "-TIP4P", "True", "-out", str(folder / "field.pkl"),
                      "-vector_out", str(folder / "vectors.pkl")])
    return folder, result


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_bar_plot_and_table(results):
    folder, result = results
    fig, tables = plot.main([str(folder / "field.pkl"), "-target", BOND, "-highlight", "43,136,205-207",
                             "-out", str(folder / "bond.png"), "-csv", str(folder / "bond.csv")])
    assert os.path.getsize(folder / "bond.png") > 10000
    rows = read_csv(folder / "bond.csv")
    assert len(rows) == 261            # the protein residues; Na+ and WAT (including water 264) are solvent
    by_resid = {int(r["resid"]): r for r in rows}
    values = np.array(result["fields"][BOND]["GLU_136"])
    assert float(by_resid[136]["mean_MV_cm"]) == pytest.approx(values.mean(), abs=1e-3)
    assert float(by_resid[136]["sd_MV_cm"]) == pytest.approx(values.std(ddof=1), abs=1e-3)
    assert {int(r["resid"]) for r in rows if r["highlighted"] == "1"} == {43, 136, 205, 206, 207}
    # Bars are signed: some above and some below zero
    means = [float(r["mean_MV_cm"]) for r in rows]
    assert min(means) < 0 < max(means)


def test_default_output_name_and_all_targets(results):
    folder, _ = results
    fig, tables = plot.main([str(folder / "field.pkl")])
    assert os.path.exists(folder / "field_residues.png")
    assert list(tables) == [BOND, POINT] and len(fig.axes) == 2


def test_point_target_projection_is_additive(results):
    folder, result = results
    _, tables = plot.main([str(folder / "field.pkl"), "-vectors", str(folder / "vectors.pkl"), "-target", POINT,
                           "-out", str(folder / "point.png")])
    fields, vectors = plot.load_pickle(str(folder / "field.pkl"), ""), plot.load_pickle(str(folder / "vectors.pkl"), "")
    residues = sum(plot.frame_values(fields, vectors, POINT, r[2]) for r in plot.residue_components(fields[POINT]))
    solvent = plot.frame_values(fields, vectors, POINT, "Solvent")
    total = plot.frame_values(fields, vectors, POINT, "Total")
    np.testing.assert_allclose(residues + solvent, total, atol=1e-6)
    np.testing.assert_allclose(total, result["fields"][POINT]["Total"], rtol=1e-9)   # = the magnitude
    assert min(r["mean"] for r in tables[POINT]) < 0                                  # signed


def test_residue_range(results):
    folder, _ = results
    _, tables = plot.main([str(folder / "field.pkl"), "-target", BOND, "-residues", "120-140",
                           "-out", str(folder / "range.png")])
    assert [r["resid"] for r in tables[BOND]] == list(range(120, 141))


@pytest.mark.parametrize("text, expected", [("40", {40}), ("40,42-44", {40, 42, 43, 44}), (" 7 , 9 ", {7, 9})])
def test_parse_residue_list(text, expected):
    assert plot.parse_residue_list(text) == expected


@pytest.mark.parametrize("args, message", [
    (["-target", "nope"], "is not in"),
    (["-highlight", "40-30"], "Invalid residue range"),
    (["-highlight", "A40"], "Invalid residue list"),
])
def test_errors(results, args, message):
    folder, _ = results
    with pytest.raises(SystemExit) as error:
        plot.main([str(folder / "field.pkl"), "-out", str(folder / "x.png"), *args])
    assert message in str(error.value)


def test_busy_port_is_reported():
    import socket
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(SystemExit) as error:
            plot.check_port("127.0.0.1", port)
    assert "Choose another port" in str(error.value)
