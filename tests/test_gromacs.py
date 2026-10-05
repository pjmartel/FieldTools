"""Tests for GROMACS input (.top/.tpr topologies, .xtc trajectories).

data/KPC.top and data/KPC.xtc are the Amber example system converted with ParmEd.
The .tpr tests use the files shipped with MDAnalysisTests (pip install MDAnalysisTests).
"""

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from fieldtools import fields as ft  # noqa: E402

mda = pytest.importorskip("MDAnalysis")

DATA = os.path.join(ROOT, "data")
EXCLUDE = (":40@HB2 :40@HB1 :40@CB :40@OG :40@C7 :40@O71 :40@C25 :40@H6 :40@CA :40@HA "
           ":264@O :264@H1 :264@H2")
TARGETS = [":40@C7 :40@O71", ":40@O71"]


def run(tmp_path, parm, nc, targets=TARGETS, exclude=EXCLUDE, extra=()):
    target_file = tmp_path / "target.dat"
    target_file.write_text("\n".join(targets) + "\n")
    argv = ["-top", str(parm), "-traj", str(nc), "-target", str(target_file),
            "-out", str(tmp_path / "field.pkl"), *extra]
    if exclude:
        exclude_file = tmp_path / "exclude.dat"
        exclude_file.write_text(exclude + "\n")
        argv += ["-exclude_atoms", str(exclude_file)]
    return ft.main(argv)


def assert_same_fields(a, b, atol):
    assert list(a) == list(b)
    for target in a:
        assert list(a[target]) == list(b[target])
        for component in a[target]:
            np.testing.assert_allclose(a[target][component], b[target][component], atol=atol,
                                       err_msg=f"{target} {component}")


@pytest.fixture(scope="module")
def gromacs(tmp_path_factory):
    return run(tmp_path_factory.mktemp("gmx"), os.path.join(DATA, "KPC.top"), os.path.join(DATA, "KPC.xtc"))


def test_gromacs_top_is_detected():
    assert ft.is_gromacs_top(os.path.join(DATA, "KPC.top"))
    assert not ft.is_gromacs_top(os.path.join(DATA, "KPC.parm7"))


def test_amber_prmtop_with_top_extension(tmp_path):
    # Amber topologies may also be called .top; they must not be read as GROMACS files
    amber = tmp_path / "KPC.top"
    os.symlink(os.path.join(DATA, "KPC.parm7"), amber)
    assert not ft.is_gromacs_top(str(amber))


def test_gromacs_topology_matches_amber_topology(tmp_path, gromacs):
    # Same coordinates (from the .xtc), so only the topology reader differs
    amber = run(tmp_path, os.path.join(DATA, "KPC.parm7"), os.path.join(DATA, "KPC.xtc"))
    assert_same_fields(gromacs["fields"], amber["fields"], atol=1e-3)


def test_gromacs_matches_amber_trajectory_within_xtc_precision(tmp_path, gromacs):
    # The .xtc stores coordinates with 0.01 A precision, which changes the fields
    # of nearby atoms slightly
    amber = run(tmp_path, os.path.join(DATA, "KPC.parm7"), os.path.join(DATA, "KPC.nc"))
    for target in amber["fields"]:
        np.testing.assert_allclose(gromacs["fields"][target]["Total"], amber["fields"][target]["Total"], rtol=0.02)


def test_auto_solvent(gromacs):
    components = list(gromacs["fields"][":40@O71"])
    assert components[:5] == ["Total", "Protein", "Solvent", "Na+", "WAT"]
    assert all(type(c) is str for c in components)


def test_gmx_include(tmp_path, monkeypatch, gromacs):
    # Move the force field part of the topology into an included file
    with open(os.path.join(DATA, "KPC.top")) as f:
        lines = f.readlines()
    split = lines.index("[ moleculetype ]\n")
    forcefield = tmp_path / "gmxtop" / "kpc.ff"
    forcefield.mkdir(parents=True)
    (forcefield / "forcefield.itp").write_text("".join(lines[:split]))
    topology = tmp_path / "system" / "KPC.top"
    topology.parent.mkdir()
    topology.write_text('#include "kpc.ff/forcefield.itp"\n\n' + "".join(lines[split:]))
    xtc = os.path.join(DATA, "KPC.xtc")

    monkeypatch.setattr(ft, "gromacs_include_dir", lambda: None)
    with pytest.raises(SystemExit) as error:
        run(tmp_path, topology, xtc)
    assert "-gmx_include" in str(error.value)

    result = run(tmp_path, topology, xtc, extra=["-gmx_include", str(tmp_path / "gmxtop")])
    assert_same_fields(result["fields"], gromacs["fields"], atol=1e-9)


def test_topology_without_charges(tmp_path):
    gro = tmp_path / "KPC.gro"
    u = mda.Universe(os.path.join(DATA, "KPC.parm7"), os.path.join(DATA, "KPC.nc"))
    u.atoms.write(str(gro))
    with pytest.raises(SystemExit) as error:
        run(tmp_path, gro, os.path.join(DATA, "KPC.xtc"))
    assert "no partial charges" in str(error.value)


def test_minimum_image_matches_mdanalysis():
    from MDAnalysis.lib.distances import minimize_vectors
    rng = np.random.default_rng(0)
    vectors = rng.uniform(-200, 200, (20000, 3))
    for dimensions in ([80, 80, 80, 60, 60, 90],                      # rhombic dodecahedron
                       [70, 70, 70, 109.4712, 109.4712, 109.4712],    # truncated octahedron
                       [50, 60, 70, 90, 90, 90]):
        wrapped = ft.minimum_image(vectors, ft.box_matrix(np.array(dimensions, dtype=float)))
        reference = minimize_vectors(vectors.astype(np.float32), np.array(dimensions, dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(wrapped, axis=1), np.linalg.norm(reference, axis=1), atol=1e-3)


def reference_pbc_field(u, target_index, excluded):
    """Independent field with whole residues at their image closest to the target."""
    from MDAnalysis.lib.distances import minimize_vectors
    box = u.dimensions
    positions = u.atoms.positions.astype(np.float64)
    for residue in u.residues:   # make residues whole (4-point water sites have no bonds for unwrap)
        idx = residue.atoms.indices
        positions[idx] = positions[idx[0]] + minimize_vectors(
            (positions[idx] - positions[idx[0]]).astype(np.float32), box)
    point = positions[target_index]
    r_vec = np.empty_like(positions)
    for residue in u.residues:
        idx = residue.atoms.indices
        shift = minimize_vectors((point - positions[idx[0]]).astype(np.float32), box) - (point - positions[idx[0]])
        r_vec[idx] = point - positions[idx] + shift
    keep = ~excluded
    r = np.linalg.norm(r_vec[keep], axis=1)
    return ft.FIELD_CONST * (u.atoms.charges[keep][:, None] * r_vec[keep] / r[:, None] ** 3).sum(axis=0)


def test_tpr_xtc_with_pbc_and_four_point_water(tmp_path):
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    # Adenylate kinase in TIP4P water (OW, HW1, HW2, MW) in a rhombic dodecahedron
    result = run(tmp_path, datafiles.TPR, datafiles.XTC, [":13@NZ", ":13@NZ :13@CE"], None,
                 ["-pbc", "True", "-TIP4P", "True", "-vector_out", str(tmp_path / "vectors.pkl")])
    assert list(result["fields"][":13@NZ"])[:5] == ["Total", "Protein", "Solvent", "SOL", "NA+"]

    u = mda.Universe(datafiles.TPR, datafiles.XTC)
    nz = u.select_atoms("resid 13 and name NZ")[0].index
    excluded = np.zeros(len(u.atoms), dtype=bool)
    excluded[u.select_atoms("resid 13").indices] = True
    for frame in range(3):
        u.trajectory[frame]
        np.testing.assert_allclose(result["vectors"][":13@NZ"]["Total"][frame],
                                   reference_pbc_field(u, nz, excluded), rtol=1e-4)


def test_pbc_repairs_broken_molecules(tmp_path):
    # Wrap every atom into the box individually, as in raw GROMACS output
    u = mda.Universe(os.path.join(DATA, "KPC.parm7"), os.path.join(DATA, "KPC.nc"))
    intact, broken = tmp_path / "intact.trr", tmp_path / "broken.trr"
    with mda.Writer(str(intact), len(u.atoms)) as w_intact, mda.Writer(str(broken), len(u.atoms)) as w_broken:
        for ts in u.trajectory[:3]:
            ts.positions += np.array([20.0, -15.0, 9.0])   # move the system off-centre
            w_intact.write(u.atoms)
            u.atoms.wrap(compound="atoms")
            w_broken.write(u.atoms)
    parm = os.path.join(DATA, "KPC.parm7")
    reference = run(tmp_path, parm, str(intact), extra=["-pbc", "True"])
    result = run(tmp_path, parm, str(broken), extra=["-pbc", "True"])
    assert_same_fields(result["fields"], reference["fields"], atol=1e-3)


def test_chains_in_tpr():
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    system = ft.load_system(datafiles.TPR, datafiles.XTC)   # adk: seg_0_AKeco, seg_1_SOL, seg_2_NA+
    first = ft.select_one_atom("AKeco/1/N", system)
    assert first == ft.select_one_atom(":1@N", system)
    assert ft.select_one_atom("seg_0_AKeco/1/N", system) == first
    assert set(ft.select_atoms("SOL/SOL/", system)) == set(ft.select_atoms(":SOL", system))
    with pytest.raises(SystemExit) as error:
        ft.select_atoms("B/1/N", system)
    assert "Available: AKeco, SOL, NA+" in str(error.value)
    with pytest.raises(SystemExit) as error:
        ft.select_atoms("SOL/1/N", system)        # residue 1 is in the protein, not in SOL
    assert "No atom" in str(error.value)


def test_tpr_coordinates_are_in_angstrom():
    # MDAnalysis <= 2.10 returns the coordinates stored in a .tpr in nm
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    if "TPR" not in mda._READERS:
        pytest.skip("this MDAnalysis version cannot read coordinates from .tpr files")
    system = ft.load_system(datafiles.TPR2024, datafiles.TPR2024)   # lysozyme, about 40 A across
    positions, _ = next(system.frames())
    assert 30 < np.ptp(positions, axis=0).max() < 60
