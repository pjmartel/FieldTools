"""Tests for FieldTools, using the KPC example system in data/.

Run with: python -m pytest tests
"""

import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
import FieldTools as ft  # noqa: E402

mda = pytest.importorskip("MDAnalysis")

DATA = os.path.join(ROOT, "data")
NC, PARM = os.path.join(DATA, "KPC.nc"), os.path.join(DATA, "KPC.parm7")
BOND, POINT = ":40@C7_:40@O71", ":40@O71"
EXCLUDE = (":40@HB2 :40@HB1 :40@CB :40@OG :40@C7 :40@O71 :40@C25 :40@H6 :40@CA :40@HA "
           ":264@O :264@H1 :264@H2")


def run(tmp_path, targets, exclude=None, extra=()):
    target_file = tmp_path / "target.dat"
    target_file.write_text("\n".join(targets) + "\n")
    argv = ["-nc", NC, "-parm", PARM, "-target", str(target_file), "-solvent", "WAT,Na+",
            "-TIP4P", "True", "-out", str(tmp_path / "field.pkl"), *extra]
    if exclude is not None:
        exclude_file = tmp_path / "exclude.dat"
        exclude_file.write_text("\n".join(exclude) + "\n")
        argv += ["-exclude_atoms", str(exclude_file)]
    return ft.main(argv)


@pytest.fixture(scope="module")
def universe():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(PARM, NC)


@pytest.fixture(scope="module")
def kpc(tmp_path_factory):
    return run(tmp_path_factory.mktemp("kpc"), [":40@C7 :40@O71", ":40@O71"], [EXCLUDE, EXCLUDE],
               ["-energy_out", "False", "-vector_out", "False"])


def reference_field(universe, frame, point, excluded):
    """Independent Coulomb sum (MV/cm) at `point`."""
    universe.trajectory[frame]
    keep = ~excluded
    r_vec = point.astype(np.float64) - universe.atoms.positions[keep].astype(np.float64)
    r = np.linalg.norm(r_vec, axis=1)
    return ft.FIELD_CONST * (universe.atoms.charges[keep][:, None] * r_vec / r[:, None] ** 3).sum(axis=0)


def exclusion_mask(universe):
    mask = np.zeros(len(universe.atoms), dtype=bool)
    sel = "(resid 40 and name HB2 HB1 CB OG C7 O71 C25 H6 CA HA) or (resid 264 and name O H1 H2)"
    mask[universe.select_atoms(sel).indices] = True
    return mask


def test_bond_field_matches_published_reference(kpc):
    # data/KPC_field.pkl was produced with the original FieldTools, which used the
    # rounded conversion factor 5140 MV/cm per atomic unit (now 5142.2)
    with open(os.path.join(DATA, "KPC_field.pkl"), "rb") as f:
        reference = pickle.load(f)[BOND]
    fields = kpc["fields"][BOND]
    assert list(fields) == list(reference)
    for component in reference:
        np.testing.assert_allclose(fields[component], np.array(reference[component]) * 5142.20674763 / 5140.0,
                                   rtol=1e-4, atol=1e-6, err_msg=component)


def test_point_field_is_magnitude_of_field_vector(kpc, universe):
    o71 = universe.select_atoms("resid 40 and name O71")[0].index
    for frame in range(3):
        universe.trajectory[frame]
        expected = reference_field(universe, frame, universe.atoms.positions[o71], exclusion_mask(universe))
        np.testing.assert_allclose(kpc["vectors"][POINT]["Total"][frame], expected, rtol=1e-6)
        np.testing.assert_allclose(kpc["fields"][POINT]["Total"][frame], np.linalg.norm(expected), rtol=1e-6)


def test_component_vectors_add_up(kpc):
    vectors = kpc["vectors"][POINT]
    residues = [c for c in vectors if c not in ("Total", "Protein", "Solvent", "WAT", "Na+")]
    np.testing.assert_allclose(vectors["Protein"] + vectors["Solvent"], vectors["Total"], atol=1e-9)
    np.testing.assert_allclose(sum(vectors[c] for c in residues), vectors["Protein"], atol=1e-9)
    np.testing.assert_allclose(vectors["WAT"] + vectors["Na+"], vectors["Solvent"], atol=1e-9)


def test_bond_field_is_projection_of_vector(kpc, universe):
    for frame in range(3):
        universe.trajectory[frame]
        c7, o71 = (universe.select_atoms(f"resid 40 and name {n}")[0].position for n in ("C7", "O71"))
        unit = (o71 - c7) / np.linalg.norm(o71 - c7)
        np.testing.assert_allclose(kpc["fields"][BOND]["Total"][frame],
                                   kpc["vectors"][BOND]["Total"][frame] @ unit, rtol=1e-6)


def test_default_exclusion_is_first_residue(tmp_path, universe):
    result = run(tmp_path, [":40@O71"])
    excluded = np.zeros(len(universe.atoms), dtype=bool)
    excluded[universe.select_atoms("resid 40").indices] = True
    o71 = universe.select_atoms("resid 40 and name O71")[0].index
    universe.trajectory[0]
    expected = reference_field(universe, 0, universe.atoms.positions[o71], excluded)
    np.testing.assert_allclose(result["fields"][POINT]["Total"][0], np.linalg.norm(expected), rtol=1e-6)


def test_single_exclusion_line_applies_to_all_targets(tmp_path, kpc):
    result = run(tmp_path, [":40@C7 :40@O71", ":40@O71"], [EXCLUDE])
    np.testing.assert_allclose(result["fields"][BOND]["Total"], kpc["fields"][BOND]["Total"])
    np.testing.assert_allclose(result["fields"][POINT]["Total"], kpc["fields"][POINT]["Total"])


def test_point_target_atom_always_excluded(tmp_path, kpc):
    result = run(tmp_path, [":40@O71"], [":40@HB2"])
    assert np.all(np.isfinite(result["fields"][POINT]["Total"]))


def test_energy_matches_coulomb_sum(tmp_path, universe):
    result = run(tmp_path, [":40@O71"], [EXCLUDE], ["-energy_out", str(tmp_path / "energy.pkl")])
    o71 = universe.select_atoms("resid 40 and name O71")[0]
    keep = ~exclusion_mask(universe)
    universe.trajectory[0]
    r = np.linalg.norm(universe.atoms.positions[keep] - o71.position, axis=1)
    expected = ft.ENERGY_CONST * o71.charge * (universe.atoms.charges[keep] / r).sum()
    np.testing.assert_allclose(result["energies"][POINT]["Total"][0], expected, rtol=1e-6)
    assert os.path.exists(tmp_path / "energy.pkl")


def test_pbc_gives_same_result_for_imaged_trajectory_near_centre(tmp_path, kpc):
    # The example trajectory is imaged and the active site is far from the box edges,
    # so the minimum-image result only differs through the few atoms near the edges
    result = run(tmp_path, [":40@C7 :40@O71"], [EXCLUDE], ["-pbc", "True"])
    np.testing.assert_allclose(result["fields"][BOND]["Protein"], kpc["fields"][BOND]["Protein"], rtol=1e-6)


def test_minimum_image():
    box = ft.box_matrix(np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]))
    wrapped = ft.minimum_image(np.array([[9.0, -6.0, 0.5]]), box)
    np.testing.assert_allclose(wrapped, [[-1.0, 4.0, 0.5]])
    octahedron = ft.box_matrix(np.array([10.0, 10.0, 10.0, 109.4712, 109.4712, 109.4712]))
    np.testing.assert_allclose(np.linalg.norm(octahedron, axis=1), [10.0, 10.0, 10.0])


@pytest.mark.parametrize("target, message", [
    (":40@XX", "No atom"),
    ("40@O71", "Invalid atom specifier"),
    (":WAT@O", "exactly one atom"),
    (":40@C7 :40@O71 :40@CA", "use one (atom) or two (bond)"),
])
def test_invalid_targets(tmp_path, capsys, target, message):
    with pytest.raises(SystemExit) as error:
        run(tmp_path, [target])
    assert message in str(error.value)


def test_exclusion_line_count_must_match(tmp_path):
    with pytest.raises(SystemExit) as error:
        run(tmp_path, [":40@C7 :40@O71", ":40@O71", ":40@C7"], [EXCLUDE, EXCLUDE])
    assert "3 targets" in str(error.value)


def test_qm_charges_replace_topology_charges(tmp_path, universe, kpc):
    qm_atoms = universe.select_atoms("resid 41")
    keys = [f"{a.name}_{a.resname}_{a.resid}" for a in qm_atoms]
    (tmp_path / "qm.dict").write_text("\n".join(keys))

    def write_charges(scale):
        lines = []
        for frame in range(1, len(universe.trajectory) + 1):
            lines.append(f"Frame {frame}")
            lines += [f"{i + 1} X {a.charge * scale:.8f}" for i, a in enumerate(qm_atoms)]
        (tmp_path / "qm.dat").write_text("\n".join(lines) + "\n")

    qm_args = ["-use_qm_charges", "True", "-qm_charges", str(tmp_path / "qm.dat"),
               "-qm_dict", str(tmp_path / "qm.dict")]
    write_charges(1.0)
    same = run(tmp_path, [":40@C7 :40@O71"], [EXCLUDE], qm_args)
    np.testing.assert_allclose(same["fields"][BOND]["Total"], kpc["fields"][BOND]["Total"], rtol=1e-6)

    write_charges(0.0)
    residue = f"{qm_atoms[0].resname}_41"
    zeroed = run(tmp_path, [":40@C7 :40@O71"], [EXCLUDE], qm_args)
    np.testing.assert_allclose(zeroed["fields"][BOND][residue], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.array(zeroed["fields"][BOND]["Total"]),
                               np.array(kpc["fields"][BOND]["Total"]) - np.array(kpc["fields"][BOND][residue]),
                               rtol=1e-6)
