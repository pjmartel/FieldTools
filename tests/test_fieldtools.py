"""Tests for FieldTools, using the KPC example system in data/.

Run with: python -m pytest tests
"""

import os
import pickle

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from fieldtools import fields as ft  # noqa: E402

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


def test_distance_filter_keeps_matching_frames(tmp_path, kpc, universe):
    o71 = universe.select_atoms("resid 40 and name O71")[0].index
    water = universe.select_atoms("resid 264 and name O")[0].index
    distances = []
    for ts in universe.trajectory:
        distances.append(np.linalg.norm(universe.atoms.positions[o71] - universe.atoms.positions[water]))
    cutoff = float(np.median(distances))
    expected = [i + 1 for i, d in enumerate(distances) if d <= cutoff]
    assert 0 < len(expected) < len(distances)

    result = run(tmp_path, [":40@C7 :40@O71", ":40@O71"], [EXCLUDE],
                 ["-filter_distance", ":40@O71", ":264@O", f"{cutoff:.6f}"])
    assert result["frames"] == expected
    for target in (BOND, POINT):
        for component in ("Total", "Protein", "WAT"):
            np.testing.assert_allclose(result["fields"][target][component],
                                       [kpc["fields"][target][component][i - 1] for i in expected])
    with open(tmp_path / "field_frames.dat") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("# frame distance::40@O71-:264@O")
    assert [int(line.split()[0]) for line in lines[1:]] == expected
    np.testing.assert_allclose([float(line.split()[1]) for line in lines[1:]],
                               [distances[i - 1] for i in expected], atol=1e-3)


def test_angle_filter_and_combined_filters(tmp_path, universe):
    def angle(a, b, c):
        u_, v_ = a - b, c - b
        return np.degrees(np.arccos(np.dot(u_, v_) / (np.linalg.norm(u_) * np.linalg.norm(v_))))

    sel = {n: universe.select_atoms(s)[0].index for n, s in
           [("O", "resid 264 and name O"), ("C7", "resid 40 and name C7"), ("O71", "resid 40 and name O71")]}
    angles = []
    for ts in universe.trajectory:
        p = universe.atoms.positions.astype(np.float64)
        angles.append(angle(p[sel["O"]], p[sel["C7"]], p[sel["O71"]]))
    low, high = np.percentile(angles, [20, 80])
    expected = [i + 1 for i, a in enumerate(angles) if low <= a <= high]

    result = run(tmp_path, [":40@O71"], [EXCLUDE],
                 ["-filter_angle", ":264@O", ":40@C7", ":40@O71", f"{low:.6f}:{high:.6f}"])
    assert result["frames"] == expected

    # A second, impossible filter removes every frame
    empty = run(tmp_path, [":40@O71"], [EXCLUDE],
                ["-filter_angle", ":264@O", ":40@C7", ":40@O71", f"{low:.6f}:{high:.6f}",
                 "-filter_distance", ":40@O71", ":264@O", "0.5"])
    assert empty["frames"] == []
    assert empty["fields"][POINT]["Total"] == []


@pytest.mark.parametrize("cutoff, message", [("abc", "Invalid distance cutoff"), ("3:2", "MIN is larger")])
def test_invalid_filter_cutoff(tmp_path, cutoff, message):
    with pytest.raises(SystemExit) as error:
        run(tmp_path, [":40@O71"], [EXCLUDE], ["-filter_distance", ":40@O71", ":264@O", cutoff])
    assert message in str(error.value)


def test_charge_report(kpc, tmp_path, capsys):
    report = kpc["charges"]
    assert report["Total"] == pytest.approx(0.0, abs=1e-4)
    assert report["Groups"]["Protein"][0] == pytest.approx(-2.0, abs=1e-4)
    assert report["Groups"]["Protein"][1] == 261          # includes the acylated serine ACA 40
    assert report["Groups"]["Na+"][:2] == (pytest.approx(2.0, abs=1e-4), 2)
    assert report["Groups"]["WAT"][2] == "solvent"

    # Residues that are neither protein nor solvent are reported as hetero groups
    target_file = tmp_path / "target.dat"
    target_file.write_text(":40@O71\n")
    result = ft.main(["-nc", NC, "-parm", PARM, "-target", str(target_file), "-solvent", "WAT",
                      "-out", str(tmp_path / "f.pkl")])
    charge, n_residues, label = result["charges"]["Groups"]["Na+"]
    assert (round(charge, 3), n_residues, label) == (2.0, 2, "resid 262,263")
    assert "Na+" in capsys.readouterr().out


def test_filter_statistics_cover_all_frames(tmp_path, universe, capsys):
    o71 = universe.select_atoms("resid 40 and name O71")[0].index
    water = universe.select_atoms("resid 264 and name O")[0].index
    distances = [np.linalg.norm(universe.atoms.positions[o71] - universe.atoms.positions[water])
                 for ts in universe.trajectory]
    result = run(tmp_path, [":40@O71"], [EXCLUDE],
                 ["-filter_distance", ":40@O71", ":264@O", "0.5",
                  "-filter_angle", ":264@O", ":40@C7", ":40@O71", "0"])
    distance, angle = result["filter_stats"]
    assert result["frames"] == []
    assert distance["min"] == pytest.approx(min(distances), abs=1e-3)
    assert distance["max"] == pytest.approx(max(distances), abs=1e-3)
    assert (distance["passed"], angle["passed"]) == (0, len(distances))
    output = capsys.readouterr().out
    assert f"Filter values over all {len(distances)} frames" in output
    assert "passed by 0 frames" in output
