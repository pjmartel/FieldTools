"""Tests for parameter files (--parameter-file, --write-parameter-file)."""

import os
import pickle

import numpy as np
import pytest

from fieldtools import fields as ft
from fieldtools import parameters

pytest.importorskip("MDAnalysis")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NC, PARM = os.path.join(DATA, "KPC.nc"), os.path.join(DATA, "KPC.parm7")
EXCLUDE = "40/HB2 40/HB1 40/CB 40/OG 40/C7 40/O71 40/C25 40/H6 40/CA 40/HA 264/"

PARAMETERS = f"""
# KPC example
[input]
top = "{PARM}"
traj = "{NC}"

[targets]
targets = ["40/C7 40/O71", "40/O71"]
exclude = ["{EXCLUDE}"]

[system]
solvent = ["WAT", "Na+"]
TIP4P = true

[filters]
distance = ["40/O71 264/O 2.6"]

[output]
out = "field.pkl"
"""


@pytest.fixture
def parameter_file(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(PARAMETERS)
    return path


def command_line_run(tmp_path, extra=()):
    (tmp_path / "target.dat").write_text("40/C7 40/O71\n40/O71\n")
    (tmp_path / "exclude.dat").write_text(EXCLUDE + "\n")
    return ft.main(["-top", PARM, "-traj", NC, "-target", str(tmp_path / "target.dat"),
                    "-exclude_atoms", str(tmp_path / "exclude.dat"), "-solvent", "WAT,Na+", "-TIP4P", "True",
                    "-filter_distance", "40/O71", "264/O", "2.6", "-out", str(tmp_path / "cli.pkl"), *extra])


def test_parameter_file_matches_command_line(tmp_path, parameter_file):
    from_file = ft.main(["--parameter-file", str(parameter_file)])
    from_cli = command_line_run(tmp_path)
    assert from_file["frames"] == from_cli["frames"] and 0 < len(from_file["frames"]) < 10
    assert from_file["fields"] == from_cli["fields"]
    # The output path is relative to the parameter file
    with open(tmp_path / "field.pkl", "rb") as f:
        assert pickle.load(f) == from_file["fields"]


def test_command_line_overrides_parameter_file(tmp_path, parameter_file):
    result = ft.main(["--parameter-file", str(parameter_file), "-out", str(tmp_path / "other.pkl"),
                      "-filter_distance", "40/O71", "264/O", "10"])
    assert os.path.exists(tmp_path / "other.pkl") and not os.path.exists(tmp_path / "field.pkl")
    assert result["frames"] == list(range(1, 11))      # the command-line filter replaces the file's


def test_target_and_exclusion_files_in_parameter_file(tmp_path, parameter_file):
    (tmp_path / "target.dat").write_text("40/C7 40/O71\n40/O71\n")
    (tmp_path / "exclude.dat").write_text(EXCLUDE + "\n")
    text = PARAMETERS.replace('targets = ["40/C7 40/O71", "40/O71"]', 'target_file = "target.dat"')
    text = text.replace(f'exclude = ["{EXCLUDE}"]', 'exclude_file = "exclude.dat"')
    parameter_file.write_text(text)
    from_file = ft.main(["--parameter-file", str(parameter_file)])
    assert from_file["fields"] == command_line_run(tmp_path)["fields"]


def test_write_parameter_file_round_trip(tmp_path, parameter_file):
    written = tmp_path / "copy" / "written.toml"
    written.parent.mkdir()
    assert ft.main(["--parameter-file", str(parameter_file), "--write-parameter-file", str(written)]) is None
    original = parameters.load_parameter_file(str(parameter_file))
    copy = parameters.load_parameter_file(str(written))
    for key, value in original.items():
        assert copy[key] == value, key
    assert ft.main(["--parameter-file", str(written)])["fields"] == ft.main(["--parameter-file", str(parameter_file)])["fields"]


def test_template_is_valid(tmp_path):
    template = tmp_path / "template.toml"
    ft.main(["--write-parameter-file", str(template)])
    values = parameters.load_parameter_file(str(template))
    assert values["pbc"] is False and values["solvent"] == "auto"


@pytest.mark.parametrize("text, message", [
    ("[inputs]\ntop = 'a'\n", "unknown section [inputs]"),
    ("[input]\ntopol = 'a'\n", "unknown key 'topol' in [input]"),
    ("[system]\npbc = 'maybe'\n", "must be true or false"),
    ("[filters]\ndistance = ['40/O71 3.2']\n", "must have 3 parts"),
    ("[targets]\ntargets = ['40/O71']\ntarget_file = 't.dat'\n", "sets the same option"),
    ("[input\n", "not valid TOML"),
])
def test_invalid_parameter_files(tmp_path, text, message):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(SystemExit) as error:
        ft.main(["--parameter-file", str(path)])
    assert message in str(error.value)


def test_missing_required_options(tmp_path, capsys):
    path = tmp_path / "partial.toml"
    path.write_text(f'[input]\ntop = "{PARM}"\n')
    with pytest.raises(SystemExit):
        ft.main(["--parameter-file", str(path)])
    assert "missing -traj, -target, -out" in capsys.readouterr().err


def test_prefix_on_command_line_renames_all_outputs(tmp_path, parameter_file):
    text = PARAMETERS.replace('out = "field.pkl"', 'out = "results/field.pkl"\nvector_out = "results/vectors.pkl"')
    (tmp_path / "results").mkdir()
    parameter_file.write_text(text)
    ft.main(["--parameter-file", str(parameter_file), "--prefix", "rep2"])
    written = sorted(os.listdir(tmp_path / "results"))
    assert written == ["rep2.log", "rep2_field.pkl", "rep2_field_frames.dat", "rep2_vectors.pkl"]


def test_prefix_without_out_uses_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "target.dat").write_text("40/O71\n")
    ft.main(["-top", PARM, "-traj", NC, "-target", "target.dat", "--prefix", "quick"])
    assert os.path.exists(tmp_path / "quick_field.pkl")


def test_prefix_in_parameter_file_and_override(tmp_path, parameter_file):
    parameter_file.write_text(PARAMETERS.replace('out = "field.pkl"', 'out = "field.pkl"\nprefix = "file"'))
    ft.main(["--parameter-file", str(parameter_file)])
    assert os.path.exists(tmp_path / "file_field.pkl")
    ft.main(["--parameter-file", str(parameter_file), "--prefix", "cli"])
    assert os.path.exists(tmp_path / "cli_field.pkl") and not os.path.exists(tmp_path / "cli_file_field.pkl")


def test_written_parameter_file_applies_prefix_once(tmp_path, parameter_file):
    written = tmp_path / "written.toml"
    ft.main(["--parameter-file", str(parameter_file), "--prefix", "once", "--write-parameter-file", str(written)])
    text = written.read_text()
    assert 'prefix = "once"' in text and 'out = "field.pkl"' in text
    ft.main(["--parameter-file", str(written)])
    assert os.path.exists(tmp_path / "once_field.pkl") and not os.path.exists(tmp_path / "once_once_field.pkl")


def test_prefix_writes_log_file(tmp_path, parameter_file, capsys):
    ft.main(["--parameter-file", str(parameter_file), "--prefix", "rep3"])
    screen = capsys.readouterr().out
    log = (tmp_path / "rep3.log").read_text()      # next to the output files
    assert log == screen
    assert "Command: fieldtools --parameter-file" in log and "--prefix rep3" in log
    assert "Frames passing the filters" in log and "Field calculation DONE" in log


def test_log_file_records_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "target.dat").write_text("XX/CA\n")
    with pytest.raises(SystemExit):
        ft.main(["-top", PARM, "-traj", NC, "-target", "target.dat", "--prefix", "bad"])
    assert "Error! No atom in the system matches 'XX/CA'." in (tmp_path / "bad.log").read_text()
