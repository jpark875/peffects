from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from peffects import cli, data, report, simulate, studies, validate
from tests.conftest import DATA_DIR, needs_data


def test_fetch_verifies_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello"
    source = data.Source(
        "toy", "toy.csv", "https://example.invalid/toy.csv", hashlib.sha256(payload).hexdigest(), ""
    )
    monkeypatch.setitem(data.SOURCES, "toy", source)
    calls: list[str] = []

    def opener(url: str) -> bytes:
        calls.append(url)
        return payload

    path = data.fetch("toy", tmp_path, opener=opener)
    assert path.read_bytes() == payload
    data.fetch("toy", tmp_path, opener=opener)
    assert len(calls) == 1  # cached copy reused
    with pytest.raises(data.DataError, match="checksum"):
        data.fetch("toy", tmp_path, force=True, opener=lambda url: b"tampered")
    assert path.read_bytes() == payload and not list(tmp_path.glob("*.part"))


def test_missing_data_is_explained(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "did", "castle"]) == 2
    assert "peffects fetch" in capsys.readouterr().err


def test_csv_command_runs_every_estimator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = simulate.staggered(n_units=60, cohorts={2003.0: 0.3, 2006.0: 0.3}, seed=3)
    frame["cohort"] = frame["cohort"].replace(float("inf"), 0)
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "out.json"
    args = [
        "csv",
        str(path),
        "--unit",
        "unit",
        "--time",
        "time",
        "--outcome",
        "y",
        "--cohort",
        "cohort",
    ]
    assert cli.main([*args, "--reps", "99", "--permutations", "19", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    for label in (
        "TWFE, wild cluster bootstrap",
        "Callaway-Sant'Anna, dynamic",
        "Sun-Abraham",
        "Goodman-Bacon",
    ):
        assert label in text
    result = json.loads(out.read_text())
    assert result["bacon"]["check"] == pytest.approx(result["twfe_unweighted"])
    bad = [
        "csv",
        str(path),
        "--unit",
        "unit",
        "--time",
        "time",
        "--outcome",
        "nope",
        "--cohort",
        "cohort",
    ]
    assert cli.main(bad) == 2


@needs_data
def test_published_numbers_are_reproduced() -> None:
    table = validate.run(DATA_DIR, placebos=False)
    assert table["passed"].all(), table[~table["passed"]]


@needs_data
def test_replication_commands(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--data-dir", str(DATA_DIR)]
    assert cli.main([*base, "did", "mpdta", "--reps", "49", "--permutations", "9"]) == 0
    assert "Callaway-Sant'Anna, simple" in capsys.readouterr().out
    assert cli.main([*base, "synth", "--no-placebos"]) == 0
    assert "Utah" in capsys.readouterr().out


@needs_data
def test_report_renders_from_results(tmp_path: Path) -> None:
    results = {
        "generated": "2026-01-01",
        "castle": studies.castle_study(DATA_DIR, reps=49, permutations=9),
        "sensitivity": report.sensitivity(DATA_DIR, 49, 0),
        "prop99": studies.prop99_study(DATA_DIR, placebos=False),
        "validation": validate.run(DATA_DIR, placebos=False).to_dict(orient="records"),
    }
    results = report._strict(results)
    json.dumps(results, allow_nan=False)
    page = report.write_page(results, tmp_path).read_text()
    assert page.count("<svg") == 3  # placebo chart needs the placebo run
    assert "36 of 36 published values reproduced" in page
    assert "<script>alert" not in page
