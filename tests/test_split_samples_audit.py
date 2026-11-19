import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "paper-figure-scripts" / "split_samples.py"


def write_reference(directory, reverse=False, impossible=False, many_conflicts=False, clean=False):
    directory.mkdir(parents=True)
    entries = {
        "real_train.txt": ["R2", "R3"],
        "real_test.txt": ["R0", "R1"],
        "mixed_train.txt": ["R2", "R3", "N2", "N3"],
        "mixed_test.txt": ["R0", "R1", "N0", "N1"],
        "synthetic_train.txt": ["R2", "R3", "N2", "N3"],
        "synthetic_test.txt": ["R0", "R1", "N0", "N1"],
        "x2ct_train.txt": (["Ctrain"] if clean else ["R0", "N0"]) + (
            ["R1", "R2", "R3"] if impossible else ["R1", "N1"] if many_conflicts else []
        ),
        "x2ct_test.txt": ["Ctest"] if clean else ["R0"],
    }
    for name, values in entries.items():
        if reverse:
            values.reverse()
        (directory / name).write_text("\n".join(values) + "\n", encoding="utf-8")


def run_audit(tmp_path, reference, name, extra_args=()):
    output = tmp_path / f"output-{name}"
    report = tmp_path / f"{name}-audit.md"
    conflicts = tmp_path / f"{name}-conflicts.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "audit",
            "--reference-dir",
            str(reference),
            "--output-dir",
            str(output),
            "--report",
            str(report),
            "--conflicts",
            str(conflicts),
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    return result, output, report, conflicts


def test_audit_rebalances_test_by_stratum_and_reports_patient_conflicts(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference)
    reference_files = {
        name: (reference / name).read_bytes()
        for name in (
            "real_train.txt", "real_test.txt", "synthetic_train.txt", "synthetic_test.txt",
            "mixed_train.txt", "mixed_test.txt", "x2ct_train.txt", "x2ct_test.txt",
        )
    }

    result, output, report, conflicts = run_audit(tmp_path, reference, "first")

    assert result.returncode == 0, result.stderr
    for name in ("real", "synthetic", "mixed"):
        train = set((output / f"{name}_train.txt").read_text().splitlines())
        test = set((output / f"{name}_test.txt").read_text().splitlines())
        assert len(test) == {"real": 2, "synthetic": 4, "mixed": 4}[name]
        assert train.isdisjoint(test)
        assert not test.intersection({"R0", "N0"})
    assert (output / "synthetic_test.txt").read_bytes() == (output / "mixed_test.txt").read_bytes()
    assert (output / "synthetic_train.txt").read_bytes() == (output / "mixed_train.txt").read_bytes()

    data = json.loads(conflicts.read_text())
    assert data["status"] == "ready"
    assert data["summary"]["initial_conflict_patient_count"] == 2
    assert data["summary"]["remaining_internal_test_x2ct_train_patient_count"] == 0
    assert data["versioned_reference"]["conflict_patient_count"] == 2
    assert all(patient["resolution"] == "fixed" for patient in data["versioned_reference"]["conflicts"])
    assert data["first_attempt"]["status"] == "not_run"
    assert data["repaired_reference"]["status"] == "ready"
    assert data["repaired_reference"]["moves"]
    assert data["summary"]["x2ct_train_test_patient_overlap_count"] == 1
    assert all((reference / name).read_bytes() == raw for name, raw in reference_files.items())
    assert "real-X-ray availability" in report.read_text()
    assert report.is_file()


def test_audit_output_is_reproducible_when_reference_lines_are_reordered(tmp_path):
    first_ref, second_ref = tmp_path / "reference-a", tmp_path / "reference-b"
    write_reference(first_ref)
    write_reference(second_ref, reverse=True)

    first, first_out, _, _ = run_audit(tmp_path, first_ref, "first")
    second, second_out, _, _ = run_audit(tmp_path, second_ref, "second")

    assert first.returncode == second.returncode == 0
    for name in (
        "real_train.txt",
        "real_test.txt",
        "synthetic_train.txt",
        "synthetic_test.txt",
        "mixed_train.txt",
        "mixed_test.txt",
    ):
        assert (first_out / name).read_bytes() == (second_out / name).read_bytes()


def test_audit_blocks_when_a_stratum_has_no_replacement_candidates(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference, impossible=True)

    result, output, report, conflicts = run_audit(tmp_path, reference, "blocked")

    assert result.returncode == 2
    assert not (output / "real_train.txt").exists()
    assert json.loads(conflicts.read_text())["status"] == "blocked"
    assert "Cannot preserve" in report.read_text()


def test_missing_x2ct_reference_still_writes_blocked_diagnostic(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference)
    (reference / "x2ct_train.txt").unlink()
    result, output, _, conflicts = run_audit(tmp_path, reference, "missing")
    assert result.returncode == 2, result.stderr
    assert not output.exists()
    data = json.loads(conflicts.read_text())
    assert data["versioned_reference"]["status"] == "not_run"
    assert "x2ct_train.txt" in data["reason"]


def test_unresolved_x2ct_series_does_not_pass_as_unrelated_patient(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference)
    (reference / "x2ct_train.txt").write_text("LIDC-IDRI-0013.unknown\n")
    result, output, _, conflicts = run_audit(tmp_path, reference, "unresolved")
    assert result.returncode == 2, result.stderr
    assert not output.exists()
    assert "unresolved X2CT" in json.loads(conflicts.read_text())["reason"]


def test_audit_repairs_more_than_three_patients_without_changing_stratum_quotas(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference, many_conflicts=True)

    result, output, _, conflicts = run_audit(tmp_path, reference, "many")

    assert result.returncode == 0, result.stderr
    data = json.loads(conflicts.read_text())
    assert data["versioned_reference"]["conflict_patient_count"] == 4
    assert data["repaired_reference"]["status"] == "ready"
    assert data["repaired_reference"]["patient_counts"]["real"]["test"] == 2
    assert data["repaired_reference"]["patient_counts"]["mixed"]["test"] == 4
    assert not set((output / "real_test.txt").read_text().splitlines()) & {"R0", "R1"}
    assert not set((output / "mixed_test.txt").read_text().splitlines()) & {"R0", "R1", "N0", "N1"}


def test_audit_reports_empty_conflict_sets_when_reference_is_clean(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference, clean=True)

    result, _, _, conflicts = run_audit(tmp_path, reference, "clean")

    assert result.returncode == 0, result.stderr
    data = json.loads(conflicts.read_text())
    assert data["versioned_reference"]["conflicts"] == []
    assert data["repaired_reference"]["moves"] == []
    assert data["first_attempt"]["status"] == "not_run"


def test_canonical_patient_id_collapses_lidc_series_suffixes(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference)
    (reference / "x2ct_train.txt").write_text(
        "LIDC-IDRI-0013.20000101.1234.1\n", encoding="utf-8"
    )
    (reference / "real_test.txt").write_text("LIDC-IDRI-0013\nR1\n", encoding="utf-8")
    (reference / "real_train.txt").write_text("R2\nR3\n", encoding="utf-8")
    (reference / "mixed_test.txt").write_text(
        "LIDC-IDRI-0013\nR1\nN0\nN1\n", encoding="utf-8"
    )
    (reference / "synthetic_test.txt").write_bytes((reference / "mixed_test.txt").read_bytes())
    (reference / "mixed_train.txt").write_text("R2\nR3\nN2\nN3\n", encoding="utf-8")
    (reference / "synthetic_train.txt").write_bytes((reference / "mixed_train.txt").read_bytes())

    result, output, _, conflicts = run_audit(tmp_path, reference, "series")

    assert result.returncode == 0, result.stderr
    data = json.loads(conflicts.read_text())
    assert any(row["patient"] == "LIDC-IDRI-0013" for row in data["versioned_reference"]["conflicts"])
    assert "LIDC-IDRI-0013" not in (output / "real_test.txt").read_text()


def test_first_attempt_inventory_is_separate_and_lists_conflicting_files(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference)
    names = [f"A{index}" for index in range(10)] + [f"D{index}" for index in range(10)]
    x2ct_train = reference / "x2ct_train.txt"
    x2ct_train.write_text(x2ct_train.read_text() + "".join(f"{name}\n" for name in names))
    real_root, derived_root = tmp_path / "real", tmp_path / "derived"
    for root, prefix in ((real_root, "A"), (derived_root, "D")):
        for index in range(10):
            sample = root / f"{prefix}{index}"
            sample.mkdir(parents=True)
            (sample / "sample.h5").write_bytes(b"")

    result, _, _, conflicts = run_audit(
        tmp_path,
        reference,
        "inventory",
        ("--real-root", str(real_root), "--derived-root", str(derived_root), "--test-size", "0.2"),
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(conflicts.read_text())
    assert data["first_attempt"]["status"] == "audited"
    assert data["first_attempt"]["conflict_patient_count"] > 0
    assert data["versioned_reference"]["conflict_patient_count"] == 2
    assert data["repaired_reference"]["status"] == "ready"
    first_conflict = data["first_attempt"]["conflicts"][0]
    assert first_conflict["physical_file_count"] >= 1
    assert first_conflict["files"]


def test_first_attempt_groups_different_hdf5_series_for_one_patient(tmp_path):
    reference = tmp_path / "reference"
    write_reference(reference, clean=True)
    patient = "LIDC-IDRI-0013"
    c_train = reference / "x2ct_train.txt"
    c_train.write_text(c_train.read_text() + f"{patient}.20000101.999.1\n", encoding="utf-8")
    real_root, derived_root = tmp_path / "real", tmp_path / "derived"
    real_sample = real_root / "REAL-001"
    real_sample.mkdir(parents=True)
    (real_sample / "real.h5").write_bytes(b"")
    for series in ("20000101.111.1", "20000101.222.1", "20000101.333.1"):
        sample = derived_root / f"{patient}.{series}"
        sample.mkdir(parents=True)
        (sample / "sample.h5").write_bytes(b"")

    result, _, _, conflicts = run_audit(
        tmp_path,
        reference,
        "series-inventory",
        ("--real-root", str(real_root), "--derived-root", str(derived_root), "--test-size", "0.67"),
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(conflicts.read_text())
    patient_conflict = next(
        row for row in data["first_attempt"]["conflicts"] if row["patient"] == patient
    )
    assert len(patient_conflict["internal_test_entries"]) == 4
    assert patient_conflict["physical_file_count"] == 2
