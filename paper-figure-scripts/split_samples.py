"""Divide samples into train and test sets using only folder names.

This script no longer inspects HDF5 internal keys. It collects patient/sample
folder names by finding all ``*.h5`` files under ``root_path`` and using the
parent folder name as the sample identifier.

Legacy usage (positional args):
    root_path save_path prefix test_size

Patient-level audit and repair:
    split_samples.py audit --reference-dir splits --output-dir splits/reprocessed

The audit mode treats the eight reference lists as read-only, can optionally
reproduce the first seed-42 attempt from real and derived HDF5 inventories, and
writes repaired internal splits and audit reports to separate destinations.

Outputs two files written to ``save_path``:
    {prefix}_train.txt
    {prefix}_test.txt

If ``prefix == mixed`` and ``real_*.txt`` plus ``synthetic_*.txt`` already
exist in ``save_path``, the script derives mixed splits from those files and
also overwrites synthetic splits so that:
    synthetic_train == mixed_train
    synthetic_test == mixed_test

The splitting is reproducible: it seeds Python's ``random`` with 42, shuffles
the (sorted) list of samples, selects the first N as test (N = floor(test_size * n)),
and writes sorted lists to the output files.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

LIDC_SERIES_ID = re.compile(r"^(LIDC-IDRI-[0-9]{4})(?:\.[0-9]+)+$")
INTERNAL_SPLITS = (
    "real_train.txt",
    "real_test.txt",
    "synthetic_train.txt",
    "synthetic_test.txt",
    "mixed_train.txt",
    "mixed_test.txt",
)
X2CT_SPLITS = ("x2ct_train.txt", "x2ct_test.txt")
SEED = 42


def canonical_patient_id(value):
    """Resolve known series IDs to patient IDs and normalize known dataset prefixes."""
    match = LIDC_SERIES_ID.fullmatch(value)
    if match:
        return match.group(1)
    for prefix in ("MIDRC-RICORD-1A-", "MIDRC-RICORD-1C-"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def read_split(path, raw=None):
    values = (path.read_bytes() if raw is None else raw).decode("utf-8").splitlines()
    if any(not value or value.strip() != value for value in values):
        raise ValueError(f"Malformed empty or whitespace-padded entry in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries in {path}")
    return values


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def plan_patient_splits(reference_dir, seed=SEED):
    """Keep test quotas by real-X-ray stratum while excluding fixed X2CT train patients."""
    files = {name: reference_dir / name for name in INTERNAL_SPLITS + X2CT_SPLITS}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError("Missing reference split files: " + ", ".join(missing))
    snapshots = {name: path.read_bytes() for name, path in files.items()}
    original = {name: read_split(path, snapshots[name]) for name, path in files.items()}
    digests = {
        name: hashlib.sha256(snapshots[name]).hexdigest()
        for name in files
    }
    sets = {name: set(values) for name, values in original.items()}

    if sets["synthetic_train.txt"] != sets["mixed_train.txt"] or sets["synthetic_test.txt"] != sets["mixed_test.txt"]:
        raise ValueError("Versioned synthetic and mixed assignments must match")

    patient_of = {entry: canonical_patient_id(entry) for name in files for entry in original[name]}
    real_pool = {patient_of[x] for x in original["real_train.txt"] + original["real_test.txt"]}
    mixed_pool = {patient_of[x] for x in original["mixed_train.txt"] + original["mixed_test.txt"]}
    real_train = {patient_of[x] for x in original["real_train.txt"]}
    real_test = {patient_of[x] for x in original["real_test.txt"]}
    mixed_train = {patient_of[x] for x in original["mixed_train.txt"]}
    mixed_test = {patient_of[x] for x in original["mixed_test.txt"]}
    if mixed_train & mixed_test or real_train & real_test:
        raise ValueError("A reference patient appears in both internal train and test")
    for name in X2CT_SPLITS:
        if not original[name] or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", entry) and not LIDC_SERIES_ID.fullmatch(entry)
            for entry in original[name]
        ):
            raise ValueError(f"Empty list or unresolved X2CT patient identity in {name}")
    if not real_pool <= mixed_pool:
        raise ValueError("Real patient universe is not a subset of mixed patient universe")
    if real_train != mixed_train & real_pool or real_test != mixed_test & real_pool:
        raise ValueError("Versioned real and mixed assignments disagree for the real-X-ray stratum")
    if len(real_test) != len(original["real_test.txt"]) or len(mixed_test) != len(original["mixed_test.txt"]):
        raise ValueError("Multiple list entries resolve to one patient; identity mapping needs review")

    x2ct_train = {patient_of[x] for x in original["x2ct_train.txt"]}
    x2ct_test = {patient_of[x] for x in original["x2ct_test.txt"]}
    conflicts = []
    for variant, split_name in (("real", "real_test.txt"), ("mixed", "mixed_test.txt"), ("synthetic", "synthetic_test.txt")):
        for entry in original[split_name]:
            patient = patient_of[entry]
            if patient in x2ct_train:
                conflicts.append({"patient_id": patient, "variant": variant, "split": "test", "entry": entry})

    new_test = set()
    strata = (
        ("real_xray", real_pool, real_test),
        ("no_real_xray", mixed_pool - real_pool, mixed_test - real_pool),
    )
    stratum_report = []
    moves = []
    for stratum, pool, previous_test in strata:
        eligible_test = previous_test - x2ct_train
        replacements = sorted(pool - previous_test - x2ct_train)
        replacement_count = len(previous_test) - len(eligible_test)
        if len(replacements) < replacement_count:
            raise ValueError(
                f"Cannot preserve {stratum} test quota: need {replacement_count} replacements, "
                f"only {len(replacements)} eligible train patients"
            )
        random.Random(seed).shuffle(replacements)
        chosen_replacements = replacements[:replacement_count]
        selected = eligible_test | set(chosen_replacements)
        if len(selected) != len(previous_test) or selected & x2ct_train:
            raise ValueError(f"Internal validation failed for {stratum}")
        for removed, replacement in zip(sorted(previous_test & x2ct_train), chosen_replacements):
            moves.append({
                "stratum": stratum,
                "patient_id": removed,
                "from": "test",
                "to": "train",
                "replacement_patient_id": replacement,
                "replacement_from": "train",
                "replacement_to": "test",
            })
        new_test.update(selected)
        stratum_report.append({
            "name": stratum,
            "universe_patient_count": len(pool),
            "test_patient_count_before": len(previous_test),
            "test_patient_count_after": len(selected),
            "replaced_patient_count": replacement_count,
            "eligible_test_patient_count_before": len(eligible_test),
        })

    outputs = {}
    for variant in ("real", "synthetic", "mixed"):
        train_name, test_name = f"{variant}_train.txt", f"{variant}_test.txt"
        all_entries = original[train_name] + original[test_name]
        outputs[train_name] = sorted(entry for entry in all_entries if patient_of[entry] not in new_test)
        outputs[test_name] = sorted(entry for entry in all_entries if patient_of[entry] in new_test)
        if len(outputs[train_name]) != len(original[train_name]) or len(outputs[test_name]) != len(original[test_name]):
            raise ValueError(f"Entry-count invariant failed for {variant}")
        if set(outputs[train_name]) | set(outputs[test_name]) != set(all_entries):
            raise ValueError(f"Patient universe changed for {variant}")
        if {patient_of[x] for x in outputs[test_name]} & x2ct_train:
            raise ValueError(f"X2CT training overlap remains in {variant} test")
        if {patient_of[x] for x in outputs[train_name]} & {patient_of[x] for x in outputs[test_name]}:
            raise ValueError(f"Patient appears in both {variant} train and test")
    if outputs["synthetic_train.txt"] != outputs["mixed_train.txt"] or outputs["synthetic_test.txt"] != outputs["mixed_test.txt"]:
        raise ValueError("Generated synthetic and mixed assignments do not match")

    return {
        "original": original,
        "digests": digests,
        "patient_of": patient_of,
        "x2ct_train": x2ct_train,
        "x2ct_test": x2ct_test,
        "conflicts": conflicts,
        "strata": stratum_report,
        "moves": moves,
        "outputs": outputs,
        "new_test_patients": new_test,
    }


def split_inventory(root, test_size, seed, inventory_name):
    """Reproduce the legacy first attempt from HDF5 sample-folder names."""
    if not 0 < test_size < 1:
        raise ValueError("Test size must be strictly between 0 and 1")
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Inventory root is not a directory: {root}")
    files_by_sample = {}
    for path in sorted(root.rglob("*.h5")):
        files_by_sample.setdefault(path.parent.name, []).append(
            f"{inventory_name}/{path.relative_to(root).as_posix()}"
        )
    samples = sorted(files_by_sample)
    if not samples:
        raise ValueError(f"No HDF5 samples found under {root}")
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    test_count = int(test_size * len(shuffled))
    test = set(shuffled[:test_count])
    return {
        "train": set(shuffled[test_count:]),
        "test": test,
        "files_by_sample": files_by_sample,
    }


def first_attempt_from_inventories(real_root, derived_root, test_size, seed):
    """Reproduce separate real/derived splits and the legacy mixed precedence."""
    real = split_inventory(real_root, test_size, seed, "real")
    synthetic = split_inventory(derived_root, test_size, seed, "derived")
    real_all = real["train"] | real["test"]
    mixed_files = {sample: list(paths) for sample, paths in synthetic["files_by_sample"].items()}
    for sample, paths in real["files_by_sample"].items():
        mixed_files.setdefault(sample, []).extend(paths)
    mixed = {
        "train": (synthetic["train"] - real_all) | real["train"],
        "test": (synthetic["test"] - real_all) | real["test"],
        "files_by_sample": mixed_files,
    }
    synthetic = {**mixed, "files_by_sample": synthetic["files_by_sample"]}
    return {"real": real, "synthetic": synthetic, "mixed": mixed}


def inventory_conflicts(attempt, x2ct_train_entries):
    x2ct_train = set(x2ct_train_entries)
    conflicts = []
    for variant, splits in attempt.items():
        for sample in sorted(splits["test"]):
            patient = canonical_patient_id(sample)
            if patient in x2ct_train:
                files = splits["files_by_sample"].get(sample, [])
                conflicts.append({
                    "patient_id": patient,
                    "variant": variant,
                    "split": "test",
                    "entry": sample,
                    "source_list": f"{variant}_test.txt",
                    "against_list": "x2ct_train.txt",
                    "against_entries": x2ct_train_entries[patient],
                    "physical_file_count": len(files),
                    "files": files,
                })
    return conflicts


def reference_conflicts(reference_dir):
    rows = {}
    for variant in ("real", "synthetic", "mixed"):
        for entry in read_split(reference_dir / f"{variant}_test.txt"):
            patient = canonical_patient_id(entry)
            rows.setdefault(patient, []).append({
                "variant": variant,
                "split": "test",
                "entry": entry,
                "source_list": f"{variant}_test.txt",
                "physical_file_count": None,
                "files": [],
            })
    x2ct_train_entries = {}
    for entry in read_split(reference_dir / "x2ct_train.txt"):
        x2ct_train_entries.setdefault(canonical_patient_id(entry), []).append(entry)
    return [
        {
            "patient_id": patient,
            "entries": entries,
            "against": {"list": "x2ct_train.txt", "entries": x2ct_train_entries[patient]},
            "resolution": "pending",
        }
        for patient, entries in sorted(rows.items())
        if patient in x2ct_train_entries
    ]


def group_conflicts(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append({
            key: value for key, value in row.items() if key != "patient_id"
        })
    return [
        {"patient_id": patient, "entries": entries}
        for patient, entries in sorted(grouped.items())
    ]


def write_audit_reports(plan, report_path, conflicts_path, seed=SEED, test_size=0.1, first_attempt=None, first_attempt_status=None):
    original = plan["original"]
    outputs = plan["outputs"]
    patient_of = plan["patient_of"]
    initial = plan["conflicts"]
    conflict_patients = sorted({row["patient_id"] for row in initial})
    x2ct_train_entries = {}
    for entry in original["x2ct_train.txt"]:
        x2ct_train_entries.setdefault(patient_of[entry], []).append(entry)
    reference_conflicts = group_conflicts([
        {
            **row,
            "source_list": f"{row['variant']}_{row['split']}.txt",
            "physical_file_count": None,
            "files": [],
            "against_entries": x2ct_train_entries[row["patient_id"]],
        }
        for row in initial
    ])
    attempt_conflicts = group_conflicts(inventory_conflicts(first_attempt, x2ct_train_entries)) if first_attempt else []
    internal_train_before = set().union(*(
        {patient_of[x] for x in original[f"{variant}_train.txt"]}
        for variant in ("real", "synthetic", "mixed")
    ))
    internal_train_after = set().union(*(
        {patient_of[x] for x in outputs[f"{variant}_train.txt"]}
        for variant in ("real", "synthetic", "mixed")
    ))

    def compact(rows, resolution):
        result = []
        for patient in rows:
            entries = patient["entries"]
            files = sorted({path for entry in entries for path in entry["files"]})
            item = {
                "patient": patient["patient_id"],
                "internal_test_entries": [f"{entry['source_list']}:{entry['entry']}" for entry in entries],
                "x2ct_train_entries": entries[0]["against_entries"],
            }
            if any(entry["physical_file_count"] is not None for entry in entries):
                item["physical_file_count"] = len(files)
                item["files"] = files
            item["resolution"] = resolution or (
                "fixed_in_versioned_reference" if patient["patient_id"] in repaired_patients
                else "audit_only_candidate_not_repaired"
            )
            result.append(item)
        return result

    repaired_patients = {move["patient_id"] for move in plan["moves"]}
    moves = []
    for move in plan["moves"]:
        moves.append({
            "stratum": move["stratum"],
            "test_to_train": move["patient_id"],
            "train_to_test": move["replacement_patient_id"],
        })
    attempt_entry_count = sum(len(row["entries"]) for row in attempt_conflicts)
    reference_entry_count = sum(len(row["entries"]) for row in reference_conflicts)
    conflict_report = {
        "seed": seed,
        "status": "ready",
        "reference_sha256": plan["digests"],
        "first_attempt": {
            "status": first_attempt_status.get("status", "audited") if first_attempt_status else ("audited" if first_attempt else "not_run"),
            "reason": first_attempt_status.get("reason") if first_attempt_status else None,
            "test_size": test_size if first_attempt else None,
            "entry_counts": {
                variant: {split: len(data[split]) for split in ("train", "test")}
                for variant, data in first_attempt.items()
            } if first_attempt else {},
            "seed": SEED,
            "conflict_patient_count": len(attempt_conflicts) if first_attempt else None,
            "conflict_entry_count": attempt_entry_count if first_attempt else None,
            "conflicts": compact(attempt_conflicts, None),
        },
        "versioned_reference": {
            "status": "audited",
            "conflict_patient_count": len(reference_conflicts),
            "conflict_entry_count": reference_entry_count,
            "entry_counts": {
                variant: {split: len(original[f"{variant}_{split}.txt"]) for split in ("train", "test")}
                for variant in ("real", "synthetic", "mixed")
            },
            "patient_counts": {
                variant: {
                    split: len({patient_of[x] for x in original[f"{variant}_{split}.txt"]})
                    for split in ("train", "test")
                }
                for variant in ("real", "synthetic", "mixed")
            },
            "patient_universe_preserved": {
                variant: {patient_of[x] for x in original[f"{variant}_train.txt"] + original[f"{variant}_test.txt"]}
                == {patient_of[x] for x in outputs[f"{variant}_train.txt"] + outputs[f"{variant}_test.txt"]}
                for variant in ("real", "synthetic", "mixed")
            },
            "conflicts": compact(reference_conflicts, "fixed"),
        },
        "repaired_reference": {
            "status": "ready",
            "scope": "split_lists_only",
            "full_cohort_verified": False,
            "remaining_internal_test_x2ct_train_patient_count": 0,
            "entry_counts": {
                variant: {split: len(outputs[f"{variant}_{split}.txt"]) for split in ("train", "test")}
                for variant in ("real", "synthetic", "mixed")
            },
            "patient_counts": {
                variant: {
                    split: len({patient_of[x] for x in outputs[f"{variant}_{split}.txt"]})
                    for split in ("train", "test")
                }
                for variant in ("real", "synthetic", "mixed")
            },
            "moves": moves,
        },
        "summary": {
            "x2ct_test_internal_train_by_variant": {
                variant: {
                    stage: len({patient_of[x] for x in lists[f"{variant}_train.txt"]} & plan["x2ct_test"])
                    for stage, lists in (("before", original), ("after", outputs))
                }
                for variant in ("real", "synthetic", "mixed")
            },
            "initial_conflict_patient_count": len(conflict_patients),
            "initial_conflict_entry_count": len(initial),
            "initial_conflict_list_count": len({
                f"{row['variant']}_{row['split']}.txt" for row in initial
            }),
            "remaining_internal_test_x2ct_train_patient_count": 0,
            "x2ct_train_test_patient_overlap_count": len(plan["x2ct_train"] & plan["x2ct_test"]),
            "x2ct_test_internal_train_variant_pair_count": sum(
                len({patient_of[x] for x in original[f"{variant}_train.txt"]} & plan["x2ct_test"])
                for variant in ("real", "synthetic", "mixed")
            ),
            "x2ct_test_internal_train_distinct_patient_count": len(plan["x2ct_test"] & internal_train_before),
            "x2ct_test_internal_train_after_variant_pair_count": sum(
                len({patient_of[x] for x in outputs[f"{variant}_train.txt"]} & plan["x2ct_test"])
                for variant in ("real", "synthetic", "mixed")
            ),
            "x2ct_test_internal_train_after_distinct_patient_count": len(plan["x2ct_test"] & internal_train_after),
        },
        "external_model_review": {
            "validation_and_checkpoint_use": {
                "status": "not_assessed",
                "reason": "Training repositories for the compared models are outside this repository.",
            },
            "pairing": [
                {"models": pair, "status": "not_assessed", "reason": "Split membership alone does not establish paired cases and comparable outcomes."}
                for pair in ("C/E", "C/M", "E/M")
            ],
        },
        "strata": plan["strata"],
    }
    write_atomic(conflicts_path, json.dumps(conflict_report, indent=1, ensure_ascii=False) + "\n")

    lines = [
        "# Split audit",
        "",
        f"Reference SHA-256 values and replacement seed `{seed}` are recorded in `{conflicts_path.name}`.",
        "These are list-verified candidates. Full-cohort membership and the reprocessed physical inventory have not been verified.",
        f"First-attempt reproduction status: `{first_attempt_status.get('status') if first_attempt_status else ('audited' if first_attempt else 'not_run')}`. The JSON separates it from the versioned reference and repaired assignments.",
        "Test patients present in fixed `x2ct_train.txt` were moved to internal train; replacements were selected from train within the same real-X-ray stratum.",
        "",
        "| Variant | Train entries before | Test entries before | Train entries after | Test entries after | Test patients in X2CT train after |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("real", "synthetic", "mixed"):
        train, test = f"{variant}_train.txt", f"{variant}_test.txt"
        after_overlap = len({patient_of[x] for x in outputs[test]} & plan["x2ct_train"])
        lines.append(
            f"| {variant} | {len(original[train])} | {len(original[test])} | {len(outputs[train])} | {len(outputs[test])} | {after_overlap} |"
        )
    lines.extend([
        "",
        "## Stratification",
        "",
        "The existing design stratifies by real-X-ray availability: real patients form one stratum; the mixed universe excluding those patients forms the no-real-X-ray stratum. Test quotas and total list-entry counts are preserved. Synthetic and mixed outputs remain identical, and real assignments retain precedence.",
        "",
        "| Stratum | Patient universe | Test patients before | Eligible original test | Replaced | Test patients after |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for stratum in plan["strata"]:
        lines.append(
            f"| {stratum['name']} | {stratum['universe_patient_count']} | {stratum['test_patient_count_before']} | {stratum['eligible_test_patient_count_before']} | {stratum['replaced_patient_count']} | {stratum['test_patient_count_after']} |"
        )
    lines.extend([
        "",
        "## Limits recorded separately",
        "",
        f"The fixed X2CT train/test patient intersection is {conflict_report['summary']['x2ct_train_test_patient_overlap_count']}; the split generator does not edit X2CT. X2CT-test/internal-train overlap changes from {conflict_report['summary']['x2ct_test_internal_train_distinct_patient_count']} to {conflict_report['summary']['x2ct_test_internal_train_after_distinct_patient_count']} distinct patients after repair. This is separate from the requested X2CT-train/internal-test exclusion.",
        "Physical HDF5 file counts are unavailable from split lists alone and are omitted when no inventory is supplied; entries and patient counts are distinct measures.",
        "The external model training repositories and Figure 2 source were not available here; validation/checkpoint use and C/E/M outcome pairing are explicitly not assessed.",
        "",
    ])
    write_atomic(report_path, "\n".join(lines))


def audit(args):
    reference_dir = Path(args.reference_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == reference_dir:
        raise ValueError("Output directory must differ from the versioned reference directory")
    report_path, conflicts_path = Path(args.report).resolve(), Path(args.conflicts).resolve()
    reference_paths = {(reference_dir / name).resolve() for name in INTERNAL_SPLITS + X2CT_SPLITS}
    output_paths = {(output_dir / name).resolve() for name in INTERNAL_SPLITS}
    destinations = {report_path, conflicts_path}
    if len(destinations) != 2 or destinations & (reference_paths | output_paths) or any(
        path == output_dir or output_dir in path.parents for path in destinations
    ):
        raise ValueError("Report and conflict JSON destinations must be distinct from split files and each other")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        error = {"seed": args.seed, "status": "blocked", "reason": f"Output directory is not empty: {output_dir}"}
        write_atomic(conflicts_path, json.dumps(error, indent=2) + "\n")
        write_atomic(report_path, f"# Split audit blocked\n\n{error['reason']}\n")
        return 2

    attempt, attempt_status = None, {"status": "not_run", "reason": "Inventory roots were not provided"}
    if args.real_root and args.derived_root:
        try:
            attempt = first_attempt_from_inventories(args.real_root, args.derived_root, args.test_size, SEED)
            attempt_status = {"status": "audited", "test_size": args.test_size}
        except (ValueError, OSError) as exc:
            attempt_status = {"status": "not_run", "reason": str(exc)}
    elif args.real_root or args.derived_root:
        attempt_status = {"status": "not_run", "reason": "Both --real-root and --derived-root are required"}
    try:
        plan = plan_patient_splits(reference_dir, args.seed)
        publish_splits(plan, reference_dir, output_dir)
    except (ValueError, OSError) as exc:
        try:
            prior_conflicts = reference_conflicts(reference_dir)
            reference_status = "audited"
        except (ValueError, OSError):
            prior_conflicts = []
            reference_status = "not_run"
        try:
            against = {}
            for entry in read_split(reference_dir / "x2ct_train.txt"):
                against.setdefault(canonical_patient_id(entry), []).append(entry)
        except (ValueError, OSError):
            attempt = None
            attempt_status = {"status": "not_run", "reason": "Unreadable X2CT training reference"}
        error = {
            "seed": args.seed,
            "status": "blocked",
            "reason": str(exc),
            "first_attempt": {**attempt_status, "conflicts": group_conflicts(inventory_conflicts(
                attempt,
                against,
            )) if attempt else []},
            "versioned_reference": {"status": reference_status, "conflicts": prior_conflicts},
            "repaired_reference": {"status": "blocked", "reason": str(exc), "moves": []},
            "conflicts": prior_conflicts,
        }
        write_atomic(conflicts_path, json.dumps(error, indent=2) + "\n")
        write_atomic(report_path, f"# Split audit blocked\n\n{exc}\n")
        return 2
    write_audit_reports(plan, report_path, conflicts_path, args.seed, args.test_size, attempt, attempt_status)
    return 0


def publish_splits(plan, reference_dir, output_dir):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        output_dir.rmdir()
    with tempfile.TemporaryDirectory(prefix="split-audit-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "splits"
        staged.mkdir()
        for name, entries in plan["outputs"].items():
            (staged / name).write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        for name, entries in plan["outputs"].items():
            if read_split(staged / name) != entries:
                raise ValueError(f"Staged output differs from validated plan: {name}")
        for name, digest in plan["digests"].items():
            if hashlib.sha256((reference_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Reference changed during split generation: {name}")
        staged.rename(output_dir)


def main(args):
    # Reproducible shuffle
    random.seed(42)

    SAVE_PATH = Path(args.save_path)
    DATA_ROOT = Path(args.root_path)
    test_size = args.test_size
    prefix = args.prefix

    if prefix == "mixed":
        real_train_file = SAVE_PATH / "real_train.txt"
        real_test_file = SAVE_PATH / "real_test.txt"
        synth_train_file = SAVE_PATH / "synthetic_train.txt"
        synth_test_file = SAVE_PATH / "synthetic_test.txt"

        if (
            real_train_file.exists()
            and real_test_file.exists()
            and synth_train_file.exists()
            and synth_test_file.exists()
        ):
            with open(real_train_file) as f:
                real_train = set(f.read().splitlines())
            with open(real_test_file) as f:
                real_test = set(f.read().splitlines())
            with open(synth_train_file) as f:
                synth_train = set(f.read().splitlines())
            with open(synth_test_file) as f:
                synth_test = set(f.read().splitlines())

            real_all = real_train.union(real_test)

            mixed_train = (synth_train - real_all).union(real_train)
            mixed_test = (synth_test - real_all).union(real_test)

            SAVE_PATH.mkdir(parents=True, exist_ok=True)

            with open(SAVE_PATH / "mixed_train.txt", "w") as f:
                out = sorted(list(mixed_train))
                lines = [f"{line}\n" for line in out]
                f.writelines(lines)

            with open(SAVE_PATH / "mixed_test.txt", "w") as f:
                out = sorted(list(mixed_test))
                lines = [f"{line}\n" for line in out]
                f.writelines(lines)

            # Keep synthetic splits aligned with mixed splits to avoid
            # cross-variant leakage when training synthetic and evaluating
            # against mixed/real test sets.
            with open(SAVE_PATH / "synthetic_train.txt", "w") as f:
                out = sorted(list(mixed_train))
                lines = [f"{line}\n" for line in out]
                f.writelines(lines)

            with open(SAVE_PATH / "synthetic_test.txt", "w") as f:
                out = sorted(list(mixed_test))
                lines = [f"{line}\n" for line in out]
                f.writelines(lines)

            return

    # Collect sample names from parent folder names of any .h5 files
    samples = []
    for filepath in sorted(list(DATA_ROOT.rglob("*.h5"))):
        samples.append(filepath.parent.name)

    # Unique sample names
    samples = sorted(set(samples))

    # Deterministic shuffle then split
    random.shuffle(samples)
    test_count = int(test_size * len(samples))
    test_set = set(samples[:test_count])
    train_set = set(samples[test_count:])

    # Sanity: disjoint and cover all
    assert test_set.isdisjoint(train_set)
    assert test_set.union(train_set) == set(samples)

    # Write outputs (sorted in output files)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with open(SAVE_PATH / f"{prefix}_train.txt", "w") as f:
        out = sorted(list(train_set))
        lines = [f"{line}\n" for line in out]
        f.writelines(lines)

    with open(SAVE_PATH / f"{prefix}_test.txt", "w") as f:
        out = sorted(list(test_set))
        lines = [f"{line}\n" for line in out]
        f.writelines(lines)


if __name__ == "__main__":
    if sys.argv[1:2] == ["audit"]:
        parser = argparse.ArgumentParser(description="Audit and repair patient-level train/test splits")
        parser.add_argument("audit")
        parser.add_argument("--reference-dir", default="splits")
        parser.add_argument("--output-dir", default="splits/reprocessed")
        parser.add_argument("--report", default="reports/split-audit.md")
        parser.add_argument("--conflicts", default="reports/split-conflicts.json")
        parser.add_argument("--seed", type=int, default=SEED)
        parser.add_argument("--real-root")
        parser.add_argument("--derived-root")
        parser.add_argument("--test-size", type=float, default=0.1)
        audit_args = parser.parse_args()
        raise SystemExit(audit(audit_args))

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path")
    parser.add_argument("save_path")
    parser.add_argument("prefix", help="dataset prefix for output files, e.g. 'synthetic'")
    parser.add_argument("test_size", type=float)
    main(parser.parse_args())
