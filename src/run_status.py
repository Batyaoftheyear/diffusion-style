"""Завершение существующего прогона с явными пропусками, без генерации."""
from datetime import datetime, timezone

from .config import data_dir, fingerprint, safe_path
from .data import validate_dataset
from .inference import complete_records, current_run, export_records
from .io import file_sha, read_json, read_jsonl, write_csv, write_json


def verified_outcomes(root, run, photos, records):
    expected = {photo["id"] for photo in photos}
    successful = {row["photo_id"] for row in records}
    skipped, seen = [], set()
    for item in run.get("skips", []):
        path = safe_path(root, item["annotation"])
        if file_sha(path) != item["sha256"]:
            raise ValueError("Skip annotation changed")
        row = read_json(path)
        photo_id = row["photo_id"]
        if row["run_id"] != run["run_id"] or photo_id not in expected or photo_id in successful or photo_id in seen:
            raise ValueError("Invalid or conflicting skip annotation")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("A skip must have an explicit reason")
        seen.add(photo_id)
        skipped.append(row)
    missing = expected - successful - seen
    if missing or not records:
        raise ValueError(f"Unfinished baseline; missing results for {sorted(missing)}. Annotate known skips explicitly.")
    if skipped:
        if not run.get("processed_complete") or run.get("completed_images") != len(records) or run.get("skipped_images") != len(skipped):
            raise ValueError("Run progress is inconsistent; finalize the existing run")
    elif not run.get("complete"):
        raise ValueError("Run is not marked complete")
    identity = fingerprint({"outputs": {row["photo_id"]: row["output_sha256"] for row in records}, "skips": run.get("skips", [])})
    return skipped, identity


def finalize_run(c, skip_photo_ids=(), reason=None):
    validate_dataset(c)
    root, run = current_run(c)
    photos = read_jsonl(data_dir(c) / "coco_manifest.jsonl")
    records = complete_records(root, photos, run["run_id"])
    successful = {row["photo_id"] for row in records}
    expected = {photo["id"] for photo in photos}
    existing = {read_json(safe_path(root, item["annotation"]))["photo_id"]: item for item in run.get("skips", [])}
    requested = set(skip_photo_ids)
    if requested and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Specify a reason for skipped photos")
    if requested - expected or requested & successful:
        raise ValueError("Cannot skip an unknown photo or replace an existing successful result")
    missing = expected - successful - set(existing) - requested
    if missing or not records:
        raise ValueError(f"Other photos still need generation: {sorted(missing)}; existing files were not modified")
    skips = dict(existing)
    for photo_id in sorted(requested):
        annotation = f"skips/{photo_id}.json"
        path = safe_path(root, annotation)
        if path.exists():
            row = read_json(path)
            if row.get("run_id") != run["run_id"] or row.get("photo_id") != photo_id or row.get("reason") != reason:
                raise ValueError("Existing skip annotation conflicts with the request")
        else:
            write_json(path, {"run_id": run["run_id"], "photo_id": photo_id, "status": "skipped", "reason": reason,
                              "source": "explicit user annotation", "created_at": datetime.now(timezone.utc).isoformat()})
        skips[photo_id] = {"annotation": annotation, "sha256": file_sha(path)}
    run = dict(run, completed_images=len(records), skipped_images=len(skips), processed_images=len(expected),
               processed_complete=True, complete=len(records) == len(expected), skips=list(skips.values()))
    skipped, outcomes_id = verified_outcomes(root, run, photos, records)
    run["outcomes_id"] = outcomes_id
    write_json(root / "run.json", run)
    reasons = {row["photo_id"]: row["reason"] for row in skipped}
    write_csv(root / "coverage.csv", [{"photo_id": photo["id"], "status": "generated" if photo["id"] in successful else "skipped",
                                      "reason": reasons.get(photo["id"], "")} for photo in photos])
    if skipped:
        write_csv(root / "skipped.csv", [{"photo_id": row["photo_id"], "reason": row["reason"]} for row in skipped])
    export_records(root, records)
    from .visualization import comparison_grid
    comparison_grid(c, root, records, max_rows=6)
    print(f"Existing run {run['run_id']}: {len(records)} generated, {len(skipped)} explicitly skipped, {len(expected)} inputs. {root}", flush=True)
    return root


def reuse_finalized_run(c):
    try:
        root, run = current_run(c)
    except (FileNotFoundError, ValueError):
        return False
    if not run.get("skips"):
        return False
    validate_dataset(c)
    photos = read_jsonl(data_dir(c) / "coco_manifest.jsonl")
    records = complete_records(root, photos, run["run_id"])
    skipped, _ = verified_outcomes(root, run, photos, records)
    print(f"Reusing finalized run {run['run_id']}: {len(records)} generated, {len(skipped)} skipped; no generation required", flush=True)
    return True
