"""Recover frozen study records and constructed features from the monthly raw files."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def recover(source, ids):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    parts, offset = [], 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100000):
        first, last = np.searchsorted(ids, [offset, offset + len(batch)])
        if last > first:
            block = batch.take(ids[first:last] - offset).to_pandas()
            block["record_id"] = ids[first:last]
            parts.append(block)
        offset += len(batch)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(ids):
        raise AssertionError("Source record IDs not found")
    return result


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from data import MODEL_FEATURES, transform_raw
    from diagnostic_sampling import add_ratios, assign_candidates
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--monthly-counts", action="store_true", help="Also scan both full months to recalculate rule-qualified populations")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    baseline = joblib.load(ROOT / "models/preprocessing/od_baseline.joblib")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    verification = []
    datasets = {
        "august_all_50000": ("2026-08", ROOT / "data/real/all_50000.parquet", "label"),
        **{key: ("2026-07", ROOT / item["path"], item["truth_column"])
           for key, item in config["evaluations"].items() if key.startswith("july")},
    }
    for name, (month, frozen_path, truth) in datasets.items():
        frozen = pd.read_parquet(frozen_path)
        raw = recover(ROOT / f"data/raw/fhvhv_tripdata_{month}.parquet", frozen.record_id)
        raw = raw.set_index("record_id").loc[frozen.record_id].reset_index()
        frame, _ = transform_raw(raw, {"month": month, "rush_hours": config["rush_hours"], "exclude_shared_accessible": True})
        frame = add_ratios(baseline.transform(frame))
        np.testing.assert_array_equal(frame.record_id, frozen.record_id)
        np.testing.assert_allclose(frame[MODEL_FEATURES], frozen[MODEL_FEATURES], rtol=1e-12, atol=1e-10)
        frame[truth] = frozen[truth].to_numpy()
        if month == "2026-08":
            for column in ["dbscan_cluster_label", "split_role", "test_subset"]:
                frame[column] = frozen[column].to_numpy()
        else:
            modes, margins, _ = assign_candidates(frame)
            np.testing.assert_array_equal(modes, frozen[truth])
            if (margins < .1).any():
                raise AssertionError("External rule margin changed")
        frame.to_parquet(out / f"{name}.parquet", index=False)
        verification.append({"dataset": name, "rows": len(frame), "features_match": True})
        print(f"Recovered raw records and verified predictors: {name}", flush=True)
    if args.monthly_counts:
        import pyarrow.parquet as pq
        for month in ["2026-07", "2026-08"]:
            counts = np.zeros(5, dtype=np.int64)
            scanned = 0
            for batch in pq.ParquetFile(ROOT / f"data/raw/fhvhv_tripdata_{month}.parquet").iter_batches(batch_size=100000):
                raw = batch.to_pandas()
                frame, _ = transform_raw(raw, {"month": month, "rush_hours": config["rush_hours"], "exclude_shared_accessible": True})
                frame = add_ratios(baseline.transform(frame))
                modes, margins, _ = assign_candidates(frame)
                counts += np.bincount(modes[(modes >= 0) & (margins >= .1)], minlength=5)
                scanned += len(raw)
            (out / f"{month}_qualified_counts.json").write_text(json.dumps({
                "raw_rows": scanned, "counts": counts.tolist(),
                "note": "Raw rule counts before duplicate/overlap removal in the frozen July evaluation cohort",
            }, indent=2), encoding="utf-8")
    (out / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
