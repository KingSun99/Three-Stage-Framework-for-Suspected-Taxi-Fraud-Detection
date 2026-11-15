"""Replay frozen upstream clustering parameters on the supplied cohort."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from memory_dbscan import MemoryBoundDBSCAN
    from behavior_features import BEHAVIOR_COLUMNS, behavior_values
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ROOT / "data/real/all_50000.parquet").sort_values("record_id").reset_index(drop=True)
    scaler = joblib.load(ROOT / "models/preprocessing/robust_scaler.joblib")
    coordinates = pd.DataFrame(behavior_values(frame), columns=BEHAVIOR_COLUMNS)
    x = scaler.transform(coordinates)
    parameters = json.loads((ROOT / "models/preprocessing/clustering_parameters.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=4):
        labels, core = MemoryBoundDBSCAN(x, max_min_samples=27).fit_predict(parameters["eps"], parameters["min_samples"])
    np.testing.assert_array_equal(labels, frame.dbscan_cluster_label)
    mapped = pd.Series(labels).map({0: 0, 1: 4, 2: 3, 3: 1, 4: 2}).to_numpy()
    np.testing.assert_array_equal(mapped, frame.label)
    pd.DataFrame({"record_id": frame.record_id, "dbscan_cluster_label": labels, "label": mapped, "is_core": core}).to_parquet(out / "labels.parquet", index=False)
    frame.groupby(["label", "class_name"]).size().rename("rows").reset_index().to_csv(out / "class_counts.csv", index=False)
    (out / "verification.json").write_text(json.dumps({"all_labels_match": True, "rows": len(frame), "clusters": 5, "noise_rows": int((labels == -1).sum())}, indent=2), encoding="utf-8")
    print("All 50,000 clustering and paper labels reproduced.", flush=True)


if __name__ == "__main__":
    main()
