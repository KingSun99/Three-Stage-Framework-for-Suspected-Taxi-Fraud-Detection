"""Portable entry point for the frozen August-prior experiment."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def protocol():
    return read_json(ROOT / "config.json")


def verify(args):
    manifest = read_json(ROOT / "MANIFEST.json")
    for name, item in manifest.items():
        path = ROOT / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"Integrity mismatch: {name}")
    print(f"Verified {len(manifest)} files.", flush=True)


def seed_split(seed):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    original = pd.read_parquet(ROOT / "data/real/train_45000.parquet")
    ids, validation_ids = train_test_split(
        np.arange(len(original)), test_size=.2, random_state=seed, stratify=original.label,
    )
    return (
        original.iloc[ids].reset_index(drop=True),
        original.iloc[validation_ids].reset_index(drop=True), original,
    )


def split(args):
    import numpy as np
    import pandas as pd
    out = fresh(args.out)
    for seed in args.seeds:
        train, validation, _ = seed_split(seed)
        folder = out / f"seed_{seed}"
        folder.mkdir()
        for frame, name in [(train, "train"), (validation, "validation")]:
            frame.to_parquet(folder / f"{name}.parquet", index=False)
        print(
            f"seed={seed}: stratified split created "
            f"({len(train):,} train / {len(validation):,} validation rows).",
            flush=True,
        )


def evaluate(model, frame, truth, destination, dataset):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
    from data import MODEL_FEATURES
    prediction = model.predict(frame[MODEL_FEATURES]).astype(int)
    y = frame[truth].to_numpy(int)
    matrix = confusion_matrix(y, prediction, labels=range(5))
    pct = np.divide(100 * matrix, matrix.sum(1, keepdims=True), out=np.zeros((5, 5)), where=matrix.sum(1, keepdims=True) > 0)
    pd.DataFrame(matrix, index=range(5), columns=range(5)).to_csv(destination / f"confusion_matrix_{dataset}.csv")
    pd.DataFrame(pct, index=range(5), columns=range(5)).to_csv(destination / f"confusion_matrix_{dataset}_row_percent.csv")
    cells = [[f"{matrix[i,j]} ({pct[i,j]:.2f}%)" for j in range(5)] for i in range(5)]
    pd.DataFrame(cells, index=range(5), columns=range(5)).to_csv(destination / f"confusion_matrix_{dataset}_count_rowpct.csv")
    pd.DataFrame({"row_index": np.arange(len(y)), "true_label": y, "predicted_label": prediction}).to_parquet(
        destination / f"predictions_{dataset}.parquet", index=False,
    )
    save_json(destination / f"class_metrics_{dataset}.json", classification_report(y, prediction, labels=range(5), output_dict=True, zero_division=0))
    return {
        "dataset": dataset, "rows": len(y),
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, labels=range(5), average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
    }, prediction


def evaluation_tables():
    import pandas as pd
    return {
        key: (pd.read_parquet(ROOT / item["path"]), item["truth_column"])
        for key, item in protocol()["evaluations"].items()
    }


def predict(args):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    out = fresh(args.out)
    tables, rows = evaluation_tables(), []
    for seed in args.seeds:
        for method in args.methods:
            source = ROOT / f"models/xgb/seed_{seed}/{method}"
            destination = out / f"seed_{seed}/{method}"
            destination.mkdir(parents=True)
            model = xgb.XGBClassifier(n_jobs=4)
            model.load_model(bytearray((source / "xgb_final.ubj").read_bytes()))
            for dataset, (frame, truth) in tables.items():
                metrics, prediction = evaluate(model, frame, truth, destination, dataset)
                frozen_path = ROOT / f"results/seed_{seed}/{method}/predictions_{dataset}.parquet"
                frozen_match = None
                if frozen_path.exists():
                    frozen = pd.read_parquet(frozen_path)
                    np.testing.assert_array_equal(prediction, frozen.predicted_label)
                    np.testing.assert_array_equal(frame[truth], frozen.true_label)
                    frozen_match = True
                rows.append({"seed": seed, "method": method, **metrics, "all_predictions_match": frozen_match})
            print(f"Verified predictions: seed={seed}, method={method}", flush=True)
    pd.DataFrame(rows).to_csv(out / "prediction_verification.csv", index=False)
    summarize(pd.DataFrame(rows), out)


def summarize(frame, out):
    frame.to_csv(out / "metrics_all_seeds.csv", index=False)
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    result = frame.groupby(["method", "dataset"])[metrics].agg(["mean", "std", "count"])
    result.columns = ["_".join(x) for x in result.columns]
    result.reset_index().to_csv(out / "metrics_mean_sd.csv", index=False)


def targets(train):
    expected = {int(k): v for k, v in protocol()["bo_target_counts"].items()}
    return {label: expected[label] - int(train.label.eq(label).sum()) for label in range(5)}


def model_table(seed, method, generated=None):
    import pandas as pd
    from data import MODEL_FEATURES
    train, validation, original = seed_split(seed)
    columns = MODEL_FEATURES + ["label"]
    if method in ("xgb", "bo_xgb"):
        return train[columns], validation[columns], original[columns]
    path = (Path(generated) / f"seed_{seed}/{method}/synthetic.parquet" if generated else
            ROOT / f"data/synthetic/seed_{seed}/{method}.parquet")
    synthetic = pd.read_parquet(path)[columns]
    bo = pd.concat([train[columns], synthetic], ignore_index=True)
    observed = {str(int(k)): int(v) for k, v in bo.label.value_counts().sort_index().items()}
    if observed != protocol()["bo_target_counts"]:
        raise AssertionError(f"Target counts changed: {observed}")
    return bo, validation[columns], pd.concat([bo, validation[columns]], ignore_index=True)


def train_xgb(args):
    import pandas as pd
    import xgboost as xgb
    from skopt import forest_minimize
    from data import MODEL_FEATURES
    from search import search_space, params_from_values
    out = fresh(args.out)
    spec, tables, all_rows = protocol(), evaluation_tables(), []
    for seed in args.seeds:
        for method in args.methods:
            folder = out / f"seed_{seed}/{method}"
            folder.mkdir(parents=True)
            bo, validation, final = model_table(seed, method, args.generated)
            base = {**spec["base_xgb"], "random_state": seed}
            trace = []
            if method == "xgb":
                best = spec["fixed_xgb"]
            elif args.retune:
                dimensions = search_space()

                def objective(values):
                    from sklearn.metrics import accuracy_score, f1_score
                    parameters = params_from_values(dimensions, values)
                    model = xgb.XGBClassifier(**{**base, **parameters})
                    started = time.perf_counter()
                    model.fit(bo[MODEL_FEATURES], bo.label)
                    prediction = model.predict(validation[MODEL_FEATURES])
                    accuracy = accuracy_score(validation.label, prediction)
                    trace.append({
                        "seed": seed, "iteration": len(trace) + 1,
                        "validation_accuracy": accuracy,
                        "validation_macro_f1": f1_score(validation.label, prediction, labels=range(5), average="macro", zero_division=0),
                        "trial_seconds": time.perf_counter() - started, **parameters,
                    })
                    pd.DataFrame(trace).to_csv(folder / "bo_trace.csv", index=False)
                    print(f"BO seed={seed} {method} {len(trace)}/50 accuracy={accuracy:.6f}", flush=True)
                    return -accuracy

                optimum = forest_minimize(
                    objective, dimensions, base_estimator="ET", acq_func="EI",
                    xi=.01, n_calls=50, n_initial_points=20, random_state=seed,
                )
                best = params_from_values(dimensions, optimum.x)
            else:
                best = read_json(ROOT / f"models/xgb/seed_{seed}/{method}/best_parameters.json")["model_parameters"]
            model = xgb.XGBClassifier(**{**base, **best})
            model.fit(final[MODEL_FEATURES], final.label)
            (folder / "xgb_final.ubj").write_bytes(model.get_booster().save_raw(raw_format="ubj"))
            save_json(folder / "best_parameters.json", {
                "seed": seed, "method": method, "model_parameters": model.get_params(),
                "bo_rows": len(bo) if method != "xgb" else 0,
                "final_fit_rows": len(final), "retuned": args.retune and method != "xgb",
            })
            for dataset, (frame, truth) in tables.items():
                row, _ = evaluate(model, frame, truth, folder, dataset)
                all_rows.append({"seed": seed, "method": method, **row})
            summarize(pd.DataFrame(all_rows), out)
            print(f"Final fit and evaluation complete: {seed}/{method}", flush=True)


def train_gan(args):
    import gc
    import joblib
    import pandas as pd
    import torch
    from joblib import parallel_backend
    from augmentation import ConstraintTransform, seed_everything
    from data import MODEL_FEATURES
    from joint_conditional_gan import JointConditionalCTGAN
    out = fresh(args.out)
    for seed in args.seeds:
        train, _, _ = seed_split(seed)
        for family in args.families:
            folder = out / f"seed_{seed}/{family}"
            folder.mkdir(parents=True)
            seed_everything(seed, 4)
            torch.backends.cuda.matmul.allow_tf32 = False
            real = train.sample(frac=1, random_state=seed).reset_index(drop=True)
            if family == "ctgan":
                prepared = real[MODEL_FEATURES + ["baseline_price_x", "baseline_drive_time_x", "baseline_drive_mile_x"]].copy()
                discrete = ["weekday", "isWeekend", "isRushHour", "request_wait_missing"]
                transform = None
            else:
                transform = ConstraintTransform()
                prepared = transform.fit_transform(real)
                discrete = list(transform.discrete)
            prepared["label"] = real.label.to_numpy(int)
            options = {**protocol()["gan_options"], "cuda": args.device == "cuda"}
            model = JointConditionalCTGAN(**options)
            model.set_random_state(seed)
            with parallel_backend("sequential"):
                model.fit(prepared, discrete_columns=discrete + ["label"])
            model.save(str(folder / "joint.pkl"))
            model.loss_values.to_csv(folder / "gan_loss.csv", index=False)
            if transform:
                joblib.dump(transform, folder / "joint_transform.joblib")
            save_json(folder / "training.json", {"seed": seed, "family": family, "rows": len(train), "epochs": 500})
            del model, prepared
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def sample(args):
    import gc
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from augmentation import seed_everything
    from ctgan import CTGAN
    from data import MODEL_FEATURES
    from exact_gan_sampling import enable_exact_numeric_decode
    from pipeline_utils import generator_digest
    out = fresh(args.out)
    for seed in args.seeds:
        train, _, _ = seed_split(seed)
        for family in args.families:
            checkpoint_root = Path(args.checkpoints) if args.checkpoints else ROOT / "models/gan"
            source = checkpoint_root / f"seed_{seed}/{family}"
            folder = out / f"seed_{seed}/{family}"
            folder.mkdir(parents=True)
            parts, audit = [], []
            for label, needed in targets(train).items():
                target = min(needed, args.limit) if args.limit is not None else needed
                seed_everything(seed, 4)
                torch.backends.cuda.matmul.allow_tf32 = False
                # map_location also permits loading the supplied GPU checkpoints on CPU.
                model = torch.load(source / "joint.pkl", map_location="cpu", weights_only=False)
                model.set_device(torch.device(args.device))
                model.set_random_state(seed)
                model._generator.eval()
                enable_exact_numeric_decode(model)
                before = generator_digest(model._generator)
                transform_path = source / "joint_transform.joblib"
                transform = joblib.load(transform_path) if transform_path.exists() else None
                blocks, obtained, drawn = [], 0, 0
                while obtained < target:
                    n = 10000
                    with torch.no_grad():
                        generated = model.sample(n, condition_column="label", condition_value=label)
                    matches = pd.to_numeric(generated.label, errors="coerce").eq(label).to_numpy()
                    decoded = transform.inverse(generated, protocol()["rush_hours"]) if transform else generated
                    block = decoded.loc[matches].head(target - obtained).copy()
                    obtained += len(block)
                    drawn += n
                    blocks.append(block)
                    if drawn > max(target, 10000) * 1000:
                        raise RuntimeError("Class conditioning failed to produce the requested label")
                if generator_digest(model._generator) != before:
                    raise AssertionError("Generator buffers changed during inference")
                result = pd.concat(blocks, ignore_index=True)
                result["label"], result["synthetic"] = label, True
                if not np.isfinite(result[MODEL_FEATURES].to_numpy(float)).all():
                    raise FloatingPointError("Nonfinite model predictors")
                parts.append(result)
                audit.append({"label": label, "rows": len(result), "drawn": drawn, "buffers_unchanged": True})
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pd.concat(parts, ignore_index=True).to_parquet(folder / "synthetic.parquet", index=False)
            save_json(folder / "sampling.json", {"seed": seed, "family": family, "generator_mode": "eval", "classes": audit})
            print(f"Sampled {seed}/{family}", flush=True)


def smote(args):
    from augmentation import generate_smote
    out = fresh(args.out)
    for seed in args.seeds:
        train, _, _ = seed_split(seed)
        generated, _ = generate_smote(train.assign(pattern_label=train.label), targets(train), {"seed": seed})
        generated = generated.rename(columns={"pattern_label": "label"})
        folder = out / f"seed_{seed}/smotenc"
        folder.mkdir(parents=True)
        generated.to_parquet(folder / "synthetic.parquet", index=False)
        print(f"SMOTENC seed={seed}: {len(generated)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify").set_defaults(func=verify)
    for name, function in [
        ("predict", predict), ("split", split), ("train-xgb", train_xgb),
        ("train-gan", train_gan), ("sample", sample), ("smote", smote),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--seeds", type=int, nargs="+", choices=range(42, 47), default=list(range(42, 47)))
        p.set_defaults(func=function)
        if name in ("predict", "train-xgb"):
            p.add_argument("--methods", nargs="+", choices=protocol()["methods"], default=protocol()["methods"])
        if name == "train-xgb":
            p.add_argument("--retune", action="store_true", help="Run the full 50-call optimization rather than replaying saved optimal parameters")
            p.add_argument("--generated", help="Root containing newly generated seed_N/method/synthetic.parquet")
        if name in ("train-gan", "sample"):
            p.add_argument("--families", nargs="+", choices=["ctgan", "pctgan"], default=["ctgan", "pctgan"])
            p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        if name == "sample":
            p.add_argument("--checkpoints")
            p.add_argument("--limit", type=int, help="Smoke test only: cap generated rows per class")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
