This is the code for the paper 'A three-stage framework for suspected taxi fraud detection and multi-dimensional interpretation on taxi order data',published in Transportation Research PartC (doi:https://doi.org/10.1016/j.trc.2026.106158).
# NYC HVFHV Supplementary Experiment

This repository snapshot contains the five NYC classifier arms retained in the manuscript's Table 7:

1. Fixed XGBoost (XGB)
2. Bayesian-optimized XGBoost without augmentation (BO-XGB)
3. SMOTE-BO-XGB, implemented with SMOTENC for the categorical predictors
4. CTGAN-BO-XGB
5. PCTGAN-BO-XGB

It is aligned to the August 2026 development cohort, the fixed August test set (`n=5,000`), and the July external evaluation cohort (`n=205,374`). It does not include the separate generator-loss ablation or the July 500-order manual-review workbook. 

## Repository Contents

- `run.py`: prediction, split, Bayesian optimization, GAN training/sampling, and SMOTENC commands.
- `preprocess.py`: reconstructs the frozen cohorts from the original monthly files and checks engineered predictors.
- `reproduce_clustering.py`: replays the frozen BMI2-DBSCAN clustering configuration.
- `code/data.py`: NYC field validation, feature construction, OD baselines, and deterministic derived fields.
- `code/behavior_features.py`: five-dimensional behavior representation used by the clustering replay.
- `code/diagnostic_sampling.py`: frozen candidate-label and interval-margin rules.
- `code/memory_dbscan.py`: memory-bounded DBSCAN implementation used for the frozen clustering replay.
- `code/augmentation.py`: PCTGAN constraint transform/reconstruction, SMOTENC generation, and seed setup.
- `code/joint_conditional_gan.py`: shared, class-conditioned CTGAN implementation used by CTGAN and PCTGAN.
- `code/search.py`: BO-XGB search space and parameter conversion.
- `data/`: frozen August cohorts, July external evaluation rows, baseline aggregates, label mapping, and data dictionaries.
- `models/xgb/`: fitted XGB classifiers and selected-parameter metadata for seeds 42–46.
- `models/preprocessing/`: frozen OD baseline, RobustScaler, and clustering parameters.
- `results/`: five-seed aggregate and per-seed metrics and selected BO parameters.

The fitted joint CTGAN and PCTGAN checkpoints for seeds 42–46 are included under `models/gan/` and can be used directly for sampling. The full monthly TLC Parquet files and generated training tables are not committed here. Obtain the July and August 2026 High Volume For-Hire Vehicle Parquet files from the [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and place them at:

```text
data/raw/fhvhv_tripdata_2026-07.parquet
data/raw/fhvhv_tripdata_2026-08.parquet
```

The included frozen cohorts allow the classifier and clustering artifacts to be inspected without rescanning the full monthly files. Raw-data reconstruction requires the two source Parquet files.

## Environment

Use Python 3.12.2. Install the classifier environment:

```powershell
python -m venv .venv_eval
.venv_eval\Scripts\python -m pip install -r requirements-evaluation.txt
.venv_eval\Scripts\python run.py verify
.venv_eval\Scripts\python run.py predict --out runs/predictions
```

`predict` uses the included fitted XGB models and frozen evaluation inputs. It writes the metrics and confusion matrices to a new output directory. The source package does not include per-order prediction fixtures; compare the generated summaries with `results/metrics_mean_sd.csv` and `results/class_metrics_individual_seeds.csv`.

Recreate the seed-specific stratified 80/20 fitting/BO-validation splits from the included 45,000-row development cohort with:

```powershell
.venv_eval\Scripts\python run.py split --out runs/splits
```

## Retraining

GAN training requires a separate environment with CUDA-enabled PyTorch when using a GPU:

```powershell
python -m venv .venv_gan
.venv_gan\Scripts\python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
.venv_gan\Scripts\python -m pip install -r requirements-gan.txt
.venv_gan\Scripts\python run.py train-gan --families ctgan pctgan --out runs/gan
.venv_gan\Scripts\python run.py sample --families ctgan pctgan --checkpoints runs/gan --out runs/generated
```

To sample with the included fitted checkpoints instead of retraining the GANs:

```powershell
.venv_gan\Scripts\python run.py sample --families ctgan pctgan --out runs/generated
```

Generate SMOTENC samples in the evaluation environment:

```powershell
.venv_eval\Scripts\python run.py smote --out runs/smotenc
```

Run BO-XGB refitting by arm. Every `--out` path must be new or empty:

```powershell
.venv_eval\Scripts\python run.py train-xgb --methods xgb bo_xgb --retune --out runs/xgb_unaugmented
.venv_eval\Scripts\python run.py train-xgb --methods smotenc --retune --generated runs/smotenc --out runs/xgb_smotenc
.venv_eval\Scripts\python run.py train-xgb --methods ctgan pctgan --retune --generated runs/generated --out runs/xgb_gan
```

Bayesian optimization uses 50 calls, 20 initial points, an ExtraTrees surrogate, expected improvement (`xi=0.01`), and real validation accuracy as the objective. Each seed has its own optimization.

## Interpretation

August cluster labels and July rule-derived labels are candidate order-pattern labels, not independently verified fraud outcomes. Class 4 is a long request-to-pickup waiting-time proxy because the NYC source lacks the original remote-dispatch surcharge field. This experiment evaluates the framework after adaptation to the available NYC fields; it does not establish that any individual trip involved verified fraud.

The reported augmentation prior is the rounded August candidate-cohort prior (10%, 85%, 1%, 2%, 2%), not the prevalence of fraud or of these behaviors among all NYC trips.

No software license has been selected. Add the research team's chosen license and confirm permission to redistribute the derived cohort files before making the repository public.
