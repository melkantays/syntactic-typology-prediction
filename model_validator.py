"""model_validator.py
 
Takes training_matrix.csv, tests a Random Forest under LOOCV
(Leave-One-Out Cross-Validation), and compares the result against a
DUMB PREDICTOR (the majority baseline).
 
Why this file exists: whether a model is "good" is not established by an
easily-flattered number like accuracy, but by whether it can beat a dumb
predictor that answers the most common label every time. If it cannot,
the model is worthless -- however sophisticated it looks.
 
Input  : training_matrix.csv  (ISO_Code + 3 X + N Y)
Output : report on screen + optimised_pipeline.joblib + validation_report.json
 
NOTE: the number of targets is NO LONGER FIXED. Whatever Y columns the
file contains are used (7 or 6 alike), so changing the target count
requires no edit to this file.
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
RANDOM_STATE = 42
ID_COLUMN = "ISO_Code"
 
# Hyperparameter combinations to try. On small data what matters is not
# the model's CAPACITY but its BRAKES against overfitting, so the grid
# sweeps the brakes: depth (max_depth) and splitting (min_samples_split).
PARAM_GRID = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [2, 3, 5, None],
    "clf__min_samples_split": [2, 3, 4],
    "clf__max_features": ["sqrt", None],
}
 
 
# --- BLOCK 1: read the data, split X from Y --------------------------- #
def load_matrix(csv_path):
    """Split the matrix into X (features) and Y (targets).
 
    The ISO_Code column is DISCARDED -- it is never used as a feature,
    because it is a unique label per language and the model could use it
    as a lookup key, producing spurious apparent accuracy. The number of
    Y columns is READ FROM THE FILE, not fixed.
    """
    frame = pd.read_csv(csv_path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
 
    # X and Y columns are NOT hard-coded; they are detected from the file:
    #   Y targets begin with "S_" (URIEL syntax features).
    #   Every remaining column is an X feature.
    # The same code therefore handles both the surface matrix (TTR,
    # Avg_Length, Length_Var) and the embedding matrix (pca_0..pca_9) --
    # running both experiments through identical code is what makes the
    # comparison fair.
    target_names = [c for c in frame.columns if c.startswith("S_")]
    if not target_names:
        raise ValueError("No Y target column starting with 'S_' was found.")
 
    feature_names = [c for c in frame.columns if c not in target_names]
    if not feature_names:
        raise ValueError("No X feature column was found.")
 
    features = frame[feature_names].to_numpy(dtype=np.float64)
    targets = frame[target_names].to_numpy()
 
    # Sanity: X must have no gaps, Y must be binary (0/1).
    if not np.isfinite(features).all():
        raise ValueError("X contains NaN/inf; clean the data first.")
    observed = np.unique(targets)
    if not np.isin(observed, [0, 1]).all():
        raise ValueError(f"Y targets must be 0/1; found: {observed[:8]}")
 
    return features, targets.astype(np.int8), feature_names, target_names
 
 
# --- BLOCK 2: compute the dumb predictor (baseline) ------------------- #
def majority_baseline(targets, target_names):
    """A predictor that blindly answers the most common label per target.
 
    This is the LINE that has to be beaten. If our model cannot improve
    on its Hamming Loss, it has added no intelligence. This also counts
    DEGENERATE targets (the same value in every language) -- those are
    free points that flatter any model.
    """
    n = targets.shape[0]
    positive_rate = targets.mean(axis=0)
    constant = (positive_rate == 0.0) | (positive_rate == 1.0)
 
    print("--- Label geometry ---")
    print(f"Samples (n)          : {n}")
    print(f"Targets              : {targets.shape[1]}")
    print(f"Degenerate labels    : {int(constant.sum())}")
    print(f"Informative labels   : {int((~constant).sum())}")
 
    majority = (positive_rate >= 0.5).astype(np.int8)
    baseline_pred = np.tile(majority, (n, 1))
    baseline_loss = hamming_loss(targets, baseline_pred)
    print(f"Majority baseline    : {baseline_loss:.4f} (Hamming Loss)\n")
    return baseline_loss
 
 
# --- BLOCK 3: build the model skeleton -------------------------------- #
def build_pipeline():
    """Combine scaling and the Random Forest into a single pipeline.
 
    Because everything lives inside the pipeline, the scaler is fitted on
    the TRAINING data of each fold only -- the held-out language never
    leaks in. This is what prevents data leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ])
 
 
# --- BLOCK 4: search for the best hyperparameters under LOOCV --------- #
def run_search(features, targets, n_jobs):
    """GridSearchCV tries every combination under LOOCV and keeps the best.
 
    LOOCV: with 31 languages, train on 30 and test on 1, repeated 31
    times. On small data this is the most accurate evaluation available.
 
    Hamming Loss is an ERROR measure (lower is better), but GridSearch
    tries to MAXIMISE its score. make_scorer(greater_is_better=False)
    negates it, so "least error" becomes "largest negative score".
    """
    scorer = make_scorer(hamming_loss, greater_is_better=False)
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=LeaveOneOut(),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(features, targets)
    return search
 
 
# --- BLOCK 5: measure the chosen config's HONEST loss ----------------- #
def honest_loss(estimator, features, targets):
    """The true LOOCV loss of the selected model.
 
    Important: GridSearch's "best score" is the lowest among DOZENS of
    configurations tried, and is therefore optimistic through selection
    bias. This function independently re-measures the SINGLE FIXED best
    configuration; that is the number that has to be defended.
    """
    predictions = cross_val_predict(
        estimator, features, targets, cv=LeaveOneOut(), n_jobs=1
    )
    return hamming_loss(targets, predictions)
 
 
# --- BLOCK 6: report and save ----------------------------------------- #
def report(search, features, targets, baseline_loss, feat_names,
           target_names, model_path, report_path):
    """Print the results, and write the model and a JSON summary to disk."""
    grid_loss = -search.best_score_
 
    print("--- Best configuration ---")
    for key in sorted(search.best_params_):
        print(f"{key:26s}: {search.best_params_[key]}")
 
    print("\n--- Performance ---")
    print(f"Model Hamming Loss (grid)         : {grid_loss:.4f}")
    print(f"Majority baseline (Hamming Loss)  : {baseline_loss:.4f}")
 
    fixed_loss = honest_loss(search.best_estimator_, features, targets)
    print(f"Model Hamming Loss (honest LOOCV) : {fixed_loss:.4f}")
 
    lift = baseline_loss - fixed_loss
    print(f"\nImprovement          : {lift:+.4f}")
    if lift <= 0:
        print("CONCLUSION: the model did NOT beat the dumb predictor. Do not "
              "present this as a positive result. It is evidence that surface\n"
              "            statistics are insufficient -- an honest negative "
              "finding.")
    else:
        print(f"CONCLUSION: the model BEAT the dumb predictor "
              f"({lift/baseline_loss:.1%} less error). A defensible positive "
              "result.")
 
    forest = search.best_estimator_.named_steps["clf"]
    print("\n--- Feature importance ---")
    for name, w in zip(feat_names, forest.feature_importances_):
        print(f"{name:14s}: {w:.4f}")
 
    joblib.dump(search.best_estimator_, model_path)
    summary = {
        "n_languages": int(features.shape[0]),
        "n_targets": len(target_names),
        "target_names": target_names,
        "grid_loss": float(grid_loss),
        "honest_loss": float(fixed_loss),
        "baseline_loss": float(baseline_loss),
        "beats_baseline": bool(lift > 0),
        "best_params": search.best_params_,
    }
    Path(report_path).write_text(json.dumps(summary, indent=2))
    print(f"\nModel  -> {model_path}")
    print(f"Report -> {report_path}")
 
 
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LOOCV validator (flexible target count)."
    )
    p.add_argument("--data", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--model-out", type=Path,
                   default=Path("optimised_pipeline.joblib"))
    p.add_argument("--report-out", type=Path,
                   default=Path("validation_report.json"))
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args(argv)
 
 
def main(argv=None):
    args = parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: {args.data} not found.", file=sys.stderr)
        return 1
 
    features, targets, feat_names, target_names = load_matrix(args.data)
    baseline_loss = majority_baseline(targets, target_names)
 
    print(f"Running LOOCV search ({features.shape[0]} languages, "
          f"{len(target_names)} targets)...\n")
    search = run_search(features, targets, args.n_jobs)
    report(search, features, targets, baseline_loss, feat_names,
           target_names, args.model_out, args.report_out)
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 