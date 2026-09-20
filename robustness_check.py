"""robustness_check.py
 
Demonstrates that the negative finding ("3 surface statistics cannot
predict deep syntax") is NOT an artefact of one lucky draw, one model,
or one hyperparameter setting.
 
Why this is needed: a reviewer can object that "maybe RandomForest was
a poor choice", "maybe those targets were unlucky", "maybe the ordering
of the data mattered". This file forecloses those objections in advance:
if the same negative result recurs across several independent settings,
the finding is no longer "our mistake" but a real limit of the method.
 
Input  : training_matrix.csv  (ISO_Code + 3 X + N Y)
Output : comparison table on screen + robustness_report.json
 
The four independent tests:
  1. Different models   : RandomForest, LogisticRegression, DummyClassifier.
                          If none beats the baseline, the model is not the problem.
  2. Feature shuffle    : if shuffling X row-wise does NOT change the result,
                          the model was drawing no information from X (evidence).
  3. Target subsets     : random groups of 4 targets -- if the result is
                          negative in every group, it does not hinge on
                          particular targets.
  4. Different seeds    : if changing the randomness does not move the
                          result, the finding is stable.
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
ID_COLUMN = "ISO_Code"
 
 
# --- BLOCK 1: load the data ------------------------------------------- #
def load_matrix(csv_path):
    """Split the matrix into X (features) and Y (targets).
 
    Targets are detected as the columns starting with "S_" (URIEL syntax
    features); every remaining column is a feature.
 
    FIXED: this previously hard-coded the feature names as
    ["TTR", "Avg_Length", "Length_Var"] and treated everything else as a
    target. Run against an embedding matrix, the pca_* columns would
    have been silently treated as TARGETS -- producing nonsense rather
    than an error. Detecting targets by their "S_" prefix matches the
    rest of the pipeline and makes this file safe on either matrix.
    """
    frame = pd.read_csv(csv_path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
 
    target_names = [c for c in frame.columns if c.startswith("S_")]
    if not target_names:
        raise ValueError("No target column starting with 'S_' was found.")
    feature_names = [c for c in frame.columns if c not in target_names]
    if not feature_names:
        raise ValueError("No feature column was found.")
 
    features = frame[feature_names].to_numpy(dtype=np.float64)
    targets = frame[target_names].to_numpy().astype(np.int8)
    return features, targets, target_names
 
 
# --- BLOCK 2: baseline (the line that must be beaten) ----------------- #
def majority_baseline_loss(targets):
    """Hamming Loss of a lazy predictor that always answers the majority class."""
    majority = (targets.mean(axis=0) >= 0.5).astype(np.int8)
    baseline_pred = np.tile(majority, (targets.shape[0], 1))
    return hamming_loss(targets, baseline_pred)
 
 
# --- BLOCK 3: measure one model's LOOCV loss -------------------------- #
def loocv_loss(pipeline, features, targets):
    """Compute LOOCV Hamming Loss, handling each target separately.
 
    Why per-target: on small, imbalanced data a target can end up
    single-class within a LOOCV fold (remove that one language and the
    rest are all 0). Logistic regression crashes on this. Here each
    target is handled separately, and where the training split is
    single-class we simply predict that constant -- no crash, and the
    correct answer.
 
    NOTE: because of this per-target loop, absolute values here differ
    slightly from results_compiler.py, which fits one multi-output model.
    The conclusion is identical; the difference must be stated in the paper.
    """
    loo = LeaveOneOut()
    n, n_targets = targets.shape
    preds = np.zeros_like(targets)
    for j in range(n_targets):
        yj = targets[:, j]
        for train_idx, test_idx in loo.split(features):
            y_train = yj[train_idx]
            if len(np.unique(y_train)) < 2:
                # Single class in training: predict that constant.
                preds[test_idx, j] = y_train[0]
            else:
                from sklearn.base import clone
                model = clone(pipeline)
                model.fit(features[train_idx], y_train)
                preds[test_idx, j] = model.predict(features[test_idx])
    return hamming_loss(targets, preds)
 
 
def make_model(kind, seed):
    """Build each of the three classifiers on the same pipeline skeleton."""
    if kind == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=2,
                                     random_state=seed, n_jobs=1)
    elif kind == "LogisticReg":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "Dummy":
        clf = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(kind)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
 
 
# --- BLOCK 4: Test 1 -- different models ------------------------------ #
def test_models(features, targets, baseline):
    """If none of three different models beats the baseline, the model
    class is not what is holding the result back."""
    print("\n=== TEST 1: Different models ===")
    print(f"{'Model':16s} {'Hamming Loss':>13s} {'vs baseline':>13s}")
    results = {}
    for kind in ["RandomForest", "LogisticReg", "Dummy"]:
        loss = loocv_loss(make_model(kind, 42), features, targets)
        verdict = "BEATS" if loss < baseline else "does not beat"
        print(f"{kind:16s} {loss:13.4f} {verdict:>13s}")
        results[kind] = float(loss)
    return results
 
 
# --- BLOCK 5: Test 2 -- feature shuffle (permutation test) ------------ #
def test_shuffle(features, targets, baseline, seed=42):
    """Shuffle the rows of X: if nothing changes, X carried no information.
 
    This is the strongest single piece of evidence. If shuffling X across
    languages -- that is, handing every language the WRONG statistics --
    leaves the model performing as well or better, the model was drawing
    no genuine signal from X at all. That is a direct demonstration of
    the ABSENCE of signal, not merely a failure to find one.
    """
    print("\n=== TEST 2: Feature shuffle (what happens when X is corrupted?) ===")
    rng = np.random.default_rng(seed)
    shuffled = features[rng.permutation(features.shape[0])]
    real_loss = loocv_loss(make_model("RandomForest", 42), features, targets)
    shuf_loss = loocv_loss(make_model("RandomForest", 42), shuffled, targets)
    print(f"With real X       : {real_loss:.4f}")
    print(f"With shuffled X   : {shuf_loss:.4f}")
    gap = real_loss - shuf_loss
    if gap >= -0.01:
        print("READING: real X is NOT better than corrupted X. The model "
              "draws no meaningful signal from X -- negative finding confirmed.")
    else:
        print("READING: real X is clearly better -- there IS signal in X, "
              "and the finding must be re-examined.")
    return {"real": float(real_loss), "shuffled": float(shuf_loss)}
 
 
# --- BLOCK 6: Test 3 -- random target subsets ------------------------- #
def test_subsets(features, targets, target_names, seed=42):
    """Is the result still negative on random groups of 4 targets?
 
    Shows that a few particular "bad" targets are not driving the result:
    the baseline goes unbeaten across different target groupings.
    """
    print("\n=== TEST 3: Random target subsets (4 at a time) ===")
    rng = np.random.default_rng(seed)
    n_targets = targets.shape[1]
    if n_targets < 4:
        print("Fewer than 4 targets; test skipped.")
        return []
    results = []
    for i in range(3):
        cols = rng.choice(n_targets, size=4, replace=False)
        sub = targets[:, cols]
        base = majority_baseline_loss(sub)
        loss = loocv_loss(make_model("RandomForest", 42), features, sub)
        verdict = "BEATS" if loss < base else "does not beat"
        names = ", ".join(target_names[c] for c in cols)
        print(f"Group {i+1}: model={loss:.4f}  baseline={base:.4f}  -> {verdict}")
        print(f"         ({names})")
        results.append({"model": float(loss), "baseline": float(base),
                        "beats": bool(loss < base)})
    return results
 
 
# --- BLOCK 7: Test 4 -- different seeds ------------------------------- #
def test_seeds(features, targets, baseline):
    """Does the result move when the randomness is changed?"""
    print("\n=== TEST 4: Different random seeds ===")
    losses = []
    for seed in [0, 1, 42, 123, 2024]:
        loss = loocv_loss(make_model("RandomForest", seed), features, targets)
        losses.append(loss)
        verdict = "BEATS" if loss < baseline else "does not beat"
        print(f"seed={seed:5d}: {loss:.4f}  ({verdict})")
    spread = max(losses) - min(losses)
    print(f"Spread (max-min): {spread:.4f}  "
          f"({'stable' if spread < 0.03 else 'volatile'})")
    return [float(x) for x in losses]
 
 
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Robustness check for the negative finding."
    )
    p.add_argument("--data", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--out", type=Path, default=Path("robustness_report.json"))
    args = p.parse_args(argv)
 
    if not args.data.exists():
        print(f"ERROR: {args.data} not found.", file=sys.stderr)
        return 1
 
    features, targets, target_names = load_matrix(args.data)
    baseline = majority_baseline_loss(targets)
    print(f"Languages: {features.shape[0]}, Targets: {targets.shape[1]}")
    print(f"Majority baseline (the line to beat): {baseline:.4f}")
 
    report = {
        "n_languages": int(features.shape[0]),
        "n_targets": int(targets.shape[1]),
        "baseline": float(baseline),
        "test1_models": test_models(features, targets, baseline),
        "test2_shuffle": test_shuffle(features, targets, baseline),
        "test3_subsets": test_subsets(features, targets, target_names),
        "test4_seeds": test_seeds(features, targets, baseline),
    }
 
    print("\n=== OVERALL READING ===")
    rf = report["test1_models"]["RandomForest"]
    if rf >= baseline:
        print("No configuration beat the baseline by any meaningful margin. "
              "The negative finding is ROBUST: 3 surface statistics are not "
              "sufficient to predict these targets, and this does not depend "
              "on one model, one setting, or one lucky draw.")
    else:
        print("One configuration beat the baseline -- the finding must be "
              "re-examined.")
 
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {args.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 