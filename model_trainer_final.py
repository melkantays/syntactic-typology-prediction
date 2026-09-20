"""model_trainer_final.py
 
Trains the final model and produces probabilities for the 7 typological
features of Elamite. Every output row carries STAMPS recording the
epistemic status of the finding.
 
EPISTEMIC STATUS (important):
This model FAILED its LOOCV validation -- in all three experiments
(surface, 10-PCA, 20-PCA) the majority baseline went unbeaten. The
predictions below are therefore NOT validated linguistic findings, but a
prototype of "what the model would say if it worked". Every output row
consequently carries the validation outcome and the fact that Elamite
has zero observed data in URIEL.
 
Input (surface mode):
  training_matrix.csv       training set (31 languages, 3 X + 7 Y)
  elamite_features.csv      Elamite's 3 X values
Input (embedding mode):
  embedding_features.csv    768 dimensions for the training languages
  elamite_embedding.csv     768 dimensions for Elamite
  training_matrix.csv       for the targets and the language set
 
Output: elamite_typology_predictions.csv (long format, stamped)
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
ID_COL = "ISO_Code"
SURFACE_COLS = ["TTR", "Avg_Length", "Length_Var"]
 
 
# --- BLOCK 1: surface-mode data --------------------------------------- #
def load_surface(train_path, query_path):
    """X for the surface experiment: 3 statistics, same 3 columns for Elamite."""
    train = pd.read_csv(train_path)
    query = pd.read_csv(query_path)
 
    target_names = [c for c in train.columns if c.startswith("S_")]
    x_train = train[SURFACE_COLS].to_numpy(np.float64)
    y_train = train[target_names].to_numpy().astype(np.int8)
    x_query = query[SURFACE_COLS].to_numpy(np.float64)
 
    if len(x_query) != 1:
        raise ValueError(f"Query file must hold 1 row; found {len(x_query)}.")
    return x_train, y_train, x_query, target_names, SURFACE_COLS
 
 
# --- BLOCK 2: embedding-mode data ------------------------------------- #
def load_embedding(emb_path, query_emb_path, train_path, n_components):
    """X for the embedding experiment: 768 dimensions -> PCA.
 
    THE MOST CRITICAL POINT: the PCA and the StandardScaler are fitted on
    the TRAINING languages only; Elamite is merely transformed. Had
    Elamite been included in the fit, the language we are predicting
    would have helped shape the transformation -- a form of data leakage
    that invalidates the result.
 
    The training languages are also intersected with the set in
    training_matrix.csv, so the SAME 31 languages are used as in the
    earlier experiments.
    """
    emb = pd.read_csv(emb_path)
    query = pd.read_csv(query_emb_path)
    train_ref = pd.read_csv(train_path)
 
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    target_names = [c for c in train_ref.columns if c.startswith("S_")]
 
    # Intersect the training set with the reference (the same 31 languages)
    ref_langs = train_ref[ID_COL].tolist()
    emb = emb[emb[ID_COL].isin(ref_langs)].copy()
    emb = emb.set_index(ID_COL).loc[ref_langs].reset_index()
 
    x_raw = emb[emb_cols].to_numpy(np.float64)
    x_query_raw = query[emb_cols].to_numpy(np.float64)
    y_train = train_ref[target_names].to_numpy().astype(np.int8)
 
    # Scaling + PCA: fitted on the training data ONLY
    scaler = StandardScaler().fit(x_raw)
    pca = PCA(n_components=min(n_components, x_raw.shape[0]),
              random_state=42).fit(scaler.transform(x_raw))
 
    x_train = pca.transform(scaler.transform(x_raw))
    x_query = pca.transform(scaler.transform(x_query_raw))
 
    print(f"PCA: {len(emb_cols)} -> {x_train.shape[1]} dimensions "
          f"(variance retained {pca.explained_variance_ratio_.sum():.1%})")
    feature_names = [f"pca_{i}" for i in range(x_train.shape[1])]
    return x_train, y_train, x_query, target_names, feature_names
 
 
# --- BLOCK 3: build and fit the model --------------------------------- #
def fit_model(x_train, y_train, seed=42):
    """Fit the final model with the configuration chosen during validation.
 
    max_depth=2 was selected in all three experiments -- the model gets
    worse as it gets deeper, so it settles on the shallowest tree. That
    is itself a sign that there is no pattern to learn; the same
    configuration is kept here for continuity.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=2, min_samples_split=2,
            max_features="sqrt", random_state=seed, n_jobs=-1)),
    ])
    model.fit(x_train, y_train)
    return model
 
 
# --- BLOCK 4: extract the probabilities safely ------------------------ #
def positive_probabilities(model, x_query, n_targets):
    """Return P(value = 1) for each target.
 
    On a multi-target model, predict_proba returns a SEPARATE array per
    target, and each array has one column per class SEEN IN TRAINING for
    that target. A target that is constant in training (say, all
    languages 0) yields a single column, and naive [0][1] indexing raises
    IndexError there. Class identity is therefore read from classes_;
    the index is never assumed.
    """
    blocks = model.predict_proba(x_query)
    classes = model.named_steps["clf"].classes_
 
    out = np.empty(n_targets, dtype=np.float64)
    for i in range(n_targets):
        cls = np.asarray(classes[i])
        block = np.asarray(blocks[i])[0]
        hit = np.flatnonzero(cls == 1)
        out[i] = float(block[hit[0]]) if hit.size else 0.0
    return out
 
 
# --- BLOCK 5: diagnose informativeness -------------------------------- #
def diagnose(probs, y_train, eps=0.02):
    """Is the prediction merely echoing the training base rate?
 
    If a prediction equals the base rate, the model has said nothing
    SPECIFIC TO ELAMITE for that target; it has reflected the corpus
    average. These are marked 'VACUOUS', so it is possible to state
    plainly how many of the predictions carry no information.
    """
    base = y_train.mean(axis=0)
    delta = np.abs(probs - base)
    degenerate = (base == 0.0) | (base == 1.0)
    vacuous = (delta < eps) & ~degenerate
    return base, delta, degenerate, vacuous
 
 
# --- BLOCK 6: write the stamped output -------------------------------- #
def write_output(path, mode, target_names, probs, base, delta,
                 degenerate, vacuous, n_train, n_tokens):
    """Write a long-format CSV in which every row carries its stamps.
 
    Long format is deliberate: in a wide 1x7 row, someone can lift a
    single probability onto a slide and leave the stamps behind. In long
    format the stamp travels with the number.
    """
    status = np.where(degenerate, "DEGENERATE_LABEL",
                      np.where(vacuous, "VACUOUS_EQUALS_PRIOR", "INFORMATIVE"))
    frame = pd.DataFrame({
        "Language": "Elamite",
        "Feature_Set": mode,
        "Typology_Feature": target_names,
        "P_positive": np.round(probs, 6),
        "Training_Base_Rate": np.round(base, 6),
        "Delta_From_Prior": np.round(delta, 6),
        "Status": status,
        "Model_Validation": "FAILED_LOOCV_NOT_ABOVE_BASELINE",
        "URIEL_Observed_Data_For_Elamite": 0,
        "Training_Languages": n_train,
        "Elamite_Tokens": n_tokens,
    })
    frame.to_csv(path, index=False)
    return frame
 
 
def read_token_count(query_path):
    """Read N_Tokens for the query language from the feature file.
 
    Returns -1 if it cannot be read, so the stamp records "unknown"
    rather than a plausible-looking wrong number.
    """
    try:
        frame = pd.read_csv(query_path)
    except (FileNotFoundError, OSError):
        print(f"WARNING: {query_path} unreadable; token count stamped as -1.")
        return -1
    if "N_Tokens" not in frame.columns or frame.empty:
        print(f"WARNING: no N_Tokens column in {query_path}; stamped as -1.")
        return -1
    return int(frame["N_Tokens"].iloc[0])
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description="Elamite typology prototype.")
    p.add_argument("--mode", choices=["surface", "embedding"], default="surface")
    p.add_argument("--train", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--query", type=Path, default=Path("elamite_features.csv"))
    p.add_argument("--emb", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--query-emb", type=Path, default=Path("elamite_embedding.csv"))
    p.add_argument("--components", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=Path("elamite_typology_predictions.csv"))
    p.add_argument("--tokens", type=int, default=None,
                   help="Token count of the Elamite corpus (for the stamp). "
                        "Read from the query file's N_Tokens column when omitted.")
    args = p.parse_args(argv)
 
    # Read the token count from the data rather than carrying a constant.
    # A hard-coded default goes stale the moment the cleaning step changes
    # the corpus, and the stale number is then stamped into every output
    # row -- which is exactly what happened once already.
    n_tokens = args.tokens
    if n_tokens is None:
        n_tokens = read_token_count(args.query)
 
    if args.mode == "surface":
        if not (args.train.exists() and args.query.exists()):
            print("ERROR: training_matrix.csv or elamite_features.csv is missing.",
                  file=sys.stderr)
            return 1
        x_tr, y_tr, x_q, targets, feats = load_surface(args.train, args.query)
    else:
        if not (args.emb.exists() and args.query_emb.exists()):
            print("ERROR: embedding files are missing.", file=sys.stderr)
            return 1
        x_tr, y_tr, x_q, targets, feats = load_embedding(
            args.emb, args.query_emb, args.train, args.components)
 
    print(f"Mode       : {args.mode}")
    print(f"Training   : {x_tr.shape[0]} languages, {x_tr.shape[1]} features")
    print(f"Targets    : {len(targets)}")
    print("WARNING    : the model FAILED validation. The predictions below")
    print("             are a prototype, NOT a validated finding.\n")
 
    model = fit_model(x_tr, y_tr)
    probs = positive_probabilities(model, x_q, len(targets))
    base, delta, degenerate, vacuous = diagnose(probs, y_tr)
 
    print(f"{'Feature':30s} {'P(1)':>7s} {'Base':>7s} {'Status':>22s}")
    for i, name in enumerate(targets):
        st = ("DEGENERATE" if degenerate[i]
              else "VACUOUS" if vacuous[i] else "INFORMATIVE")
        print(f"{name:30s} {probs[i]:7.3f} {base[i]:7.3f} {st:>22s}")
 
    n_inf = int((~degenerate & ~vacuous).sum())
    print(f"\nInformative predictions : {n_inf}/{len(targets)}")
    print(f"Base-rate echoes        : {int(vacuous.sum())}")
    print(f"Degenerate targets      : {int(degenerate.sum())}")
 
    write_output(args.out, args.mode, targets, probs, base, delta,
                 degenerate, vacuous, x_tr.shape[0], n_tokens)
    print(f"\nOutput -> {args.out}")
    print("Every row carries the validation status and the fact that Elamite")
    print("has 0 observed URIEL features; no single number can be taken out")
    print("of context.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 