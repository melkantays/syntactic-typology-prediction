"""embedding_compiler.py
 
Reduces the 768-dimensional language representations to 10 dimensions
with PCA and joins them to the same 7 URIEL targets, producing the
training matrix for the second experiment.
 
Why PCA: 768 features but only ~31 languages. At that ratio a model
memorises NOISE rather than any real pattern (the curse of
dimensionality). Compressing 768 down to the 10 most informative axes
preserves the substance of the rich representation while bringing the
sample-to-dimension ratio back to something defensible.
 
Why the same language set: this experiment will be compared against the
surface experiment (3 statistics). Run on different language sets, any
difference in the result could not be attributed to the features rather
than to the sample. We therefore INTERSECT with the languages in
training_matrix.csv -- which additionally filters out files with corrupt
content, such as ara/jpn/vie.
 
Input  : embedding_features.csv  (ISO_Code + emb_0..emb_767)
         training_matrix.csv     (for the surface experiment's language set)
Output : training_matrix_emb.csv (ISO_Code + pca_0..pca_9 + 7 targets)
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import lang2vec.lang2vec as l2v
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
TARGET_FEATURES = [
    "S_SOV", "S_SVO", "S_VSO",
    "S_ADPOSITION_BEFORE_NOUN", "S_ADPOSITION_AFTER_NOUN",
    "S_CASE_SUFFIX", "S_ADJECTIVE_BEFORE_NOUN",
]
 
 
# --- BLOCK 1: read the embedding file --------------------------------- #
def load_embeddings(path):
    """Read embedding_features.csv: ISO codes and the 768-dim matrix."""
    frame = pd.read_csv(path)
    emb_cols = [c for c in frame.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No column starting with emb_ was found.")
    return frame["ISO_Code"].tolist(), frame[emb_cols].to_numpy(np.float64)
 
 
# --- BLOCK 2: restrict to the same language set ----------------------- #
def restrict_to_reference(iso_codes, matrix, reference_path):
    """Intersect with the languages used in the surface experiment.
 
    This does two jobs at once:
      1) Fair comparison: both experiments run on identical languages.
      2) Hygiene: files with corrupt content (for example an ara.txt
         containing nothing but dashes) were already excluded from the
         surface experiment, so they drop out here automatically --
         meaningless vectors never enter the data.
    """
    if not Path(reference_path).exists():
        print(f"WARNING: {reference_path} not found; no restriction applied.")
        return iso_codes, matrix
 
    ref = pd.read_csv(reference_path)
    ref_langs = set(ref["ISO_Code"].tolist())
 
    keep = [i for i, code in enumerate(iso_codes) if code in ref_langs]
    dropped = [c for c in iso_codes if c not in ref_langs]
    if dropped:
        print(f"Dropped {len(dropped)} language(s) absent from the reference "
              f"set: {dropped}")
 
    return [iso_codes[i] for i in keep], matrix[keep]
 
 
# --- BLOCK 3: dimensionality reduction with PCA ----------------------- #
def reduce_dimensions(matrix, n_components):
    """Reduce 768 dimensions down to n_components.
 
    StandardScaler first: the dimensions may be on different scales and
    PCA is scale-sensitive. Then PCA: the data is re-expressed along the
    axes carrying the most variance, with the leading axes carrying the
    most information.
 
    NOTE (disclosure): PCA is fitted here on ALL of X. PCA is
    unsupervised -- it never sees the targets (Y) -- so this does not
    constitute classical label leakage. The strictest protocol would
    nonetheless fit PCA INSIDE the cross-validation folds; this choice
    must be stated in the paper.
    """
    scaled = StandardScaler().fit_transform(matrix)
    n_components = min(n_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
 
    explained = pca.explained_variance_ratio_
    print(f"\nPCA: {matrix.shape[1]} dimensions -> {n_components} dimensions")
    print(f"Variance retained (total): {explained.sum():.1%}")
    print("Share of the first 5 axes: " +
          ", ".join(f"{v:.1%}" for v in explained[:5]))
    return reduced, explained
 
 
# --- BLOCK 4: fetch the URIEL targets --------------------------------- #
def fetch_targets(langs):
    """Fetch the same 7 observed (wals) targets; drop incomplete languages."""
    data = l2v.get_features(langs, "syntax_wals", header=True)
    index = {name: i for i, name in enumerate(data["CODE"])}
 
    rows, dropped = [], []
    for lang in langs:
        if lang not in data:
            dropped.append(lang)
            continue
        vector = data[lang]
        row, complete = {"ISO_Code": lang}, True
        for feat in TARGET_FEATURES:
            value = vector[index[feat]]
            if value == "--":
                complete = False
                break
            row[feat] = float(value)
        (rows if complete else dropped).append(row if complete else lang)
 
    if dropped:
        print(f"Dropped for missing target data: {dropped}")
    return pd.DataFrame(rows)
 
 
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Embedding + PCA training-matrix compiler."
    )
    p.add_argument("--emb", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--reference", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--out", type=Path, default=Path("training_matrix_emb.csv"))
    p.add_argument("--components", type=int, default=10)
    args = p.parse_args(argv)
 
    if not args.emb.exists():
        print(f"ERROR: {args.emb} not found.", file=sys.stderr)
        return 1
 
    iso_codes, matrix = load_embeddings(args.emb)
    print(f"Read: {len(iso_codes)} languages, {matrix.shape[1]} dimensions")
 
    iso_codes, matrix = restrict_to_reference(iso_codes, matrix, args.reference)
    print(f"After restriction: {len(iso_codes)} languages")
 
    reduced, _ = reduce_dimensions(matrix, args.components)
 
    x_frame = pd.DataFrame(
        reduced, columns=[f"pca_{i}" for i in range(reduced.shape[1])]
    )
    x_frame.insert(0, "ISO_Code", iso_codes)
 
    print(f"\n[{len(iso_codes)} languages] fetching {len(TARGET_FEATURES)} "
          f"targets from URIEL...")
    y_frame = fetch_targets(iso_codes)
 
    final = pd.merge(x_frame, y_frame, on="ISO_Code")
    final.to_csv(args.out, index=False)
 
    print(f"\n[SUCCESS] {args.out}")
    print(f"Shape: {final.shape}  "
          f"(1 ISO + {reduced.shape[1]} PCA + {len(TARGET_FEATURES)} targets)")
    print(f"Languages: {len(final)}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 