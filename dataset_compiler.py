"""dataset_compiler.py
 
Builds the training matrix: joins the 3 X features (TTR, Avg_Length,
Length_Var) with the 7 selected Y targets (the core structural
properties of a language).
 
NOTE: we now use 7 targets rather than 103. These 7 were found by
feature_selector.py to be >=90% filled across all training languages and
variable between them, AND they are non-redundant dimensions that the
typological literature treats as defining the core structure of a
language (word order, adposition direction, morphology).
"""
 
import pandas as pd
import lang2vec.lang2vec as l2v
import sys
 
# --- BLOCK 1: read the X features from file --------------------------- #
# This is the file feature_engineer.py is expected to produce: TTR,
# Avg_Length and Length_Var for each language. This block stops
# immediately with a clear error if the file is missing -- it does not
# silently continue on empty data.
try:
    df_features = pd.read_csv("extracted_features.csv")
except FileNotFoundError:
    print("ERROR: extracted_features.csv not found.")
    print("Run feature_engineer.py first.")
    sys.exit(1)
 
columns_to_keep = ["ISO_Code", "TTR", "Avg_Length", "Length_Var"]
df_features = df_features[[c for c in columns_to_keep if c in df_features.columns]]
 
# --- BLOCK 2: define the 7 selected targets --------------------------- #
# This list is the project's final Y target set because it takes 7 of the
# 25 robust features found by feature_selector.py that are linguistically
# non-redundant. Not an arbitrary 7: the word-order family (SOV/SVO/VSO),
# adposition direction (two opposing values), morphology (case_suffix)
# and adjective order -- four distinct linguistic dimensions.
TARGET_FEATURES = [
    "S_SOV",
    "S_SVO",
    "S_VSO",
    "S_ADPOSITION_BEFORE_NOUN",
    "S_ADPOSITION_AFTER_NOUN",
    "S_CASE_SUFFIX",
    "S_ADJECTIVE_BEFORE_NOUN",
]
 
# --- BLOCK 3: pull the OBSERVED (wals) data from URIEL ---------------- #
# syntax_wals, NOT syntax_knn: kNN fills gaps by inference, WALS returns
# only values that were actually recorded. A language returning "--" for
# a target is DROPPED from the dataset (below) rather than imputed.
langs = df_features["ISO_Code"].tolist()
print(f"[{len(langs)} languages] fetching {len(TARGET_FEATURES)} selected "
      f"features from URIEL (observed / wals)...")
 
try:
    syntax_data = l2v.get_features(langs, "syntax_wals", header=True)
except Exception as e:
    print("URIEL API error:", e)
    sys.exit(1)
 
feature_index = {name: i for i, name in enumerate(syntax_data["CODE"])}
 
# --- BLOCK 4: collect the 7 selected values for each language --------- #
# This loop walks every language, takes only the 7 columns listed in
# TARGET_FEATURES, and discards the entire row on seeing a "--". The
# resulting matrix therefore contains NO missing and NO imputed values:
# either a language is present with a fully observed row, or it is not
# present at all.
y_rows = []
dropped = []
for lang in langs:
    if lang not in syntax_data:
        dropped.append(lang)
        continue
    vector = syntax_data[lang]
    row = {"ISO_Code": lang}
    complete = True
    for feat in TARGET_FEATURES:
        value = vector[feature_index[feat]]
        if value == "--":
            complete = False
            break
        row[feat] = float(value)
    if complete:
        y_rows.append(row)
    else:
        dropped.append(lang)
 
if dropped:
    print(f"Warning: {len(dropped)} language(s) dropped for missing data: {dropped}")
 
df_targets = pd.DataFrame(y_rows)
 
# --- BLOCK 5: join X and Y, then save --------------------------------- #
final_matrix = pd.merge(df_features, df_targets, on="ISO_Code")
final_matrix.to_csv("training_matrix.csv", index=False)
 
print(f"\n[SUCCESS] Matrix shape: {final_matrix.shape}")
print("Expected column count: 11 (1 ISO_Code + 3 X features + 7 Y targets)")
print(f"Languages retained: {len(final_matrix)}")
 