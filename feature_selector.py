"""feature_selector.py
 
Determines which of the 103 typological features are ACTUALLY observed
(not kNN-imputed) and usable across our training languages.
 
Purpose: to perform the "103 -> a small number of features" reduction on
evidence rather than by assertion. Output: a short but defensible list of
target features.
"""
 
import lang2vec.lang2vec as l2v
 
# --- BLOCK 1: our training languages ---------------------------------- #
# ISO codes of the languages the model will be trained on. Elamite (elx)
# is DELIBERATELY ABSENT: it is the language to be predicted. Including
# it in the training set would be reading the answer key before the exam
# (data leakage). Only languages with genuine observed data appear here.
TRAIN_LANGS = [
    "tur", "eng", "ara", "fin", "hun", "eus", "fas", "hin",
    "jpn", "kor", "heb", "ind", "tam", "rus", "deu", "fra",
    "ita", "spa", "ces", "pol", "nld", "por", "swe", "bul",
    "dan", "ell", "ron", "slk", "hrv", "srp", "slv", "est",
    "lit", "lav",
]
 
# --- BLOCK 2: which data source we use -------------------------------- #
# "syntax_wals" = OBSERVED data, recorded by human linguists.
# "syntax_knn"  = the same set with gaps filled in by inference from
#                 neighbouring languages.
# We want the observed set, because only WALS tells us whether a feature
# is genuinely attested; kNN reports everything as filled, but much of it
# is guesswork. header=True additionally returns the feature NAMES.
FEATURE_SET = "syntax_wals"
 
 
def load_observed(langs, feature_set):
    """Fetch the observed feature vector for each language, plus the names.
 
    l2v.get_features(...) returns a dictionary:
      { "CODE": [feature names], "tur": [values], "eng": [...], ... }
    The "CODE" key is present because of header=True and holds the names.
    """
    data = l2v.get_features(langs, feature_set, header=True)
    feature_names = data["CODE"]          # names of the 103 features
    vectors = {lang: data[lang] for lang in langs if lang in data}
    return feature_names, vectors
 
 
def coverage_per_feature(feature_names, vectors):
    """For each feature: in how many languages is it FILLED (observed)?
 
    The marker "--" means no data for that language-feature pair. We walk
    each feature column and count how many languages carry a real value
    (0.0 or 1.0). A feature filled in many languages is a reliable target.
    """
    n_langs = len(vectors)
    rows = []
    for col_index, name in enumerate(feature_names):
        filled = 0
        positives = 0
        for lang in vectors:
            value = vectors[lang][col_index]
            if value != "--":                  # is it filled?
                filled += 1
                if float(value) == 1.0:        # is the feature PRESENT?
                    positives += 1
        rows.append({
            "feature": name,
            "filled": filled,                  # how many languages know it
            "coverage": filled / n_langs,      # proportion (0-1)
            "positives": positives,            # how many hold the value 1
        })
    return rows
 
 
def select_features(rows, min_coverage=0.90):
    """Select the usable features.
 
    Both conditions must hold:
      1) coverage >= min_coverage : filled in nearly every language. No
         model can be built on a feature that is half empty.
      2) 0 < positives < n_filled : the feature must take both the value
         1 and the value 0. A feature with the same value in every
         language is DEGENERATE -- there is nothing to learn from it, the
         model memorises it and produces spurious apparent accuracy.
    """
    selected = []
    for r in rows:
        n_filled = r["filled"]
        varies = 0 < r["positives"] < n_filled   # does it take both 1 and 0?
        if r["coverage"] >= min_coverage and varies:
            selected.append(r)
    # Sort from most-filled downwards, so the most reliable come first.
    selected.sort(key=lambda r: (-r["coverage"], -r["positives"]))
    return selected
 
 
def main():
    print(f"Training languages : {len(TRAIN_LANGS)}")
    print(f"Data source        : {FEATURE_SET} (observed)\n")
 
    feature_names, vectors = load_observed(TRAIN_LANGS, FEATURE_SET)
    print(f"Total features     : {len(feature_names)}")
    print(f"Languages returned : {len(vectors)}\n")
 
    rows = coverage_per_feature(feature_names, vectors)
    selected = select_features(rows, min_coverage=0.90)
 
    print(f"--- Selected features ({len(selected)}) ---")
    print(f"{'feature':32s} {'filled':>6s} {'cover':>6s} {'positive':>9s}")
    for r in selected:
        print(f"{r['feature']:32s} {r['filled']:6d} "
              f"{r['coverage']:6.2f} {r['positives']:9d}")
 
    print(f"\nSummary: {len(selected)} of the 103 features are both "
          f"sufficiently\nobserved and variable across languages. This is "
          f"our candidate target list.")
 
 
if __name__ == "__main__":
    main()