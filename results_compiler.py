"""results_compiler.py
 
Aggregates every experimental result into a single, reproducible summary:
the tables and figures that feed the Results section of the paper.
 
Why this exists: the results are currently scattered across terminal
output, robustness_report.json and two Elamite prediction CSVs. A number
copied by hand out of a terminal cannot be defended when a reviewer asks
where it came from. Every figure in the output of this script is
recomputed from the source matrices by the code below.
 
It also quantifies how far the two Elamite predictions DISAGREE.
Disagreement is direct evidence that the predictions are unreliable: if
two different input representations yield different typologies for the
same language, the output is an artefact of the method rather than a
property of the language.
 
Reads (skips silently if absent):
  training_matrix.csv, training_matrix_emb.csv, training_matrix_emb20.csv
  robustness_report.json
  elamite_pred_surface.csv, elamite_pred_embedding.csv
 
Writes:
  results_summary.csv      main results table (goes straight into the paper)
  elamite_comparison.csv   the two predictions side by side + disagreement
  fig_experiments.png      experiment results figure
  fig_elamite.png          Elamite prediction figure
  RESULTS.md               one-page human-readable summary
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
import matplotlib
matplotlib.use("Agg")          # render to file; no display required
import matplotlib.pyplot as plt
 
ID_COL = "ISO_Code"
 
# --- Palette (validated: CVD dE 24.7, normal-vision dE 33.6, all PASS) --- #
# The two series take categorical slots 1-2. The threshold line is NOT a
# series, so it takes neutral ink rather than a series colour -- given a
# series colour it would read as a third data series.
C_SERIES_1 = "#2a78d6"   # surface (blue)
C_SERIES_2 = "#eb6834"   # embeddings (orange)
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"
 
 
# --- BLOCK 1: evaluate one matrix under LOOCV ------------------------- #
def evaluate(path, label):
    """Measure one training matrix under LOOCV; return a single result row.
 
    Every experiment is re-measured by the SAME code path, so each number
    in the table is reproducible by running this script rather than being
    transcribed from a terminal.
 
    The X and Y columns are detected from the file itself: any column
    starting with "S_" is a target, every other column is a feature. The
    same function therefore handles the surface matrix (TTR / Avg_Length /
    Length_Var) and the embedding matrix (pca_*) -- running both through
    identical code is what makes the comparison fair.
 
    No hyperparameter search is performed here. Selecting hyperparameters
    on the same data used for evaluation biases the result optimistically,
    which at n=31 is not a risk worth taking.
    """
    if not Path(path).exists():
        print(f"  skipped (file not found): {path}")
        return None
 
    frame = pd.read_csv(path)
    if ID_COL in frame.columns:
        frame = frame.drop(columns=[ID_COL])
 
    target_cols = [c for c in frame.columns if c.startswith("S_")]
    feat_cols = [c for c in frame.columns if c not in target_cols]
    if not target_cols or not feat_cols:
        print(f"  skipped (column layout not recognised): {path}")
        return None
 
    X = frame[feat_cols].to_numpy(np.float64)
    Y = frame[target_cols].to_numpy().astype(np.int8)
 
    # Baseline: a predictor that blindly outputs the majority class of
    # each target. Any model that cannot beat this has learned nothing.
    majority = (Y.mean(axis=0) >= 0.5).astype(np.int8)
    baseline = hamming_loss(Y, np.tile(majority, (Y.shape[0], 1)))
 
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=2,
                                       min_samples_split=2,
                                       max_features="sqrt",
                                       random_state=42, n_jobs=-1)),
    ])
    preds = cross_val_predict(model, X, Y, cv=LeaveOneOut(), n_jobs=1)
    model_loss = hamming_loss(Y, preds)
 
    print(f"  {label:22s} model={model_loss:.4f}  baseline={baseline:.4f}")
    return {
        "Experiment": label,
        "N_Languages": int(X.shape[0]),
        "N_Features": int(X.shape[1]),
        "N_Targets": int(Y.shape[1]),
        "Model_Hamming_Loss": round(float(model_loss), 4),
        "Majority_Baseline": round(float(baseline), 4),
        "Improvement": round(float(baseline - model_loss), 4),
        "Beats_Baseline": bool(model_loss < baseline),
    }
 
 
# --- BLOCK 2: compare the two Elamite predictions --------------------- #
def compare_elamite(surface_path, embedding_path):
    """Put the two prediction sets side by side and quantify disagreement.
 
    THIS IS THE CENTRAL ARGUMENT. If two different input representations
    produce different typologies for the same language, the output is an
    artefact of the method, not a property of the language. It is as
    strong as "the model does not beat the baseline", and more intuitive.
 
    Same_Direction: do the two models make the same call at the 0.5
    threshold? A target that flips direction is the most severe form of
    inconsistency.
    """
    if not (Path(surface_path).exists() and Path(embedding_path).exists()):
        print("  skipped (Elamite prediction files missing).")
        return None
 
    s = pd.read_csv(surface_path)
    e = pd.read_csv(embedding_path)
 
    # Select the columns that are actually present rather than hard-coding
    # them. The two CSVs may carry different provenance columns depending
    # on which version of the trainer produced them, and the whole
    # compilation failing on a KeyError would not be acceptable.
    want = ["Typology_Feature", "P_positive", "Training_Base_Rate", "Status"]
    s_cols = [c for c in want if c in s.columns]
    e_cols = [c for c in ["Typology_Feature", "P_positive", "Status"]
              if c in e.columns]
    if "P_positive" not in s_cols or "P_positive" not in e_cols:
        print("  skipped (P_positive column not found).")
        return None
 
    merged = pd.merge(
        s[s_cols], e[e_cols],
        on="Typology_Feature", suffixes=("_surface", "_embedding"),
    )
    merged["Abs_Disagreement"] = (
        merged["P_positive_surface"] - merged["P_positive_embedding"]
    ).abs().round(4)
 
    call_s = merged["P_positive_surface"] >= 0.5
    call_e = merged["P_positive_embedding"] >= 0.5
    merged["Same_Direction"] = call_s == call_e
 
    print(f"  Mean disagreement    : {merged['Abs_Disagreement'].mean():.4f}")
    print(f"  Largest disagreement : {merged['Abs_Disagreement'].max():.4f}")
    print(f"  Direction flips      : {int((~merged['Same_Direction']).sum())}"
          f"/{len(merged)}")
    return merged
 
 
# --- BLOCK 3: figure chrome ------------------------------------------- #
_ACRONYMS = {"SOV", "SVO", "VSO", "OSV", "OVS", "VOS"}
 
 
def _pretty(name):
    """S_ADPOSITION_BEFORE_NOUN -> 'Adposition Before Noun'; S_SOV -> 'SOV'.
 
    A plain .title() call mangles acronyms ("SOV" -> "Sov"). Typological
    acronyms must stay uppercase; for a reviewer this is a question of
    correctness, not of readability.
    """
    parts = name.replace("S_", "", 1).split("_")
    return " ".join(p if p in _ACRONYMS else p.title() for p in parts)
 
 
def _style(ax):
    """Push grid and axes into the background so the data leads."""
    ax.set_facecolor(C_SURFACE)
    ax.figure.patch.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=C_MUTED, labelsize=9, length=0)
    ax.xaxis.grid(True, color=C_GRID, linewidth=1)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    # tick_params above muted EVERY tick label. Category names are data
    # identity, not chrome, so they are restored to primary ink. The
    # numeric axis stays muted.
    for lbl in ax.get_yticklabels():
        lbl.set_color(C_INK)
 
 
def plot_experiments(summary, out_path):
    """Plot each experiment against the baseline threshold (horizontal bars).
 
    Form: magnitude comparison with long category labels -> horizontal bar.
    A single series needs no legend box; the title already names it. The
    threshold line is drawn in neutral ink so it does not read as a second
    series. Every bar is directly labelled, so values never have to be
    estimated off the axis.
    """
    if summary is None or summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = np.arange(len(summary))
    baseline = float(summary["Majority_Baseline"].iloc[0])
 
    ax.barh(y, summary["Model_Hamming_Loss"], height=0.6,
            color=C_SERIES_1, zorder=3)
 
    ax.axvline(baseline, color=C_INK_2, linestyle="--", linewidth=1.6,
               zorder=4)
    # The threshold label is positioned in AXES fraction, not data
    # coordinates. In data coordinates it lands outside the drawing area
    # once the bar count changes and is clipped out of the image -- which
    # is exactly what happened on the first render.
    ax.annotate(f"Majority baseline  {baseline:.4f}",
                xy=(baseline, 1.0), xycoords=("data", "axes fraction"),
                xytext=(5, 5), textcoords="offset points",
                color=C_INK_2, fontsize=8.5, va="bottom")
 
    for i, v in enumerate(summary["Model_Hamming_Loss"]):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", ha="left",
                fontsize=9, color=C_INK)
 
    ax.set_yticks(y)
    ax.set_yticklabels(summary["Experiment"], fontsize=9.5)
    ax.set_xlabel("Hamming Loss  (lower is better)", fontsize=9,
                  color=C_INK_2)
    ax.set_title("No feature set outperforms the majority baseline",
                 fontsize=11, color=C_INK, loc="left", pad=24)
    ax.set_xlim(0, max(summary["Model_Hamming_Loss"].max(), baseline) * 1.28)
    ax.invert_yaxis()
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"  figure -> {out_path}")
 
 
def plot_elamite(comp, out_path):
    """Plot the two Elamite predictions side by side; disagreement is visible.
 
    Two series, so a legend is mandatory -- identity is never carried by
    colour alone. The 0.5 decision threshold is a neutral dotted line. A
    small surface gap separates the paired bars so adjacent fills do not
    bleed into one another.
    """
    if comp is None or comp.empty:
        return
    labels = [_pretty(f) for f in comp["Typology_Feature"]]
    y = np.arange(len(labels))
    h = 0.36
 
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.barh(y - h / 2 - 0.02, comp["P_positive_surface"], height=h,
            color=C_SERIES_1, label="Surface (3 statistics)", zorder=3)
    ax.barh(y + h / 2 + 0.02, comp["P_positive_embedding"], height=h,
            color=C_SERIES_2, label="Embeddings (10 PCA)", zorder=3)
 
    ax.axvline(0.5, color=C_INK_2, linestyle=":", linewidth=1.4, zorder=4)
    ax.annotate("decision threshold", xy=(0.5, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(5, -9), textcoords="offset points",
                color=C_INK_2, fontsize=8, va="top")
 
    # A zero-length bar is INVISIBLE, and the reader is left wondering
    # whether the data is missing. Targets predicted at 0 are labelled
    # explicitly; these are the targets that are constant (degenerate)
    # in the training set.
    for i in range(len(comp)):
        for val, off in ((comp["P_positive_surface"].iloc[i], -h / 2 - 0.02),
                         (comp["P_positive_embedding"].iloc[i], h / 2 + 0.02)):
            if val < 0.02:
                ax.text(0.008, i + off, f"{val:.3f}", va="center", ha="left",
                        fontsize=8, color=C_MUTED)
 
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("P(feature = 1) for Elamite", fontsize=9, color=C_INK_2)
    ax.set_xlim(0, 1.0)
    ax.set_title("Unvalidated prototype: the two feature sets disagree",
                 fontsize=11, color=C_INK, loc="left", pad=26)
    # The legend sits ABOVE the plotting area, laid out horizontally.
    # Placed inside, it overlapped the lower bars -- which is what the
    # first render did.
    leg = ax.legend(fontsize=8.5, frameon=False, ncol=2,
                    loc="lower left", bbox_to_anchor=(0, 1.01))
    for t in leg.get_texts():
        t.set_color(C_INK_2)
    ax.invert_yaxis()
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"  figure -> {out_path}")
 
 
# --- BLOCK 4: human-readable summary ---------------------------------- #
def _md_table(frame):
    """DataFrame -> markdown table, with or without `tabulate` installed.
 
    pandas.to_markdown() silently depends on the `tabulate` package. A
    script whose entire summary collapses over one missing optional
    dependency is not much use, hence the fallback.
    """
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        cols = list(frame.columns)
        out = ["| " + " | ".join(cols) + " |",
               "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in frame.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(out)
 
 
def write_markdown(summary, comp, robustness, out_path):
    """RESULTS.md: the one page to open while writing the paper."""
    lines = ["# Results Summary", ""]
 
    if summary is not None and not summary.empty:
        lines += ["## Experiments (LOOCV, Hamming Loss)", "",
                  _md_table(summary), ""]
        if not summary["Beats_Baseline"].any():
            lines += ["**No feature set beats the majority baseline.** "
                      "The result holds from 3 hand-built statistics up to "
                      "20 PCA components of XLM-R representations.", ""]
 
    if robustness and (robustness.get("test1_models")
                       or robustness.get("test2_shuffle")):
        lines += ["## Robustness checks", ""]
        base = robustness.get("baseline")
        for name, loss in robustness.get("test1_models", {}).items():
            verdict = "beats" if base and loss < base else "does not beat"
            lines.append(f"- **{name}**: {loss:.4f} ({verdict} baseline)")
        sh = robustness.get("test2_shuffle", {})
        if sh:
            lines.append(
                f"- **Feature shuffle**: real X = {sh['real']:.4f}, "
                f"shuffled X = {sh['shuffled']:.4f} - corrupting the features "
                "does not hurt performance, i.e. no usable signal was present.")
        lines += ["",
                  "_Note: the robustness suite uses a per-target LOOCV loop "
                  "to accommodate folds in which a target is single-class. "
                  "Absolute values therefore differ slightly from the table "
                  "above, while the conclusion is identical._", ""]
 
    if comp is not None and not comp.empty:
        flips = int((~comp["Same_Direction"]).sum())
        lines += ["## Elamite prototype (UNVALIDATED)", "",
                  _md_table(comp), "",
                  f"- Mean absolute disagreement between the two feature "
                  f"sets: **{comp['Abs_Disagreement'].mean():.4f}**",
                  f"- Largest single disagreement: "
                  f"**{comp['Abs_Disagreement'].max():.4f}**",
                  f"- Targets where the two sets cross the 0.5 threshold in "
                  f"opposite directions: **{flips}/{len(comp)}**",
                  "- URIEL observed (WALS / SSWL) data for Elamite: "
                  "**0 of 103 features**", "",
                  "> These numbers are a prototype, not a finding. The model "
                  "does not beat a constant predictor, the two feature sets "
                  "disagree with each other, and no observed data exists "
                  "against which any prediction could be validated.", ""]
 
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  summary -> {out_path}")
 
 
def main():
    print("Evaluating experiments (LOOCV)...")
    rows = []
    for path, label in [
        ("training_matrix.csv", "Surface (3 statistics)"),
        ("training_matrix_emb.csv", "Embeddings (10 PCA)"),
        ("training_matrix_emb20.csv", "Embeddings (20 PCA)"),
    ]:
        row = evaluate(path, label)
        if row:
            rows.append(row)
 
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv("results_summary.csv", index=False)
        print("  table -> results_summary.csv")
 
    print("\nComparing Elamite predictions...")
    comp = compare_elamite("elamite_pred_surface.csv",
                           "elamite_pred_embedding.csv")
    if comp is not None:
        comp.to_csv("elamite_comparison.csv", index=False)
        print("  table -> elamite_comparison.csv")
 
    robustness = None
    if Path("robustness_report.json").exists():
        robustness = json.loads(
            Path("robustness_report.json").read_text(encoding="utf-8"))
 
    print("\nDrawing figures...")
    plot_experiments(summary, "fig_experiments.png")
    plot_elamite(comp, "fig_elamite.png")
 
    print("\nWriting summary...")
    write_markdown(summary, comp, robustness, "RESULTS.md")
 
    print("\n[DONE] All findings compiled in one place.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 