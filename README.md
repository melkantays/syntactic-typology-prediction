# Predicting Syntactic Typology from Surface Text Statistics

A controlled negative result on whether corpus-level text statistics and
pretrained multilingual representations carry enough signal to predict
observed syntactic typology, evaluated on 31 languages and probed on a
language isolate (Elamite).

## Finding

No feature set outperformed a majority-class baseline.

| Experiment | Features | Hamming Loss | Majority baseline |
|---|---|---|---|
| Surface statistics | 3 | 0.2949 | 0.2488 |
| XLM-R embeddings, 10 PCA | 10 | 0.2995 | 0.2488 |
| XLM-R embeddings, 20 PCA | 20 | 0.2488 | 0.2488 |

The 20-component result lands exactly on the baseline: at that setting
the model rediscovers constant prediction. Four independent robustness
checks (three model classes, a feature-permutation test, random target
subsets, and five random seeds) reproduce the same outcome. In the
permutation test, corrupting the features by shuffling them across
languages does not hurt performance — direct evidence that no usable
signal was present in the first place.

Full numbers, including the Elamite prototype, are in `RESULTS.md`, and
are regenerated from the source matrices by `results_compiler.py`.

## Data

All corpora and derived matrices are committed to this repository, so
every number reported here can be reproduced without downloading
anything: `UD_Corpus/` holds the per-language text used for feature
extraction, `ario-json/` holds the raw ORACC Elamite source, and
`UD_Elx/elx.txt` is the cleaned Elamite corpus derived from it.

## Pipeline

Run in this order. Each script names its inputs and outputs in its
module docstring.

| # | Script | Produces |
|---|---|---|
| 1 | `corpus_downloader.py` | `UD_Corpus/*.txt` |
| 2 | `ario-json/oracc_extractor.py` | `melkan_elamite_dataset.csv` |
| 3 | `data_cleaner.py` | `melkan_elamite_cleaned.csv` |
| 4 | `csv_to_txt.py` | `UD_Elx/elx.txt` |
| 5 | `feature_engineer.py` | `extracted_features.csv` |
| 6 | `feature_selector.py` | the usable-target report (stdout) |
| 7 | `dataset_compiler.py` | `training_matrix.csv` |
| 8 | `embedding_extractor.py` | `embedding_features.csv` |
| 9 | `embedding_compiler.py` | `training_matrix_emb.csv`, `training_matrix_emb20.csv` |
| 10 | `model_validator.py` | `validation_report.json` |
| 11 | `robustness_check.py` | `robustness_report.json` |
| 12 | `elamite_check.py` | URIEL coverage report for Elamite (stdout) |
| 13 | `model_trainer_final.py` | `elamite_pred_surface.csv`, `elamite_pred_embedding.csv` |
| 14 | `results_compiler.py` | `results_summary.csv`, `RESULTS.md`, both figures |

`lang_audit.py` is a standalone check on the source data: it reports the
language composition of the ORACC corpus, confirming that the Elamite
extraction did not pull in Old Persian or Akkadian.

Elamite is processed as a query language, not a training language:

```
py feature_engineer.py --corpus UD_Elx --out elamite_features.csv --query-mode
py embedding_extractor.py --corpus UD_Elx --out elamite_embedding.csv
py model_trainer_final.py --mode surface   --out elamite_pred_surface.csv
py model_trainer_final.py --mode embedding --out elamite_pred_embedding.csv
```

## Install

```
py -m pip install -r requirements.txt
```

`requirements.txt` pins `setuptools<82`: later versions removed
`pkg_resources`, which `lang2vec` imports at load time. Exact versions
used to produce the reported numbers are in `requirements-lock.txt`.

## Method notes

- **Targets.** 7 of URIEL's 103 syntactic features, selected by
  `feature_selector.py` as >=90% filled across the training languages and
  variable between them. `syntax_wals` (observed) is used throughout,
  never `syntax_knn` (kNN-imputed); a language missing any target is
  dropped rather than imputed.
- **Evaluation.** Leave-one-out cross-validation with Hamming Loss,
  against a per-target majority-class baseline. Scaling and PCA are
  fitted inside the training split only.
- **No hyperparameter search on the evaluation data.** A fixed
  configuration is used for all reported experiments. Selecting
  hyperparameters on the same data used for evaluation is optimistically
  biased at n=31.
- **Two evaluation protocols.** The robustness suite fits one classifier
  per target, to accommodate folds where a target is single-class;
  `results_compiler.py` fits one multi-output classifier. Absolute
  values differ slightly between the two; the conclusion does not.

## On the Elamite predictions

`elamite_pred_*.csv` are a **prototype, not a finding**, and every row is
stamped as such. Three facts constrain how they may be read:

1. The model does not beat a constant predictor.
2. Elamite has **0 of 103** observed URIEL features (WALS 0, SSWL 0), so
   no observed data exists against which any prediction could be
   validated. For comparison, Basque has 82 and Turkish 86.
3. The two feature sets disagree with each other on the same language,
   which is itself evidence that the output reflects the method rather
   than the language.

Two further limitations bear specifically on the embedding branch: the
Elamite corpus is a concatenation of disconnected transliterated
fragments rather than running text, and XLM-R has never seen Elamite —
it reads the Latin-script transliteration as a corrupted Latin-script
language.

## Citing

Melkan Taysi, Department of Artificial Intelligence Engineering,
Hacettepe University, Ankara. Work supervised by Res. Asst. Orhan
Demirci, Hacettepe University.

## License

The contents of `ario-json/` are redistributed from ORACC / ARIO under
CC BY-SA 3.0, and files derived from them remain under that license.
See `ario-json/README.md` for attribution.

A license for the code in this repository is to be determined.
