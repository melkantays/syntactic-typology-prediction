# Source data: ORACC / ARIO

This directory is an **unmodified download** from ORACC (The Open
Richly Annotated Cuneiform Corpus), project **ARIO — Achaemenid Royal
Inscriptions Online**. Nothing in it was edited; it is included so the
Elamite branch of this pipeline can be reproduced from its actual
source rather than from a derived file.

- Project page: http://oracc.museum.upenn.edu/ario/
- Downloaded: August 2026
- License: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)

## What this pipeline reads

Only `corpusjson/*.json`. The remaining files are part of the same
distribution and are kept for provenance:

| Path | Used | Contents |
|---|---|---|
| `corpusjson/*.json` | **yes** | 139 inscriptions; the lemma-level annotation the extractor reads |
| `gloss-akk.json`, `gloss-peo.json`, `gloss-qpn.json` | no | Akkadian / Old Persian / proper-noun glossaries |
| `index-*.json`, `catalogue.json`, `corpus.json`, `metadata.json`, `ario-portal.json`, `cat.geojson`, `sortcodes.json` | no | ORACC's own indices and catalogue |

## Language composition

The Achaemenid royal inscriptions are trilingual. Running
`lang_audit.py` from the repository root reports, over all 139 files:

| lang | language | lemmas | share |
|---|---|---|---|
| peo | Old Persian | 15,333 | 81.0% |
| akk | Akkadian | 2,265 | 12.0% |
| elx | **Elamite** | **1,330** | **7.0%** |

`oracc_extractor.py` filters on `lang == "elx"`; the 1,330 Elamite
lemmas correspond to the 1,327 rows of `melkan_elamite_dataset.csv`
(the three-row difference is empty forms, dropped on extraction).

Two consequences are carried into the paper's limitations:

1. **Sample size.** Elamite is only 7% of this corpus. After cleaning,
   the Elamite text yields 1,322 tokens against the 5,000-token budget
   used for every training language — hence the `Budget_Shortfall=1`
   stamp. Larger Elamite corpora exist (notably the Persepolis
   Fortification Archive) but are separate ORACC projects.

2. **Genre mismatch.** The 31 training languages are drawn from UD
   corpora spanning news, encyclopedic and fictional prose. Elamite
   here is drawn entirely from royal inscriptions, a highly formulaic
   register. Type-token ratio and token length are genre-sensitive, so
   the Elamite surface features are not measuring the same thing as
   the training languages' surface features.
