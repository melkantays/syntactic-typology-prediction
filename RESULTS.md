# Results Summary

## Experiments (LOOCV, Hamming Loss)

| Experiment             |   N_Languages |   N_Features |   N_Targets |   Model_Hamming_Loss |   Majority_Baseline |   Improvement | Beats_Baseline   |
|:-----------------------|--------------:|-------------:|------------:|---------------------:|--------------------:|--------------:|:-----------------|
| Surface (3 statistics) |            31 |            3 |           7 |               0.2949 |              0.2488 |       -0.0461 | False            |
| Embeddings (10 PCA)    |            31 |           10 |           7 |               0.2995 |              0.2488 |       -0.0507 | False            |
| Embeddings (20 PCA)    |            31 |           20 |           7 |               0.2488 |              0.2488 |        0      | False            |

**No feature set beats the majority baseline.** The result holds from 3 hand-built statistics up to 20 PCA components of XLM-R representations.

## Robustness checks

- **RandomForest**: 0.3134 (does not beat baseline)
- **LogisticReg**: 0.2949 (does not beat baseline)
- **Dummy**: 0.2488 (does not beat baseline)
- **Feature shuffle**: real X = 0.3134, shuffled X = 0.2857 - corrupting the features does not hurt performance, i.e. no usable signal was present.

## Elamite prototype (UNVALIDATED)

| Typology_Feature         |   P_positive_surface |   Training_Base_Rate | Status_surface       |   P_positive_embedding | Status_embedding   |   Abs_Disagreement | Same_Direction   |
|:-------------------------|---------------------:|---------------------:|:---------------------|-----------------------:|:-------------------|-------------------:|:-----------------|
| S_SOV                    |             0.207472 |             0.290323 | INFORMATIVE          |               0.321464 | INFORMATIVE        |             0.114  | True             |
| S_SVO                    |             0.915146 |             0.806452 | INFORMATIVE          |               0.731415 | INFORMATIVE        |             0.1837 | True             |
| S_VSO                    |             0        |             0        | DEGENERATE_LABEL     |               0        | DEGENERATE_LABEL   |             0      | True             |
| S_ADPOSITION_BEFORE_NOUN |             0.880972 |             0.741935 | INFORMATIVE          |               0.699382 | INFORMATIVE        |             0.1816 | True             |
| S_ADPOSITION_AFTER_NOUN  |             0.121721 |             0.290323 | INFORMATIVE          |               0.340823 | INFORMATIVE        |             0.2191 | True             |
| S_CASE_SUFFIX            |             0.56112  |             0.580645 | VACUOUS_EQUALS_PRIOR |               0.52885  | INFORMATIVE        |             0.0323 | True             |
| S_ADJECTIVE_BEFORE_NOUN  |             0.72406  |             0.709677 | VACUOUS_EQUALS_PRIOR |               0.572494 | INFORMATIVE        |             0.1516 | True             |

- Mean absolute disagreement between the two feature sets: **0.1260**
- Largest single disagreement: **0.2191**
- Targets where the two sets cross the 0.5 threshold in opposite directions: **0/7**
- URIEL observed (WALS / SSWL) data for Elamite: **0 of 103 features**

> These numbers are a prototype, not a finding. The model does not beat a constant predictor, the two feature sets disagree with each other, and no observed data exists against which any prediction could be validated.
