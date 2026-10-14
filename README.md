# Diabetes Classification

Exploratory analysis and classification on a public Chinese diabetes dataset. The goal is to detect undiagnosed prediabetes/diabetes using only non-glycemic features (anthropometrics, blood pressure, lipids, liver/kidney function, blood count, lifestyle).

## Dataset

| | |
|---|---|
| Source | Li J., Zheng H., Zhou Y., …, Jiang F. *A bimodal dataset for diabetes research.* Scientific Data 13, 652 (2026). [doi:10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y) |
| Data | Zenodo, [doi:10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337) (CC BY 4.0) |
| Origin | Dept. of Endocrinology & Metabolism, Shanghai Sixth People's Hospital (Shanghai Jiao Tong University), China |
| Period | 10 Feb – 7 Apr 2012 (cross-sectional) |
| Size | 5,922 records × 190 columns, 1 row = 1 person, no duplicate IDs |
| Encoding | GBK (`pd.read_csv(..., encoding="gbk")`); blank cells are whitespace strings |

### Files (data/)

- Dataset for diabetes research.csv: the raw data.
- Supplementary Information.docx: Supplementary Table 1, the data dictionary (variable, label, value coding).

### Column groups

Demographics (age, gender) · anthropometrics (height, weight, BMI, waist, hip, WHR, body fat) · blood pressure and heart rate · glycemia (FPG, PG2h, HbA1c, HPLC-HbA1c, glycated albumin) · insulin/C-peptide and HOMA indices · liver function · lipids · kidney function (creatinine, uric acid, urine albumin, eGFR) · complete blood count · thalassemia flags · medical history, smoking, drinking, tea · complications (retinopathy, neuropathy, nephropathy, cardiovascular) · family history · questionnaire items · many derived categorical cut-offs.

Three columns are text: Data (survey date), EscDrinkD, EyeDia.

### Target

`PGclass3`: 0 = NGR (normal), 1 = prediabetes, 2 = diabetes (WHO 1999).

## Known Issues in the Dataset

- **Population is ambiguous.** The paper calls the records "diabetes patients", and Zenodo calls them "follow-up records". However, 75.7% of the records are NGR and only 49 people have a prior diabetes diagnosis. The cohort looks more like a screening or OGTT-referral population. No inclusion criteria are published.
- **Gender coding mismatch.** The dictionary says 0/1, but the data uses 1/2. Height, hemoglobin and body fat indicate **1 = male, 2 = female**.
- **ID gaps.** IDs go up to 6,571 but there are only 5,922 rows, so 649 IDs are missing.
- **Undocumented leaking columns.** `VAR00001` has no description. See the leakage section.
- **Implausible kidney values.** 330 people have creatinine < 35 µmol/L, which in turn produces 376 eGFR values > 200.
- **"9 = unknown" coding.** 689 people answered 9 on all 15 disease-history items. This creates spurious correlations of r ≥ 0.9 between those items.

## Main EDA Findings (`eda-diabetes.ipynb`)

**Screening subset.** Excluding people already diagnosed with diabetes (`DMhis = 1`) leaves **5,873** people:

| Class | n | % |
|---|---|---|
| NGR | 4,444 | 75.7 |
| Prediabetes | 1,121 | 19.1 |
| DM | 308 | 5.2 |

The classes are imbalanced (NGR : DM ≈ 14 : 1).

**Label leakage.**

1. **The label is deterministic.** Applying WHO 1999 rules to `FPG`/`PG2h`/`DMhis` reproduces `PGclass3` for **5,922 / 5,922** rows. Any model that sees glucose is only relearning a threshold rule.
2. **`VAR00001` (undocumented) leaks the label.** Its value is 1 or 9, and 9 never occurs in NGR (0 / 4,444) but does occur in prediabetes (484) and DM (111).
3. **`ThalussemiaAND` leaks the label.** A value of 1 appears in only 2 NGR records versus 371 prediabetes/DM records. It appears to be filled only for people with abnormal glucose.

As a result, 17 columns are dropped as leakage. These are the glucose columns, derived indices (HOMA, ISI, OGTT insulin/C-peptide), the glucose flags, the other label versions, `VAR00001` and `ThalussemiaAND`. A further 20 columns are dropped because they are other glycemic tests (HbA1c, GA), study markers, complications, or mostly-unknown questionnaire items. After also removing columns that are >50% missing or constant, **135 candidate features** remain.

**Signal without glucose (median NGR → Pre → DM).**

| Feature | NGR | Pre | DM |
|---|---|---|---|
| Age | 38 | 43 | 47 |
| Waist (cm) | 73 | 76 | 79 |
| Systolic BP | 110 | 120 | 120 |
| GGT | 24 | 34 | 61 |
| ALT | 21 | 24 | 30 |
| Male (%) | 43.6 | 52.5 | 69.5 |
| High BP (%) | 18.8 | 30.2 | 51.9 |

DM separates well from NGR. Prediabetes differs only slightly from NGR and is expected to be the hardest class.

**Redundancy.** There are 56 feature pairs with |Spearman r| ≥ 0.9. Six BMI variants are cut-offs of the same BMI. Other pairs include Age/AgeGroup, GFR category variants, %/count blood cells, TBIL/IDBIL and HGB/HCT. These will be pruned before modeling.

## Contribution

As of Oct 2026, no published study has used this dataset (0 citations on OpenAlex), and the data descriptor reports no ML baseline. This work contributes:

1. **The first ML baseline** on this dataset.
2. **Documented, reproducible label leakage.** The target is fully recomputable from FPG/PG2h, and two undocumented or under-documented columns (`VAR00001`, `ThalussemiaAND`) leak it.
3. **A leakage-free setup for undiagnosed diabetes screening without glucose tests.** Known diabetics are excluded, and all glycemic, derived and leaking columns are removed.

Every claim above can be checked directly from the raw CSV with the notebook.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook eda-diabetes.ipynb
```

## Status

- [x] EDA: missing values, target, leakage, distributions, outliers, redundancy
- [ ] Preprocessing (redundancy pruning, creatinine/eGFR handling, imputation)
- [ ] Baselines (logistic regression, RF, XGBoost) with class imbalance handling
- [ ] Evaluation (macro-F1, per-class recall, AUROC) and feature importance
