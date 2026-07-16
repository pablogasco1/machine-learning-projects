# Data Science interview practice

Seven notebooks of tasks + worked solutions, built to match the pipeline in
`pipeline_guide.html`. Datasets are local — nothing needs the network at practice time.

Open a notebook, read the task, write code in the empty cell under it. The matching
notebook in `solutions/` has a worked answer for every task, plus the *reasoning* an
interviewer is listening for.

---

## The notebooks

| # | Notebook | Covers | Time |
|---|---|---|---|
| 01 | `01_cleaning_titanic.ipynb` | Load, dtypes, missing values, duplicates, outliers | ~35 min |
| 02 | `02_eda_titanic.ipynb` | Univariate, bivariate, correlation, reading distributions | ~30 min |
| 03 | `03_features_titanic.ipynb` | Engineering, encoding, scaling, VIF, **leakage** | ~35 min |
| 04 | `04_modeling_titanic.ipynb` | Baselines, CV, metrics, tuning, over/underfitting, importances | ~40 min |
| 05 | `05_regression_housing.ipynb` | The whole pipeline again, but **regression** | ~45 min |
| 06 | `06_pandas_drills.ipynb` | groupby, merge, pivot, dates, window functions | ~30 min |
| 07 | `07_mock_interview.ipynb` | **Timed, cold start, no hints.** Do this last | 60 min |

**Suggested order:** 01 → 02 → 03 → 04 → 05 → 06, then 07 as a dress rehearsal.
Short on time? Do **06** (fluency), **03** (leakage) and **07** (the real thing).

---

## The datasets

| File | Rows | What it's for |
|---|---|---|
| `titanic_raw.csv` | 914 | Deliberately dirty: `$` fares, mixed date formats, dupes, impossible ages, inconsistent casing |
| `titanic_clean.csv` | 891 | The original Kaggle Titanic — still has real missing values |
| `housing_raw.csv` | 20,680 | California housing + synthetic `Neighbourhood`/`HouseType`/`Condition`, lightly messed up |
| `housing_clean.csv` | 20,640 | Same, unmessed |
| `orders.csv` / `customers.csv` | 5,000 / 400 | Two joinable tables for wrangling and the churn mock |

Two things were planted on purpose, and finding them is part of the exercise:
`housing`'s target is **censored at $500k**, and `orders` has **~3% orphan rows** whose
`customer_id` isn't in `customers.csv`.

---

## Environment

Every solution notebook was executed end to end against the `homl3` conda env
(pandas 2.1.4, sklearn 1.3.2) — the kernel VS Code picks by default here. They also run
on pandas 3.x, with one caveat worth knowing: `resample('M')` (pandas < 2.2) became
`resample('ME')` (2.2+) and each version rejects the other's alias. `dt.to_period('M')`
works on both, so that's what the notebooks use.

Needs: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `statsmodels`, `joblib`,
`scipy`. All already present in `homl3`.

---

## How to get value out of this

1. **Say it out loud.** Every task has a number as its answer and a *sentence* as its real
   answer. The sentence is what gets scored. Practise talking while typing — it's harder
   than it sounds and it's most of what a live round tests.
2. **Don't peek early.** A wrong answer you then correct sticks; a read solution doesn't.
3. **Time yourself.** Running out of time while narrating clearly beats silence.
4. **Say what you distrust.** Several of these datasets contain features that are pure
   noise, and one has no signal at all. Noticing is the skill.

### The five things most likely to come up

1. `groupby().agg(name=('col', 'fn'))` and `transform` vs `agg` — pure muscle memory.
2. **Split before you fit.** Any statistic (median, mode, scaler, encoder) is learned on
   train only. Say this unprompted at least once.
3. **Baseline first.** `DummyClassifier` before anything else, so a score has meaning.
4. **Accuracy is usually the wrong metric** — be ready to explain why, in terms of the
   positive rate.
5. **Check every merge.** Row count before, row count after.

### The traps planted for you

- **01** — `cabin` is 77% missing and looks like an easy drop. Its missingness carries most
  of the signal.
- **02** — `age` correlates ~-0.06 with survival and is genuinely predictive anyway.
- **03** — every imputation is a leak waiting to happen; also, no VIF here exceeds 10 even
  though two features are 84% correlated.
- **04** — the tuned random forest wins CV and then *loses* on test. Work out why.
- **05** — the target is censored, and `Condition` is random noise dressed up as a feature.
- **07** — compute recency without a temporal cutoff and you'll score 0.99 AUC on a
  dataset with no signal in it whatsoever.
