# State-Space Echo State Networks for Multi-Horizon Financial Forecasting

> **Course:** DS & AI Lab • **Repo status:** Milestone 1 submitted (Problem Definition & Literature Review)
> We’ll keep this README updated as we progress through subsequent milestones.

---

## 🔎 Project Overview

This project investigates **Echo State Networks (ESNs)** framed explicitly as **state-space models (SSMs)** for **multi-horizon financial forecasting** (returns/prices).
The ESN reservoir provides a fixed, contractive latent transition; only the linear readout(s) are trained. We evaluate **point**, **probabilistic**, and **decision-aware** performance under **leakage-safe, walk-forward** validation.

* **Modality/Tags:** DL · Time-Series · Finance · SSM · Reservoir Computing · Probabilistic ML
* **Data Sources (open):** Yahoo Finance, Stooq, Alpha Vantage (free tier), FRED (macro exogenous series)

---

## 🎯 Objectives (Milestone 1)

1. **Problem Definition**
   Formalize ESN as a state-space model with contractive reservoir and multi-horizon readouts.
2. **Literature Review**
   ESNs/RC (Jaeger; Lukoševičius & Jaeger), SSMs (Durbin & Koopman), deep SSMs (S4), forecasting evaluation (Hyndman & Athanasopoulos), proper scoring (Gneiting & Raftery), backtest hygiene (Bailey et al.), DM tests (Diebold–Mariano).
3. **Gaps & Opportunities**

   * Variance control via fixed dynamics in noisy regimes
   * Explicit SSM framing of ESNs with probabilistic outputs
   * Unified, leakage-safe evaluation across point, probabilistic, and trading metrics
   * Systematic use of exogenous macro factors

> 📄 The Milestone 1 PDF (Problem Definition & Literature Review) is in `docs/` (will be updated as we iterate).

---

## 🗂️ Repository Structure (as of Milestone 1)

```
.
├── docs/
│   └── milestone_1_problem_litreview.pdf   # This document
├── notebooks/
│   └── 00_data_sources_eda.ipynb           # Sketch/EDA scaffold (to be expanded)
├── src/                                    # Placeholder: will host core library soon
│   ├── __init__.py
│   └── es_ssm/                             # Planned: reservoir, readouts, validation
├── data/                                   # .gitignore: raw/processed data not tracked
│   ├── raw/
│   └── processed/
├── .gitignore
├── LICENSE
└── README.md
```

**Why this layout?**
We aim for clean separation of concerns: `src/` for reusable code, `notebooks/` for exploration/EDA, `docs/` for milestone artifacts, and `data/` (ignored) for local datasets.

---

## 📦 Setup (lightweight for M1)

We’re not shipping a full package yet—just a minimal environment to run EDA and data downloads.

```bash
# Option A: Conda
conda create -n es-ssm python=3.11 -y
conda activate es-ssm
pip install -r requirements.txt

# Option B: venv
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**`requirements.txt` (initial):**

```
numpy
pandas
scikit-learn
matplotlib
yfinance
requests
tqdm
statsmodels
```

> We’ll pin versions and add experiment tooling (e.g., hydra/lightning/wandb) in later milestones.

---

## 📥 Data Sources (Open)

* **Yahoo Finance** — end-of-day prices/volumes (via `yfinance`)
* **Stooq** — free historical equities/FX/indices (CSV)
* **Alpha Vantage** — free API tier for equities/FX/crypto (API key required)
* **FRED** — macro series for exogenous covariates (e.g., rates, CPI, industrial production)

> All downloads will honor **walk-forward** splits and **no look-ahead** transformations.

---

## 🧪 Evaluation (defined in M1)

* **Point:** RMSE, MAE, sMAPE
* **Probabilistic:** interval coverage/width
* **Decision-aware:** cost-adjusted P&L, Sharpe, max drawdown, turnover
* **Stats:** Diebold–Mariano tests for predictive accuracy differences
* **Backtesting hygiene:** rolling/expanding splits; scalers/feature transforms fit on train only

---

## 🧭 Roadmap & Milestones

* **M1 (Oct 3):** ✅ Problem Definition & Literature Review
* **M2 (Oct 10):** Dataset preparation, feature schema, leakage-safe splits
* **M3 (Oct 17):** Model architecture (reservoir design, leak rate, horizon heads)
* **M4 (Oct 31):** Training runs, hyperparam sweeps, ablations
* **M5 (Nov 7):** Evaluation, error analysis, limitations
* **M6 (Nov 14):** Minimal demo/API, documentation, final report

*(Dates per course portal; we’ll adjust if the schedule changes.)*

---

## 🧰 How to Reproduce Milestone 1

1. Clone the repo and set up the environment.
2. Open `notebooks/00_data_sources_eda.ipynb` to preview data source access and schema checks.
3. Read `docs/milestone_1_problem_litreview.pdf` for the formal problem, literature, and evaluation plan.

> M1 focuses on design & review—no official results are published yet.

---

## 🔒 Ethics & Responsible Use

* We do **not** make investment recommendations.
* All results will be **educational**, using historical data with explicit assumptions about costs/slippage.
* We will document limitations and **avoid backtest overfitting** practices.

---

## 📚 Key References (M1)

* Jaeger (2001); Lukoševičius & Jaeger (2009) — ESNs & Reservoir Computing
* Jaeger et al. (2007) — Leaky ESNs
* Durbin & Koopman (2012) — State-Space Methods
* Gneiting & Raftery (2007) — Proper Scoring & Calibration
* Hyndman & Athanasopoulos (FPP3) — Forecasting principles
* Diebold & Mariano (1995) — Predictive accuracy tests
* Bailey et al. (2014) — Backtest overfitting

(A full reference list appears in the Milestone 1 PDF.)

---

## 👥 Contributors

* **Team:** GROUP-3



---

## 💬 Contact

For questions or issues, please open a GitHub Issue or reach out to the team maintainers.
