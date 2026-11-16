# FraudLens | Public Research Demo

FraudLens is an explainable digital-payment anomaly-screening research dashboard
built using **Streamlit, Isolation Forest, SHAP and PaySim**.

> **Research demonstration only.** A flag does not establish that a transaction
> is fraudulent. The model was evaluated on synthetic PaySim records, not
> real Philippine banking transactions. Do not upload real customer data.

## Deployment on Streamlit Community Cloud

1. Create a **new GitHub repository** for the contents of this directory (files at repository root).
2. Commit **`app.py`**, `requirements.txt`, `src/`, **`output/fraudlens.db`**,
   **`output/model.joblib`**, **`output/scaler.joblib`**,
   **`output/feature_cols.joblib`**, **`output/model_config.json`**, and
   **`data/train.csv`**. The latter is needed for the exact SHAP explainer's
   fixed training background; it contains synthetic PaySim data.
3. Do **not** include a `venv/`, the full PaySim dataset, GitHub credentials,
   or `output/analyst_evaluation/`.
4. At https://share.streamlit.io, choose the GitHub repository, branch `main`,
   entrypoint `app.py`, and a Python version compatible with the pinned requirements.
5. Let Streamlit install the versions in `requirements.txt`. Run all six pages
   in your deployed app. If installation fails, check package availability and
   the deployment logs; do not silently change model-library versions.

You can run the same demo locally:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Six dashboard pages

- **Overview**: working-sample and evaluation summaries
- **Transaction Analytics**: four patterns (amount, balances, transaction type, time)
- **Fraud Detection**: stored review queue, numerical anomaly scores and SHAP explanations
- **Batch Screening**: user-supplied synthetic PaySim-compatible CSV, scored by the saved model
- **Model Evaluation**: test-set metrics, scenario probes and actual detection examples
- **Explainability**: SHAP attributions and risk-tier explanation

The public demo does **not** provide the earlier `analyst_stage1` and
`analyst_stage2` workflows. Those modes are intentionally blocked, because
masking a few fields in a public UI is not proper access control. If the research
team needs staged analyst assessment, it must use a separately secured local
workflow and keep researcher-only records private.

## Exact model and results

- Dataset: 58,213-record fraud-enriched working sample, including 8,213 labelled
  fraud cases; **58,213 stored predictions**.
- Scope: TRANSFER and CASH_OUT only; other transaction types are not scored.
- Saved Isolation Forest: **9 features**, trained on eligible training records.
- Working-sample flags: **2,906**; all correspond to stored scores and decisions.
- Held-out test: **348 TP, 232 FP, 1,296 FN, 9,782 TN**.
- Test precision **60.00%**, recall **21.17%**, F1 **31.29%**, false-positive
  rate **2.32%**. These results reflect an artificially fraud-enriched test split.
- At the original PaySim fraud prevalence, the *separate stored prevalence-adjusted
  evaluation* reports about **1.20% precision** (not 60%).
- The saved model configuration and output database are authoritative; the public
  demo never retrains or modifies them.
- The constructed wealthy/smaller-balance cases demonstrate a major model
  limitation: legitimate transactions from high-balance accounts can be
  incorrectly flagged. They are not confirmed fraud cases.

## Public upload and compute limits

- **5,000 CSV rows** maximum per upload, enforced in `src/batch_screen.py`.
- **8 MiB file size** maximum, checked before parsing in `app.py`.
- **10 on-demand exact SHAP explanations per browser session**; the other
  transactions can still be screened and downloaded without SHAP.
- Required columns: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,
  `oldbalanceDest`, `newbalanceDest`. An optional `isFraud` is **never used
  to score** and is only used to compare predictions with known labels.
- Uploaded files are not written into the study database by the app; uploaded
  data may remain in process memory for the user's active session.
- These limits protect typical demo usage, **not** a security guarantee or
  absolute server-wide quota. Free hosting can still experience resource pressure.

## Reproducibility and data handling

`output/fraudlens.db` contains 30 frozen tables. The research paper documents
experimental methodology and limitations. This *deployment* package intentionally
excludes the original 6,362,620-row PaySim dataset, preprocessing pipeline, private
analyst evidence, and full train/validation/test split archive. For complete
reproduction of the paper, use the separate research archive and obtain the
PaySim population from https://www.kaggle.com/datasets/ealaxi/paysim1.

`data/train.csv` in this package is used **only** as the frozen background for
on-demand SHAP explanations. It does not trigger training.

## Troubleshooting

- **Missing files**: verify GitHub contains `output/*.joblib`, `output/fraudlens.db`,
  `output/model_config.json` and `data/train.csv`.
- **Model load warning**: install the package versions from `requirements.txt`.
  Pickle/joblib artifacts must only be loaded from this trusted project.
- **Unavailable analyst mode**: expected for the public release; unset
  `FRAUDLENS_MODE` and restart the app.
- **Uploads rejected**: use PaySim-compatible synthetic CSV below 8 MiB and
  at most 5,000 rows. Larger research batches belong in the local research build.
