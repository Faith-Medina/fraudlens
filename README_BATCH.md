# Public Batch Screening

Batch Screening applies the original saved Isolation Forest model, scaler,
feature order and score thresholds to PaySim-compatible synthetic CSV files.
It never fits, refits, or changes a model. An `isFraud` label is optional and
is not used to score uploaded rows.

Public demo limits: 5,000 rows, 8 MiB per CSV, and 10 exact on-demand SHAP
requests per session. These are hosting-resource guardrails and do not change
the published research performance metrics.

For full research evaluation and tests, use the separate original research
archive containing data/test.csv and the experiment scripts.
