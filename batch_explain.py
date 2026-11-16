"""
On-demand SHAP explanations for batch-screened transactions.

WHY ON DEMAND
-------------
The explainer is the exact model-agnostic one, which evaluates all 2^9 = 512
feature coalitions per transaction. Measured on this machine that is about
94 ms per transaction, so explaining a few hundred flagged rows up front
would add tens of seconds to every upload for explanations most users will
never open. Screening is therefore immediate and an explanation is computed
when a reviewer asks for one.

The reference group is the same one the research pipeline uses: a sample of
ordinary in-scope transactions from the TRAINING split, scaled with the saved
scaler. Using the same reference means a batch explanation answers the same
question as an explanation in the rest of the system -- what makes this
transaction unusual compared with normal activity -- rather than a different
question that happens to look similar.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_NAMES = {
    "Amount": "Transaction amount",
    "step": "Hours since the start of the period",
    "hour": "Hour of the simulated day (0-23)",
    "oldbalanceOrg": "Sender's balance before",
    "newbalanceOrig": "Sender's balance after",
    "oldbalanceDest": "Recipient's balance before",
    "newbalanceDest": "Recipient's balance after",
    "type_CASH_OUT": "Transaction type is cash-out",
    "type_TRANSFER": "Transaction type is transfer",
}
MONEY = {"Amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
         "newbalanceDest"}


def format_value(name, raw):
    if name.startswith("type_"):
        return None
    if name in MONEY:
        return f"{raw:,.2f}"
    if name == "step":
        return f"{raw:,.0f}"
    return f"{raw:,.4f}"


@functools.lru_cache(maxsize=1)
def _explainer(_key: str):
    """Build the explainer once per process. Cached on the artifact paths."""
    import shap
    import joblib
    from batch_screen import load_artifacts, SCOPE_TYPES

    art = load_artifacts()
    train_path = os.path.join(HERE, "data", "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "data/train.csv is required to build the explanation reference "
            "group. Without it an explanation would describe a different "
            "baseline from the rest of the system.")

    bg = pd.read_csv(train_path)
    bg = bg[bg["type"].astype(str).str.upper().isin(SCOPE_TYPES)]
    bg = bg.drop(columns=[c for c in ("nameOrig", "nameDest",
                                      "isFlaggedFraud") if c in bg.columns])
    bg = bg.rename(columns={"amount": "Amount", "isFraud": "Class"})
    bg = pd.concat([bg, pd.get_dummies(bg["type"], prefix="type")], axis=1)
    bg["hour"] = bg["step"] % 24
    for c in art.feature_cols:
        if c not in bg.columns:
            bg[c] = 0
    sample = bg[art.feature_cols].sample(min(100, len(bg)), random_state=42)
    background = pd.DataFrame(art.scaler.transform(sample.astype(float)),
                              columns=art.feature_cols)
    return shap.Explainer(art.model.decision_function, background), art


def explain_row(row: pd.Series, art=None) -> dict:
    """Explain one screened transaction. Returns the parts, not prose."""
    explainer, loaded = _explainer("default")
    art = art or loaded
    raw = {c: float(row[c]) if c in row and pd.notna(row[c]) else 0.0
           for c in art.feature_cols}
    raw["hour"] = float(row.get("hour", row.get("step", 0) % 24))
    for t in ("type_CASH_OUT", "type_TRANSFER"):
        raw[t] = 1.0 if str(row.get("type", "")).upper() == t[5:] else 0.0

    X = pd.DataFrame([[raw[c] for c in art.feature_cols]],
                     columns=art.feature_cols)
    Xs = pd.DataFrame(art.scaler.transform(X.astype(float)),
                      columns=art.feature_cols)
    sv = explainer(Xs)
    values = sv.values[0]
    base = float(sv.base_values[0])

    pairs = sorted(zip(art.feature_cols, values),
                   key=lambda p: abs(p[1]), reverse=True)
    total_abs = float(np.sum(np.abs(values))) or 1.0
    top3 = pairs[:3]
    share = sum(abs(v) for _, v in top3) / total_abs * 100

    factors = []
    for name, val in top3:
        shown = format_value(name, raw[name])
        label = FEATURE_NAMES.get(name, name)
        # A one-hot column contributes whether it is 1 or 0, so the label has
        # to say which. "Transaction type is cash-out" on a TRANSFER row
        # states the opposite of the truth.
        if name.startswith("type_"):
            kind = name[5:].replace("_", "-").lower()
            label = (f"Transaction type is {kind}" if raw[name] == 1.0
                     else f"Transaction type is not {kind}")
        factors.append({
            "feature": name,
            "label": label,
            "value": shown,
            "contribution": float(val),
            "direction": ("pushed the score down, making the transaction "
                          "more unusual") if val < 0 else
                         ("pushed the score up, making the transaction look "
                          "more ordinary"),
        })
    return {
        "anomaly_score": float(row.get("anomaly_score", np.nan)),
        "threshold": art.flag_threshold,
        "risk_level": row.get("risk_level", ""),
        "base_value": base,
        "shap_sum": float(np.sum(values)),
        "top3_share_pct": share,
        "factors": factors,
    }


DISCLAIMER = (
    "A contribution measures how much a feature moved the anomaly score. It "
    "is not a probability that the transaction is fraudulent."
)
