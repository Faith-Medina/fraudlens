"""
CSV batch screening for FraudLens.

WHAT THIS FILE IS FOR
---------------------
It lets a user upload a PaySim-compatible CSV and screen those transactions
with the model that was already trained. Nothing here fits, refits or tunes
anything. It loads the saved artifacts and applies them:

    output/model.joblib          the fitted Isolation Forest (nine features)
    output/scaler.joblib         the StandardScaler fitted on in-scope
                                 training rows only
    output/feature_cols.joblib   the nine feature names, in order
    output/model_config.json     the flagging cut-off and the risk boundary

The preprocessing below is the same chain the training pipeline used. It is
written out here rather than imported because the training scripts read from
fixed paths and run end to end; the rules, not the scripts, are what must
match. The regression test in tests/test_batch_screen.py re-screens the
stored test split through this module and asserts the confusion matrix is
unchanged, which is what keeps the two honest.

WHAT A FLAG MEANS
-----------------
A flag means the transaction is unusual relative to the patterns the model
learned. It is not a finding of fraud. Nothing in this module decides that a
transaction is fraudulent, and the labels, if a file carries them, are used
only to report how the screening would have scored against them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(HERE, "output")

# PaySim's own column names, and what this system calls them internally.
CANONICAL = {
    "step": "step",
    "amount": "Amount",
    "oldbalanceorg": "oldbalanceOrg",
    "newbalanceorig": "newbalanceOrig",
    "oldbalancedest": "oldbalanceDest",
    "newbalancedest": "newbalanceDest",
    "type": "type",
    "isfraud": "Class",
}

# Columns a file must carry to be screenable at all.
REQUIRED = ["step", "amount", "type", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]

NUMERIC = ["step", "Amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest"]

# Types the model was fitted on. Anything else is not evaluated.
SCOPE_TYPES = ("TRANSFER", "CASH_OUT")
KNOWN_TYPES = ("TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT")

# Public-demo ceiling to protect shared hosting. The research pipeline is unchanged.
# End-to-end testing covered files up to about 4,000 rows.
MAX_ROWS = 5_000


class BatchError(Exception):
    """A problem with the uploaded file that the user needs to fix."""


@dataclass
class Artifacts:
    model: object
    scaler: object
    feature_cols: list
    flag_threshold: float
    risk_boundary: float
    contamination: float


def load_artifacts(output_dir: str = OUTPUT) -> Artifacts:
    """Load the trained model and its configuration. Nothing is fitted."""
    need = ["model.joblib", "scaler.joblib", "feature_cols.joblib",
            "model_config.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(output_dir, n))]
    if missing:
        raise BatchError(
            "The trained model is not available: missing "
            + ", ".join(missing)
            + ". Run the training pipeline before screening a file.")
    cfg = json.load(open(os.path.join(output_dir, "model_config.json")))
    return Artifacts(
        model=joblib.load(os.path.join(output_dir, "model.joblib")),
        scaler=joblib.load(os.path.join(output_dir, "scaler.joblib")),
        feature_cols=list(joblib.load(os.path.join(output_dir,
                                                   "feature_cols.joblib"))),
        flag_threshold=float(cfg["flag_threshold"]),
        risk_boundary=float(cfg["risk_high_priority_below"]),
        contamination=float(cfg["contamination"]),
    )


# ---------------------------------------------------------------- validation
@dataclass
class Validation:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)
    n_rows: int = 0
    has_labels: bool = False


def _canonical_name(col: str) -> str | None:
    return CANONICAL.get(str(col).strip().lower().replace(" ", ""))


def validate(df: pd.DataFrame) -> Validation:
    """Check an uploaded frame before anything is screened.

    Column mapping is by name only, and only where the meaning is the same
    (PaySim's own spellings, case and spacing ignored). No column is renamed
    on a guess about what it might mean.
    """
    v = Validation(ok=True, n_rows=len(df))

    if len(df) == 0:
        v.ok = False
        v.errors.append("The file contains no rows.")
        return v
    if len(df) > MAX_ROWS:
        v.ok = False
        v.errors.append(
            f"The file has {len(df):,} rows. The limit is {MAX_ROWS:,}. "
            "Split it into smaller files and screen them separately.")
        return v

    for col in df.columns:
        canon = _canonical_name(col)
        if canon:
            v.mapping[col] = canon

    present = set(v.mapping.values())
    missing = [c for c in REQUIRED
               if c not in present
               and CANONICAL.get(c.lower()) not in present]
    # REQUIRED mixes raw and canonical spellings; normalise before reporting
    missing = []
    for raw in ["step", "amount", "type", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]:
        canon = CANONICAL[raw.lower()]
        if canon not in present:
            missing.append(raw)
    if missing:
        v.ok = False
        v.errors.append(
            "These required columns are missing: " + ", ".join(missing)
            + ". A screenable file needs PaySim's columns: step, type, "
              "amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, "
              "newbalanceDest.")
        return v

    work = df.rename(columns=v.mapping)
    v.has_labels = "Class" in work.columns

    bad_numeric = []
    for c in NUMERIC:
        coerced = pd.to_numeric(work[c], errors="coerce")
        n_bad = int(coerced.isna().sum() - work[c].isna().sum())
        if n_bad > 0:
            bad_numeric.append(f"{c} ({n_bad:,} values)")
    if bad_numeric:
        v.ok = False
        v.errors.append(
            "These columns contain values that are not numbers: "
            + "; ".join(bad_numeric) + ".")

    n_missing = int(work[NUMERIC].isna().sum().sum())
    if n_missing:
        v.warnings.append(
            f"{n_missing:,} missing numeric values were found. Rows with a "
            "missing value in any model input are reported as Not evaluated "
            "rather than guessed at.")

    types_present = set(work["type"].dropna().astype(str).str.upper().unique())
    unknown = sorted(types_present - set(KNOWN_TYPES))
    if unknown:
        v.warnings.append(
            "These transaction types are not part of PaySim and are not "
            "scored: " + ", ".join(unknown) + ".")
    if not (types_present & set(SCOPE_TYPES)):
        v.ok = False
        v.errors.append(
            "No TRANSFER or CASH_OUT transactions are present. The model "
            "scores only those two types, so there is nothing to screen in "
            "this file.")

    if v.has_labels:
        v.warnings.append(
            "An isFraud column was found. It is not given to the model. It "
            "is used only to report how the screening compares with those "
            "labels.")
    return v


# ---------------------------------------------------------------- screening
def _prepare(df: pd.DataFrame, mapping: dict, feature_cols: list):
    """Apply the training preprocessing rules to an uploaded frame."""
    work = df.rename(columns=mapping).copy()
    work = work.drop(columns=[c for c in ("nameOrig", "nameDest",
                                          "isFlaggedFraud")
                              if c in work.columns])
    for c in NUMERIC:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work["type"] = work["type"].astype(str).str.upper().str.strip()

    # hour is derived exactly as in training: the hour of day implied by step
    work["hour"] = work["step"] % 24

    # one-hot the single categorical column, then supply any level this file
    # does not contain so the matrix always has the nine training columns
    dummies = pd.get_dummies(work["type"], prefix="type")
    for col in feature_cols:
        if col.startswith("type_") and col not in dummies.columns:
            dummies[col] = 0
    work = pd.concat([work, dummies], axis=1)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0
    return work


def screen(df: pd.DataFrame, art: Artifacts, mapping: dict | None = None
           ) -> pd.DataFrame:
    """Screen an uploaded frame. Returns one row per uploaded transaction.

    Every uploaded row comes back, in its original order, with a status of
    Flagged, Not flagged, or Not evaluated. Nothing is dropped silently.
    """
    if mapping is None:
        mapping = {c: _canonical_name(c) for c in df.columns
                   if _canonical_name(c)}
    work = _prepare(df, mapping, art.feature_cols)

    in_scope = work["type"].isin(SCOPE_TYPES)
    complete = work[art.feature_cols].notna().all(axis=1)
    evaluable = in_scope & complete

    out = pd.DataFrame(index=work.index)
    out["row"] = np.arange(1, len(work) + 1)
    out["type"] = work["type"].values
    out["Amount"] = work["Amount"].values
    out["step"] = work["step"].values
    out["hour"] = work["hour"].values
    for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
              "newbalanceDest"):
        out[c] = work[c].values
    out["anomaly_score"] = np.nan
    out["status"] = "Not evaluated"
    out["risk_level"] = ""
    out["reason_not_evaluated"] = ""

    out.loc[~in_scope, "reason_not_evaluated"] = (
        "Transaction type is outside the detection scope")
    out.loc[in_scope & ~complete, "reason_not_evaluated"] = (
        "A required value is missing")

    if evaluable.any():
        X = art.scaler.transform(
            work.loc[evaluable, art.feature_cols].astype(float).values)
        scores = art.model.score_samples(X)
        out.loc[evaluable, "anomaly_score"] = scores
        flagged = scores < art.flag_threshold
        idx = out.index[evaluable]
        out.loc[idx, "status"] = np.where(flagged, "Flagged", "Not flagged")
        out.loc[idx, "risk_level"] = np.where(
            flagged,
            np.where(scores < art.risk_boundary, "High Priority",
                     "Other Flagged"),
            "")

    if "Class" in work.columns:
        out["known_label"] = pd.to_numeric(work["Class"],
                                           errors="coerce").values
    return out.reset_index(drop=True)


def summarise(res: pd.DataFrame, art: Artifacts) -> dict:
    """Headline counts for an upload, with both rates the user needs."""
    total = len(res)
    evaluated = int((res.status != "Not evaluated").sum())
    flagged = int((res.status == "Flagged").sum())
    hp = int((res.risk_level == "High Priority").sum())
    s = {
        "total": total,
        "evaluated": evaluated,
        "not_evaluated": total - evaluated,
        "eligible_pct": (evaluated / total * 100) if total else 0.0,
        "flagged": flagged,
        "flagged_pct_of_total": (flagged / total * 100) if total else 0.0,
        "flagged_pct_of_evaluated": (flagged / evaluated * 100)
                                    if evaluated else 0.0,
        "high_priority": hp,
        "other_flagged": flagged - hp,
        "contamination": art.contamination,
    }
    if "known_label" in res.columns and res.known_label.notna().any():
        ev = res[res.status != "Not evaluated"]
        lab = pd.to_numeric(ev.known_label, errors="coerce")
        tp = int(((ev.status == "Flagged") & (lab == 1)).sum())
        fp = int(((ev.status == "Flagged") & (lab == 0)).sum())
        fn = int(((ev.status != "Flagged") & (lab == 1)).sum())
        s["labels"] = {
            "fraud_in_file": int((pd.to_numeric(res.known_label,
                                                errors="coerce") == 1).sum()),
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn,
            "precision": (tp / (tp + fp) * 100) if (tp + fp) else 0.0,
            "recall": (tp / (tp + fn) * 100) if (tp + fn) else 0.0,
        }
    return s


FLAG_RATE_NOTE = (
    "The contamination setting of 0.05 was chosen on this study's own data, "
    "where it flags about five per cent of all transactions. It is a fixed "
    "score cut-off, not a quota: the share flagged in any uploaded file "
    "depends on how many of its transactions are eligible for screening and "
    "on how unusual they are. A file need not come out at five per cent, and "
    "a different figure is not a fault."
)
