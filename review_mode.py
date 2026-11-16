"""Who is allowed to see what in the dashboard.

WHY THIS FILE EXISTS
--------------------
The fraud analyst evaluation has two stages. In Stage 1 the analyst judges
transactions on their own, without seeing what FraudLens decided. If they see
the system's output first, their judgement is no longer independent and the
comparison in Chapter 4 is worthless.

Every rule about what each audience may see is written here, in one place. If
these rules were spread across app.py, a later change to one screen could
leave another screen still revealing something, and nobody would notice until
after the analyst session.

THE THREE MODES
---------------
researcher      The research group. Everything, exactly as the dashboard has
                always worked. This is the default.

analyst_stage1  The analyst's independent assessment. Only the transaction
                record itself: time step, type, amount and the four balance
                figures. No prediction, no risk level, no anomaly score, no
                explanation, no dataset label.

analyst_stage2  After the Stage 1 answers have been collected on paper. The
                analyst now sees what the system decided and why: the risk
                level, the anomaly score, the explanation and the feature
                contributions. The dataset's fraud label STAYS HIDDEN, in
                both analyst modes, because the analyst is never asked to
                check their answers against it.

THE DATASET LABEL
-----------------
The PaySim fraud label is visible in researcher mode only. It is listed in
ALWAYS_HIDDEN_FROM_ANALYST and is filtered out of every analyst view by
strip_hidden(), whatever stage is running.

HOW THE MODE IS SET
-------------------
By the FRAUDLENS_MODE environment variable, read once when the dashboard
starts:

    FRAUDLENS_MODE=analyst_stage1 streamlit run app.py

It is deliberately NOT a control inside the page. A dropdown could be clicked
by accident during a session, and the analyst could move themselves into
researcher mode without the researcher noticing. Changing the mode requires
restarting the dashboard, which cannot happen by accident.

If the variable is not set, or is set to something unrecognised, the mode is
researcher and the dashboard behaves exactly as before.
"""

import os

RESEARCHER = "researcher"
STAGE1 = "analyst_stage1"
STAGE2 = "analyst_stage2"
VALID_MODES = (RESEARCHER, STAGE1, STAGE2)

# The dataset's fraud label, under every name it goes by in the data.
# Never shown to the analyst in either stage.
ALWAYS_HIDDEN_FROM_ANALYST = {"Class", "isFraud", "is_fraud"}

# What the system decided. Hidden in Stage 1, revealed in Stage 2.
SYSTEM_OUTPUT_FIELDS = {
    "prediction", "predicted_fraud", "risk_level", "risk_rank",
    "risk_display", "anomaly_score", "explanation", "top_features",
    "top_feature_impacts",
}

# Bookkeeping columns that mean nothing to a reviewer.
INTERNAL_FIELDS = {"split", "in_scope", "sample_group"}


def current_mode():
    """The mode this dashboard is running in. Defaults to researcher."""
    m = os.environ.get("FRAUDLENS_MODE", RESEARCHER).strip().lower()
    return m if m in VALID_MODES else RESEARCHER


def is_analyst(mode=None):
    return (mode or current_mode()) in (STAGE1, STAGE2)


def hidden_fields(mode=None):
    """The set of column names this mode must not display."""
    mode = mode or current_mode()
    if mode == RESEARCHER:
        return set()
    hidden = set(ALWAYS_HIDDEN_FROM_ANALYST) | set(INTERNAL_FIELDS)
    if mode == STAGE1:
        hidden |= set(SYSTEM_OUTPUT_FIELDS)
    return hidden


def may_show(field, mode=None):
    return field not in hidden_fields(mode)


def strip_hidden(df, mode=None):
    """Return df without any column this mode must not display.

    Used as the last step before anything is rendered, so that a column added
    to the database in future is hidden by default rather than leaking.
    """
    drop = [c for c in df.columns if c in hidden_fields(mode)]
    return df.drop(columns=drop) if drop else df


def may_show_section(section, mode=None):
    """Whether a whole area of the dashboard may be displayed.

    Some screens cannot be made safe by hiding columns, because the aggregate
    itself gives the answer away. A chart of 'how many flagged transactions
    were fraud' tells the analyst what the system found even though no single
    transaction is named. These are hidden from the analyst entirely.
    """
    mode = mode or current_mode()
    if mode == RESEARCHER:
        return True
    ANALYST_MAY_SEE = {
        "transaction_list",     # the 30 sample transactions
        "transaction_detail",   # one transaction, filtered by strip_hidden
    }
    return section in ANALYST_MAY_SEE


def banner_text(mode=None):
    """A line shown at the top of the page so nobody is unsure which mode is
    running. An analyst session where the researcher believes Stage 1 is
    active but it is not would be unrecoverable."""
    mode = mode or current_mode()
    if mode == STAGE1:
        return ("STAGE 1 — INDEPENDENT ASSESSMENT.  The system's decisions are "
                "hidden. Please record your own judgement on the paper form.")
    if mode == STAGE2:
        return ("STAGE 2 — SYSTEM REVIEW.  You can now see what FraudLens "
                "decided and why. Your Stage 1 answers have already been "
                "collected and cannot be changed.")
    return None


def sample_ids(path="output/analyst_evaluation/sample_master_RESEARCHERS_ONLY.csv"):
    """The transaction ids in the analyst sample, or None if there is no
    sample file. In an analyst mode the dashboard shows only these, so the
    analyst cannot browse the other 2,881 flagged transactions."""
    import os as _os
    if not _os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    col = "transaction_id" if "transaction_id" in df.columns else df.columns[0]
    return set(df[col].tolist())
