"""
cleaning.py  --  turn a raw extract into analysis-quality payments
==================================================================

Same idea as zlab/cleaning.py: each step is one small function, and
``clean()`` chains them and returns the *cut flow*.
"""

import pandas as pd

from .bank import CARD_LIMIT


def remove_duplicates(df):
    """The same payment (txn_id) must appear only once."""
    return df.drop_duplicates(subset=["txn_id"])


def remove_missing(df):
    """Drop payments with a missing amount, currency or exchange rate."""
    return df.dropna(subset=["amount", "currency", "fx_rate"])


def live_payments(df):
    """Remove the shop's own test-mode payments."""
    return df[df["is_test"] == 0]


def valid_amounts(df):
    """An amount must be positive and below the card limit (no 0.00 or 99999999)."""
    return df[(df["amount"] > 0) & (df["amount"] < CARD_LIMIT)]


def purchases_only(df):
    """Refunds are not new payments: keep purchases only."""
    return df[df["type"] == "purchase"]


def approved_only(df):
    """We measure the fraud that went through: keep approved payments."""
    return df[df["status"] == "approved"]


CLEANING_STEPS = [
    ("remove duplicates", remove_duplicates),
    ("remove missing values", remove_missing),
    ("live payments (no test mode)", live_payments),
    ("valid amounts", valid_amounts),
    ("purchases only (no refunds)", purchases_only),
    ("approved only", approved_only),
]


def clean(df, verbose=True):
    """
    Apply every cleaning step in order.

    Returns
    -------
    clean_df : pandas.DataFrame   the surviving payments
    cutflow  : pandas.DataFrame   payments left after each step
    """
    rows = [("raw rows", len(df))]
    for name, step in CLEANING_STEPS:
        df = step(df)
        rows.append((name, len(df)))
    cutflow = pd.DataFrame(rows, columns=["step", "payments"])
    cutflow["kept (%)"] = (100.0 * cutflow["payments"] / cutflow["payments"].iloc[0]).round(1)
    if verbose:
        print(cutflow.to_string(index=False))
    return df.reset_index(drop=True), cutflow
