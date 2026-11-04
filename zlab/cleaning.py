"""
cleaning.py  --  turn a raw run into analysis-quality data
==========================================================

Each cleaning step is one small function, so you can read (and test)
them one at a time.  ``clean()`` chains them and returns a *cut flow*:
a table of how many events survive each step (Week 1 Lesson 2: Pandas;
"don't repeat yourself" -> write the steps once, use them for every run).
"""

import numpy as np
import pandas as pd

from .detector import BEAM_ENERGY

PT_MIN = 20.0      # GeV, trigger threshold for each muon
ETA_MAX = 2.4      # tracker acceptance |eta| < 2.4

MUON_COLUMNS = ["mu1_pt", "mu1_eta", "mu1_phi", "mu1_charge",
                "mu2_pt", "mu2_eta", "mu2_phi", "mu2_charge"]


def remove_duplicates(df):
    """The same (run, event) must appear only once."""
    return df.drop_duplicates(subset=["run", "event"])


def remove_missing(df):
    """Drop events with any missing (NaN) muon quantity."""
    return df.dropna(subset=MUON_COLUMNS)


def good_lumi(df):
    """Keep only luminosity sections certified as good ('golden JSON')."""
    return df[df["lumi_ok"] == 1]


def physical_values(df):
    """A muon pT must be positive and cannot exceed the beam energy."""
    ok = ((df["mu1_pt"] > 0) & (df["mu1_pt"] < BEAM_ENERGY) &
          (df["mu2_pt"] > 0) & (df["mu2_pt"] < BEAM_ENERGY))
    return df[ok]


def opposite_charge(df):
    """A Z boson is neutral, so it decays to mu+ mu- (charges sum to zero)."""
    return df[df["mu1_charge"] + df["mu2_charge"] == 0]


def acceptance(df, pt_min=PT_MIN, eta_max=ETA_MAX):
    """Both muons inside the tracker and above the trigger threshold."""
    ok = ((df["mu1_pt"] > pt_min) & (df["mu2_pt"] > pt_min) &
          (df["mu1_eta"].abs() < eta_max) & (df["mu2_eta"].abs() < eta_max))
    return df[ok]


CLEANING_STEPS = [
    ("remove duplicates", remove_duplicates),
    ("remove missing values", remove_missing),
    ("good luminosity sections", good_lumi),
    ("physical pT values", physical_values),
    ("opposite charge", opposite_charge),
    ("acceptance + trigger", acceptance),
]


def clean(df, verbose=True):
    """
    Apply every cleaning step in order.

    Returns
    -------
    clean_df : pandas.DataFrame   the surviving events
    cutflow  : pandas.DataFrame   events left after each step
    """
    rows = [("raw rows", len(df))]
    for name, step in CLEANING_STEPS:
        df = step(df)
        rows.append((name, len(df)))
    cutflow = pd.DataFrame(rows, columns=["step", "events"])
    cutflow["kept (%)"] = 100.0 * cutflow["events"] / cutflow["events"].iloc[0]
    cutflow["kept (%)"] = cutflow["kept (%)"].round(1)
    if verbose:
        print(cutflow.to_string(index=False))
    return df.reset_index(drop=True), cutflow
