"""Sanity tests for the fraud lab:   python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from fraudlab import (BOT_TARGET_TRUE, FRAUD_RATE_TRUE, clean, fit_checklist, fit_spike,
                      repeat_experiment, take_data, to_myr, weighted_average)


def test_conversion_is_amount_times_rate():
    df = pd.DataFrame({"amount": [1.00, 10.0], "fx_rate": [4.40, 1.0]})
    assert to_myr(df) == pytest.approx([4.40, 10.0])


def test_cleaning_removes_all_problems(tmp_path):
    raw = pd.read_csv(take_data(1, outdir=tmp_path, verbose=False))
    good, cutflow = clean(raw, verbose=False)
    assert not good.duplicated(subset=["txn_id"]).any()
    assert not good[["amount", "currency", "fx_rate"]].isna().any().any()
    assert (good["is_test"] == 0).all()
    assert ((good["amount"] > 0) & (good["amount"] < 50_000)).all()
    assert (good["type"] == "purchase").all() and (good["status"] == "approved").all()
    assert cutflow["payments"].is_monotonic_decreasing


def test_fit_recovers_truth():
    """On the largest batch the fit must agree with the simulation truth within 3 sigma."""
    a = to_myr(clean(take_data(3, save=False), verbose=False)[0])
    r = fit_spike(a)
    assert r.valid
    assert abs(r.values["mu"] - BOT_TARGET_TRUE) < 3 * r.errors["mu"]
    assert abs(r.values["fraud_rate"] - FRAUD_RATE_TRUE) < 3 * r.errors["fraud_rate"]


def test_checklist_passes_good_fit_and_catches_wrong_model():
    a = to_myr(clean(take_data(3, save=False), verbose=False)[0])
    good = fit_checklist(fit_spike(a), a, verbose=False)
    assert (good["verdict"] == "✓").all()
    wrong = fit_checklist(fit_spike(a, flat_background=True), a, verbose=False)
    assert (wrong["verdict"] == "⚠").any()


def test_monte_carlo_spread_matches_fit_error():
    toys = repeat_experiment(2, n_toys=40)
    ratio = toys["d_fraud_rate"].mean() / toys["fraud_rate"].std(ddof=1)
    assert 0.7 < ratio < 1.4
