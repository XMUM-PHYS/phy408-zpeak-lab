"""Sanity tests:   python -m pytest -q   (Week 1 Lesson 2: test your code with known values)"""
import numpy as np
import pandas as pd
import pytest

from zlab import fit_checklist, repeat_experiment, M_Z_TRUE, GAMMA_Z_TRUE, clean, fit_zpeak, invariant_mass, take_data, weighted_average
from zlab.detector import M_MU


def test_invariant_mass_of_z_at_rest():
    """Back-to-back muons from a Z at rest must give back M_Z (Week 2 Lesson 2, slide 16)."""
    p = np.sqrt(M_Z_TRUE**2 / 4 - M_MU**2)
    # pT = p, eta = 0, opposite phi
    df = pd.DataFrame({"mu1_pt": [p], "mu1_eta": [0.0], "mu1_phi": [0.0],
                       "mu2_pt": [p], "mu2_eta": [0.0], "mu2_phi": [np.pi]})
    assert invariant_mass(df)[0] == pytest.approx(M_Z_TRUE, abs=1e-9)


def test_weighted_average_textbook():
    """Two equal errors: plain mean, error / sqrt(2)."""
    c = weighted_average([1.0, 3.0], [1.0, 1.0])
    assert c.value == pytest.approx(2.0)
    assert c.error_raw == pytest.approx(1 / np.sqrt(2))
    assert c.chi2 == pytest.approx(2.0)


def test_cleaning_removes_all_problems(tmp_path):
    raw = pd.read_csv(take_data(1, outdir=tmp_path, verbose=False))
    good, cutflow = clean(raw, verbose=False)
    assert not good.duplicated(subset=["run", "event"]).any()
    assert not good.isna().any().any()
    assert (good["lumi_ok"] == 1).all()
    assert (good["mu1_charge"] + good["mu2_charge"] == 0).all()
    assert (good[["mu1_pt", "mu2_pt"]] > 20).all().all()
    assert cutflow["events"].is_monotonic_decreasing


def test_fit_recovers_truth(tmp_path):
    """On the largest run the fit must agree with the simulation truth within 3 sigma."""
    raw = pd.read_csv(take_data(3, outdir=tmp_path, verbose=False))
    r = fit_zpeak(invariant_mass(clean(raw, verbose=False)[0]))
    assert r.valid
    assert abs(r.values["M_Z"] - M_Z_TRUE) < 3 * r.errors["M_Z"]
    assert abs(r.values["Gamma_Z"] - GAMMA_Z_TRUE) < 3 * r.errors["Gamma_Z"]


def test_checklist_passes_good_fit_and_catches_wrong_model(tmp_path):
    raw = pd.read_csv(take_data(3, outdir=tmp_path, verbose=False))
    m = invariant_mass(clean(raw, verbose=False)[0])
    good = fit_checklist(fit_zpeak(m), m, verbose=False)
    assert (good["verdict"] == "✓").all()
    wrong = fit_checklist(fit_zpeak(m, sigma_res=2.0), m, verbose=False)
    assert (wrong["verdict"] == "⚠").any()


def test_monte_carlo_spread_matches_fit_error():
    toys = repeat_experiment(2, n_toys=40)
    ratio = toys["dM_Z"].mean() / toys["M_Z"].std(ddof=1)
    assert 0.7 < ratio < 1.4
