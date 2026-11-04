"""
montecarlo.py  --  check the fit error by repeating the experiment (Week 5 Lesson 2)
====================================================================================

The fit gives an error from ONE data set (the curvature of chi2).
But what does "M_Z = 91.32 +- 0.11 GeV" really mean?  It means:

    if we could repeat the whole experiment many times, the fitted M_Z
    would scatter around the true value with a standard deviation of 0.11 GeV.

With a simulation we CAN repeat the experiment.  We record K new copies
of a run (same size, different random numbers), fit each one, and take
the standard deviation of the K fitted values:

    sigma_MC = sqrt( 1/(K-1) * sum_k (theta_k - theta_mean)^2 )

If the fit error is honest, sigma_fit and sigma_MC agree (within the
statistical precision of sigma_MC itself, about 1/sqrt(2(K-1)), i.e. 7 %
for K = 100).
"""

import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import clean
from .detector import GAMMA_Z_TRUE, M_Z_TRUE, take_data
from .fitting import fit_zpeak
from .physics import invariant_mass

TRUTH = {"M_Z": M_Z_TRUE, "Gamma_Z": GAMMA_Z_TRUE}


def repeat_experiment(run, n_toys=100, first_seed=10_000):
    """
    Record and fit the same run n_toys times with different random seeds.

    Returns a DataFrame with one row per pseudo-experiment.
    """
    tmp = tempfile.mkdtemp()
    rows = []
    for k in range(n_toys):
        raw = pd.read_csv(take_data(run=run, seed=first_seed + k, outdir=tmp, verbose=False))
        with warnings.catch_warnings():              # keep the screen clean for 100 fits
            warnings.simplefilter("ignore")
            r = fit_zpeak(invariant_mass(clean(raw, verbose=False)[0]))
        rows.append({"toy": k, "N": r.n_events,
                     "M_Z": r.values["M_Z"], "dM_Z": r.errors["M_Z"],
                     "Gamma_Z": r.values["Gamma_Z"], "dGamma_Z": r.errors["Gamma_Z"]})
    return pd.DataFrame(rows)


def compare_errors(fits, toys):
    """
    fits : {run: FitResult}  -- your own fit of each run
    toys : {run: DataFrame}  -- output of repeat_experiment for each run
    """
    rows = []
    for run in sorted(toys):
        t, f = toys[run], fits[run]
        K = len(t)
        for par in ("M_Z", "Gamma_Z"):
            s_mc = t[par].std(ddof=1)
            rows.append({"run": run, "events": f.n_events, "parameter": par,
                         "sigma_fit (your fit)": f.errors[par],
                         "sigma_MC (spread of toys)": s_mc,
                         "± on sigma_MC": s_mc / np.sqrt(2 * (K - 1)),
                         "ratio fit/MC": f.errors[par] / s_mc})
    return pd.DataFrame(rows)


def plot_mc_errors(fits, toys, par="M_Z"):
    """One panel per run: histogram of the toy results vs the Gaussian the fit predicts."""
    runs = sorted(toys)
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4))
    axes = np.atleast_1d(axes)
    name = {"M_Z": r"$M_Z$", "Gamma_Z": r"$\Gamma_Z$"}[par]
    for ax, run in zip(axes, runs):
        vals, s_fit = toys[run][par], fits[run].errors[par]
        s_mc = vals.std(ddof=1)
        lo, hi = TRUTH[par] - 4 * s_fit, TRUTH[par] + 4 * s_fit
        ax.hist(vals, bins=np.linspace(lo, hi, 21), density=True, alpha=0.6,
                label=f"{len(vals)} repeated experiments")
        x = np.linspace(lo, hi, 300)
        ax.plot(x, np.exp(-0.5 * ((x - TRUTH[par]) / s_fit) ** 2) / (np.sqrt(2 * np.pi) * s_fit),
                "r-", lw=2, label=r"Gaussian with $\sigma_\mathrm{fit}$")
        ax.axvline(TRUTH[par], color="0.3", ls="--", lw=1)
        ax.set_title(f"run {run} ({fits[run].n_events:,} events)\n"
                     rf"$\sigma_\mathrm{{fit}}$ = {s_fit:.3f},   $\sigma_\mathrm{{MC}}$ = {s_mc:.3f} GeV",
                     fontsize=10)
        ax.set_xlabel(f"fitted {name} [GeV]")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig
