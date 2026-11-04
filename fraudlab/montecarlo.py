"""
montecarlo.py  --  check the fit error by repeating the experiment
==================================================================

Same idea as zlab/montecarlo.py: replay a batch K times with new random
numbers, fit every copy, and compare

    sigma_MC = sqrt( 1/(K-1) * sum_k (theta_k - theta_mean)^2 )

with the error your fit predicted.  sigma_MC is itself known to about
1/sqrt(2(K-1)), i.e. 7 % for K = 100.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bank import BOT_SPREAD_TRUE, BOT_TARGET_TRUE, FRAUD_RATE_TRUE, take_data
from .cleaning import clean
from .conversion import to_myr
from .fitting import fit_spike

TRUTH = {"mu": BOT_TARGET_TRUE, "w": BOT_SPREAD_TRUE, "fraud_rate": FRAUD_RATE_TRUE}
NAMES = {"mu": r"bot target $\mu$ [RM]", "w": r"bot spread $w$ [RM]",
         "fraud_rate": r"fraud rate $f$ [%]"}


def repeat_experiment(batch, n_toys=100, first_seed=10_000):
    """Receive and fit the same batch n_toys times with different random seeds."""
    rows = []
    for k in range(n_toys):
        raw = take_data(batch=batch, seed=first_seed + k, save=False)
        with warnings.catch_warnings():              # keep the screen clean for 100 fits
            warnings.simplefilter("ignore")
            r = fit_spike(to_myr(clean(raw, verbose=False)[0]))
        rows.append({"toy": k, "N": r.n_events,
                     "mu": r.values["mu"], "d_mu": r.errors["mu"],
                     "fraud_rate": r.values["fraud_rate"], "d_fraud_rate": r.errors["fraud_rate"]})
    return pd.DataFrame(rows)


def compare_errors(fits, toys):
    """fits: {batch: FitResult};  toys: {batch: DataFrame from repeat_experiment}."""
    rows = []
    for b in sorted(toys):
        t, f = toys[b], fits[b]
        K = len(t)
        for par in ("mu", "fraud_rate"):
            s_mc = t[par].std(ddof=1)
            rows.append({"batch": b, "in window": f.n_events, "parameter": par,
                         "sigma_fit (your fit)": f.errors[par],
                         "sigma_MC (spread of toys)": s_mc,
                         "± on sigma_MC": s_mc / np.sqrt(2 * (K - 1)),
                         "ratio fit/MC": f.errors[par] / s_mc})
    return pd.DataFrame(rows)


def plot_mc_errors(fits, toys, par="mu"):
    """One panel per batch: histogram of the toy results vs the Gaussian the fit predicts."""
    batches = sorted(toys)
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, batches):
        vals, s_fit = toys[b][par], fits[b].errors[par]
        s_mc = vals.std(ddof=1)
        lo, hi = TRUTH[par] - 4 * s_fit, TRUTH[par] + 4 * s_fit
        ax.hist(vals, bins=np.linspace(lo, hi, 21), density=True, alpha=0.6,
                label=f"{len(vals)} repeated batches")
        x = np.linspace(lo, hi, 300)
        ax.plot(x, np.exp(-0.5 * ((x - TRUTH[par]) / s_fit) ** 2) / (np.sqrt(2 * np.pi) * s_fit),
                "r-", lw=2, label=r"Gaussian with $\sigma_\mathrm{fit}$")
        ax.axvline(TRUTH[par], color="0.3", ls="--", lw=1)
        ax.set_title(f"batch {b} ({fits[b].n_events:,} payments in window)\n"
                     rf"$\sigma_\mathrm{{fit}}$ = {s_fit:.3f},   $\sigma_\mathrm{{MC}}$ = {s_mc:.3f}",
                     fontsize=10)
        ax.set_xlabel(f"fitted {NAMES[par]}")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig
