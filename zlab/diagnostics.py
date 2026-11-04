"""
diagnostics.py  --  should I trust this fit?  (Week 4 Lesson 1, Steps 1-4)
=========================================================================

A fit always returns numbers, even when the fit is bad.  Before you quote
M_Z +- sigma, go through five simple checks.  ``fit_checklist(result)``
does them all and prints one line per check:

    1. Did MINUIT find a minimum?            (convergence)
    2. Does the model describe the data?     (chi2/ndf and p-value)
    3. Do the bin pulls look like N(0, 1)?   (about 95 % within +-2)
    4. Are the errors symmetric?             (Hesse vs MINOS)
    5. Is the result stable?                 (refit with 2 GeV bins)

The thresholds were tuned on toy experiments: a correct fit passes every
check about 95 % of the time or more, so one warning is not a disaster.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import PARAMETERS, expected_counts, fit_zpeak


def bin_pulls(result):
    """pull_i = (n_i - mu_i) / sqrt(mu_i) for every bin of the fit."""
    mu = expected_counts(result.edges, *[result.values[p] for p in PARAMETERS],
                         sigma=result.sigma_res)
    return (result.counts - mu) / np.sqrt(mu)


def minos_errors(result, par):
    """Asymmetric errors from the Delta chi2 = 1 crossings (MINOS)."""
    m = result.minuit
    m.minos(par)
    e = m.merrors[par]
    return e.lower, e.upper


def fit_checklist(result, masses, verbose=True):
    """
    Run the five checks on one fit.

    Parameters
    ----------
    result : FitResult from fit_zpeak
    masses : the same masses that went into the fit (needed for check 5)

    Returns
    -------
    pandas.DataFrame with one row per check.
    """
    rows = []

    def add(check, value, ok, meaning_ok, meaning_bad):
        rows.append({"check": check, "value": value, "verdict": "✓" if ok else "⚠",
                     "what it means": meaning_ok if ok else meaning_bad})

    # 1) convergence -------------------------------------------------------
    m = result.minuit
    add("1. converged?", f"valid = {m.valid}, EDM = {m.fmin.edm:.1e}",
        m.valid and m.fmin.edm < 1e-3,
        "MINUIT found a true minimum.",
        "MINUIT did not converge: do not trust any number below.")

    # 2) goodness of fit ---------------------------------------------------
    add("2. good fit?", f"chi2/ndf = {result.chi2:.1f}/{result.ndf}, p = {result.pvalue:.2f}",
        result.pvalue > 0.05,
        "The model describes the data (p > 0.05).",
        "p < 0.05: the model may be missing something (or it is 1-in-20 bad luck).")

    # 3) pulls -------------------------------------------------------------
    pulls = bin_pulls(result)
    frac2 = np.mean(np.abs(pulls) < 2)
    add("3. pulls Gaussian?", f"{100 * frac2:.0f}% within ±2, spread = {np.std(pulls, ddof=1):.2f}",
        frac2 >= 0.85,
        "Pulls behave like N(0,1): no region of the spectrum is badly described.",
        "Too many large pulls: look at the pull panel to see where the model fails.")

    # 4) symmetric errors? -------------------------------------------------
    texts, ok = [], True
    for par in ("M_Z", "Gamma_Z"):
        lo, hi = minos_errors(result, par)
        texts.append(f"{par}: -{abs(lo):.3f}/+{hi:.3f} (Hesse ±{result.errors[par]:.3f})")
        ok &= abs(abs(hi) - abs(lo)) / result.errors[par] < 0.2
    add("4. errors symmetric?", ";  ".join(texts), ok,
        "Hesse ± and MINOS agree within 20%: chi2 is close to a parabola, quoting ± is fine.",
        "chi2 is not a parabola: quote the asymmetric MINOS errors -lo/+hi.")

    # 5) stability against an arbitrary choice ------------------------------
    alt = fit_zpeak(masses, label="2 GeV bins", fit_range=(result.edges[0], result.edges[-1]),
                    n_bins=len(result.counts) // 2, sigma_res=result.sigma_res)
    shifts = {p: (alt.values[p] - result.values[p]) / result.errors[p] for p in ("M_Z", "Gamma_Z")}
    add("5. stable?", "shift with 2 GeV bins: " +
        ",  ".join(f"{p} {s:+.2f}σ" for p, s in shifts.items()),
        all(abs(s) < 1.0 for s in shifts.values()),
        "Changing the bin width moves the result by much less than 1σ.",
        "The result depends on the binning: investigate before quoting it.")

    table = pd.DataFrame(rows)
    if verbose:
        n_ok = (table["verdict"] == "✓").sum()
        print(f"Fit checklist for '{result.label}':  {n_ok}/5 checks passed")
        for _, r in table.iterrows():
            print(f"  {r['verdict']} {r['check']:22s} {r['value']}")
            print(f"    {'':22s} -> {r['what it means']}")
    return table


def plot_pull_histogram(result):
    """Histogram of the bin pulls with the expected N(0,1) curve."""
    pulls = bin_pulls(result)
    x = np.linspace(-4, 4, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pulls, bins=np.linspace(-4, 4, 17), density=True, alpha=0.6,
            label=f"bin pulls of '{result.label}' ({len(pulls)} bins)")
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r-", lw=2, label=r"expected: $\mathcal{N}(0,1)$")
    ax.axvspan(-2, 2, color="0.5", alpha=0.1, label="±2: should hold about 95%")
    ax.set_xlabel(r"pull $=(n_i-\mu_i)/\sqrt{\mu_i}$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig
