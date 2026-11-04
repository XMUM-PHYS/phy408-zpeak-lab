"""
plotting.py  --  every figure of the fraud lab (same figures as the Z lab)
=========================================================================
"""

import matplotlib.pyplot as plt
import numpy as np

from zlab.fitting import _simpson_bins
from zlab.plotting import COLOURS, _forest          # same vertical plot as the Z lab

from .bank import BOT_TARGET_TRUE, FRAUD_RATE_TRUE
from .fitting import (FIT_RANGE, N_BINS, PARAMETERS, UNITS, background_shape,
                      chi2_profile, expected_counts, signal_shape)


# ----------------------------------------------------------------------
# 1) show the data (and the fit, once we have one)
# ----------------------------------------------------------------------
def plot_amounts(amounts, result=None, title="", fit_range=FIT_RANGE, n_bins=N_BINS):
    """Payment-amount histogram with sqrt(N) error bars; add the fit and pulls if given."""
    edges = np.linspace(*fit_range, n_bins + 1)
    counts, _ = np.histogram(amounts, bins=edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]

    if result is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax_pull = None
    else:
        fig, (ax, ax_pull) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="o", ms=3.5,
                color=COLOURS["data"], label=f"payments ({counts.sum():,} in window)")

    if result is not None:
        v = result.values
        fine = np.linspace(*fit_range, 400)
        s_norm = _simpson_bins(lambda x: signal_shape(x, v["mu"], v["w"], result.sigma_res), edges).sum()
        b_norm = _simpson_bins(lambda x: background_shape(x, v["slope"]), edges).sum()
        sig = v["N_fraud"] * width * signal_shape(fine, v["mu"], v["w"], result.sigma_res) / s_norm
        bkg = v["N_legit"] * width * background_shape(fine, v["slope"]) / b_norm
        ax.plot(fine, sig + bkg, color=COLOURS["fit"], lw=2, label="fit: fraud + normal")
        ax.plot(fine, sig, color=COLOURS["sig"], ls="--", label="card-testing spike")
        ax.fill_between(fine, bkg, color=COLOURS["bkg"], alpha=0.3, label="normal customers")

        mu = expected_counts(edges, *[v[p] for p in PARAMETERS], sigma=result.sigma_res)
        pulls = (counts - mu) / np.sqrt(mu)
        ax_pull.bar(centres, pulls, width=width, color="0.5")
        ax_pull.axhline(0, color="k", lw=0.8)
        for y in (-2, 2):
            ax_pull.axhline(y, color="k", ls=":", lw=0.8)
        ax_pull.set_ylim(-4, 4)
        ax_pull.set_ylabel("pull")
        ax_pull.set_xlabel("payment amount [RM]")
        txt = (rf"bot target $\mu = {v['mu']:.3f} \pm {result.errors['mu']:.3f}$ RM" "\n"
               rf"fraud rate $f = {v['fraud_rate']:.2f} \pm {result.errors['fraud_rate']:.2f}$ %" "\n"
               rf"$\chi^2/\mathrm{{ndf}} = {result.chi2:.1f}/{result.ndf}$, $p = {result.pvalue:.2f}$")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
    else:
        ax.set_xlabel("payment amount [RM]")

    ax.set_ylabel(f"payments / RM {width:.2f}")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2) Delta chi2 profile
# ----------------------------------------------------------------------
def plot_profile(result, par="mu"):
    x, dchi2 = chi2_profile(result, par)
    best, err = result.values[par], result.errors[par]
    unit = UNITS[par]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, dchi2, color=COLOURS["sig"], lw=2, label=r"$\Delta\chi^2$ profile")
    ax.plot(x, ((x - best) / err) ** 2, color="0.5", ls="--",
            label=r"parabola from Hesse: $((\theta-\hat\theta)/\sigma)^2$")
    ax.axhline(1.0, color=COLOURS["fit"], ls=":", label=r"$\Delta\chi^2 = 1$")
    ax.axvspan(best - err, best + err, color=COLOURS["fit"], alpha=0.1)
    ax.set_ylim(0, 6)
    ax.set_xlabel(f"{par} [{unit}]")
    ax.set_ylabel(r"$\Delta\chi^2 = \chi^2(\theta) - \chi^2_{\min}$")
    ax.set_title(f"{result.label}: {par} = {best:.3f} ± {err:.3f} {unit}")
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3) the vertical plot of all measurements
# ----------------------------------------------------------------------
def forest_plot(cumulative_results, batch_results, comb_mu, comb_f):
    """Top: topping up (cumulative). Bottom: each batch alone + their weighted average."""
    rows_mu, rows_f = [], []

    def add(label, r, kind):
        rows_mu.append(dict(label=label, value=r.values["mu"], error=r.errors["mu"], kind=kind))
        rows_f.append(dict(label=label, value=r.values["fraud_rate"],
                           error=r.errors["fraud_rate"], kind=kind))

    for i, r in enumerate(cumulative_results, start=1):
        add(f"iteration {i}: {r.label}  (N={r.n_events:,})", r, "cumul")
    for r in batch_results:
        add(f"{r.label} alone  (N={r.n_events:,})", r, "run")
    rows_mu.append(dict(label="weighted average of batches", value=comb_mu.value,
                        error=comb_mu.error, kind="comb"))
    rows_f.append(dict(label="weighted average of batches", value=comb_f.value,
                       error=comb_f.error, kind="comb"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows_mu) + 2.2), sharey=True)
    _forest(axes[0], rows_mu, BOT_TARGET_TRUE, r"bot target amount $\mu$ [RM]",
            truth_label="truth of the simulation")
    _forest(axes[1], rows_f, FRAUD_RATE_TRUE, r"fraud rate $f$ [%]",
            truth_label="truth of the simulation")
    axes[1].tick_params(labelleft=False)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Card-testing measurements: topping up (orange), each batch on its own (blue), "
                 "weighted average of the batches (purple)", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    return fig


# ----------------------------------------------------------------------
# 4) how fast does the error shrink?
# ----------------------------------------------------------------------
def plot_precision_vs_n(results, pars=("mu", "fraud_rate")):
    """Error vs number of payments in the window, with the 1/sqrt(N) expectation."""
    N = np.array([r.n_events for r in results], float)
    n_line = np.geomspace(N.min() / 1.5, N.max() * 1.5, 50)
    names = {"mu": r"$\mu$ [RM]", "fraud_rate": r"$f$ [%]", "w": r"$w$ [RM]"}
    fig, axes = plt.subplots(1, len(pars), figsize=(5.5 * len(pars), 4))
    axes = np.atleast_1d(axes)
    for ax, par in zip(axes, pars):
        err = np.array([r.errors[par] for r in results])
        ax.loglog(N, err, "o", ms=8, color=COLOURS["cumul"], label=f"fitted error on {names[par]}")
        ax.loglog(n_line, err[0] * np.sqrt(N[0] / n_line), "k--",
                  label=r"$\sigma \propto 1/\sqrt{N}$ (scaled from first point)")
        for n, e, r in zip(N, err, results):
            ax.annotate(r.label, (n, e), textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("number of payments in the fit window, N")
        ax.set_ylabel(f"error on {names[par]}")
        ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return fig
