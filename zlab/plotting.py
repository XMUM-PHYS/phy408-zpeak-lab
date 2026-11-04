"""
plotting.py  --  every figure of the lab
========================================
"""

import matplotlib.pyplot as plt
import numpy as np

from .detector import GAMMA_Z_TRUE, M_Z_TRUE
from .fitting import (FIT_RANGE, N_BINS, _simpson_bins, background_shape,
                      expected_counts, signal_shape)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 11, "legend.frameon": False})

COLOURS = {"data": "black", "fit": "#d62728", "sig": "#1f77b4", "bkg": "#2ca02c",
           "run": "#1f77b4", "cumul": "#ff7f0e", "comb": "#9467bd", "truth": "0.3"}


# ----------------------------------------------------------------------
# 1) show the data (and the fit, once we have one)
# ----------------------------------------------------------------------
def plot_mass(masses, result=None, title="", fit_range=FIT_RANGE, n_bins=N_BINS):
    """Dimuon mass histogram with sqrt(N) error bars; add the fit and pulls if given."""
    edges = np.linspace(*fit_range, n_bins + 1)
    counts, _ = np.histogram(masses, bins=edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]

    if result is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax_pull = None
    else:
        fig, (ax, ax_pull) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="o", ms=3.5,
                color=COLOURS["data"], label=f"data ({counts.sum():,} events)")

    if result is not None:
        v = result.values
        fine = np.linspace(*fit_range, 400)
        s = signal_shape(fine, v["M_Z"], v["Gamma_Z"], result.sigma_res)
        b = background_shape(fine, v["slope"])
        # normalise the smooth curves to "events per bin" (same as in the fit)
        s_norm = _simpson_bins(lambda m: signal_shape(m, v["M_Z"], v["Gamma_Z"],
                                                      result.sigma_res), edges).sum()
        b_norm = _simpson_bins(lambda m: background_shape(m, v["slope"]), edges).sum()
        sig = v["N_sig"] * width * s / s_norm
        bkg = v["N_bkg"] * width * b / b_norm
        ax.plot(fine, sig + bkg, color=COLOURS["fit"], lw=2, label="fit: signal + background")
        ax.plot(fine, sig, color=COLOURS["sig"], ls="--", label="signal (Voigt)")
        ax.fill_between(fine, bkg, color=COLOURS["bkg"], alpha=0.3, label="background (exp)")

        mu = expected_counts(edges, *[v[p] for p in ("M_Z", "Gamma_Z", "N_sig", "N_bkg", "slope")],
                             sigma=result.sigma_res)
        pulls = (counts - mu) / np.sqrt(mu)
        ax_pull.bar(centres, pulls, width=width, color="0.5")
        ax_pull.axhline(0, color="k", lw=0.8)
        for y in (-2, 2):
            ax_pull.axhline(y, color="k", ls=":", lw=0.8)
        ax_pull.set_ylim(-4, 4)
        ax_pull.set_ylabel("pull")
        ax_pull.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
        txt = (rf"$M_Z = {v['M_Z']:.3f} \pm {result.errors['M_Z']:.3f}$ GeV" "\n"
               rf"$\Gamma_Z = {v['Gamma_Z']:.3f} \pm {result.errors['Gamma_Z']:.3f}$ GeV" "\n"
               rf"$\chi^2/\mathrm{{ndf}} = {result.chi2:.1f}/{result.ndf}$, $p = {result.pvalue:.2f}$")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
    else:
        ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")

    ax.set_ylabel(f"events / {width:.0f} GeV")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2) Delta chi2 profile (graphical error, Week 4 Lesson 2)
# ----------------------------------------------------------------------
def plot_profile(result, par="M_Z", unit="GeV"):
    from .fitting import chi2_profile

    x, dchi2 = chi2_profile(result, par)
    best, err = result.values[par], result.errors[par]
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
# 3) the vertical ("forest") plot of all measurements
# ----------------------------------------------------------------------
def _forest(ax, rows, truth, xlabel, truth_label="PDG (truth of the simulation)"):
    """rows: list of dicts with keys label, value, error, kind ('run','cumul','comb')."""
    n = len(rows)
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows):
        style = {"run": dict(fmt="o", color=COLOURS["run"]),
                 "cumul": dict(fmt="s", color=COLOURS["cumul"]),
                 "comb": dict(fmt="D", color=COLOURS["comb"], ms=8)}[r["kind"]]
        ax.errorbar(r["value"], yi, xerr=r["error"], capsize=3, lw=1.8, **style)
        ax.annotate(f"{r['value']:.3f} ± {r['error']:.3f}", (1.01, yi),
                    xycoords=("axes fraction", "data"), va="center", fontsize=9)
    ax.axvline(truth, color=COLOURS["truth"], ls="--", lw=1.2, label=truth_label)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0)
    # horizontal separators between the groups
    kinds = [r["kind"] for r in rows]
    for i in range(1, n):
        if kinds[i] != kinds[i - 1]:
            ax.axhline(y[i] + 0.5, color="0.6", lw=0.8)


def forest_plot(cumulative_results, run_results, comb_M, comb_G):
    """
    One row per measurement, vertical layout (like the PDG ideogram).

    Top block    : topping up -- the fit after each iteration (cumulative data)
    Bottom block : each run fitted on its own (independent) + their weighted average
    """
    rows_M, rows_G = [], []

    def add(label, r, kind):
        rows_M.append(dict(label=label, value=r.values["M_Z"], error=r.errors["M_Z"], kind=kind))
        rows_G.append(dict(label=label, value=r.values["Gamma_Z"], error=r.errors["Gamma_Z"], kind=kind))

    for i, r in enumerate(cumulative_results, start=1):
        add(f"iteration {i}: {r.label}  (N={r.n_events:,})", r, "cumul")
    for r in run_results:
        add(f"{r.label} alone  (N={r.n_events:,})", r, "run")
    rows_M.append(dict(label="weighted average of runs", value=comb_M.value,
                       error=comb_M.error, kind="comb"))
    rows_G.append(dict(label="weighted average of runs", value=comb_G.value,
                       error=comb_G.error, kind="comb"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows_M) + 2.2), sharey=True)
    _forest(axes[0], rows_M, M_Z_TRUE, r"$M_Z$ [GeV]")
    _forest(axes[1], rows_G, GAMMA_Z_TRUE, r"$\Gamma_Z$ [GeV]")
    axes[1].tick_params(labelleft=False)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Z boson measurements: topping up (orange), each run on its own (blue), "
                 "weighted average of the runs (purple)", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    return fig


# ----------------------------------------------------------------------
# 4) how fast does the error shrink?
# ----------------------------------------------------------------------
def plot_precision_vs_n(results, pars=("M_Z", "Gamma_Z")):
    """Error vs number of events on log-log axes, with the 1/sqrt(N) expectation."""
    N = np.array([r.n_events for r in results], float)
    n_line = np.geomspace(N.min() / 1.5, N.max() * 1.5, 50)
    names = {"M_Z": r"$M_Z$", "Gamma_Z": r"$\Gamma_Z$"}
    fig, axes = plt.subplots(1, len(pars), figsize=(5.5 * len(pars), 4))
    axes = np.atleast_1d(axes)
    for ax, par in zip(axes, pars):
        err = np.array([r.errors[par] for r in results])
        ax.loglog(N, err, "o", ms=8, color=COLOURS["cumul"], label=f"fitted error on {names[par]}")
        ax.loglog(n_line, err[0] * np.sqrt(N[0] / n_line), "k--",
                  label=r"$\sigma \propto 1/\sqrt{N}$ (scaled from first point)")
        for n, e, r in zip(N, err, results):
            ax.annotate(r.label, (n, e), textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("number of events in the fit window, N")
        ax.set_ylabel(f"error on {names[par]} [GeV]")
        ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return fig
