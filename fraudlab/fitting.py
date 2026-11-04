"""
fitting.py  --  fit the card-testing spike and measure the fraud
================================================================

Model (same structure as the Z lab, "mixture of PDFs"):

    expected payments in bin i:   mu_i = N_fraud * s_i  +  N_legit * b_i

    s_i : Gaussian spike centred on the bot's target amount mu.
          The bot randomises its amounts with spread w, and the currency
          conversion blurs them by sigma_res, so the observed width is
                sigma_obs = sqrt( w^2 + sigma_res^2 )
    b_i : falling exponential exp(-slope * amount)  (normal customers)

    both shapes are normalised to 1 inside the fit window.

Cost function: exactly the Poisson chi^2 of the Z lab (we import it).

The headline number is the FRAUD RATE:

    f = N_fraud / N_clean      (N_clean = all clean payments, any amount)
    sigma_f = sigma(N_fraud) / N_clean
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy import stats

from zlab.fitting import _simpson_bins, poisson_chi2   # same engine as the Z lab

# RM 3 to 7 in 40 bins of RM 0.10.  Payments come in whole sen, so we put the
# bin edges at HALF a sen (2.995, 3.095, ...): no payment then sits exactly on
# an edge, where floating-point rounding would drop it into the wrong bin.
FIT_RANGE = (2.995, 6.995)   # RM
N_BINS = 40

# Conversion blur at the spike, fixed in the fit.  In real life this comes
# from calibration: compare our table rate with the network's settled rates.
SIGMA_RES = 0.150            # RM

PARAMETERS = ["mu", "w", "N_fraud", "N_legit", "slope"]
UNITS = {"mu": "RM", "w": "RM", "N_fraud": "payments", "N_legit": "payments",
         "slope": "1/RM", "fraud_rate": "%"}


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def signal_shape(x, mu, w, sigma=None):
    """Gaussian spike: bot spread w combined with conversion blur sigma."""
    sigma = SIGMA_RES if sigma is None else sigma
    return stats.norm.pdf(x, mu, np.sqrt(w**2 + sigma**2))


def background_shape(x, slope, x0=FIT_RANGE[0]):
    """Falling exponential (not normalised)."""
    return np.exp(-slope * (x - x0))


def expected_counts(edges, mu, w, N_fraud, N_legit, slope, sigma=None):
    """mu_i for every bin."""
    s = _simpson_bins(lambda x: signal_shape(x, mu, w, sigma), edges)
    b = _simpson_bins(lambda x: background_shape(x, slope), edges)
    return N_fraud * s / s.sum() + N_legit * b / b.sum()


# ----------------------------------------------------------------------
# result container
# ----------------------------------------------------------------------
@dataclass
class FitResult:
    label: str
    n_events: int            # payments inside the fit window
    n_clean: int             # all clean payments (denominator of the fraud rate)
    values: dict
    errors: dict
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    chi2: float
    ndf: int
    pvalue: float
    valid: bool
    counts: np.ndarray = field(repr=False)
    edges: np.ndarray = field(repr=False)
    minuit: Minuit = field(repr=False)
    sigma_res: float = SIGMA_RES
    options: dict = field(default_factory=dict, repr=False)

    def summary(self):
        lines = [f"Fit: {self.label}   ({self.n_events:,} payments in window, "
                 f"{self.n_clean:,} clean payments in total)",
                 f"  valid minimum : {self.valid}"]
        for p in PARAMETERS:
            fixed = "  (fixed)" if self.minuit.fixed[p] else ""
            lines.append(f"  {p:8s} = {self.values[p]:10.4f} +- {self.errors[p]:.4f}  {UNITS[p]}{fixed}")
        lines.append(f"  -> fraud rate f = {self.values['fraud_rate']:.3f} +- "
                     f"{self.errors['fraud_rate']:.3f} %")
        lines.append(f"  chi2 / ndf = {self.chi2:.1f} / {self.ndf}  ->  p-value = {self.pvalue:.3f}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


# ----------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------
def fit_spike(amounts, label="data", fit_range=FIT_RANGE, n_bins=N_BINS,
              sigma_res=None, flat_background=False):
    """
    Binned Poisson-likelihood fit of the payment-amount spectrum.

    Parameters
    ----------
    amounts : ALL clean amounts in MYR (the window is applied here, and the
              total number of amounts is the denominator of the fraud rate)
    label   : a name for this fit, e.g. "batch 1" or "batches 1+2"
    sigma_res       : conversion blur in RM (default: SIGMA_RES)
    flat_background : True fixes slope = 0 (used to show a WRONG model)

    Returns
    -------
    FitResult  (print it to see the numbers)
    """
    amounts = np.asarray(amounts)
    n_clean = amounts.size
    edges = np.linspace(*fit_range, n_bins + 1)
    counts, _ = np.histogram(amounts, bins=edges)
    n_tot = counts.sum()

    def chi2(mu, w, N_fraud, N_legit, slope):
        return poisson_chi2(counts, expected_counts(edges, mu, w, N_fraud, N_legit, slope, sigma_res))

    m = Minuit(chi2, mu=5.0, w=0.3, N_fraud=0.2 * n_tot, N_legit=0.8 * n_tot,
               slope=0.0 if flat_background else 0.1)
    m.errordef = Minuit.LEAST_SQUARES            # Delta chi2 = 1  ->  1 sigma
    m.limits["mu"] = (fit_range[0] + 0.5, fit_range[1] - 0.5)
    m.limits["w"] = (0.01, 1.5)
    m.limits["N_fraud"] = (0.0, None)
    m.limits["N_legit"] = (0.0, None)
    m.limits["slope"] = (-1.0, 1.0)
    if flat_background:
        m.fixed["slope"] = True
    m.migrad()
    m.hesse()

    values = {p: m.values[p] for p in PARAMETERS}
    errors = {p: m.errors[p] for p in PARAMETERS}
    # derived quantity: the fraud rate in percent
    values["fraud_rate"] = 100.0 * values["N_fraud"] / n_clean
    errors["fraud_rate"] = 100.0 * errors["N_fraud"] / n_clean

    cov = pd.DataFrame(np.array(m.covariance), index=PARAMETERS, columns=PARAMETERS)
    sig = np.sqrt(np.diag(cov.values))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sig, sig)

    n_free = len(PARAMETERS) - sum(m.fixed[p] for p in PARAMETERS)
    ndf = n_bins - n_free
    return FitResult(label=label, n_events=int(n_tot), n_clean=int(n_clean), values=values,
                     errors=errors, covariance=cov, correlation=corr,
                     chi2=m.fval, ndf=ndf, pvalue=stats.chi2.sf(m.fval, ndf),
                     valid=m.valid, counts=counts, edges=edges, minuit=m,
                     sigma_res=SIGMA_RES if sigma_res is None else sigma_res,
                     options=dict(fit_range=fit_range, sigma_res=sigma_res,
                                  flat_background=flat_background))


def chi2_profile(result, par, n_sigma=3, size=41):
    """Delta chi2 profile for one parameter (other parameters re-minimised)."""
    m = result.minuit
    x, fval, _ = m.mnprofile(par, bound=n_sigma, size=size)
    return np.asarray(x), np.asarray(fval) - m.fval


def results_table(results):
    """Collect FitResult objects into one tidy DataFrame."""
    rows = []
    for r in results:
        rows.append({"fit": r.label, "clean payments": r.n_clean, "in window": r.n_events,
                     "mu [RM]": r.values["mu"], "d mu": r.errors["mu"],
                     "fraud rate [%]": r.values["fraud_rate"], "d rate": r.errors["fraud_rate"],
                     "w [RM]": r.values["w"], "d w": r.errors["w"],
                     "chi2/ndf": f"{r.chi2:.1f}/{r.ndf}", "p-value": round(r.pvalue, 3)})
    return pd.DataFrame(rows)
