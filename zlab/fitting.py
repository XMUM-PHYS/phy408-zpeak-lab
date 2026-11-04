"""
fitting.py  --  fit the Z peak and extract M_Z and Gamma_Z with errors
======================================================================

Model (Week 3 Lesson 2, "mixtures of PDFs"):

    expected counts in bin i:   mu_i = N_sig * s_i  +  N_bkg * b_i

    s_i : Voigt profile  = Breit-Wigner(M_Z, Gamma_Z)  (x)  Gaussian(sigma_res)
          (the Z line shape smeared by the detector resolution)
    b_i : falling exponential  exp(-slope * m)  (Drell-Yan continuum)

    both shapes are normalised so that their bins sum to 1 inside the fit
    window, so N_sig and N_bkg are numbers of events in the window.

Cost function (Week 4 Lesson 1):
    The bins in the tails hold only a few events, so we use the Poisson
    version of chi^2 (Baker-Cousins):

        chi2 = 2 * sum_i [ mu_i - n_i + n_i * ln(n_i / mu_i) ]

    Minimising it IS the maximum-likelihood fit for Poisson counts, and
    it behaves like an ordinary chi^2: its minimum tests goodness of fit
    and  Delta chi2 = 1  gives the 1-sigma errors (Week 4 Lesson 2).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy import stats
from scipy.special import voigt_profile

FIT_RANGE = (70.0, 110.0)    # GeV
N_BINS = 40                  # 1 GeV per bin

# Detector mass resolution, fixed in the fit.  In a real experiment this
# comes from calibration (simulation + J/psi, Upsilon, Z control samples).
SIGMA_RES = 0.97             # GeV

PARAMETERS = ["M_Z", "Gamma_Z", "N_sig", "N_bkg", "slope"]


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def _simpson_bins(f, edges):
    """Integral of f over each bin with Simpson's rule (3 points per bin)."""
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    return (hi - lo) / 6.0 * (f(lo) + 4.0 * f(mid) + f(hi))


def signal_shape(m, M_Z, Gamma_Z, sigma=None):
    """Voigt profile: Breit-Wigner of full width Gamma_Z smeared by sigma."""
    sigma = SIGMA_RES if sigma is None else sigma
    return voigt_profile(m - M_Z, sigma, 0.5 * Gamma_Z)


def background_shape(m, slope, m0=FIT_RANGE[0]):
    """Falling exponential (not normalised)."""
    return np.exp(-slope * (m - m0))


def expected_counts(edges, M_Z, Gamma_Z, N_sig, N_bkg, slope, sigma=None):
    """mu_i for every bin."""
    s = _simpson_bins(lambda m: signal_shape(m, M_Z, Gamma_Z, sigma), edges)
    b = _simpson_bins(lambda m: background_shape(m, slope), edges)
    return N_sig * s / s.sum() + N_bkg * b / b.sum()


def poisson_chi2(n, mu):
    """Baker-Cousins chi^2 for Poisson counts (n ln(n/mu) -> 0 when n = 0)."""
    mu = np.clip(mu, 1e-300, None)
    log_term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)
    return 2.0 * np.sum(mu - n + log_term)


# ----------------------------------------------------------------------
# result container
# ----------------------------------------------------------------------
@dataclass
class FitResult:
    label: str
    n_events: int
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

    def summary(self):
        lines = [f"Fit: {self.label}   ({self.n_events:,} events in window)",
                 f"  valid minimum : {self.valid}"]
        for p in PARAMETERS:
            unit = {"M_Z": "GeV", "Gamma_Z": "GeV", "slope": "1/GeV"}.get(p, "events")
            lines.append(f"  {p:8s} = {self.values[p]:10.4f} +- {self.errors[p]:.4f}  {unit}")
        lines.append(f"  chi2 / ndf = {self.chi2:.1f} / {self.ndf}"
                     f"  ->  p-value = {self.pvalue:.3f}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


# ----------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------
def fit_zpeak(masses, label="data", fit_range=FIT_RANGE, n_bins=N_BINS,
              sigma_res=None):
    """
    Binned Poisson-likelihood fit of the dimuon mass spectrum.

    Parameters
    ----------
    masses : array of invariant masses in GeV (any range; the window is applied here)
    label  : a name for this fit, e.g. "run 1" or "runs 1+2"
    sigma_res : detector mass resolution in GeV (default: SIGMA_RES)

    Returns
    -------
    FitResult  (print it to see the numbers)
    """
    masses = np.asarray(masses)
    edges = np.linspace(*fit_range, n_bins + 1)
    counts, _ = np.histogram(masses, bins=edges)
    n_tot = counts.sum()

    def chi2(M_Z, Gamma_Z, N_sig, N_bkg, slope):
        mu = expected_counts(edges, M_Z, Gamma_Z, N_sig, N_bkg, slope, sigma_res)
        return poisson_chi2(counts, mu)

    m = Minuit(chi2, M_Z=91.0, Gamma_Z=2.5, N_sig=0.9 * n_tot,
               N_bkg=0.1 * n_tot, slope=0.04)
    m.errordef = Minuit.LEAST_SQUARES          # Delta chi2 = 1  ->  1 sigma
    m.limits["M_Z"] = (85.0, 97.0)
    m.limits["Gamma_Z"] = (0.1, 10.0)
    m.limits["N_sig"] = (0.0, None)
    m.limits["N_bkg"] = (0.0, None)
    m.limits["slope"] = (-0.2, 0.5)
    m.migrad()          # find the minimum
    m.hesse()           # curvature (Hessian) at the minimum -> covariance

    values = {p: m.values[p] for p in PARAMETERS}
    errors = {p: m.errors[p] for p in PARAMETERS}
    cov = pd.DataFrame(np.array(m.covariance), index=PARAMETERS, columns=PARAMETERS)
    sig = np.sqrt(np.diag(cov.values))
    corr = cov / np.outer(sig, sig)

    ndf = n_bins - len(PARAMETERS)
    return FitResult(label=label, n_events=int(n_tot), values=values,
                     errors=errors, covariance=cov, correlation=corr,
                     chi2=m.fval, ndf=ndf, pvalue=stats.chi2.sf(m.fval, ndf),
                     valid=m.valid, counts=counts, edges=edges, minuit=m,
                     sigma_res=SIGMA_RES if sigma_res is None else sigma_res)


def chi2_profile(result, par, n_sigma=3, size=41):
    """
    Delta chi2 profile for one parameter (Week 4 Lesson 2, graphical method).

    At every scan point the OTHER parameters are re-minimised, so the
    crossing of Delta chi2 = 1 gives the full error, correlations included.
    Returns (x, delta_chi2).
    """
    m = result.minuit
    x, fval, _ = m.mnprofile(par, bound=n_sigma, size=size)
    return np.asarray(x), np.asarray(fval) - m.fval


def results_table(results):
    """Collect a list of FitResult objects into one tidy DataFrame."""
    rows = []
    for r in results:
        rows.append({"fit": r.label, "events": r.n_events,
                     "M_Z": r.values["M_Z"], "dM_Z": r.errors["M_Z"],
                     "Gamma_Z": r.values["Gamma_Z"], "dGamma_Z": r.errors["Gamma_Z"],
                     "chi2/ndf": f"{r.chi2:.1f}/{r.ndf}", "p-value": round(r.pvalue, 3)})
    return pd.DataFrame(rows)
