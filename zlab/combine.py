"""
combine.py  --  PDG-style weighted average (Week 2 Lesson 2, slide 38)
======================================================================

For INDEPENDENT measurements x_i +- sigma_i:

    w_i    = 1 / sigma_i^2
    x_bar  = sum(w_i x_i) / sum(w_i)
    sigma  = 1 / sqrt(sum(w_i))
    chi2   = sum( w_i (x_i - x_bar)^2 ),   ndf = N - 1
    S      = sqrt(chi2 / ndf);  if S > 1, multiply sigma by S   (PDG rule)

WARNING: the recipe assumes the measurements share no data and no
correlated errors.  Fits to overlapping (cumulative) data sets are NOT
independent, so they must not be averaged with this formula.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Combination:
    value: float
    error: float           # after PDG scale factor (if S > 1)
    error_raw: float       # 1/sqrt(sum w), before scaling
    chi2: float
    ndf: int
    S: float
    pvalue: float
    weights: np.ndarray    # weight share of each input (sums to 1)

    def __str__(self):
        txt = (f"weighted average = {self.value:.4f} +- {self.error:.4f}\n"
               f"  chi2/ndf = {self.chi2:.2f}/{self.ndf}  (p-value = {self.pvalue:.3f}),"
               f"  scale factor S = {self.S:.2f}")
        if self.S > 1:
            txt += f"\n  (error scaled up from {self.error_raw:.4f} because S > 1)"
        share = ", ".join(f"{100 * w:.1f}%" for w in self.weights)
        return txt + f"\n  weight shares: {share}"


def weighted_average(values, errors):
    """PDG inverse-variance weighted average of independent measurements."""
    x = np.asarray(values, float)
    s = np.asarray(errors, float)
    w = 1.0 / s**2
    xbar = np.sum(w * x) / np.sum(w)
    err = 1.0 / np.sqrt(np.sum(w))
    chi2 = float(np.sum(w * (x - xbar) ** 2))
    ndf = len(x) - 1
    S = np.sqrt(chi2 / ndf) if ndf > 0 else 1.0
    err_scaled = err * S if S > 1 else err
    p = stats.chi2.sf(chi2, ndf) if ndf > 0 else 1.0
    return Combination(value=xbar, error=err_scaled, error_raw=err, chi2=chi2,
                       ndf=ndf, S=S, pvalue=p, weights=w / w.sum())
