"""
zlab -- the PHY408 Z-peak "top-up" lab
=====================================

    from zlab import take_data, clean, invariant_mass, fit_zpeak, weighted_average
"""

from .detector import take_data, RUN_PLAN, DEFAULT_SEED, M_Z_TRUE, GAMMA_Z_TRUE
from .cleaning import clean, CLEANING_STEPS
from .physics import invariant_mass, four_vector
from .fitting import fit_zpeak, chi2_profile, results_table, SIGMA_RES
from .combine import weighted_average
from .diagnostics import fit_checklist
from .montecarlo import repeat_experiment, compare_errors
from . import plotting, diagnostics, montecarlo

__all__ = ["take_data", "RUN_PLAN", "DEFAULT_SEED", "M_Z_TRUE", "GAMMA_Z_TRUE", "clean",
           "CLEANING_STEPS", "invariant_mass", "four_vector", "fit_zpeak",
           "chi2_profile", "results_table", "SIGMA_RES", "weighted_average",
           "plotting", "fit_checklist", "repeat_experiment", "compare_errors",
           "diagnostics", "montecarlo"]
