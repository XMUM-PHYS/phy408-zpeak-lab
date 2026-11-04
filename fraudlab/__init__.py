"""
fraudlab -- the PHY408 card-testing fraud "top-up" lab
=====================================================

Same workflow as zlab (the Z-peak lab), same fit engine, new observable:

    from fraudlab import take_data, clean, to_myr, fit_spike, weighted_average
"""

from zlab.combine import weighted_average          # the PDG average is the same

from .bank import (take_data, RUN_PLAN, DEFAULT_SEED, FX_TABLE,
                   BOT_TARGET_TRUE, BOT_SPREAD_TRUE, FRAUD_RATE_TRUE)
from .cleaning import clean, CLEANING_STEPS
from .conversion import to_myr
from .fitting import fit_spike, chi2_profile, results_table, SIGMA_RES
from .diagnostics import fit_checklist
from .montecarlo import repeat_experiment, compare_errors
from . import plotting, diagnostics, montecarlo

__all__ = ["take_data", "RUN_PLAN", "DEFAULT_SEED", "FX_TABLE", "BOT_TARGET_TRUE",
           "BOT_SPREAD_TRUE", "FRAUD_RATE_TRUE", "clean", "CLEANING_STEPS", "to_myr",
           "fit_spike", "chi2_profile", "results_table", "SIGMA_RES", "weighted_average",
           "fit_checklist", "repeat_experiment", "compare_errors",
           "plotting", "diagnostics", "montecarlo"]
