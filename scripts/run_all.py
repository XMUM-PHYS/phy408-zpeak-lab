"""
run_all.py  --  the whole lab in one command (no notebook needed)
=================================================================

    python scripts/run_all.py                 # class data set
    python scripts/run_all.py --seed 1234567  # your own data (student ID)
    python scripts/run_all.py --toys 100      # also the Monte Carlo error check

Figures go to figures/, the results table to results/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # write files, don't open windows
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zlab import (DEFAULT_SEED, clean, fit_checklist, fit_zpeak, invariant_mass,  # noqa: E402
                  montecarlo, plotting, results_table, take_data, weighted_average)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (e.g. your student ID)")
    parser.add_argument("--toys", type=int, default=0,
                        help="also check the fit errors with this many Monte Carlo repeats per run (e.g. 100)")
    args = parser.parse_args()
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    masses, run_fits, cumul = {}, [], []
    for run in (1, 2, 3):
        print(f"\n================ ITERATION {run} ================")
        raw = pd.read_csv(take_data(run=run, seed=args.seed))            # 1) top up
        good, _ = clean(raw)                                              # 2) clean
        masses[run] = invariant_mass(good)
        pooled = np.concatenate([masses[k] for k in range(1, run + 1)])
        label = "run 1" if run == 1 else "runs " + "+".join(str(k) for k in range(1, run + 1))
        run_fits.append(fit_zpeak(masses[run], label=f"run {run}"))       # 4) fit (run alone)
        cumul.append(fit_zpeak(pooled, label=label))                      # 4) fit (topped up)
        print(cumul[-1])                                                  # 5) measure
        fit_checklist(cumul[-1], pooled)                                  # 6) check the fit
        plotting.plot_mass(pooled, cumul[-1], title=f"Iteration {run}: {label}") \
            .savefig(ROOT / f"figures/fit_iteration{run}.png", dpi=150)    # 3) show

    comb_M = weighted_average([r.values["M_Z"] for r in run_fits], [r.errors["M_Z"] for r in run_fits])
    comb_G = weighted_average([r.values["Gamma_Z"] for r in run_fits], [r.errors["Gamma_Z"] for r in run_fits])
    print("\n================ COMBINATION ================")
    print("M_Z    :", comb_M)
    print("Gamma_Z:", comb_G)

    if args.toys:
        print(f"\n================ MONTE CARLO CHECK ({args.toys} repeats per run) ================")
        toys = {run: montecarlo.repeat_experiment(run, n_toys=args.toys) for run in (1, 2, 3)}
        fits = {run: r for run, r in zip((1, 2, 3), run_fits)}
        comparison = montecarlo.compare_errors(fits, toys)
        print(comparison.round(4).to_string(index=False))
        comparison.to_csv(ROOT / "results/fit_vs_mc_errors.csv", index=False)
        montecarlo.plot_mc_errors(fits, toys, "M_Z").savefig(ROOT / "figures/mc_errors_M_Z.png", dpi=150)

    plotting.forest_plot(cumul, run_fits, comb_M, comb_G).savefig(
        ROOT / "figures/forest_plot.png", dpi=150, bbox_inches="tight")
    plotting.plot_precision_vs_n(cumul).savefig(ROOT / "figures/precision_vs_N.png", dpi=150)
    table = results_table(run_fits + cumul)
    table.to_csv(ROOT / "results/zpeak_results.csv", index=False)
    print("\n", table.to_string(index=False))
    print("\nFigures written to figures/, table to results/zpeak_results.csv")


if __name__ == "__main__":
    main()
