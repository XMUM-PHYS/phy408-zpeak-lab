"""
run_fraud.py  --  the whole fraud lab in one command (no notebook needed)
=========================================================================

    python scripts/run_fraud.py                 # class data set
    python scripts/run_fraud.py --seed 1234567  # your own data (student ID)
    python scripts/run_fraud.py --toys 100      # also the Monte Carlo error check

Figures go to figures/, the results tables to results/.
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
from fraudlab import (DEFAULT_SEED, clean, fit_checklist, fit_spike, montecarlo,  # noqa: E402
                      plotting, results_table, take_data, to_myr, weighted_average)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (e.g. your student ID)")
    parser.add_argument("--toys", type=int, default=0,
                        help="also check the fit errors with this many Monte Carlo repeats per batch (e.g. 100)")
    args = parser.parse_args()
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    amounts, batch_fits, cumul = {}, [], []
    for batch in (1, 2, 3):
        print(f"\n================ ITERATION {batch} ================")
        raw = pd.read_csv(take_data(batch=batch, seed=args.seed))        # 1) top up
        good, _ = clean(raw)                                              # 2) clean
        amounts[batch] = to_myr(good)
        pooled = np.concatenate([amounts[k] for k in range(1, batch + 1)])
        label = "batch 1" if batch == 1 else "batches " + "+".join(str(k) for k in range(1, batch + 1))
        batch_fits.append(fit_spike(amounts[batch], label=f"batch {batch}"))   # 4) fit (batch alone)
        cumul.append(fit_spike(pooled, label=label))                           # 4) fit (topped up)
        print(cumul[-1])                                                       # 5) measure
        fit_checklist(cumul[-1], pooled)                                       # 6) check the fit
        plotting.plot_amounts(pooled, cumul[-1], title=f"Iteration {batch}: {label}") \
            .savefig(ROOT / f"figures/fraud_fit_iteration{batch}.png", dpi=150)  # 3) show

    comb_mu = weighted_average([r.values["mu"] for r in batch_fits], [r.errors["mu"] for r in batch_fits])
    comb_f = weighted_average([r.values["fraud_rate"] for r in batch_fits],
                              [r.errors["fraud_rate"] for r in batch_fits])
    print("\n================ COMBINATION ================")
    print("bot target mu:", comb_mu)
    print("fraud rate f :", comb_f)

    if args.toys:
        print(f"\n================ MONTE CARLO CHECK ({args.toys} repeats per batch) ================")
        toys = {b: montecarlo.repeat_experiment(b, n_toys=args.toys) for b in (1, 2, 3)}
        fits = {b: r for b, r in zip((1, 2, 3), batch_fits)}
        comparison = montecarlo.compare_errors(fits, toys)
        print(comparison.round(4).to_string(index=False))
        comparison.to_csv(ROOT / "results/fraud_fit_vs_mc_errors.csv", index=False)
        montecarlo.plot_mc_errors(fits, toys, "fraud_rate").savefig(
            ROOT / "figures/fraud_mc_errors_rate.png", dpi=150)

    plotting.forest_plot(cumul, batch_fits, comb_mu, comb_f).savefig(
        ROOT / "figures/fraud_forest_plot.png", dpi=150, bbox_inches="tight")
    plotting.plot_precision_vs_n(cumul).savefig(ROOT / "figures/fraud_precision_vs_N.png", dpi=150)
    table = results_table(batch_fits + cumul)
    table.to_csv(ROOT / "results/fraud_results.csv", index=False)
    print("\n", table.round(4).to_string(index=False))
    print("\nFigures written to figures/, table to results/fraud_results.csv")


if __name__ == "__main__":
    main()
