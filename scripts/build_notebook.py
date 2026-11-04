"""
Builds Z_peak_lab.ipynb from the cells below (instructor tool).
Students never need to run this:  python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================
md(r"""
# PHY408 Lab: Measuring the Z boson, one run at a time

**Goal (60 min).** You will measure the mass $M_Z$ and the width $\Gamma_Z$ of the Z boson from dimuon events. The detector delivers data in three runs of increasing size. After each run you **top up** your data set and repeat the full chain, then watch your measurement get sharper:

```
   ┌──────────────┐   ┌───────┐   ┌──────┐   ┌─────┐   ┌──────────────────┐
   │ take data    │ → │ clean │ → │ show │ → │ fit │ → │ M_Z ± σ, Γ_Z ± σ │ ──┐
   │ (top up)     │   └───────┘   └──────┘   └─────┘   └──────────────────┘   │
   └──────────────┘ ←─────────────────── repeat for run 2 and run 3 ──────────┘
```

At the end you put all your results on **one vertical plot** and **combine** them.

| Part | What you do | Time | Lecture link |
|---|---|---|---|
| 0 | Set up | 5 min | W1 L2: Python, NumPy, Pandas, Git |
| 1 | Iteration 1: the full chain on run 1 | 20 min | W2 L2 four-vectors · W3 L2 Breit–Wigner, mixtures · W4 L1 χ², MLE, p-value · W4 L2 covariance, Δχ² = 1 |
| 2 | Iterations 2 and 3: top up and refit | 12 min | W2 L1 law of large numbers, estimators |
| 3 | Show the improvement | 8 min | W2 L2 PDG ideogram · W3 L1 CLT, 1/√N |
| 4 | Combine the measurements | 10 min | W2 L2 PDG weighted average · W5 L1 correlations |
| 5 | Save and push to GitHub | 5 min | W1 L2 Git |
| Bonus | Monte Carlo toys · systematic error | at home | W5 L1 error propagation · W5 L2 Monte Carlo |

**How to work.** Run the cells from top to bottom with **Shift + Enter**. You only need to *edit* the cells marked ✍️. Cells marked 💡 explain the physics or statistics. Write your answers to the questions directly in the markdown cells that say *Your answer*.
""")

# =====================================================================
md(r"""
---
## Part 0: Set up (5 min)

▶️ Run the next cell. If you are on **Google Colab** it installs `iminuit` and downloads the lab; on your own computer it does nothing.
""")
code(r"""
import sys, os
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip -q install iminuit
    if not os.path.exists("phy408-zpeak-lab"):
        !git clone -q https://github.com/SiewYan/phy408-zpeak-lab.git
    %cd phy408-zpeak-lab
print("Running on Colab" if IN_COLAB else "Running locally in", os.getcwd())
""")
code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zlab
from zlab import (take_data, clean, invariant_mass, fit_zpeak, results_table,
                  weighted_average, plotting, M_Z_TRUE, GAMMA_Z_TRUE)

print("zlab is ready.  The three runs will contain", zlab.RUN_PLAN, "events (before cleaning).")
""")
md(r"""
✍️ **Choose your data.** With the default seed everybody gets the same "class" data set, which is handy for comparing with your neighbour. If you want your **own** data set, replace the number by your student ID (digits only). Then run the cell.
""")
code(r"""
SEED = zlab.DEFAULT_SEED      # e.g. SEED = 1234567  (your student ID) for your own data
print("Using seed", SEED)
""")

# =====================================================================
md(r"""
---
## Part 1: Iteration 1, the full chain on run 1 (20 min)

### 1.1 Take data

The function `take_data` is our pretend CMS detector. Each call records one run and writes it to `data/raw/run<k>.csv`. One row is one event with two muons.
""")
code(r"""
path = take_data(run=1, seed=SEED)
raw1 = pd.read_csv(path)
raw1.head()
""")
code(r"""
raw1.info()
""")
md(r"""
💡 **Never trust raw data.** Before any physics, look for problems. The next cell gives you four quick diagnostics. Read the output carefully.
""")
code(r"""
print("1) duplicated (run, event) pairs :", raw1.duplicated(subset=["run", "event"]).sum())
print("2) missing values per column    :")
print(raw1.isna().sum()[raw1.isna().sum() > 0].to_string())
print("3) events in bad lumi sections  :", (raw1["lumi_ok"] == 0).sum())
print("4) pT summary (look at min and max!):")
raw1[["mu1_pt", "mu2_pt"]].describe().round(2)
""")
md(r"""
✍️ **Question 1.1.** Which values in the pT summary are clearly not physical? What do you think caused them?

*Your answer:*
""")

md(r"""
### 1.2 Clean the data

We clean in small, separate steps so that each one can be checked. Here is every step written out in Pandas. Read each line before running.
""")
code(r"""
BEAM_ENERGY = 4000.0   # GeV: no muon can carry more than the beam energy

df = raw1.copy()
print(f"{'raw rows':28s}: {len(df):6d}")

df = df.drop_duplicates(subset=["run", "event"])
print(f"{'remove duplicates':28s}: {len(df):6d}")

df = df.dropna()
print(f"{'remove missing values':28s}: {len(df):6d}")

df = df[df["lumi_ok"] == 1]
print(f"{'good luminosity sections':28s}: {len(df):6d}")

df = df[(df["mu1_pt"] > 0) & (df["mu1_pt"] < BEAM_ENERGY) &
        (df["mu2_pt"] > 0) & (df["mu2_pt"] < BEAM_ENERGY)]
print(f"{'physical pT values':28s}: {len(df):6d}")

df = df[df["mu1_charge"] + df["mu2_charge"] == 0]          # Z is neutral: mu+ mu-
print(f"{'opposite charge':28s}: {len(df):6d}")

df = df[(df["mu1_pt"] > 20) & (df["mu2_pt"] > 20) &        # trigger threshold
        (df["mu1_eta"].abs() < 2.4) & (df["mu2_eta"].abs() < 2.4)]   # tracker acceptance
print(f"{'acceptance + trigger':28s}: {len(df):6d}")
""")
md(r"""
💡 **Don't repeat yourself.** We will clean two more runs, so the same steps live in one function, `zlab.clean()`, which also returns the **cut flow** table. Check that it agrees with what you did by hand:
""")
code(r"""
clean1, cutflow1 = clean(raw1)
assert len(clean1) == len(df), "the function and the hand-made cleaning disagree!"
print("\n✓ zlab.clean() gives the same result as the hand-made cleaning.")
""")
md(r"""
💡 **Why the opposite-charge cut?** The next cell computes the mass of the pairs that the charge cut *removed*, which are the same-sign pairs, and draws them next to the pairs we kept.
""")
code(r"""
from zlab.cleaning import remove_duplicates, remove_missing, good_lumi, physical_values, acceptance

pre = acceptance(physical_values(good_lumi(remove_missing(remove_duplicates(raw1)))))
same_sign = pre[pre["mu1_charge"] == pre["mu2_charge"]]

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(60, 130, 71)
ax.hist(invariant_mass(clean1), bins=bins, histtype="step", lw=2, label="opposite sign (kept)")
ax.hist(invariant_mass(same_sign), bins=bins, histtype="step", lw=2, label="same sign (removed)")
ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]"); ax.set_ylabel("events / 1 GeV"); ax.legend();
""")
md(r"""
✍️ **Question 1.2.** (a) In the cut flow, which step removes the most events? (b) Why is there no peak in the same-sign spectrum?

*Your answer:*
""")

md(r"""
### 1.3 Show the data: from two muons to one number

For each muon we build the four-vector $(E, p_x, p_y, p_z)$ from $(p_T, \eta, \phi)$:

$$p_x = p_T\cos\phi,\quad p_y = p_T\sin\phi,\quad p_z = p_T\sinh\eta,\quad E=\sqrt{|\vec p|^2+m_\mu^2},$$

add the two four-vectors and take $m_{\mu\mu}=\sqrt{E^2-|\vec p|^2}$. Look at `zlab/physics.py`: it is only a few lines.
""")
code(r"""
m1 = invariant_mass(clean1)
print(f"{len(m1)} clean events;  median mass = {np.median(m1):.2f} GeV")
fig = plotting.plot_mass(m1, title="Run 1: dimuon mass in the fit window (70 to 110 GeV)")
""")
md(r"""
✍️ **Question 1.3.** Read a rough estimate of $M_Z$ and of the full width at half maximum (FWHM) off the plot, *before* fitting. Is the FWHM equal to $\Gamma_Z$? (Hint: Week 2 Lesson 2, "why the window matters".)

*Your answer:*
""")

md(r"""
### 1.4 Fit the data

💡 **The model** (Week 3 Lesson 2, mixtures of PDFs). The expected number of events in bin $i$ is

$$\mu_i = N_\mathrm{sig}\, s_i(M_Z, \Gamma_Z) + N_\mathrm{bkg}\, b_i(\lambda),$$

where $s$ is a Breit–Wigner smeared by the detector resolution (a *Voigt* profile, with the resolution $\sigma_\mathrm{res}=0.97$ GeV fixed from calibration) and $b \propto e^{-\lambda m}$ is the Drell–Yan background. There are five free parameters.

💡 **The cost function** (Week 4 Lesson 1). Many bins in the tails have only a few events, so the Gaussian approximation behind the ordinary $\chi^2$ fails there. We use its Poisson version,

$$\chi^2 = 2\sum_i\left[\mu_i - n_i + n_i\ln\frac{n_i}{\mu_i}\right],$$

which is $-2\ln(\text{likelihood ratio})$. Minimising it is the **maximum-likelihood** fit, and it is used exactly like the $\chi^2$ you know: its minimum tests the goodness of fit and $\Delta\chi^2 = 1$ gives the errors. The minimiser is MINUIT (`iminuit`).
""")
code(r"""
r1 = fit_zpeak(m1, label="run 1")
print(r1)
""")
code(r"""
fig = plotting.plot_mass(m1, r1, title="Run 1: fit")
""")
md(r"""
💡 **Reading the pull panel.** pull $=(n_i-\mu_i)/\sqrt{\mu_i}$. If the model is good, the pulls scatter around 0 and about 95 % of them lie within $\pm2$.
""")

md(r"""
### 1.5 Where does the "±" come from?

Week 4 Lesson 2: MINUIT measures the **curvature** of $\chi^2$ at the minimum (the Hessian $H$). The covariance matrix is $C = 2H^{-1}$, the errors are $\sigma_i=\sqrt{C_{ii}}$, and the correlations are $\rho_{ij}=C_{ij}/(\sigma_i\sigma_j)$.
""")
code(r"""
print("Covariance matrix C:")
display(r1.covariance.round(4))
print("Check: sqrt of the diagonal =", np.sqrt(np.diag(r1.covariance)).round(4))
print("Correlation matrix rho:")
display(r1.correlation.round(2))
""")
md(r"""
💡 **The graphical method.** Scan one parameter, re-minimise all the others at every point, and find where $\Delta\chi^2=\chi^2(\theta)-\chi^2_{\min}$ crosses 1. The grey dashed parabola is what the Hesse error assumes.
""")
code(r"""
fig = plotting.plot_profile(r1, "M_Z")
fig = plotting.plot_profile(r1, "Gamma_Z")
""")
md(r"""
✍️ **Question 1.4.** (a) Which parameter is most strongly correlated with $\Gamma_Z$? Explain in one sentence why this makes physical sense. (b) Is the $\Gamma_Z$ profile symmetric? What does that tell you about the Hesse error with only one run of data? (c) Is the fit acceptable? Quote $\chi^2/\mathrm{ndf}$ and the $p$-value.

*Your answer:*
""")
md(r"""
### 1.6 Record the result of iteration 1

We keep three dictionaries, each keyed by the run number $k$:
* `masses[k]`: the clean masses of run $k$, kept so that we can top up later
* `run_fits[k]`: run $k$ fitted **on its own** (independent results)
* `cumulative_fits[k]`: iteration $k$, the fit to **all data so far**, runs 1 to $k$ (the "topping up" view)

After iteration 1 both kinds of fit are the same.
""")
code(r"""
masses = {1: m1}
run_fits = {1: r1}
cumulative_fits = {1: r1}
results_table(cumulative_fits.values())
""")

# =====================================================================
md(r"""
---
## Part 2: Iterations 2 and 3, top up and refit (12 min)

✍️ **Question 2.1. Predict first!** Run 2 is 4× bigger than run 1, and run 3 is 4× bigger than run 2. If the error shrinks like $1/\sqrt N$, what error on $M_Z$ do you expect after topping up run 2 (runs 1+2), and after run 3 (runs 1+2+3)? Write down your two numbers *before* you run the next cells.

*Your prediction:*

▶️ This helper repeats Part 1 (take data, clean, compute the mass) for any run:
""")
code(r"""
def take_and_clean(run):
    raw = pd.read_csv(take_data(run=run, seed=SEED))
    good, cutflow = clean(raw, verbose=False)
    print(f"run {run}: {len(raw):,} raw rows -> {len(good):,} clean events")
    return invariant_mass(good)
""")
md(r"""
### Iteration 2: top up with run 2
""")
code(r"""
run = 2
masses[run] = take_and_clean(run)

# (a) the new run on its own
run_fits[run] = fit_zpeak(masses[run], label=f"run {run}")

# (b) top up: pool runs 1..run and fit everything together
pooled = np.concatenate([masses[k] for k in range(1, run + 1)])
cumulative_fits[run] = fit_zpeak(pooled, label="runs " + "+".join(str(k) for k in range(1, run + 1)))

print(cumulative_fits[run])
fig = plotting.plot_mass(pooled, cumulative_fits[run], title=f"Iteration {run}: " + cumulative_fits[run].label)
""")
md(r"""
### Iteration 3: top up with run 3

✍️ This cell is **identical** to the one above except for the run number. Change `run = ...` to `3`, then run it.
""")
code(r"""
run = 2       # ✍️ change me to 3

masses[run] = take_and_clean(run)
run_fits[run] = fit_zpeak(masses[run], label=f"run {run}")
pooled = np.concatenate([masses[k] for k in range(1, run + 1)])
cumulative_fits[run] = fit_zpeak(pooled, label="runs " + "+".join(str(k) for k in range(1, run + 1)))

print(cumulative_fits[run])
fig = plotting.plot_mass(pooled, cumulative_fits[run], title=f"Iteration {run}: " + cumulative_fits[run].label)
""")
code(r"""
# safety check: did you change the run number?
assert sorted(masses) == [1, 2, 3], "Iteration 3 needs run = 3 in the cell above. Fix it and rerun that cell."

# from now on we use plain lists, ordered by run number
run_list = [run_fits[k] for k in (1, 2, 3)]
cumul_list = [cumulative_fits[k] for k in (1, 2, 3)]
print("All three runs are in. Topping up (cumulative) results:")
results_table(cumul_list)
""")

# =====================================================================
md(r"""
---
## Part 3: Show the improvement (8 min)

### 3.1 The vertical plot

One row per measurement, as in the PDG top-quark ideogram of Week 2 Lesson 2. The dashed line is the PDG value used as "truth" in the simulation.
""")
code(r"""
comb_M = weighted_average([r.values["M_Z"] for r in run_list], [r.errors["M_Z"] for r in run_list])
comb_G = weighted_average([r.values["Gamma_Z"] for r in run_list], [r.errors["Gamma_Z"] for r in run_list])

fig_forest = plotting.forest_plot(cumul_list, run_list, comb_M, comb_G)
""")
md(r"""
### 3.2 How fast does the error shrink?
""")
code(r"""
fig_prec = plotting.plot_precision_vs_n(cumul_list)
for r in cumul_list:
    print(f"{r.label:11s} N = {r.n_events:6,d}   dM_Z = {r.errors['M_Z']:.4f} GeV   "
          f"dGamma_Z = {r.errors['Gamma_Z']:.4f} GeV")
""")
md(r"""
✍️ **Question 3.1.** (a) Compare the errors with your prediction in Question 2.1. (b) The error shrinks as $1/\sqrt N$, but the central value does **not** move steadily towards the truth. Why not? (c) Count how many of the orange points have an error bar that covers the dashed line. How many would you expect?

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 4: Combine the measurements (10 min)

### 4.1 PDG weighted average of the three independent runs

Recipe from Week 2 Lesson 2: $w_i = 1/\sigma_i^2$, $\bar x = \sum w_i x_i/\sum w_i$, $\sigma_{\bar x}=1/\sqrt{\sum w_i}$, then check $\chi^2/\mathrm{ndf}$ and apply the scale factor $S=\sqrt{\chi^2/\mathrm{ndf}}$ only if $S>1$. The code is in `zlab/combine.py`.
""")
code(r"""
print("M_Z    :", comb_M, "\n")
print("Gamma_Z:", comb_G, "\n")

final = cumulative_fits[3]
print(f"Compare with ONE fit to all data (iteration 3):\n"
      f"  M_Z     = {final.values['M_Z']:.4f} +- {final.errors['M_Z']:.4f} GeV\n"
      f"  Gamma_Z = {final.values['Gamma_Z']:.4f} +- {final.errors['Gamma_Z']:.4f} GeV")
""")
md(r"""
✍️ **Question 4.1.** (a) Does the weighted average agree with the single fit to all the data? Should it? (b) Which run dominates the average, and why? (c) Ask your neighbour (who has a different seed) for their $S$. If yours is above 1, what does that tell you?

*Your answer:*
""")
md(r"""
### 4.2 ⚠️ A trap: averaging the topping-up results

Someone suggests: *"We have three results from the iterations, so let's average those too!"* Let's try it.
""")
code(r"""
naive = weighted_average([r.values["M_Z"] for r in cumul_list],
                         [r.errors["M_Z"] for r in cumul_list])
print(f"naive average of the 3 cumulative fits: M_Z = {naive.value:.4f} +- {naive.error_raw:.4f} GeV")
print(f"honest answer (fit to all data)       : M_Z = {final.values['M_Z']:.4f} +- {final.errors['M_Z']:.4f} GeV")
print(f"\nThe naive error is {final.errors['M_Z'] / naive.error_raw:.2f}x too small.")
""")
md(r"""
✍️ **Question 4.2.** Why is the naive error too small? (Hint: how many times has each run-1 event been used? Which assumption of the weighted-average formula is broken? Week 5 Lesson 1, correlations.)

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 5: Save your work and push it to GitHub (5 min)
""")
code(r"""
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
table = results_table(run_list + cumul_list)
table.to_csv("results/zpeak_results.csv", index=False)
fig_forest.savefig("figures/forest_plot.png", dpi=150, bbox_inches="tight")
fig_prec.savefig("figures/precision_vs_N.png", dpi=150, bbox_inches="tight")
print(f"FINAL (seed {SEED}):  M_Z = {final.values['M_Z']:.3f} +- {final.errors['M_Z']:.3f} GeV,"
      f"  Gamma_Z = {final.values['Gamma_Z']:.3f} +- {final.errors['Gamma_Z']:.3f} GeV")
print("Saved results/zpeak_results.csv, figures/forest_plot.png, figures/precision_vs_N.png")
""")
md(r"""
Save the notebook (**Ctrl + S**). Then, in a terminal inside the lab folder (your clone of **your fork**, see the README), commit your notebook with your answers, your results and your figures:

```bash
git checkout -b PHY408_<studentID>
git add Z_peak_lab.ipynb results/ figures/
git commit -m "Z peak lab: M_Z and Gamma_Z from three runs"
git push -u origin PHY408_<studentID>
```

Then open a pull request on GitHub. **On Colab?** Use *File → Download → Download .ipynb*, and upload the file to your fork on GitHub with *Add file → Upload files*.

---
## Summary: what you just did

* **Clean before you analyse.** Six small, testable cuts turned messy raw rows into physics events, and the cut flow documents each one.
* **A fit gives a number *and* a covariance.** The "±" is the curvature of $\chi^2$; the profile $\Delta\chi^2=1$ crossing is the same error read graphically, correlations included.
* **More data → smaller error, as $1/\sqrt N$.** Topping up from about 500 to 11 000 events cut the error by about a factor of $\sqrt{22}\approx 4.7$.
* **Combine only independent results.** The weighted average of the three runs reproduces the fit to all the data. Averaging cumulative fits counts the same events more than once and gives an error that is far too small.
""")

# =====================================================================
md(r"""
---
## Bonus A (at home): is the error honest? Monte Carlo toys (W5 L2)

Repeat the run-1 experiment 200 times with different random seeds, fit each, and histogram the **pull** $(\hat\theta - \theta_\mathrm{true})/\hat\sigma$. If the errors are right, the pulls follow a standard Gaussian: mean 0 and width 1. (About 30 s.)
""")
code(r"""
import tempfile, warnings
warnings.filterwarnings("ignore")
toy_dir = tempfile.mkdtemp()
pulls_M, pulls_G = [], []
for toy in range(200):
    raw = pd.read_csv(take_data(run=1, seed=10_000 + toy, outdir=toy_dir, verbose=False))
    r = fit_zpeak(invariant_mass(clean(raw, verbose=False)[0]))
    pulls_M.append((r.values["M_Z"] - M_Z_TRUE) / r.errors["M_Z"])
    pulls_G.append((r.values["Gamma_Z"] - GAMMA_Z_TRUE) / r.errors["Gamma_Z"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.linspace(-4, 4, 200)
for ax, p, name in zip(axes, [pulls_M, pulls_G], [r"$M_Z$", r"$\Gamma_Z$"]):
    ax.hist(p, bins=np.linspace(-4, 4, 25), density=True, alpha=0.6, label="200 toy experiments")
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r-", label=r"$\mathcal{N}(0,1)$")
    ax.set_title(f"pull of {name}: mean = {np.mean(p):+.2f}, width = {np.std(p):.2f}")
    ax.set_xlabel("pull"); ax.legend()
fig.tight_layout()
""")
md(r"""
✍️ **Question A.** Are the pulls consistent with mean 0 and width 1? What would a width of 1.5 have meant?

*Your answer:*
""")
md(r"""
## Bonus B (at home): when does more data stop helping? A systematic error (W5 L1)

The resolution $\sigma_\mathrm{res} = 0.97$ GeV was fixed in the fit, but the calibration only knows it to $\pm 0.03$ GeV. Refit with $\sigma_\mathrm{res}$ shifted up and down and take the shift in the result as the systematic error. Unlike the statistical error, it does **not** shrink when you top up.
""")
code(r"""
SIGMA_NOMINAL, SIGMA_ERR = zlab.SIGMA_RES, 0.03

rows = []
for k, r_nom in cumulative_fits.items():
    pooled_k = np.concatenate([masses[j] for j in range(1, k + 1)])
    up = fit_zpeak(pooled_k, sigma_res=SIGMA_NOMINAL + SIGMA_ERR)
    dn = fit_zpeak(pooled_k, sigma_res=SIGMA_NOMINAL - SIGMA_ERR)
    for par in ["M_Z", "Gamma_Z"]:
        syst = 0.5 * abs(up.values[par] - dn.values[par])
        stat = r_nom.errors[par]
        rows.append({"fit": r_nom.label, "parameter": par, "value": r_nom.values[par],
                     "stat": stat, "syst": syst, "total": np.hypot(stat, syst)})
budget = pd.DataFrame(rows)
display(budget.round(4))

fig, ax = plt.subplots(figsize=(6, 4))
g = budget[budget.parameter == "Gamma_Z"]
N = [r.n_events for r in cumul_list]
ax.loglog(N, g["stat"], "o-", label="statistical (shrinks as 1/√N)")
ax.loglog(N, g["syst"], "s-", label="systematic from σ_res (does not shrink)")
ax.set_xlabel("events in fit window, N"); ax.set_ylabel(r"error on $\Gamma_Z$ [GeV]"); ax.legend()
fig.tight_layout()
""")
md(r"""
✍️ **Question B.** (a) Why does $\sigma_\mathrm{res}$ change $\Gamma_Z$ so much but $M_Z$ hardly at all? (b) Extrapolate the statistical error on $\Gamma_Z$ with $1/\sqrt N$: how many events would you need before the systematic error dominates? What would you then do to improve the measurement?

*Your answer:*
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "Z_peak_lab.ipynb"
nbf.write(nb, out)
print("wrote", out)
