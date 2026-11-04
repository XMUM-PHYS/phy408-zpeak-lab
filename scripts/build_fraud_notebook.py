"""
Builds Fraud_lab.ipynb from the cells below (instructor tool).
Students never need to run this:  python scripts/build_fraud_notebook.py

The notebook mirrors Z_peak_lab.ipynb part by part and question by question.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================
md(r"""
# PHY408 Lab: Catching a card-testing bot, one batch at a time

**Goal (60 min).** You are a data analyst at a bank. A small online shop that uses your payment gateway is being attacked by a **card-testing bot**: criminals with a list of stolen card numbers check which cards still work by making many tiny purchases of about the same amount. You will measure **where the bot charges** (its target amount $\mu$) and **how much of the traffic is fraud** (the fraud rate $f$) from the payment amounts. The data arrive in three batches of increasing size. After each batch you **top up** your data set and repeat the full chain, then watch your measurement get sharper:

```
   ┌───────────┐   ┌───────┐   ┌──────┐   ┌─────┐   ┌───────────┐   ┌────────────────┐
   │ take data │ → │ clean │ → │ show │ → │ fit │ → │ check fit │ → │ μ ± σ,  f ± σ  │ ──┐
   │ (top up)  │   └───────┘   └──────┘   └─────┘   └───────────┘   └────────────────┘   │
   └───────────┘ ←───────────────────────── repeat for batch 2 and batch 3 ───────────────┘
```

At the end you put all your results on **one vertical plot**, check the fit errors with **Monte Carlo**, and **combine** the batches.

> 💡 **Same lab, different data.** This notebook follows *exactly* the same steps as the physics lab `Z_peak_lab.ipynb`. There, the Z boson is a peak in the dimuon mass on top of a smooth background. Here, the bot is a spike in the payment amounts on top of the smooth spending of normal customers. Data-intensive science is the method, not the subject.

| Part | What you do | Time | Lecture link |
|---|---|---|---|
| 0 | Set up | 5 min | W1 L2: Python, NumPy, Pandas, Git |
| 1 | Iteration 1: the full chain on batch 1, then the fit checklist | 22 min | W1 L2 Pandas cleaning · W3 L2 mixtures of PDFs · W4 L1 χ², MLE, p-value, pulls · W4 L2 covariance, Δχ² = 1 |
| 2 | Iterations 2 and 3: top up, refit, check | 10 min | W2 L1 law of large numbers, estimators |
| 3 | Show the improvement | 6 min | W2 L2 PDG ideogram · W3 L1 CLT, 1/√N |
| 4 | Is the fit error right? Check it with Monte Carlo | 7 min | W5 L2 Monte Carlo methods |
| 5 | Combine the measurements | 7 min | W2 L2 PDG weighted average · W5 L1 correlations |
| 6 | Save and push to GitHub | 3 min | W1 L2 Git |
| Bonus | A systematic error | at home | W5 L1 error propagation |

**How to work.** Run the cells from top to bottom with **Shift + Enter**. You only need to *edit* the cells marked ✍️. Cells marked 💡 explain the ideas. Write your answers to the questions directly in the markdown cells that say *Your answer*.

*All data are simulated, and the shop is fictional.*
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

import fraudlab
from fraudlab import (take_data, clean, to_myr, fit_spike, results_table,
                      weighted_average, fit_checklist, plotting, diagnostics, montecarlo)

print("fraudlab is ready.  The three batches will contain about", fraudlab.RUN_PLAN,
      "payment records (before cleaning).")
""")
md(r"""
✍️ **Choose your data.** With the default seed everybody gets the same "class" data set, which is handy for comparing with your neighbour. If you want your **own** data set, replace the number by your student ID (digits only). Then run the cell.
""")
code(r"""
SEED = fraudlab.DEFAULT_SEED      # e.g. SEED = 1234567  (your student ID) for your own data
print("Using seed", SEED)
""")

# =====================================================================
md(r"""
---
## Part 1: Iteration 1, the full chain on batch 1 (22 min)

### 1.1 Take data

The function `take_data` is our pretend payment gateway. Each call delivers one batch of card payments and writes it to `data/raw/fraud_batch<k>.csv`. One row is one payment attempt.
""")
code(r"""
path = take_data(batch=1, seed=SEED)
raw1 = pd.read_csv(path)
raw1.head()
""")
code(r"""
raw1.info()
""")
md(r"""
💡 **Never trust raw data.** Before any analysis, look for problems. The next cell gives you five quick diagnostics. Read the output carefully.
""")
code(r"""
print("1) duplicated payments (same txn_id):", raw1.duplicated(subset=["txn_id"]).sum())
print("2) missing values per column        :")
print(raw1.isna().sum()[raw1.isna().sum() > 0].to_string())
print("3) test-mode payments by the shop   :", (raw1["is_test"] == 1).sum())
print("4) payment types and statuses       :")
print(raw1.groupby(["type", "status"]).size().to_string())
print("5) amount summary (look at min and max!):")
raw1[["amount"]].describe().round(2)
""")
md(r"""
✍️ **Question 1.1.** Which values in the amount summary are clearly not real payments? What do you think caused them?

*Your answer:*
""")

md(r"""
### 1.2 Clean the data

We clean in small, separate steps so that each one can be checked. Here is every step written out in Pandas. Read each line before running.
""")
code(r"""
CARD_LIMIT = 50_000.0   # no single payment can be larger than this

df = raw1.copy()
print(f"{'raw rows':30s}: {len(df):6d}")

df = df.drop_duplicates(subset=["txn_id"])
print(f"{'remove duplicates':30s}: {len(df):6d}")

df = df.dropna()
print(f"{'remove missing values':30s}: {len(df):6d}")

df = df[df["is_test"] == 0]                          # the shop's own test payments
print(f"{'live payments (no test mode)':30s}: {len(df):6d}")

df = df[(df["amount"] > 0) & (df["amount"] < CARD_LIMIT)]
print(f"{'valid amounts':30s}: {len(df):6d}")

df = df[df["type"] == "purchase"]                    # refunds are not new payments
print(f"{'purchases only (no refunds)':30s}: {len(df):6d}")

df = df[df["status"] == "approved"]                  # the fraud that went through
print(f"{'approved only':30s}: {len(df):6d}")
""")
md(r"""
💡 **Don't repeat yourself.** We will clean two more batches, so the same steps live in one function, `fraudlab.clean()`, which also returns the **cut flow** table. Check that it agrees with what you did by hand:
""")
code(r"""
clean1, cutflow1 = clean(raw1)
assert len(clean1) == len(df), "the function and the hand-made cleaning disagree!"
print("\n✓ fraudlab.clean() gives the same result as the hand-made cleaning.")
""")
md(r"""
💡 **Why remove refunds and test payments?** The next cell draws the amounts of the payments we kept next to two groups we removed: the refunds and the shop's own test payments.
""")
code(r"""
from fraudlab.cleaning import remove_duplicates, remove_missing, valid_amounts

pre = valid_amounts(remove_missing(remove_duplicates(raw1)))
refunds = pre[pre["type"] == "refund"]
tests = pre[pre["is_test"] == 1]

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(2.995, 6.995, 41)
ax.hist(to_myr(clean1), bins=bins, histtype="step", lw=2, label="approved purchases (kept)")
ax.hist(to_myr(refunds), bins=bins, histtype="step", lw=2, label="refunds (removed)")
ax.hist(to_myr(tests), bins=bins, histtype="step", lw=2, label="shop's test payments (removed)")
ax.set_xlabel("payment amount [RM]"); ax.set_ylabel("payments / RM 0.10"); ax.legend(loc="upper left", fontsize=9);
""")
md(r"""
✍️ **Question 1.2.** (a) In the cut flow, which step removes the most payments? (b) Why is there no spike in the refunds? What would the test payments have done to our measurement if we had kept them? (c) The last step throws away the declined payments, and many of those come from the bot (most stolen cards are dead). Why might a fraud analyst still want to watch them?

*Your answer:*
""")

md(r"""
### 1.3 Show the data: from a raw record to one number

In the Z lab we built four-vectors to get one number per event, the invariant mass. Here each payment is recorded in the **card's currency**, together with the bank's reference exchange rate at that moment, so the one number we need is

$$\text{amount in MYR} = \text{amount} \times \text{rate},$$

where the rate is the column `fx_rate` (exactly 1 for MYR cards).

Look at `fraudlab/conversion.py`: it is one line.

💡 **A detail about bins.** Payments come in whole sen (RM 0.01), so we put the bin edges at *half* a sen (RM 2.995, 3.095, …). An amount such as RM 5.30 then never sits exactly on an edge, where rounding in the computer could drop it into the wrong bin.
""")
code(r"""
a1 = to_myr(clean1)
print(f"{len(a1)} clean payments;  median amount = RM {np.median(a1):.2f}")
fig = plotting.plot_amounts(a1, title="Batch 1: payment amounts in the fit window (RM 3 to 7)")
""")
md(r"""
✍️ **Question 1.3.** Read a rough estimate of the spike's position and of its full width at half maximum (FWHM) off the plot, *before* fitting. Is that width the same as the bot's own randomisation $w$? (Hint: what else blurs a foreign-card amount?)

*Your answer:*
""")

md(r"""
### 1.4 Fit the data

💡 **The model** (Week 3 Lesson 2, mixtures of PDFs). The expected number of payments in bin $i$ is

$$\mu_i = N_\mathrm{fraud}\, s_i(\mu, w) + N_\mathrm{legit}\, b_i(\lambda),$$

where $s$ is a Gaussian spike at the bot's target amount $\mu$, and $b \propto e^{-\lambda x}$ is the smooth spending of normal customers. The bot randomises its amounts with spread $w$, and the currency conversion blurs them by $\sigma_\mathrm{res} = 0.15$ RM (fixed from calibration), so the observed width is

$$\sigma_\mathrm{obs} = \sqrt{w^2 + \sigma_\mathrm{res}^2}.$$

There are five free parameters. The headline number is derived from one of them:

$$f = \frac{N_\mathrm{fraud}}{N_\mathrm{clean}}, \qquad \sigma_f = \frac{\sigma(N_\mathrm{fraud})}{N_\mathrm{clean}},$$

where $N_\mathrm{clean}$ is the number of all clean payments, at any amount.

💡 **The cost function** is the same Poisson $\chi^2$ as in the Z lab (Week 4 Lesson 1),

$$\chi^2 = 2\sum_i\left[\mu_i - n_i + n_i\ln\frac{n_i}{\mu_i}\right],$$

minimised by MINUIT (`iminuit`). Its minimum tests the goodness of fit, and $\Delta\chi^2 = 1$ gives the errors.
""")
code(r"""
r1 = fit_spike(a1, label="batch 1")
print(r1)
""")
code(r"""
fig = plotting.plot_amounts(a1, r1, title="Batch 1: fit")
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
fig = plotting.plot_profile(r1, "mu")
fig = plotting.plot_profile(r1, "w")
""")
md(r"""
✍️ **Question 1.4.** (a) Which parameter is most strongly correlated with the bot's spread $w$? Explain in one sentence why this makes sense. (b) Is the $w$ profile symmetric? What does that tell you about the Hesse error with only one batch of data? (c) Is the fit acceptable? Quote $\chi^2/\mathrm{ndf}$ and the $p$-value.

*Your answer:*
""")
md(r"""
### 1.6 Check the fit: the five-point checklist

💡 **A fit always gives you numbers, even when it is wrong.** So before we write down $f \pm \sigma$, we go through a checklist, the way a pilot does before take-off. These are the steps of Week 4 Lesson 1, done for you by one function:

| # | Question | What we look at | ✓ if… |
|---|---|---|---|
| 1 | Did MINUIT find a minimum? | `valid` flag and EDM (estimated distance to minimum) | valid and EDM tiny |
| 2 | Does the model describe the data? | $\chi^2/\mathrm{ndf}$ and $p$-value | $p > 0.05$ |
| 3 | Do the bin pulls look like $\mathcal{N}(0,1)$? | fraction of pulls within $\pm2$ | about 95 % (at least 85 %) |
| 4 | Are the errors symmetric? | Hesse $\pm\sigma$ vs MINOS $-\sigma_\mathrm{lo}/+\sigma_\mathrm{hi}$ | they agree within 20 % |
| 5 | Is the result stable? | refit with RM 0.20 bins instead of RM 0.10 | shift well below $1\sigma$ |

▶️ Run the checklist on the batch-1 fit:
""")
code(r"""
checks1 = fit_checklist(r1, a1)
""")
md(r"""
▶️ Check 3 in a picture: the 40 bin pulls should look like a standard Gaussian.
""")
code(r"""
fig = diagnostics.plot_pull_histogram(r1)
""")
md(r"""
✍️ **Question 1.5.** (a) How many checks passed? (b) In check 4, which parameter has the more asymmetric errors? Does that agree with what you saw in the $w$ profile plot? (c) Check 5 changes only the bin width, and the data are the same. Why should a good result not depend on that choice?

*Your answer:*
""")
md(r"""
### 1.7 Record the result of iteration 1

We keep three dictionaries, each keyed by the batch number $k$:
* `amounts[k]`: the clean amounts of batch $k$, kept so that we can top up later
* `batch_fits[k]`: batch $k$ fitted **on its own** (independent results)
* `cumulative_fits[k]`: iteration $k$, the fit to **all data so far**, batches 1 to $k$ (the "topping up" view)

After iteration 1 both kinds of fit are the same.
""")
code(r"""
amounts = {1: a1}
batch_fits = {1: r1}
cumulative_fits = {1: r1}
results_table(cumulative_fits.values())
""")

# =====================================================================
md(r"""
---
## Part 2: Iterations 2 and 3, top up, refit, check (10 min)

✍️ **Question 2.1. Predict first!** Batch 2 is 4× bigger than batch 1, and batch 3 is 4× bigger than batch 2. If the error shrinks like $1/\sqrt N$, what error on the fraud rate $f$ do you expect after topping up batch 2 (batches 1+2), and after batch 3 (batches 1+2+3)? Write down your two numbers *before* you run the next cells.

*Your prediction:*

▶️ This helper repeats the first steps of Part 1 (take data, clean, convert to MYR) for any batch:
""")
code(r"""
def take_and_clean(batch):
    raw = pd.read_csv(take_data(batch=batch, seed=SEED))
    good, cutflow = clean(raw, verbose=False)
    print(f"batch {batch}: {len(raw):,} raw rows -> {len(good):,} clean payments")
    return to_myr(good)
""")
md(r"""
### Iteration 2: top up with batch 2
""")
code(r"""
batch = 2
amounts[batch] = take_and_clean(batch)

# (a) the new batch on its own
batch_fits[batch] = fit_spike(amounts[batch], label=f"batch {batch}")

# (b) top up: pool batches 1..batch and fit everything together
pooled = np.concatenate([amounts[k] for k in range(1, batch + 1)])
cumulative_fits[batch] = fit_spike(pooled, label="batches " + "+".join(str(k) for k in range(1, batch + 1)))

print(cumulative_fits[batch])
fig = plotting.plot_amounts(pooled, cumulative_fits[batch], title=f"Iteration {batch}: " + cumulative_fits[batch].label)

# (c) check the fit before we trust it
checks = fit_checklist(cumulative_fits[batch], pooled)
""")
md(r"""
### Iteration 3: top up with batch 3

✍️ This cell is **identical** to the one above except for the batch number. Change `batch = ...` to `3`, then run it.
""")
code(r"""
batch = 2       # ✍️ change me to 3

amounts[batch] = take_and_clean(batch)
batch_fits[batch] = fit_spike(amounts[batch], label=f"batch {batch}")
pooled = np.concatenate([amounts[k] for k in range(1, batch + 1)])
cumulative_fits[batch] = fit_spike(pooled, label="batches " + "+".join(str(k) for k in range(1, batch + 1)))

print(cumulative_fits[batch])
fig = plotting.plot_amounts(pooled, cumulative_fits[batch], title=f"Iteration {batch}: " + cumulative_fits[batch].label)
checks = fit_checklist(cumulative_fits[batch], pooled)
""")
code(r"""
# safety check: did you change the batch number?
assert sorted(amounts) == [1, 2, 3], "Iteration 3 needs batch = 3 in the cell above. Fix it and rerun that cell."

# from now on we use plain lists, ordered by batch number
batch_list = [batch_fits[k] for k in (1, 2, 3)]
cumul_list = [cumulative_fits[k] for k in (1, 2, 3)]
print("All three batches are in. Topping up (cumulative) results:")
results_table(cumul_list)
""")
md(r"""
### What does a failing checklist look like?

So far every check has passed. To see the checklist catch a mistake, we make one on purpose. A tempting shortcut is to assume that normal customers spend evenly across RM 3 to 7, a **flat background**. We fit with that wrong model twice, first on batch 1 only and then on all the data.

▶️ Wrong model, batch 1 only (about 1 300 payments in the window):
""")
code(r"""
wrong_small = fit_spike(amounts[1], label="WRONG model, batch 1", flat_background=True)
checks = fit_checklist(wrong_small, amounts[1])
""")
md(r"""
▶️ The same wrong model on all the data (about 25 000 payments in the window):
""")
code(r"""
wrong_big = fit_spike(pooled, label="WRONG model, batches 1+2+3", flat_background=True)
checks = fit_checklist(wrong_big, pooled)
fig = plotting.plot_amounts(pooled, wrong_big, title="A wrong model: look at the pull panel")
print(f"bot target with the wrong model: {wrong_big.values['mu']:.3f} ± {wrong_big.errors['mu']:.3f} RM"
      f"   (right model: {cumul_list[-1].values['mu']:.3f} ± {cumul_list[-1].errors['mu']:.3f} RM)")
""")
md(r"""
✍️ **Question 2.2.** (a) Which checks fail for the wrong model on all the data? Describe the pattern in the pull panel: where is the model too high, and where too low? How did the wrong background move the bot's target $\mu$? (b) The same wrong model passed on batch 1. Why can a small data set not detect the mistake? What does that tell you about a fit that "passes" on very little data?

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 3: Show the improvement (6 min)

### 3.1 The vertical plot

One row per measurement, as in the PDG top-quark ideogram of Week 2 Lesson 2. The dashed line is the truth used in the simulation.
""")
code(r"""
comb_mu = weighted_average([r.values["mu"] for r in batch_list], [r.errors["mu"] for r in batch_list])
comb_f = weighted_average([r.values["fraud_rate"] for r in batch_list], [r.errors["fraud_rate"] for r in batch_list])

fig_forest = plotting.forest_plot(cumul_list, batch_list, comb_mu, comb_f)
""")
md(r"""
### 3.2 How fast does the error shrink?
""")
code(r"""
fig_prec = plotting.plot_precision_vs_n(cumul_list)
for r in cumul_list:
    print(f"{r.label:13s} N = {r.n_events:6,d}   d_mu = {r.errors['mu']:.4f} RM   "
          f"d_f = {r.errors['fraud_rate']:.3f} %")
""")
md(r"""
✍️ **Question 3.1.** (a) Compare the errors with your prediction in Question 2.1. (b) The error shrinks as $1/\sqrt N$, but the central value does **not** move steadily towards the truth. Why not? (c) Count how many of the orange points have an error bar that covers the dashed line. How many would you expect?

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 4: Is the fit error right? Check it with Monte Carlo (7 min)

💡 **What does "$f = 5.7 \pm 0.8$ %" really mean?** It means: *if the same month could happen many times, the fitted $f$ would scatter around the true value with a standard deviation of 0.8 %.* The fit predicts this number from **one** data set, using the curvature of $\chi^2$.

With a simulation we **can** replay the month (Week 5 Lesson 2). That gives an independent check of the fit error, in three steps:

1. Receive 100 new copies of each batch: same size, new random numbers.
2. Fit every copy exactly as you fitted your own data.
3. The spread (standard deviation) of the 100 fitted values is the **Monte Carlo error**:

$$\sigma_\mathrm{MC} = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(\hat\theta_k - \bar{\hat\theta}\right)^2}, \qquad K = 100.$$

If the fit is honest, $\sigma_\mathrm{fit} \approx \sigma_\mathrm{MC}$. With $K=100$, $\sigma_\mathrm{MC}$ itself is only known to about $1/\sqrt{2(K-1)} \approx 7\,\%$, so "agree" means within about 10 to 15 %.

▶️ **Step 1 and 2:** replay each batch 100 times and fit every copy (about 30 s, so be patient):
""")
code(r"""
toys = {}
for batch in (1, 2, 3):
    toys[batch] = montecarlo.repeat_experiment(batch, n_toys=100)
    print(f"batch {batch}: 100 replayed batches fitted")
""")
md(r"""
▶️ **Step 3:** compare the error from your fit with the spread of the replayed batches:
""")
code(r"""
comparison = montecarlo.compare_errors(batch_fits, toys)
comparison.round(4)
""")
md(r"""
▶️ The same comparison as a picture. Each histogram shows the 100 replayed results. The red curve is a Gaussian centred on the truth, with the width **your fit** predicted. If the fit error is right, the curve follows the histogram.
""")
code(r"""
fig = montecarlo.plot_mc_errors(batch_fits, toys, "mu")
fig = montecarlo.plot_mc_errors(batch_fits, toys, "fraud_rate")
""")
md(r"""
✍️ **Question 4.1.** (a) Look at the column `ratio fit/MC`. Do $\sigma_\mathrm{fit}$ and $\sigma_\mathrm{MC}$ agree? Use the column `± on sigma_MC` to decide what "agree" means. (b) From batch 1 to batch 3, by what factor does $\sigma_\mathrm{MC}$ shrink? Compare with $1/\sqrt{N}$. (c) A real bank cannot replay November 100 times. Where could the repeated data sets come from instead?

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 5: Combine the measurements (7 min)

### 5.1 PDG weighted average of the three independent batches

Recipe from Week 2 Lesson 2: $w_i = 1/\sigma_i^2$, $\bar x = \sum w_i x_i/\sum w_i$, $\sigma_{\bar x}=1/\sqrt{\sum w_i}$, then check $\chi^2/\mathrm{ndf}$ and apply the scale factor $S=\sqrt{\chi^2/\mathrm{ndf}}$ only if $S>1$. It is literally the same code as in the Z lab (`zlab/combine.py`).
""")
code(r"""
print("bot target mu:", comb_mu, "\n")
print("fraud rate f :", comb_f, "\n")

final = cumulative_fits[3]
print(f"Compare with ONE fit to all data (iteration 3):\n"
      f"  mu = {final.values['mu']:.4f} +- {final.errors['mu']:.4f} RM\n"
      f"  f  = {final.values['fraud_rate']:.3f} +- {final.errors['fraud_rate']:.3f} %")
""")
md(r"""
✍️ **Question 5.1.** (a) Does the weighted average agree with the single fit to all the data? Should it? (b) Which batch dominates the average, and why? (c) Ask your neighbour (who has a different seed) for their $S$. If yours is above 1, what does that tell you?

*Your answer:*
""")
md(r"""
### 5.2 ⚠️ A trap: averaging the topping-up results

Your manager suggests: *"We have three fraud-rate reports from the iterations, so let's average those too!"* Banks really do produce year-to-date reports like these. Let's try it.
""")
code(r"""
naive = weighted_average([r.values["fraud_rate"] for r in cumul_list],
                         [r.errors["fraud_rate"] for r in cumul_list])
print(f"naive average of the 3 cumulative fits: f = {naive.value:.3f} +- {naive.error_raw:.3f} %")
print(f"honest answer (fit to all data)       : f = {final.values['fraud_rate']:.3f} +- {final.errors['fraud_rate']:.3f} %")
print(f"\nThe naive error is {final.errors['fraud_rate'] / naive.error_raw:.2f}x too small.")
""")
md(r"""
✍️ **Question 5.2.** Why is the naive error too small? (Hint: how many times has each batch-1 payment been used? Which assumption of the weighted-average formula is broken? Week 5 Lesson 1, correlations.)

*Your answer:*
""")

# =====================================================================
md(r"""
---
## Part 6: Save your work and push it to GitHub (3 min)
""")
code(r"""
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
table = results_table(batch_list + cumul_list)
table.to_csv("results/fraud_results.csv", index=False)
comparison.to_csv("results/fraud_fit_vs_mc_errors.csv", index=False)
fig_forest.savefig("figures/fraud_forest_plot.png", dpi=150, bbox_inches="tight")
fig_prec.savefig("figures/fraud_precision_vs_N.png", dpi=150, bbox_inches="tight")
print(f"FINAL (seed {SEED}):  bot target = RM {final.values['mu']:.3f} +- {final.errors['mu']:.3f},"
      f"  fraud rate = {final.values['fraud_rate']:.2f} +- {final.errors['fraud_rate']:.2f} %")
print("Saved results/fraud_results.csv, results/fraud_fit_vs_mc_errors.csv,\n"
      "      figures/fraud_forest_plot.png, figures/fraud_precision_vs_N.png")
""")
md(r"""
Save the notebook (**Ctrl + S**). Then, in a terminal inside the lab folder (your clone of **your fork**, see the README), commit your notebook with your answers, your results and your figures:

```bash
git checkout -b PHY408_<studentID>
git add Fraud_lab.ipynb results/ figures/
git commit -m "Fraud lab: bot target and fraud rate from three batches"
git push -u origin PHY408_<studentID>
```

Then open a pull request on GitHub. **On Colab?** Use *File → Download → Download .ipynb*, and upload the file to your fork on GitHub with *Add file → Upload files*.

---
## Summary: what you just did

* **Clean before you analyse.** Six small, testable cuts turned messy gateway records into real purchases, and the cut flow documents each one. One cut removed the shop's own test payments, which would otherwise have faked a second spike.
* **A fit gives a number *and* a covariance.** The "±" is the curvature of $\chi^2$; the profile $\Delta\chi^2=1$ crossing is the same error read graphically, correlations included.
* **Check a fit before you trust it.** Five quick checks caught the wrong flat-background model, but only once there were enough data to see it, and that wrong model had quietly shifted the bot's target.
* **More data → smaller error, as $1/\sqrt N$.** Topping up from about 1 300 to 25 000 payments in the window cut the error by about a factor of $\sqrt{20}\approx 4.5$.
* **The fit error means "the spread over repeated experiments".** Monte Carlo replays of each batch confirmed that $\sigma_\mathrm{fit} \approx \sigma_\mathrm{MC}$.
* **Combine only independent results.** The weighted average of the three batches reproduces the fit to all the data. Averaging cumulative reports counts the same payments more than once and gives an error that is too small.
* **The same method measured a Z boson and a fraud rate.** That is the point of data-intensive science.
""")

# =====================================================================
md(r"""
---
## Bonus (at home): when does more data stop helping? A systematic error (W5 L1)

The conversion blur $\sigma_\mathrm{res} = 0.15$ RM was fixed in the fit, but the calibration only knows it to $\pm 0.015$ RM. Refit with $\sigma_\mathrm{res}$ shifted up and down and take the shift in the bot's spread $w$ as the systematic error. Unlike the statistical error, it does **not** shrink when you top up.
""")
code(r"""
SIGMA_NOMINAL, SIGMA_ERR = fraudlab.SIGMA_RES, 0.015

rows = []
for k, r_nom in cumulative_fits.items():
    pooled_k = np.concatenate([amounts[j] for j in range(1, k + 1)])
    up = fit_spike(pooled_k, sigma_res=SIGMA_NOMINAL + SIGMA_ERR)
    dn = fit_spike(pooled_k, sigma_res=SIGMA_NOMINAL - SIGMA_ERR)
    for par in ["mu", "w", "fraud_rate"]:
        syst = 0.5 * abs(up.values[par] - dn.values[par])
        stat = r_nom.errors[par]
        rows.append({"fit": r_nom.label, "parameter": par, "value": r_nom.values[par],
                     "stat": stat, "syst": syst, "total": np.hypot(stat, syst)})
budget = pd.DataFrame(rows)
display(budget.round(4))

fig, ax = plt.subplots(figsize=(6, 4))
g = budget[budget.parameter == "w"]
N = [r.n_events for r in cumul_list]
ax.loglog(N, g["stat"], "o-", label="statistical (shrinks as 1/√N)")
ax.loglog(N, g["syst"], "s-", label="systematic from σ_res (does not shrink)")
ax.set_xlabel("payments in fit window, N"); ax.set_ylabel(r"error on $w$ [RM]"); ax.legend()
fig.tight_layout()
""")
md(r"""
✍️ **Bonus question.** (a) Why does $\sigma_\mathrm{res}$ change $w$ but hardly change $\mu$ or $f$? (b) Since $\sigma_\mathrm{obs} = \sqrt{w^2 + \sigma_\mathrm{res}^2}$, show with the propagation formula that $\delta w \approx (\sigma_\mathrm{res}/w)\,\delta\sigma_\mathrm{res}$, and compare with the table. (c) Extrapolate the statistical error on $w$ with $1/\sqrt N$: how many payments would you need before the systematic error dominates? What would you then do to improve the measurement?

*Your answer:*
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "Fraud_lab.ipynb"
nbf.write(nb, out)
print("wrote", out)
