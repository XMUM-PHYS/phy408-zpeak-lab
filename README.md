# PHY408 Lab: Measuring the Z boson, one run at a time

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SiewYan/phy408-zpeak-lab/blob/main/Z_peak_lab.ipynb)
![tests](https://github.com/SiewYan/phy408-zpeak-lab/actions/workflows/tests.yml/badge.svg)

A one-hour hands-on session that ties together **Weeks 1 to 5** of PHY408 Data Intensive Science.

You are the analyst of a (simulated) CMS-like experiment. The detector delivers dimuon data in **three runs** of increasing size (1 000, 4 000 and 16 000 events). After each run you go round the same loop:

```
 top up data → clean data → show data → fit data → measure M_Z ± σ and Γ_Z ± σ
      ↑                                                                    │
      └──────────────────────── repeat for run 2 and run 3 ────────────────┘
```

At the end you draw every result on **one vertical plot**, watch the error bars shrink like $1/\sqrt{N}$, and **combine** the runs into a single measurement of the Z boson mass and width.

---

## What you will practise

| Step in the lab | Where you met it |
|---|---|
| Load and inspect data with Pandas, run code from a Git repository | Week 1 Lesson 2: UNIX, Git and Python essentials |
| Estimators get better with more data (law of large numbers) | Week 2 Lesson 1: Probability theory |
| Four-vectors, invariant mass, the Z peak, the PDG weighted average | Week 2 Lesson 2: From collision to number |
| Why the error falls as $1/\sqrt{N}$ (Central Limit Theorem) | Week 3 Lesson 1: Continuous distributions |
| Breit–Wigner line shape, exponential background, mixture of PDFs | Week 3 Lesson 2: Probability distributions in physics |
| χ² as a cost function, maximum likelihood, goodness of fit and p-value | Week 4 Lesson 1: Least squares, likelihood and regression |
| Hessian → covariance matrix → errors and correlations; Δχ² = 1 | Week 4 Lesson 2: Parameter uncertainties and covariance |
| Correlated measurements cannot be averaged naively; systematic errors | Week 5 Lesson 1: Error propagation |
| Monte Carlo toy experiments to check that the errors are honest | Week 5 Lesson 2: Monte Carlo methods |

---

## Quick start: pick ONE of the three options

### Option A: Google Colab (nothing to install, recommended if your laptop is not set up)

1. Click the **Open in Colab** badge at the top of this page.
2. Sign in with a Google account.
3. Run the first code cell. It installs `iminuit` and downloads this lab for you.
4. Continue from **Part 0** of the notebook.

### Option B: your own laptop (the environment you built in Week 1)

You need a terminal: **macOS** Terminal, **Linux** terminal, or **Windows** Ubuntu (WSL).

**Step 1. Fork this repository** (so you can push your own work later). On GitHub, click **Fork** at the top right. You now have your own copy at `github.com/<your-username>/phy408-zpeak-lab`.

**Step 2. Clone your fork and enter the folder:**

```bash
git clone https://github.com/<your-username>/phy408-zpeak-lab.git
cd phy408-zpeak-lab
```

**Step 3. Install the packages.** Choose one of these:

```bash
# (a) recommended: a fresh environment just for this lab
mamba env create -f environment.yml
mamba activate phy408-zlab

# (b) or add the one missing package to your Week 1 environment
mamba install -c conda-forge iminuit

# (c) or, without mamba
pip install -r requirements.txt
```

**Step 4. Check that everything works:**

```bash
python scripts/check_setup.py
```

You should see a column of ✓ and the line `Test fit: M_Z = ... GeV -> everything works.`

**Step 5. Start Jupyter and open the notebook:**

```bash
jupyter lab
```

In the browser window that opens, double-click **`Z_peak_lab.ipynb`**.

### Option C: no notebook at all

The whole lab also runs as one script; the figures appear in `figures/`:

```bash
python scripts/run_all.py
python scripts/run_all.py --seed 1234567     # your own data set (student ID)
```

This is handy for checking your notebook results, but the questions are only in the notebook.

---

## The plan for the hour

| Time | Part | You will… |
|---|---|---|
| 0–5 min | 0. Set up | open the notebook, choose your seed |
| 5–25 min | 1. Iteration 1 | take run 1, find the bad data, clean it, plot the mass, fit, read the covariance matrix and the Δχ² profile |
| 25–37 min | 2. Iterations 2 and 3 | **predict** the new error, top up with run 2 then run 3, refit |
| 37–45 min | 3. Show the improvement | draw the vertical plot and the error-versus-N plot |
| 45–55 min | 4. Combine | PDG weighted average; discover why averaging the top-up results is wrong |
| 55–60 min | 5. Save and push | commit your notebook, results and figures |
| at home | Bonus A, B | Monte Carlo toys; a systematic error that more data cannot fix |

**Tips.** Run cells from top to bottom with **Shift + Enter**. Only edit the cells marked ✍️. If something goes wrong, use *Kernel → Restart Kernel and Run All Cells*: the data are re-created identically every time.

---

## What is in this repository

```
phy408-zpeak-lab/
├── Z_peak_lab.ipynb        ← START HERE: the lab notebook with all instructions and questions
├── README.md               ← this file
├── environment.yml         ← mamba environment (Option B, step 3a)
├── requirements.txt        ← pip packages (Option B, step 3c)
├── zlab/                   ← the analysis code: short, commented, worth reading
│   ├── detector.py         ←   the pretend detector: take_data() records one run (you don't edit this)
│   ├── cleaning.py         ←   six cleaning steps and the cut flow
│   ├── physics.py          ←   four-vectors and the invariant mass
│   ├── fitting.py          ←   the signal + background model and the fit with iminuit
│   ├── combine.py          ←   the PDG weighted average with the scale factor S
│   └── plotting.py         ←   every figure of the lab
├── scripts/
│   ├── check_setup.py      ←   "is my computer ready?"
│   ├── run_all.py          ←   the whole lab in one command (Option C)
│   └── build_notebook.py   ←   (instructor tool) rebuilds the notebook
├── tests/test_zlab.py      ← sanity tests:  python -m pytest -q
├── data/raw/               ← run1.csv, run2.csv, run3.csv appear here when you take data
├── figures/                ← your plots are saved here
└── results/                ← your results table is saved here
```

---

## The physics and statistics on one page

**Invariant mass.** For each muon, $p_x=p_T\cos\phi$, $p_y=p_T\sin\phi$, $p_z=p_T\sinh\eta$ and $E=\sqrt{|\vec p|^2+m_\mu^2}$. For the pair, $m_{\mu\mu}=\sqrt{(E_1+E_2)^2-|\vec p_1+\vec p_2|^2}$.

**Model.** In each bin, expected events $=N_\mathrm{sig}\times$ (Breit–Wigner of mass $M_Z$ and width $\Gamma_Z$, smeared by the detector resolution $\sigma_\mathrm{res}=0.97$ GeV) $+\,N_\mathrm{bkg}\times e^{-\lambda m}$. That gives five free parameters.

**Fit.** Minimise the Poisson χ², $\chi^2=2\sum_i[\mu_i-n_i+n_i\ln(n_i/\mu_i)]$, with MINUIT. Its minimum gives the goodness of fit (χ²/ndf and p-value); its curvature gives the covariance matrix; $\Delta\chi^2=1$ gives the 1σ errors.

**Combination of independent results.** $w_i=1/\sigma_i^2$, $\bar x=\sum w_ix_i/\sum w_i$, $\sigma_{\bar x}=1/\sqrt{\sum w_i}$; if $S=\sqrt{\chi^2/\mathrm{ndf}}>1$, multiply $\sigma_{\bar x}$ by $S$.

**The truth.** The simulation uses the PDG values $M_Z=91.1876$ GeV and $\Gamma_Z=2.4955$ GeV, so you can check whether your measurement is right. In a real experiment you never get to do that.

---

## How to hand in your work

1. Save the notebook with your answers filled in (**Ctrl + S**).
2. In the terminal, inside the lab folder:

   ```bash
   git checkout -b PHY408_<studentID>
   git add Z_peak_lab.ipynb results/ figures/
   git commit -m "Z peak lab: M_Z and Gamma_Z from three runs"
   git push -u origin PHY408_<studentID>
   ```

3. On GitHub, open your fork and click **Compare & pull request**.

**On Colab?** *File → Download → Download .ipynb*, then on your fork on GitHub use *Add file → Upload files*.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'iminuit'` | `mamba install -c conda-forge iminuit` (or `pip install iminuit`), then restart the kernel. |
| `ModuleNotFoundError: No module named 'zlab'` | Jupyter was started outside the lab folder. Close it, `cd phy408-zpeak-lab`, and run `jupyter lab` again. |
| Packages are installed but the notebook can't find them | The notebook is using a different Python. Run `mamba activate phy408-zlab` **before** `jupyter lab`, or choose the right kernel at the top right of the notebook. |
| `AssertionError: Iteration 3 needs run = 3` | In the Iteration 3 cell, change `run = 2` to `run = 3` and rerun that cell. |
| Plots don't appear | Rerun the cell. On a script, figures are saved to `figures/` instead of shown. |
| My numbers differ from my neighbour's | Good! If you changed `SEED`, you have a different data set. Your results should agree within the error bars. |
| `git push` says *permission denied* | You cloned the original repository instead of your fork. Fork it on GitHub, then `git remote set-url origin https://github.com/<your-username>/phy408-zpeak-lab.git`. |
| Windows without WSL | Use Option A (Colab). |

Still stuck? Raise your hand, or open an *Issue* on this repository with the full error message.

---

*PHY408 Data Intensive Science, Xiamen University Malaysia. The code is released under the MIT licence.*
