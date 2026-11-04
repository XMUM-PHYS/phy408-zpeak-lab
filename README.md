# PHY408 Lab: Measuring the Z boson, one run at a time

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SiewYan/phy408-zpeak-lab/blob/main/Z_peak_lab.ipynb)
![tests](https://github.com/SiewYan/phy408-zpeak-lab/actions/workflows/tests.yml/badge.svg)

A one-hour hands-on session that ties together **Weeks 1 to 5** of PHY408 Data Intensive Science.

You are the analyst of a (simulated) CMS-like experiment. The detector delivers dimuon data in **three runs** of increasing size (1 000, 4 000 and 16 000 events). After each run you go round the same loop:

```
 top up data → clean data → show data → fit data → check fit → measure M_Z ± σ and Γ_Z ± σ
      ↑                                                                                │
      └──────────────────────────── repeat for run 2 and run 3 ────────────────────────┘
```

At the end you draw every result on **one vertical plot** and watch the error bars shrink like $`1/\sqrt{N}`$. You then check that the fit errors are honest by **repeating the experiment with Monte Carlo**, and **combine** the runs into a single measurement of the Z boson mass $`M_Z`$ and width $`\Gamma_Z`$.

---

## What you will practise

| Step in the lab | Where you met it |
|---|---|
| Load and inspect data with Pandas, run code from a Git repository | Week 1 Lesson 2: UNIX, Git and Python essentials |
| Estimators get better with more data (law of large numbers) | Week 2 Lesson 1: Probability theory |
| Four-vectors, invariant mass, the Z peak, the PDG weighted average | Week 2 Lesson 2: From collision to number |
| Why the error falls as $`1/\sqrt{N}`$ (Central Limit Theorem) | Week 3 Lesson 1: Continuous distributions |
| Breit–Wigner line shape, exponential background, mixture of PDFs | Week 3 Lesson 2: Probability distributions in physics |
| $`\chi^2`$ as a cost function, maximum likelihood, goodness of fit and $`p`$-value | Week 4 Lesson 1: Least squares, likelihood and regression |
| A five-point fit checklist: convergence, $`p`$-value, pulls, symmetric errors, stability | Week 4 Lesson 1: the fit-validation steps |
| Hessian $`\to`$ covariance matrix $`\to`$ errors and correlations; $`\Delta\chi^2 = 1`$ | Week 4 Lesson 2: Parameter uncertainties and covariance |
| Monte Carlo: repeat the experiment 100 times and compare $`\sigma_\mathrm{MC}`$ with $`\sigma_\mathrm{fit}`$ | Week 5 Lesson 2: Monte Carlo methods |
| Correlated measurements cannot be averaged naively; systematic errors | Week 5 Lesson 1: Error propagation |

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
python scripts/run_all.py --toys 100         # add the Monte Carlo error check (about 30 s)
```

This is handy for checking your notebook results, but the questions are only in the notebook.

---

## The plan for the hour

| Time | Part | You will… |
|---|---|---|
| 0–5 min | 0. Set up | open the notebook, choose your seed |
| 5–27 min | 1. Iteration 1 | take run 1, find the bad data, clean it, plot the mass, fit, read the covariance matrix and the $`\Delta\chi^2`$ profile, run the fit checklist |
| 27–37 min | 2. Iterations 2 and 3 | **predict** the new error, top up with run 2 then run 3, refit and check; see a wrong model fail the checklist |
| 37–43 min | 3. Show the improvement | draw the vertical plot and the error-versus-$`N`$ plot |
| 43–50 min | 4. Monte Carlo check | repeat each run 100 times and compare $`\sigma_\mathrm{MC}`$ with $`\sigma_\mathrm{fit}`$ |
| 50–57 min | 5. Combine | PDG weighted average; discover why averaging the top-up results is wrong |
| 57–60 min | 6. Save and push | commit your notebook, results and figures |
| at home | Bonus | a systematic error that more data cannot fix |

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
│   ├── diagnostics.py      ←   the five-point fit checklist and the pull histogram
│   ├── montecarlo.py       ←   repeat the experiment: sigma_MC versus sigma_fit
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

Every formula below is a few lines of code in `zlab/`; each heading names the file where you will find it.

### 1. From two muons to one mass (`physics.py`)

Each muon is measured as $`(p_T, \eta, \phi)`$. Its four-vector, with $`c = 1`$, is

```math
p_x = p_T\cos\phi, \qquad p_y = p_T\sin\phi, \qquad p_z = p_T\sinh\eta, \qquad E = \sqrt{p_x^2 + p_y^2 + p_z^2 + m_\mu^2}.
```

Adding the two four-vectors gives the invariant mass of the pair, which is the same in every reference frame:

```math
m_{\mu\mu} = \sqrt{\left(E_1 + E_2\right)^2 - \left\lVert \vec{p}_1 + \vec{p}_2 \right\rVert^2}.
```

### 2. The model (`fitting.py`)

The Z line shape is a Breit–Wigner of mass $`M_Z`$ and full width $`\Gamma_Z`$. The detector smears it with a Gaussian resolution $`\sigma_\mathrm{res}`$, and the result is a *Voigt profile*:

```math
V(m) = \int_{-\infty}^{+\infty}
\underbrace{\frac{1}{\pi}\,\frac{\Gamma_Z/2}{\left(m' - M_Z\right)^2 + \left(\Gamma_Z/2\right)^2}}_{\text{Breit–Wigner (physics)}}
\;
\underbrace{\frac{1}{\sqrt{2\pi}\,\sigma_\mathrm{res}}\exp\!\left[-\frac{\left(m - m'\right)^2}{2\sigma_\mathrm{res}^2}\right]}_{\text{Gaussian (detector)}}
\,\mathrm{d}m' .
```

The resolution is fixed at $`\sigma_\mathrm{res} = 0.97`$ GeV from calibration. The Drell–Yan background falls exponentially, $`B(m) \propto e^{-\lambda m}`$. The expected number of events in bin $`i`$ is a mixture of the two:

```math
\mu_i(\boldsymbol{\theta}) = N_\mathrm{sig}\, s_i + N_\mathrm{bkg}\, b_i,
\qquad
s_i = \frac{\int_{\mathrm{bin}\ i} V\,\mathrm{d}m}{\int_{70}^{110} V\,\mathrm{d}m},
\qquad
b_i = \frac{\int_{\mathrm{bin}\ i} B\,\mathrm{d}m}{\int_{70}^{110} B\,\mathrm{d}m}.
```

Because $`\sum_i s_i = \sum_i b_i = 1`$, the yields $`N_\mathrm{sig}`$ and $`N_\mathrm{bkg}`$ are numbers of events in the 70–110 GeV window. The model has five free parameters:

```math
\boldsymbol{\theta} = \left(M_Z,\ \Gamma_Z,\ N_\mathrm{sig},\ N_\mathrm{bkg},\ \lambda\right).
```

### 3. The fit (`fitting.py`)

The data are the counts $`n_i`$ in 40 bins of 1 GeV. Many tail bins hold only a few events, so we use the Poisson form of $`\chi^2`$ (Baker–Cousins):

```math
\chi^2(\boldsymbol{\theta}) = 2\sum_{i=1}^{40}\left[\mu_i(\boldsymbol{\theta}) - n_i + n_i \ln\frac{n_i}{\mu_i(\boldsymbol{\theta})}\right].
```

This is $`-2\ln`$ of a likelihood ratio, so minimising it is the maximum-likelihood fit. MINUIT returns the best estimate $`\hat{\boldsymbol{\theta}}`$ and the minimum $`\chi^2_\mathrm{min} = \chi^2(\hat{\boldsymbol{\theta}})`$.

### 4. Where the "±" comes from (`fitting.py`)

The curvature of $`\chi^2`$ at the minimum is the Hessian matrix. Its inverse gives the covariance matrix, and from that the errors and correlations follow:

```math
H_{jk} = \left.\frac{\partial^2 \chi^2}{\partial\theta_j\,\partial\theta_k}\right|_{\hat{\boldsymbol{\theta}}},
\qquad
C = 2H^{-1},
\qquad
\sigma_j = \sqrt{C_{jj}},
\qquad
\rho_{jk} = \frac{C_{jk}}{\sigma_j\,\sigma_k}.
```

The same error can be read graphically. Scan one parameter, re-minimise all the others at every point, and find where the profile crosses one unit above the minimum:

```math
\Delta\chi^2(\theta_j) = \chi^2(\theta_j) - \chi^2_\mathrm{min} = 1
\quad\Longrightarrow\quad
\theta_j = \hat{\theta}_j \pm \sigma_j .
```

### 5. Should I trust this fit? The checklist (`diagnostics.py`)

A fit always returns numbers, even when it is wrong. With $`\nu = 40 - 5 = 35`$ degrees of freedom, the goodness of fit is the $`p`$-value

```math
p = \int_{\chi^2_\mathrm{min}}^{\infty} f_{\chi^2}(t;\nu)\,\mathrm{d}t ,
```

and the pull of each bin should behave like a draw from $`\mathcal{N}(0,1)`$:

```math
\mathrm{pull}_i = \frac{n_i - \mu_i(\hat{\boldsymbol{\theta}})}{\sqrt{\mu_i(\hat{\boldsymbol{\theta}})}} .
```

`fit_checklist()` runs five checks and prints ✓ or ⚠ for each:

| # | Check | ✓ if… |
|---|---|---|
| 1 | MINUIT converged | the minimum is valid and the EDM is tiny |
| 2 | The model describes the data | $`p > 0.05`$ |
| 3 | The pulls look Gaussian | at least 85 % of the pulls lie within $`\pm 2`$ (about 95 % expected) |
| 4 | The errors are symmetric | the Hesse $`\pm\sigma`$ and MINOS $`-\sigma_\mathrm{lo}/+\sigma_\mathrm{hi}`$ agree within 20 % |
| 5 | The result is stable | refitting with 2 GeV bins moves $`M_Z`$ and $`\Gamma_Z`$ by less than $`1\sigma`$ |

### 6. Topping up: how fast does the error shrink?

The statistical error of an estimator falls with the number of events $`N`$ in the fit window:

```math
\sigma \propto \frac{1}{\sqrt{N}}
\qquad\Longrightarrow\qquad
\frac{\sigma_\mathrm{after}}{\sigma_\mathrm{before}} = \sqrt{\frac{N_\mathrm{before}}{N_\mathrm{after}}} .
```

### 7. Checking the error with Monte Carlo (`montecarlo.py`)

The error $`\sigma_\mathrm{fit}`$ is a prediction: *if the experiment were repeated, the fitted value would scatter by this much.* With a simulation we can repeat it. We record $`K = 100`$ new copies of a run, fit each one, and take the spread of the results:

```math
\sigma_\mathrm{MC} = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(\hat{\theta}_k - \bar{\hat{\theta}}\right)^2},
\qquad
\frac{\delta\sigma_\mathrm{MC}}{\sigma_\mathrm{MC}} \approx \frac{1}{\sqrt{2(K-1)}} \approx 7\,\% .
```

An honest fit gives $`\sigma_\mathrm{fit} / \sigma_\mathrm{MC} \approx 1`$, within the 7 % precision of $`\sigma_\mathrm{MC}`$ itself.

### 8. Combining independent runs (`combine.py`)

For **independent** measurements $`x_i \pm \sigma_i`$, the PDG weighted average is

```math
w_i = \frac{1}{\sigma_i^2},
\qquad
\bar{x} = \frac{\sum_i w_i\, x_i}{\sum_i w_i},
\qquad
\sigma_{\bar{x}} = \frac{1}{\sqrt{\sum_i w_i}} .
```

Check consistency with

```math
\chi^2 = \sum_i w_i \left(x_i - \bar{x}\right)^2,
\qquad
\nu = N_\mathrm{meas} - 1,
\qquad
S = \sqrt{\chi^2/\nu}.
```

If $`S > 1`$, the measurements scatter more than their errors allow, and $`\sigma_{\bar{x}}`$ is multiplied by $`S`$. The formula assumes the inputs share **no** events; Part 5.2 of the notebook shows what happens when they do.

### 9. Statistical and systematic errors (Bonus)

Independent error sources add in quadrature:

```math
\sigma_\mathrm{tot} = \sqrt{\sigma_\mathrm{stat}^2 + \sigma_\mathrm{syst}^2} .
```

### The truth

The simulation uses the PDG values

```math
M_Z = 91.1876\ \mathrm{GeV}, \qquad \Gamma_Z = 2.4955\ \mathrm{GeV},
```

so you can check whether your measurement is right. In a real experiment you never get to do that.

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
| The equations in this README show as raw code | You are reading it outside GitHub. Open the repository page on github.com, where the LaTeX is rendered. |

Still stuck? Raise your hand, or open an *Issue* on this repository with the full error message.

---

*PHY408 Data Intensive Science, Xiamen University Malaysia. The code is released under the MIT licence.*
