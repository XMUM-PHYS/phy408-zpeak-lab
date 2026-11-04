"""
detector.py  --  our pretend CMS detector
=========================================

You do NOT need to edit this file.  Think of it as the experiment:
every time you call ``take_data(run=..., n_events=...)`` the "detector"
records a new batch (a *run*) of dimuon events and writes it to
``data/raw/run<k>.csv``.

What is inside (so you know what to look for when you clean the data):

* Z -> mu mu decays: Breit-Wigner mass (Week 3 Lesson 2), isotropic decay
  in the Z rest frame and a boost to the lab (Week 2 Lesson 2), and a
  muon pT resolution of 1.5 %.
* Drell-Yan continuum (opposite-sign muons, falling exponential in mass).
* Fake "QCD" pairs (same-sign muons): these must be removed by a cut.
* Real-life data problems, injected on purpose:
    - duplicated events (the same run/event number written twice),
    - missing values (NaN),
    - read-out errors (pT = -999 or absurdly large),
    - luminosity sections flagged as bad by the data-quality team,
    - muons outside the tracker acceptance or below the trigger threshold.

The true values used by the simulation are the PDG values below, so at
the end of the lab you can check whether your measurement is right.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- "truth" used by the simulation (PDG values, GeV) ----------------
M_Z_TRUE = 91.1876
GAMMA_Z_TRUE = 2.4955
M_MU = 0.1057

PT_RESOLUTION = 0.015          # 1.5 % relative muon pT resolution
BEAM_ENERGY = 4000.0           # 8 TeV collisions -> 4 TeV per beam

# composition of a raw batch (before cleaning)
FRAC_Z, FRAC_DY, FRAC_QCD = 0.78, 0.12, 0.10

# the three runs used in the lab: each one is 4x bigger than the last
RUN_PLAN = {1: 1_000, 2: 4_000, 3: 16_000}

DEFAULT_SEED = 31              # the "class" data set; use your student ID for your own

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ----------------------------------------------------------------------
# small kinematics helpers
# ----------------------------------------------------------------------
def _boost(E, px, py, pz, bx, by, bz):
    """Lorentz-boost the four-vector (E, p) by the velocity vector beta."""
    b2 = bx**2 + by**2 + bz**2
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = bx * px + by * py + bz * pz
    # (gamma - 1)/beta^2, written safely for beta -> 0
    g2 = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    E_new = gamma * (E + bp)
    k = g2 * bp + gamma * E
    return E_new, px + k * bx, py + k * by, pz + k * bz


def _decay_to_muons(rng, m):
    """Parent of mass m -> mu+ mu-, returned as (pt, eta, phi) for both muons."""
    n = m.size
    # parent kinematics in the lab
    pt_parent = rng.exponential(7.0, n)                 # GeV
    y_parent = rng.normal(0.0, 1.6, n)                  # rapidity
    phi_parent = rng.uniform(-np.pi, np.pi, n)
    mT = np.sqrt(m**2 + pt_parent**2)
    E_P = mT * np.cosh(y_parent)
    px_P = pt_parent * np.cos(phi_parent)
    py_P = pt_parent * np.sin(phi_parent)
    pz_P = mT * np.sinh(y_parent)
    bx, by, bz = px_P / E_P, py_P / E_P, pz_P / E_P

    # isotropic two-body decay in the rest frame (Week 2 Lesson 2, slide 48)
    p_star = np.sqrt(np.clip(m**2 / 4.0 - M_MU**2, 0.0, None))
    E_star = np.sqrt(p_star**2 + M_MU**2)
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi = rng.uniform(-np.pi, np.pi, n)
    nx, ny, nz = sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t

    muons = []
    for sign in (+1.0, -1.0):
        E, px, py, pz = _boost(E_star, sign * p_star * nx, sign * p_star * ny,
                               sign * p_star * nz, bx, by, bz)
        pt = np.hypot(px, py)
        eta = np.arcsinh(pz / pt)
        phi_mu = np.arctan2(py, px)
        # detector resolution: smear pT
        pt = pt * (1.0 + PT_RESOLUTION * rng.normal(size=n))
        muons.append((pt, eta, phi_mu))
    return muons


def _sample_breit_wigner(rng, n, lo=60.0, hi=130.0):
    """Non-relativistic Breit-Wigner (Cauchy), truncated to [lo, hi] GeV."""
    out = np.empty(0)
    while out.size < n:
        m = M_Z_TRUE + 0.5 * GAMMA_Z_TRUE * rng.standard_cauchy(2 * n)
        out = np.concatenate([out, m[(m > lo) & (m < hi)]])
    return out[:n]


def _sample_exponential(rng, n, slope, lo=60.0, hi=130.0):
    """Falling exponential exp(-slope * m), truncated to [lo, hi] GeV."""
    out = np.empty(0)
    while out.size < n:
        m = lo + rng.exponential(1.0 / slope, 2 * n)
        out = np.concatenate([out, m[m < hi]])
    return out[:n]


# ----------------------------------------------------------------------
# the function you call
# ----------------------------------------------------------------------
def take_data(run, n_events=None, seed=DEFAULT_SEED, outdir=None, verbose=True):
    """
    Record one run of dimuon data and save it as a CSV file.

    Parameters
    ----------
    run      : int   run number (1, 2, 3, ...)
    n_events : int   number of events to record (default: RUN_PLAN[run])
    seed     : int   random seed. Use your student ID to get your own data!
    outdir   : path  where to write the file (default: data/raw/)

    Returns
    -------
    pathlib.Path of the CSV file that was written.
    """
    if n_events is None:
        n_events = RUN_PLAN[run]
    rng = np.random.default_rng([int(seed), int(run)])

    n_z = rng.binomial(n_events, FRAC_Z)
    n_dy = rng.binomial(n_events - n_z, FRAC_DY / (1 - FRAC_Z))
    n_qcd = n_events - n_z - n_dy

    masses = np.concatenate([
        _sample_breit_wigner(rng, n_z),
        _sample_exponential(rng, n_dy, slope=0.045),
        _sample_exponential(rng, n_qcd, slope=0.030),
    ])
    same_sign = np.concatenate([np.zeros(n_z + n_dy, bool), np.ones(n_qcd, bool)])

    (pt1, eta1, phi1), (pt2, eta2, phi2) = _decay_to_muons(rng, masses)
    q1 = rng.choice([-1, 1], n_events)
    q2 = np.where(same_sign, q1, -q1)

    df = pd.DataFrame({
        "run": run,
        "lumi": rng.integers(1, 200, n_events),
        "event": rng.choice(10**8, n_events, replace=False) + 1,
        "lumi_ok": 1,
        "mu1_pt": pt1, "mu1_eta": eta1, "mu1_phi": phi1, "mu1_charge": q1,
        "mu2_pt": pt2, "mu2_eta": eta2, "mu2_phi": phi2, "mu2_charge": q2,
    })

    # ---- inject the data-quality problems ----------------------------
    # 1) bad luminosity sections (whole lumi blocks flagged by DQ shifters)
    bad_lumis = rng.choice(np.arange(1, 200), 6, replace=False)
    df.loc[df["lumi"].isin(bad_lumis), "lumi_ok"] = 0
    # 2) read-out errors: sentinel -999 or absurdly large pT
    idx = rng.choice(n_events, max(1, n_events // 200), replace=False)
    df.loc[idx[: len(idx) // 2], "mu1_pt"] = -999.0
    df.loc[idx[len(idx) // 2:], "mu2_pt"] = 1.0e6
    # 3) missing values
    for col in ["mu1_pt", "mu2_eta", "mu2_phi", "mu1_charge"]:
        idx = rng.choice(n_events, max(1, n_events // 400), replace=False)
        df.loc[idx, col] = np.nan
    # 4) duplicated events (same run/event written twice by the DAQ)
    dup = df.sample(frac=0.02, random_state=int(rng.integers(1 << 31)))
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    df = df.round({"mu1_pt": 4, "mu2_pt": 4, "mu1_eta": 5, "mu2_eta": 5,
                   "mu1_phi": 5, "mu2_phi": 5})

    outdir = Path(outdir) if outdir is not None else DATA_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"run{run}.csv"
    df.to_csv(path, index=False)
    if verbose:
        print(f"[detector] run {run}: recorded {len(df):,} rows -> {path.name}")
    return path
