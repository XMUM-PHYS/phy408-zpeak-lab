"""
bank.py  --  our pretend payment gateway
========================================

You do NOT need to edit this file.  Think of it as the bank's system:
every time you call ``take_data(batch=...)`` the gateway hands you a new
extract (a *batch*) of card payments made at a small online shop, written
to ``data/raw/fraud_batch<k>.csv``.

The story (all numbers are simulated, the shop is fictional):

* Normal customers buy digital goods (e-books, top-ups, apps). Their
  payment amounts form a smooth, falling distribution.
* A card-testing bot is attacking the shop. It holds a list of stolen card
  numbers and checks which ones still work with tiny purchases of about
  RM 4.90. To dodge simple "block RM 4.90" rules, it randomises the amount
  a little around its target.
* The stolen cards are foreign (USD, EUR). Each record carries the amount
  in the card's currency and the bank's reference exchange rate at the
  moment of payment (`fx_rate`). The card network, however, charged at its own rate with
  a variable markup, so converting back to MYR blurs each small foreign
  amount by about RM 0.15 -- exactly like a detector resolution.
* Real-life data problems, injected on purpose:
    - duplicated payments (the gateway retried and logged twice),
    - missing values (NaN) in the amount or the currency,
    - test payments made by the shop's own developers (test mode),
    - gateway errors: amount 0.00 or 99999999,
    - refunds mixed in with purchases,
    - declined payments (many from the bot: most stolen cards are dead).

The true values used by the simulation are below, so at the end of the lab
you can check whether your measurement is right.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- "truth" used by the simulation -------------------------------------
BOT_TARGET_TRUE = 4.90         # RM, the amount the bot aims for
BOT_SPREAD_TRUE = 0.25         # RM, how much the bot randomises the amount
FRAUD_RATE_TRUE = 5.0          # %, share of approved purchases that are card tests

FX_BLUR = 0.15                 # RM: network rate vs the bank's reference rate (~3 % at RM 5)
FX_DRIFT = 0.01                # reference rates move by about 1 % during the month
LEGIT_MIN, LEGIT_MEAN = 1.00, 11.0   # RM, normal purchases: 1 + exponential(11)
CARD_LIMIT = 50_000.0          # no single payment can exceed this

# typical exchange rates to MYR (simulated values)
FX_TABLE = {"MYR": 1.00, "SGD": 3.30, "USD": 4.40, "EUR": 4.80}
LEGIT_CURRENCIES = (["MYR", "SGD", "USD", "EUR"], [0.75, 0.10, 0.10, 0.05])
FRAUD_CURRENCIES = (["USD", "EUR"], [0.70, 0.30])

# raw rows per batch: each one is 4x bigger than the last
# (the bank rolls the monitoring out from a pilot, to a region, to the country)
RUN_PLAN = {1: 5_000, 2: 20_000, 3: 80_000}
DEFAULT_SEED = 115            # the "class" data set; use your student ID for your own

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _record(rng, amount_myr, currencies, blurred=True):
    """Pick a card currency and write the amount as the card network would."""
    n = amount_myr.size
    cur = rng.choice(currencies[0], size=n, p=currencies[1])
    # reference rate at the moment of payment (MYR cards: exactly 1)
    table = np.array([FX_TABLE[c] for c in cur])
    ref = np.where(cur == "MYR", 1.0, table * (1.0 + FX_DRIFT * rng.normal(size=n)))
    # the network's rate differs from our reference rate by a random markup
    blur = np.where(cur == "MYR", 0.0, FX_BLUR) if blurred else np.zeros(n)
    amount = np.round((amount_myr + blur * rng.normal(size=n)) / ref, 2)   # charged in cents
    return cur, np.round(ref, 6), amount


def _legit_amounts(rng, n):
    return LEGIT_MIN + rng.exponential(LEGIT_MEAN, n)


def _bot_amounts(rng, n):
    return rng.normal(BOT_TARGET_TRUE, BOT_SPREAD_TRUE, n)


def take_data(batch, n_rows=None, seed=DEFAULT_SEED, outdir=None, verbose=True, save=True):
    """
    Receive one batch of card payments.

    Parameters
    ----------
    batch  : int   batch number (1, 2, 3, ...)
    n_rows : int   approximate number of raw rows (default: RUN_PLAN[batch])
    seed   : int   random seed. Use your student ID to get your own data!
    outdir : path  where to write the file (default: data/raw/)
    save   : bool  False returns the DataFrame without writing a file (used by the toys)

    Returns
    -------
    pathlib.Path of the CSV file (or the DataFrame if save=False).
    """
    if n_rows is None:
        n_rows = RUN_PLAN[batch]
    rng = np.random.default_rng([int(seed), 7, int(batch)])

    # approved purchases: a fraction FRAUD_RATE_TRUE of them are card tests
    n_approved = int(round(n_rows / 1.20))
    n_fraud = rng.binomial(n_approved, FRAUD_RATE_TRUE / 100)
    n_legit = n_approved - n_fraud
    # declined attempts: most stolen cards are dead (bot approval rate ~35 %)
    n_dec_fraud = rng.poisson(n_fraud * 0.65 / 0.35)
    n_dec_legit = rng.poisson(0.03 * n_legit)
    n_refund = rng.poisson(0.03 * n_legit)
    n_test = rng.poisson(0.03 * n_approved)

    parts = []

    def add(amount_myr, currencies, kind, status, is_test=0, blurred=True):
        cur, ref, amt = _record(rng, amount_myr, currencies, blurred)
        parts.append(pd.DataFrame({"type": kind, "status": status, "is_test": is_test,
                                   "currency": cur, "amount": amt, "fx_rate": ref}))

    add(_legit_amounts(rng, n_legit), LEGIT_CURRENCIES, "purchase", "approved")
    add(_bot_amounts(rng, n_fraud), FRAUD_CURRENCIES, "purchase", "approved")
    add(_legit_amounts(rng, n_dec_legit), LEGIT_CURRENCIES, "purchase", "declined")
    add(_bot_amounts(rng, n_dec_fraud), FRAUD_CURRENCIES, "purchase", "declined")
    add(_legit_amounts(rng, n_refund), LEGIT_CURRENCIES, "refund", "approved")
    # the developers always test with exactly RM 5.00 in MYR
    add(np.full(n_test, 5.00), (["MYR"], [1.0]), "purchase", "approved", is_test=1, blurred=False)

    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    df.insert(0, "batch", batch)
    df.insert(1, "txn_id", rng.choice(10**9, n, replace=False) + 10**9)
    start = np.datetime64("2026-11-01T00:00:00")
    df.insert(2, "time", (start + rng.integers(0, 30 * 86400, n).astype("timedelta64[s]")).astype(str))
    df.insert(3, "card_id", [f"{x:08x}" for x in rng.integers(0, 2**32, n)])

    # ---- inject the data-quality problems ------------------------------
    # gateway errors: 0.00 or 99999999
    idx = rng.choice(n, max(2, n // 200), replace=False)
    df.loc[idx[: len(idx) // 2], "amount"] = 0.0
    df.loc[idx[len(idx) // 2:], "amount"] = 99_999_999.0
    # missing values
    for col in ["amount", "currency", "fx_rate"]:
        idx = rng.choice(n, max(1, n // 400), replace=False)
        df.loc[idx, col] = np.nan
    # duplicated payments (gateway retry logged twice)
    dup = df.sample(frac=0.02, random_state=int(rng.integers(1 << 31)))
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    if not save:
        return df
    outdir = Path(outdir) if outdir is not None else DATA_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"fraud_batch{batch}.csv"
    df.to_csv(path, index=False)
    if verbose:
        print(f"[bank] batch {batch}: received {len(df):,} payment records -> {path.name}")
    return path
