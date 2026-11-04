"""
conversion.py  --  from a raw record to one number: the amount in MYR
=====================================================================

In the Z lab we built four-vectors to get one number per event, the
invariant mass.  Here each payment is recorded in the card's currency,
together with the bank's reference exchange rate of that day, so

        amount_MYR = amount x fx_rate          (fx_rate = 1 for MYR cards)
"""

import numpy as np


def to_myr(df):
    """Amount in MYR for every row of a cleaned DataFrame."""
    return np.asarray(df["amount"] * df["fx_rate"], dtype=float)
