"""
physics.py  --  from two muons to one number: the invariant mass
================================================================

Week 2 Lesson 2: build each muon's four-vector (E, px, py, pz) from
(pT, eta, phi), add the two four-vectors, and take

        m = sqrt( E^2 - |p|^2 )            (natural units, c = 1)
"""

import numpy as np

from .detector import M_MU


def four_vector(pt, eta, phi, mass=M_MU):
    """(pT, eta, phi, m)  ->  (E, px, py, pz) as NumPy arrays."""
    pt, eta, phi = np.asarray(pt), np.asarray(eta), np.asarray(phi)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E, px, py, pz


def invariant_mass(df):
    """Dimuon invariant mass (GeV) for every row of a cleaned DataFrame."""
    E1, px1, py1, pz1 = four_vector(df["mu1_pt"], df["mu1_eta"], df["mu1_phi"])
    E2, px2, py2, pz2 = four_vector(df["mu2_pt"], df["mu2_eta"], df["mu2_phi"])
    E, px, py, pz = E1 + E2, px1 + px2, py1 + py2, pz1 + pz2
    m2 = E**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.clip(m2, 0.0, None))
