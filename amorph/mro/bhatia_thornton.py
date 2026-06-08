"""
bhatia_thornton.py — Bhatia–Thornton structure-factor decomposition.

Separates topology from chemistry for an effective binary A/B:
  S_NN(Q) number–number (network topology),
  S_CC(Q) concentration–concentration (chemical (de)mixing; S_CC>c_Ac_B ⇒
          clustering / phase separation),
  S_NC(Q) cross term.

Implementation: relabel the system into the two groups, compute the (time-
averaged) effective-binary partial g(r), sine-transform to Faber–Ziman partial
S_ij(Q), then combine:
  S_NN = c_A²S_AA + c_B²S_BB + 2c_Ac_B S_AB
  S_NC = c_Ac_B[c_A(S_AA−S_AB) − c_B(S_BB−S_AB)]
  S_CC = c_Ac_B[1 + c_Ac_B(S_AA+S_BB−2S_AB)]

For SiCN the natural split is network (Si+N) vs free carbon (C).
"""
from __future__ import annotations
import numpy as np

from ..sro.rdf import partial_rdf
from ..core._compat import trapezoid


def _sine_transform(r, g, rho, ci, cj, q, lorch=True):
    """Faber–Ziman partial S_ij(Q) (without the Kronecker δ, added by caller)."""
    w = np.sinc(r / r[-1]) if lorch else np.ones_like(r)   # sinc(x)=sin(πx)/(πx)
    pre = 4.0 * np.pi * rho * np.sqrt(ci * cj)
    integrand = r * (g - 1.0) * w
    S = np.empty_like(q)
    for k, qq in enumerate(q):
        if qq < 1e-9:
            S[k] = pre * trapezoid(integrand * r, r)
        else:
            S[k] = pre * trapezoid(integrand * np.sin(qq * r) / qq, r)
    return S


def bhatia_thornton(traj, group_A, group_B, r_max=10.0, nbins=500,
                    q=None, n_blocks=5, lorch=True):
    """Time-averaged Bhatia–Thornton components for groups A vs B.

    Parameters
    ----------
    group_A, group_B : list[str]   element symbols forming each pseudo-species.
    q : array | None   Q grid (Å⁻¹); default linspace(0.2, 20, 400).

    Returns
    -------
    dict: q, S_NN, S_CC, S_NC, S_AA, S_BB, S_AB, c_A, c_B, S_CC_ideal, labels
    """
    if q is None:
        q = np.linspace(0.2, 20.0, 400)
    mapping = {**{e: "A" for e in group_A}, **{e: "B" for e in group_B}}
    rtraj = [fr.relabeled(mapping) for fr in traj]

    R = partial_rdf(rtraj, pairs=[("A", "A"), ("A", "B"), ("B", "B")],
                    r_max=r_max, nbins=nbins, n_blocks=n_blocks)
    r = R["r"]
    rho = R["rho0"]
    cA = R["rho"].get("A", 0.0) / rho if rho > 0 else 0.0
    cB = R["rho"].get("B", 0.0) / rho if rho > 0 else 0.0
    if cA < 1e-9 or cB < 1e-9:
        raise ValueError("One pseudo-binary group is empty — check group_A/group_B.")

    S_AA = 1.0 + _sine_transform(r, R[("A", "A")]["g"], rho, cA, cA, q, lorch)
    S_BB = 1.0 + _sine_transform(r, R[("B", "B")]["g"], rho, cB, cB, q, lorch)
    S_AB = _sine_transform(r, R[("A", "B")]["g"], rho, cA, cB, q, lorch)

    S_NN = cA**2 * S_AA + cB**2 * S_BB + 2 * cA * cB * S_AB
    S_NC = cA * cB * (cA * (S_AA - S_AB) - cB * (S_BB - S_AB))
    S_CC = cA * cB * (1.0 + cA * cB * (S_AA + S_BB - 2 * S_AB))

    return dict(q=q, S_NN=S_NN, S_CC=S_CC, S_NC=S_NC,
                S_AA=S_AA, S_BB=S_BB, S_AB=S_AB,
                c_A=cA, c_B=cB, S_CC_ideal=cA * cB,
                label_A="+".join(group_A), label_B="+".join(group_B))
