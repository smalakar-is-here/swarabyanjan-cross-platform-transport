"""E4: decomposition of the source-to-target risk gap for a frozen source model f.

For target environment e (regime B) with R6 label Y = 1[L>=1] and construct-aligned soft
label G = g(L) (construct.py), and a proper loss R (NLL or Brier):

    R00 = R_e(f;        Y)      R6 evaluation
    RA  = R_e(f;        G)      construct aligned
    RP  = R_e(f^{pi(Y)}; Y)     prior corrected to the R6 prevalence
    RAP = R_e(f^{pi(G)}; G)     both
    RS  = R_S(f;        Y_S)    source test risk

f^{pi} is the Bayes odds correction of f from the source prevalence to pi (priors.prior_adjust).
The gap D = R00 - RS splits exactly into
    phi_A = 1/2[(R00 - RA) + (RP - RAP)]       construct component (two-player Shapley)
    phi_P = 1/2[(R00 - RP) + (RA - RAP)]       prior component
    rho   = RAP - RS                             residual (cue/covariate shift + remaining concept shift)
with phi_A + phi_P + rho = D. phi_A is identified under A-link; phi_P uses the oracle
prevalence (an upper bound on what unlabelled correction can recover; the deployable version
uses an MLLS estimate and is reported separately). rho is not split further from frozen
predictions: covariate and concept shift are not separately identified without target training.
AUROC is invariant to prior correction, so for AUROC only phi_A and rho are defined.
"""
from __future__ import annotations
import numpy as np
from . import metrics as M
from .priors import prior_adjust, mlls_em

LOSSES = {"nll": M.nll, "brier": M.brier}


def components(p, y, gsoft, pi_s, rs: dict, rs_auc: float) -> dict:
    out = {}
    pY, pG = float(np.mean(y)), float(np.mean(gsoft))
    fY, fG = prior_adjust(p, pi_s, pY), prior_adjust(p, pi_s, pG)
    for name, L in LOSSES.items():
        R00, RA, RP, RAP = L(y, p), L(gsoft, p), L(y, fY), L(gsoft, fG)
        phiA = 0.5 * ((R00 - RA) + (RP - RAP)); phiP = 0.5 * ((R00 - RP) + (RA - RAP))
        out.update({f"{name}_R00": R00, f"{name}_RA": RA, f"{name}_RP": RP, f"{name}_RAP": RAP,
                    f"{name}_RS": rs[name], f"{name}_gap": R00 - rs[name],
                    f"{name}_construct": phiA, f"{name}_prior": phiP, f"{name}_residual": RAP - rs[name]})
        # deployable prior correction (no target labels): MLLS estimate of the target prevalence
        pi_hat = mlls_em(p, pi_s)
        out[f"{name}_R00_mlls"] = L(y, prior_adjust(p, pi_s, pi_hat))
        out[f"{name}_RA_mlls"] = L(gsoft, prior_adjust(p, pi_s, pi_hat))
    a00, aA = M.auroc(y, p), M.auroc(gsoft, p)
    out.update({"auroc_R00": a00, "auroc_RA": aA, "auroc_RS": rs_auc, "auroc_gap": rs_auc - a00,
                "auroc_construct": aA - a00, "auroc_residual": rs_auc - aA,
                "prev_r6": pY, "prev_aligned": pG, "pi_hat_mlls": mlls_em(p, pi_s)})
    return out


def source_risk(p, y) -> tuple[dict, float]:
    return {k: L(y, p) for k, L in LOSSES.items()}, M.auroc(y, p)
