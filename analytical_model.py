"""
Analytical model: Captive Factoring and Supply Network Resilience
================================================================

Re-derives the closed-form Stackelberg equilibrium of a focal buyer with a
captive factoring subsidiary financing a capital-constrained supplier.

Structure of the model (two-stage Stackelberg game):

  Stage t=0  : the focal buyer B (controlling factor F with stake gamma) sets
               the baseline payment term T in (0, T_max] and the factoring
               discount rate r_f >= r_0.
  Stage t=1  : the supplier S chooses production q; its cash position after
               full-recourse factoring is M = B_0 + w q (1 - tau) - (k/2) q^2,
               where tau := r_f * T is the *effective discount cost*.
  Stage t=2  : a liquidity shock xi ~ U[0, xi_bar] hits. The supplier survives
               iff xi <= M.

Resilience  : R(gamma) = P(survive) * q*/q_FB, with q_FB = p/k (first-best).

Key corrections relative to the research outline (documented, not silent):
  1. Supplier objective uses E[(M - xi) 1{xi <= M}] = M^2/(2 xi_bar)
     (the outline wrote M^2/xi_bar; the factor 1/2 does not affect the FOC).
  2. r_f*(gamma) is NOT decreasing in gamma. It equals r_0 for gamma <= gamma_tilde
     and is strictly increasing for gamma > gamma_tilde (rent extraction).
  3. The payment-term cap T_max does NOT unconditionally improve resilience.
     Because d(tau*)/dT_max = (gamma r_0 - r_B)/(2 gamma) < 0 under r_B > gamma r_0,
     tightening the cap RAISES the discount burden (regulatory evasion). Only a
     combined instrument (cap + factoring-rate ceiling) raises resilience.

Outputs (written to ../results and ../figures):
  fig1_lemma1.png     supplier reaction function q*(tau), buffer M*(tau)
  fig2_prop1.png      tau*(gamma), r_f*(gamma) with threshold gamma_tilde
  fig3_prop2.png      inverted-U resilience R(gamma) with critical gamma*
  fig4_prop3.png      payment-cap evasion + combined-governance comparison
  fig5_decomp.png     liquidity index Omega(gamma) decomposition (buffer vs drain)
  analytical_results.json   all numerical quantities for the paper
"""

import json
import os

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "serif",
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESDIR = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(RESDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# ----------------------------------------------------------------------
# Model primitives
# ----------------------------------------------------------------------
PARAMS = dict(
    p=0.75,       # retail price (normalised)
    w=0.60,       # wholesale price
    k=1.00,       # quadratic cost coefficient, c(q) = (k/2) q^2
    B0=0.08,      # supplier initial working capital, 0 < B0 < w^2/(2k)
    r0=0.04,      # factor's external (risk-free) funding cost
    rB=0.10,      # buyer's opportunity cost of deferring payables, rB > r0
    Tmax=1.00,    # regulatory payment-term ceiling (years)
    xi_bar=0.22,  # upper bound of liquidity shock U[0, xi_bar]
    beta=3.40,    # saturation speed of credit penetration lambda(gamma)
)

# Parameter-consistency checks (from the outline's own assumptions)
p, w, k, B0, r0, rB, Tmax, xi_bar, beta = (PARAMS[k] for k in
    ("p", "w", "k", "B0", "r0", "rB", "Tmax", "xi_bar", "beta"))
assert 0 < B0 < w ** 2 / (2 * k), "B0 must lie in (0, w^2/(2k))"
assert xi_bar > w ** 2 / (2 * k), "xi_bar must be large enough for interior default"
assert rB > r0, "buyer's deferral benefit must exceed factor's funding cost"


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def credit_penetration(gamma):
    """Credit penetration lambda(gamma) = 1 - exp(-beta*gamma)."""
    return 1.0 - np.exp(-beta * gamma)


def K_const():
    """K := p - w + rB * w * Tmax > 0."""
    return p - w + rB * w * Tmax


def gamma_tilde():
    """Threshold ownership above which rent extraction dominates.

    gamma_tilde = K / (w * (1 - r0 * Tmax))
    """
    return K_const() / (w * (1.0 - r0 * Tmax))


def tau_hat(gamma):
    """Interior effective discount cost (unconstrained, gamma > gamma_tilde).

    tau_hat = 1/2 - (p-w)/(2 gamma w) + (gamma r0 - rB) Tmax / (2 gamma)
    """
    return 0.5 - (p - w) / (2.0 * gamma * w) + (gamma * r0 - rB) * Tmax / (2.0 * gamma)


def tau_star(gamma):
    """Equilibrium effective discount cost with the participation constraint."""
    g_t = gamma_tilde()
    if gamma <= g_t:
        return r0 * Tmax            # cost-priced factoring (pure credit pass-through)
    return max(tau_hat(gamma), r0 * Tmax)


def liquidity_index(gamma):
    """Omega(gamma) = lambda(gamma) * (1 - tau*(gamma))."""
    return credit_penetration(gamma) * (1.0 - tau_star(gamma))


def q_star(gamma):
    """Equilibrium supplier production q*(gamma) = w * Omega / k."""
    return w * liquidity_index(gamma) / k


def buffer(gamma):
    """Equilibrium financial buffer M*(gamma) = B0 + w^2 Omega^2 / (2k)."""
    Om = liquidity_index(gamma)
    return B0 + w ** 2 * Om ** 2 / (2.0 * k)


def resilience(gamma):
    """R(gamma) = [M*/xi_bar] * [q*/q_FB], q_FB = p/k."""
    q_fb = p / k
    return (buffer(gamma) / xi_bar) * (q_star(gamma) / q_fb)


def resilience_of_Omega(Om):
    """R as an explicit cubic in Omega (used for the monotonicity lemma)."""
    return (w * B0 / (p * xi_bar)) * Om + (w ** 3 / (2.0 * k * p * xi_bar)) * Om ** 3


def dlogOmega(gamma):
    """d log Omega / d gamma = lambda'/lambda - rent-extraction term.

    Valid for gamma > gamma_tilde (the high-ownership regime).
    """
    lam = credit_penetration(gamma)
    dlam = beta * np.exp(-beta * gamma)          # lambda'(gamma)
    enable = dlam / lam                          # marginal credit-enabling (+)
    K = K_const()
    drain = K / (gamma * (gamma * w * (1.0 - r0 * Tmax) + K))   # marginal rent drain (-)
    return enable - drain


def solve_gamma_star():
    """Unique critical gamma* in (gamma_tilde, 1) solving d log Omega / d gamma = 0."""
    g_t = gamma_tilde()
    lo = g_t + 1e-6
    hi = 1.0 - 1e-6
    # guard against no-internal-root (boundary solutions)
    if dlogOmega(lo) <= 0:
        return g_t                      # peak at the boundary
    if dlogOmega(hi) >= 0:
        return 1.0
    return brentq(dlogOmega, lo, hi, xtol=1e-12)


# ----------------------------------------------------------------------
# Verify the closed-form quantities
# ----------------------------------------------------------------------
def verify():
    g_t = gamma_tilde()
    g_star = solve_gamma_star()
    tau0 = tau_star(0.5 * g_t)
    tau_hi = tau_star(0.9)
    tau_1 = tau_star(1.0)
    checks = {
        "gamma_tilde": g_t,
        "gamma_star": g_star,
        "tau*(0.5 gamma_tilde) == r0 Tmax": abs(tau0 - r0 * Tmax) < 1e-9,
        "tau* increasing in gamma (high regime)": tau_hi < tau_1,
        "dlogOmega(gamma*) ~ 0": abs(dlogOmega(g_star)) < 1e-6,
        "resilience concave around gamma*": (
            resilience(g_star) >= resilience(g_t)
            and resilience(g_star) >= resilience(1.0)),
        "R increasing before gamma*": resilience(g_t) < resilience(g_star),
        "R decreasing after gamma*": resilience(g_star) > resilience(1.0),
    }
    return checks


# ----------------------------------------------------------------------
# Figure 1: Lemma 1 -- supplier reaction
# ----------------------------------------------------------------------
def fig1_lemma1():
    tau = np.linspace(0.0, 0.95 * r0 * Tmax + 0.9 * (0.9), 400)
    tau = np.linspace(0.0, 0.9, 400)
    q = w * (1 - tau) / k
    M = B0 + w ** 2 * (1 - tau) ** 2 / (2 * k)

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    ax1.plot(tau, q, color="#1f77b4", lw=2, label=r"$q^*(\tau)=w(1-\tau)/k$")
    ax1.set_xlabel(r"Effective discount cost $\tau = r_f T$")
    ax1.set_ylabel(r"Production $q^*(\tau)$", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(tau, M, color="#d62728", lw=2, ls="--", label=r"$M^*(\tau)$")
    ax2.set_ylabel(r"Financial buffer $M^*(\tau)$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # mark the no-factoring limited quantity q_NF = sqrt(2 B0 / k)
    q_nf = np.sqrt(2 * B0 / k)
    ax1.axhline(q_nf, color="grey", lw=1, ls=":")
    ax1.text(0.62, q_nf + 0.01, r"$q^{\mathrm{NF}}=\sqrt{2B_0/k}$", color="grey", fontsize=9)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_lemma1.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: Proposition 1 -- optimal contract
# ----------------------------------------------------------------------
def fig2_prop1():
    g = np.linspace(0.01, 1.0, 600)
    tau = np.array([tau_star(x) for x in g])
    rf = tau / Tmax
    g_t = gamma_tilde()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)

    ax1.plot(g, tau, lw=2, color="#2ca02c")
    ax1.axvline(g_t, color="grey", ls="--", lw=1)
    ax1.set_xlabel(r"Ownership stake $\gamma$")
    ax1.set_ylabel(r"Effective discount cost $\tau^*(\gamma)$")
    ax1.set_title(r"(a) $\tau^*(\gamma)$", loc="left", fontsize=11)
    ax1.text(g_t + 0.01, tau.max() * 0.35, r"$\tilde{\gamma}$",
             color="grey", fontsize=10)

    ax2.plot(g, rf, lw=2, color="#ff7f0e")
    ax2.axhline(r0, color="grey", ls=":", lw=1)
    ax2.axvline(g_t, color="grey", ls="--", lw=1)
    ax2.set_xlabel(r"Ownership stake $\gamma$")
    ax2.set_ylabel(r"Factoring rate $r_f^*(\gamma)$")
    ax2.set_title(r"(b) $r_f^*(\gamma)$", loc="left", fontsize=11)
    ax2.text(g_t + 0.01, r0 + 0.002, r"$r_0$ (cost price)", color="grey", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_prop1.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Proposition 2 -- inverted-U resilience
# ----------------------------------------------------------------------
def fig3_prop2():
    g = np.linspace(0.01, 1.0, 800)
    R = np.array([resilience(x) for x in g])
    g_t = gamma_tilde()
    g_star = solve_gamma_star()
    R_star = resilience(g_star)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(g, R, lw=2.4, color="#9467bd")
    ax.axvline(g_t, color="grey", ls="--", lw=1)
    ax.axvline(g_star, color="#d62728", ls="--", lw=1)

    ax.fill_between(g, 0, R, where=(g <= g_t),
                    color="#1f77b4", alpha=0.08)
    ax.fill_between(g, 0, R, where=(g > g_t),
                    color="#d62728", alpha=0.08)

    ax.plot([g_star], [R_star], "o", color="#d62728", ms=7, zorder=5)

    ax.set_xlabel(r"Ownership stake $\gamma$")
    ax.set_ylabel(r"Supply-chain resilience $R(\gamma)$")
    ax.text(g_t * 0.42, R.max() * 0.82, "Liquidity\nbuffer", color="#1f77b4",
            fontsize=10, ha="center")
    ax.text(0.5 * (g_star + 1.0), R.min() * 1.6, "Systemic\nfragility",
            color="#d62728", fontsize=10, ha="center")
    ax.text(g_star + 0.012, R_star - 0.02 * R.max(), r"$\gamma^*$",
            color="#d62728", fontsize=11)
    ax.text(g_t - 0.012, R.max() * 0.5, r"$\tilde{\gamma}$",
            color="grey", fontsize=10, ha="right", rotation=90)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_prop2.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: Proposition 3 -- payment-cap evasion vs combined governance
# ----------------------------------------------------------------------
def fig4_prop3():
    g = np.linspace(0.01, 1.0, 600)

    def R_under(Tmax_v, rbar=None):
        """Resilience under payment cap Tmax_v and optional rate ceiling rbar."""
        def Om_of(x):
            lam = 1.0 - np.exp(-beta * x)
            # recompute tau* with this Tmax_v
            K = p - w + rB * w * Tmax_v
            g_t = K / (w * (1.0 - r0 * Tmax_v))
            if x <= g_t:
                tau = r0 * Tmax_v
            else:
                tau = 0.5 - (p - w) / (2 * x * w) + (x * r0 - rB) * Tmax_v / (2 * x)
            tau = max(tau, r0 * Tmax_v)
            if rbar is not None:
                tau = min(tau, rbar * Tmax_v)
            return lam * (1.0 - tau)

        R = []
        for x in g:
            Om = Om_of(x)
            R.append((B0 + w ** 2 * Om ** 2 / (2 * k)) / xi_bar * (w * Om / k) / (p / k))
        return np.array(R)

    R_base = R_under(Tmax)
    R_cap = R_under(0.5)                       # cap tightened alone (evasion)
    R_comb = R_under(0.5, rbar=0.10)           # cap + rate ceiling (combined)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(g, R_base, lw=2, color="#9467bd",
            label=r"$T_{\max}=1.0$ (laissez-faire)")
    ax.plot(g, R_cap, lw=2, color="#d62728", ls="--",
            label=r"$T_{\max}=0.5$ (cap only: evasion)")
    ax.plot(g, R_comb, lw=2.4, color="#2ca02c",
            label=r"$T_{\max}=0.5,\;r_f\leq0.10$ (combined)")
    ax.set_xlabel(r"Ownership stake $\gamma$")
    ax.set_ylabel(r"Resilience $R(\gamma)$")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_prop3.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5: decomposition of Omega(gamma) into buffer vs drain forces
# ----------------------------------------------------------------------
def fig5_decomp():
    g = np.linspace(0.01, 1.0, 600)
    lam = credit_penetration(g)
    one_minus_tau = np.array([1.0 - tau_star(x) for x in g])
    Om = lam * one_minus_tau
    g_t = gamma_tilde()
    g_star = solve_gamma_star()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(g, lam, lw=2, color="#1f77b4", label=r"$\lambda(\gamma)$ (penetration)")
    ax.plot(g, one_minus_tau, lw=2, color="#d62728",
            label=r"$1-\tau^*(\gamma)$ (discount burden)")
    ax.plot(g, Om, lw=2.4, color="#2ca02c",
            label=r"$\Omega(\gamma)=\lambda(1-\tau^*)$")
    ax.axvline(g_t, color="grey", ls="--", lw=1)
    ax.axvline(g_star, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"Ownership stake $\gamma$")
    ax.set_ylabel("Liquidity index components")
    ax.text(g_star + 0.01, 0.05, r"$\gamma^*$", color="grey", fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_decomp.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    checks = verify()

    # quantitative summary for the paper
    g_t = gamma_tilde()
    g_star = solve_gamma_star()
    summary = {
        "params": PARAMS,
        "gamma_tilde": g_t,
        "gamma_star": g_star,
        "tau_star_at_gamma_star": tau_star(g_star),
        "rf_star_at_gamma_star": tau_star(g_star) / Tmax,
        "rf_star_at_gamma_1": tau_star(1.0) / Tmax,
        "resilience_peak": resilience(g_star),
        "resilience_at_gamma_1": resilience(1.0),
        "resilience_at_gamma_tilde": resilience(g_t),
        "resilience_drop_ratio": resilience(g_star) / resilience(1.0),
        "lambda_at_gamma_star": credit_penetration(g_star),
        "Omega_at_gamma_star": liquidity_index(g_star),
        "buffer_at_gamma_star": buffer(g_star),
        "q_star_at_gamma_star": q_star(g_star),
        "q_FB": p / k,
        "d_tau_dTmax_analytical": (gamma_tilde() * r0 - rB) / (2 * gamma_tilde()),
        "checks": checks,
    }
    with open(os.path.join(RESDIR, "analytical_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    fig1_lemma1()
    fig2_prop1()
    fig3_prop2()
    fig4_prop3()
    fig5_decomp()

    print("=== Analytical model verification ===")
    for key, v in checks.items():
        print(f"  {key:50s}: {v}")
    print()
    print(f"gamma_tilde        = {g_t:.4f}")
    print(f"gamma_star         = {g_star:.4f}")
    print(f"tau*(gamma_star)   = {tau_star(g_star):.4f}")
    print(f"r_f*(gamma_star)   = {tau_star(g_star)/Tmax:.4f}")
    print(f"r_f*(1.0)          = {tau_star(1.0)/Tmax:.4f}")
    print(f"R(gamma_star)      = {resilience(g_star):.4f}  (peak)")
    print(f"R(1.0)             = {resilience(1.0):.4f}")
    print(f"d tau*/d Tmax      = {(g_t*r0 - rB)/(2*g_t):.4f}  (negative -> evasion)")
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
