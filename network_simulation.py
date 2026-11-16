"""
Agent-on-Networks simulation: captive factoring and supply-network resilience
============================================================================

Extends the analytical Stackelberg model to a directed supply network
G = (V, E) with a focal buyer B as the terminal sink. Three classic topologies
are generated (Erdos-Renyi, Scale-Free, Small-World) to test how network
structure moderates the liquidity-buffer / systemic-fragility trade-off.

Mechanics (per node i, mirroring the analytical core):
  - credit penetration decays with hop distance d(b,i):
        Omega_i = lambda(gamma) * exp(-mu * d(b,i)) * (1 - tau*(gamma))
  - production q_i* = w Omega_i / k ; buffer M_i* = B0 + w^2 Omega_i^2 / (2k)
  - idiosyncratic shock xi_i ~ U[0, xi_bar]: node defaults iff M_i < xi_i
  - systemic shock (prob p_sys(gamma) = 1 - exp(-delta gamma)): the core buyer's
    liquidity freezes and it fails to honour receivables, so every first-tier
    supplier (d = 1) loses its receivable and defaults; this cascades upstream
    through the Leontief complementarity.
  - cascading disruption (Leontief): any upstream default halts all downstream
    dependents, propagating default until a steady state.

Resilience metric: R_net = sum_i q_i* I(i active) / sum_i q_i^FB.

Hypotheses:
  H1: R_net(gamma) retains an inverted-U shape; the peak gamma* is moderated by
      the degree-distribution variance of the network.
  H2: scale-free networks are "robust-yet-fragile": higher baseline resilience
      (hub-mediated penetration) but the largest systemic-shock resilience drop.
  H3: a payment-term cap alone is evaded (discount rate rises); only a combined
      cap + factoring-rate ceiling raises resilience.

Outputs: network_results.json, fig6..fig11.
"""

import json
import os

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytical_model import (credit_penetration, tau_star, PARAMS,
                              K_const, gamma_tilde)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9.5, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "font.family": "serif",
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESDIR = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(RESDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# network / simulation parameters
N = 150                 # number of supplier nodes
MU = 0.15               # deep-tier penetration friction
DELTA = 1.5             # systemic-fragility coefficient (systemic shock prob)
N_MC = 400              # Monte Carlo draws per (topology, gamma)
GAMMA_GRID = np.linspace(0.02, 1.0, 41)

p, w, k, B0, r0, rB, Tmax, xi_bar, beta = (PARAMS[k] for k in
    ("p", "w", "k", "B0", "r0", "rB", "Tmax", "xi_bar", "beta"))
q_fb = p / k


# ----------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------
def build_network(topology, seed):
    """Directed supply network with core node 'B' as the sink."""
    if topology == "er":
        g = nx.erdos_renyi_graph(N, 0.03, seed=seed)
    elif topology == "sf":
        g = nx.barabasi_albert_graph(N, 2, seed=seed)
    elif topology == "sw":
        g = nx.watts_strogatz_graph(N, 4, 0.1, seed=seed)
    else:
        raise ValueError(topology)

    lcc = max(nx.connected_components(g), key=len)
    g = g.subgraph(lcc).copy()

    g.add_node("B")
    hubs = sorted(g.degree, key=lambda x: x[1], reverse=True)
    hubs = [h for h, _ in hubs if h != "B"][:4]
    for h in hubs:
        g.add_edge(h, "B")

    dist = _bfs_distance(g, "B")
    D = nx.DiGraph()
    D.add_nodes_from(g.nodes())
    for u, v in g.edges():
        du, dv = dist.get(u, np.inf), dist.get(v, np.inf)
        if du > dv:
            D.add_edge(u, v)
        elif dv > du:
            D.add_edge(v, u)
        else:
            if g.degree(u) >= g.degree(v):
                D.add_edge(u, v)
            else:
                D.add_edge(v, u)
    D.graph["dist"] = dist
    return D


def _bfs_distance(g, source):
    dist = {source: 0}
    frontier = [source]
    d = 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in g[u]:
                if v not in dist:
                    dist[v] = d + 1
                    nxt.append(v)
        frontier = nxt
        d += 1
    return dist


def degree_variance(D):
    degs = [D.degree(u) for u in D.nodes() if u != "B"]
    return float(np.var(degs))


# ----------------------------------------------------------------------
# Node-level economics
# ----------------------------------------------------------------------
def node_liquidity(D, gamma):
    dist = D.graph["dist"]
    nodes = [u for u in D.nodes() if u != "B"]
    lam = credit_penetration(gamma)
    tau = tau_star(gamma)
    Om = np.array([lam * np.exp(-MU * dist[u]) * (1.0 - tau) for u in nodes])
    q = w * Om / k
    M = B0 + w ** 2 * Om ** 2 / (2.0 * k)
    return nodes, Om, q, M


# ----------------------------------------------------------------------
# Cascading disruption
# ----------------------------------------------------------------------
def disruption_cascade(D, nodes, active):
    """Upstream default -> downstream halt (Leontief complementarity)."""
    idx = {u: i for i, u in enumerate(nodes)}
    active = active.copy()
    changed = True
    while changed:
        changed = False
        for u in nodes:
            i = idx[u]
            if not active[i]:
                continue
            preds = [v for v in D.predecessors(u) if v != "B"]
            if preds and any(not active[idx[v]] for v in preds):
                active[i] = False
                changed = True
    return active


def financial_contagion(D, nodes, active):
    """Systemic shock: core B defaults, receivable default propagates upstream.

    A supplier v defaults if all of its downstream customers (successors) have
    defaulted. Redundant networks (ER/SW) let a supplier survive via alternative
    customers; tree-like scale-free chains propagate the default all the way up.
    """
    idx = {u: i for i, u in enumerate(nodes)}
    active = active.copy()
    # direct suppliers of B (tier-1) lose their receivable
    for u in D.predecessors("B"):
        i = idx.get(u)
        if i is not None:
            active[i] = False
    changed = True
    while changed:
        changed = False
        for v in nodes:
            i = idx[v]
            if not active[i]:
                continue
            customers = [s for s in D.successors(v) if s != "B"]
            if customers and all(not active[idx[s]] for s in customers):
                active[i] = False
                changed = True
    return active


def simulate_once(D, gamma, rng, sys_shock_prob, cap_T=None, cap_rf=None):
    """One Monte Carlo realisation -> realised network resilience."""
    nodes, Om, q, M = node_liquidity(D, gamma)

    if cap_T is not None or cap_rf is not None:
        lam = credit_penetration(gamma)
        Tmax_v = cap_T if cap_T is not None else Tmax
        K = p - w + rB * w * Tmax_v
        g_t = K / (w * (1.0 - r0 * Tmax_v))
        if gamma <= g_t:
            tau = r0 * Tmax_v
        else:
            tau = 0.5 - (p - w) / (2 * gamma * w) + (gamma * r0 - rB) * Tmax_v / (2 * gamma)
        tau = max(tau, r0 * Tmax_v)
        if cap_rf is not None:
            tau = min(tau, cap_rf * Tmax_v)
        dist = D.graph["dist"]
        Om = np.array([lam * np.exp(-MU * dist[u]) * (1.0 - tau) for u in nodes])
        q = w * Om / k
        M = B0 + w ** 2 * Om ** 2 / (2.0 * k)

    xi = rng.uniform(0, xi_bar, size=len(nodes))
    # 1. idiosyncratic default
    active = (M >= xi)
    # 2. systemic shock: core defaults -> upstream financial contagion
    if rng.random() < sys_shock_prob:
        active = financial_contagion(D, nodes, active)
    # 3. supply-disruption cascade
    active = disruption_cascade(D, nodes, active)
    return float(np.sum(q * active) / (len(nodes) * q_fb))


def sys_shock_prob(gamma):
    return 1.0 - np.exp(-DELTA * gamma)


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def sweep(topology, seed=0, cap_T=None, cap_rf=None, systemic=True):
    D = build_network(topology, seed)
    out = {"gamma": [], "R_mean": [], "R_std": []}
    rng = np.random.default_rng(seed + 1)
    for gamma in GAMMA_GRID:
        sp = sys_shock_prob(gamma) if systemic else 0.0
        vals = [simulate_once(D, gamma, rng, sp, cap_T, cap_rf)
                for _ in range(N_MC)]
        out["gamma"].append(float(gamma))
        out["R_mean"].append(float(np.mean(vals)))
        out["R_std"].append(float(np.std(vals)))
    return D, out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig6_topology():
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, topo in zip(axes, ["er", "sf", "sw"]):
        D = build_network(topo, seed=0)
        degs = [D.degree(u) for u in D.nodes() if u != "B"]
        ax.hist(degs, bins=range(0, max(degs) + 2), color="#4c72b0",
                edgecolor="white", alpha=0.85)
        ax.set_title({"er": "Erdos-Renyi", "sf": "Scale-free",
                      "sw": "Small-world"}[topo], fontsize=11)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_topology.png"), dpi=200)
    plt.close(fig)


def fig7_invertedU(sweeps):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = {"er": "#1f77b4", "sf": "#d62728", "sw": "#2ca02c"}
    labels = {"er": "Erdos-Renyi", "sf": "Scale-free", "sw": "Small-world"}
    peaks = {}
    for topo, (D, out) in sweeps.items():
        g = np.array(out["gamma"]); R = np.array(out["R_mean"])
        ax.plot(g, R, lw=2, color=colors[topo], label=labels[topo])
        peaks[topo] = float(g[np.argmax(R)])
    ax.set_xlabel(r"Ownership stake $\gamma$")
    ax.set_ylabel(r"Network resilience $R_{\mathrm{net}}(\gamma)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_invertedU.png"), dpi=200)
    plt.close(fig)
    return peaks


def fig8_systemic_drop(baseline, systemic):
    """H2 robust-yet-fragile: baseline resilience vs systemic-shock loss."""
    topo_order = ["er", "sw", "sf"]
    labels = ["Erdos-Renyi", "Small-world", "Scale-free"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(3)
    base_vals = [baseline[t] for t in topo_order]
    sys_vals = [systemic[t] for t in topo_order]
    loss_vals = [baseline[t] - systemic[t] for t in topo_order]

    width = 0.35
    ax1.bar(x - width/2, base_vals, width, color="#1f77b4",
            label="Baseline")
    ax1.bar(x + width/2, sys_vals, width, color="#d62728",
            label="Systemic shock")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"$R_{\mathrm{net}}(\gamma=1)$")
    ax1.set_title("(a) Resilience level", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5)

    ax2.bar(x, loss_vals, width * 0.8, color="#ff7f0e")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"Absolute resilience loss $\Delta\,R_{\mathrm{net}}$")
    ax2.set_title("(b) Systemic-shock loss", loc="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig8_systemic_drop.png"), dpi=200)
    plt.close(fig)


def fig9_peak_vs_variance(peaks, deg_vars, peak_heights):
    """H1: degree-distribution variance lifts the inverted-U peak height."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    order = ["sw", "er", "sf"]
    xs = [deg_vars[t] for t in order]
    ys = [peak_heights[t] for t in order]
    ax.plot(xs, ys, "o-", color="#4c72b0", lw=2, ms=8)
    for t, x, y in zip(order, xs, ys):
        lab = {"er": "ER", "sw": "SW", "sf": "SF"}[t]
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=10)
    ax.set_xlabel("Degree-distribution variance")
    ax.set_ylabel(r"Peak resilience $\max_\gamma R_{\mathrm{net}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig9_peak_vs_variance.png"), dpi=200)
    plt.close(fig)


def fig10_governance():
    D = build_network("sf", seed=0)

    def sweep_policy(cap_T, cap_rf):
        out = {"gamma": [], "R_mean": []}
        rng = np.random.default_rng(99)
        for gamma in GAMMA_GRID:
            sp = sys_shock_prob(gamma)
            vals = [simulate_once(D, gamma, rng, sp, cap_T, cap_rf)
                    for _ in range(N_MC)]
            out["gamma"].append(float(gamma))
            out["R_mean"].append(float(np.mean(vals)))
        return out

    base = sweep_policy(None, None)
    cap = sweep_policy(0.5, None)
    comb = sweep_policy(0.5, 0.10)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(base["gamma"], base["R_mean"], lw=2, color="#9467bd",
            label=r"$T_{\max}=1.0$ (laissez-faire)")
    ax.plot(cap["gamma"], cap["R_mean"], lw=2, color="#d62728", ls="--",
            label=r"$T_{\max}=0.5$ (cap only: evasion)")
    ax.plot(comb["gamma"], comb["R_mean"], lw=2.4, color="#2ca02c",
            label=r"$T_{\max}=0.5,\;r_f\leq0.10$ (combined)")
    ax.set_xlabel(r"Ownership stake $\gamma$")
    ax.set_ylabel(r"Network resilience $R_{\mathrm{net}}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig10_governance.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    sweeps = {}
    deg_vars = {}
    for topo in ["er", "sf", "sw"]:
        D, out = sweep(topo, seed=0, systemic=True)
        sweeps[topo] = (D, out)
        deg_vars[topo] = degree_variance(D)
        print(f"[{topo}] degree variance = {deg_vars[topo]:.2f}, "
              f"peak R_net = {max(out['R_mean']):.4f} at "
              f"gamma = {out['gamma'][int(np.argmax(out['R_mean']))]:.3f}")

    peaks = fig7_invertedU(sweeps)
    peak_heights = {t: max(o["R_mean"]) for t, (D, o) in sweeps.items()}
    fig6_topology()
    fig9_peak_vs_variance(peaks, deg_vars, peak_heights)
    fig10_governance()

    # H2: robust-yet-fragile -- baseline vs systemic resilience at gamma=1
    baseline = {}
    systemic = {}
    for topo in ["er", "sf", "sw"]:
        _, out_base = sweep(topo, seed=5, systemic=False)
        _, out_sys = sweep(topo, seed=5, systemic=True)
        i1 = int(np.argmin(np.abs(np.array(out_base["gamma"]) - 1.0)))
        baseline[topo] = out_base["R_mean"][i1]
        systemic[topo] = out_sys["R_mean"][i1]
    fig8_systemic_drop(baseline, systemic)

    results = {
        "params": {**PARAMS, "N": N, "MU": MU, "DELTA": DELTA, "N_MC": N_MC},
        "peaks": peaks,
        "peak_heights": peak_heights,
        "degree_variance": deg_vars,
        "baseline_R_at_gamma1": baseline,
        "systemic_R_at_gamma1": systemic,
        "absolute_loss": {t: baseline[t] - systemic[t] for t in baseline},
        "systemic_drop": {t: (baseline[t] - systemic[t]) / baseline[t]
                          for t in baseline},
        "sweeps": {t: {"gamma": o["gamma"], "R_mean": o["R_mean"],
                       "R_std": o["R_std"]} for t, (D, o) in sweeps.items()},
    }
    with open(os.path.join(RESDIR, "network_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n=== H2: robust-yet-fragile (gamma=1) ===")
    for topo in ["er", "sf", "sw"]:
        print(f"  {topo:4s}: baseline={baseline[topo]:.4f}  "
              f"systemic={systemic[topo]:.4f}  "
              f"drop={results['systemic_drop'][topo]:.2%}")
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
