# Captive Factoring and Supply Network Resilience: Liquidity Buffer or Cascading Fragility?

Replication code and figures for the paper *Captive Factoring and Supply Network
Resilience: Liquidity Buffer or Cascading Fragility?* (targeted at Production and
Operations Management).

## Overview

The paper studies **captive factoring**, in which a focal buyer both sets its
suppliers' payment terms and runs the factoring subsidiary that discounts their
receivables. We model the dual role as a two-stage Stackelberg game and extend it
to a directed supply network.

Three results drive the paper:

1. The buyer always extends the payment term to the regulatory ceiling, but prices
   factoring at cost below a threshold ownership stake and above it extracts an
   increasing spread (Proposition 1).
2. Supply-chain resilience is an **inverted U** in the buyer's ownership stake, with
   a unique interior critical stake separating a liquidity-buffer regime from a
   systemic-fragility regime (Proposition 2).
3. A payment-term cap alone is **evaded** through the discount rate and can lower
   resilience; a combined cap plus discount-rate ceiling restores it (Proposition 3).

Across Erdos-Renyi, scale-free, and small-world supply networks, the inverted U
survives, its height rises with degree heterogeneity, and scale-free networks are
robust-yet-fragile.

## Repository contents

```
.
├── analytical_model.py        # closed-form Stackelberg model (Lemma 1, Prop. 1-3)
├── network_simulation.py      # agent-on-networks simulation (3 topologies, dual contagion)
├── verify_references.py       # Crossref verification of the reference list
├── verify_dois.py             # DOI-level verification of key references
├── verify_precise.py          # author+title verification of specific references
├── verify_captive_refs.py     # verification of the captive-finance references
├── figures/                   # all figures in the manuscript (fig1..fig10)
├── results/                   # numerical results and reference-verification logs
└── README.md
```

The paper-generation script is intentionally excluded; the manuscript itself is
available from the corresponding author.

## Key results

Analytical model (baseline calibration, Table 1 of the paper):

| Quantity | Value |
|---|---|
| Cost-price threshold $\tilde\gamma$ | 0.365 |
| Critical ownership stake $\gamma^*$ | 0.379 |
| Peak resilience $\mathcal R(\gamma^*)$ | 0.406 |
| Resilience at full ownership $\mathcal R(1)$ | 0.350 |
| Discount rate at full ownership $r_f^*(1)$ | 0.345 |

Network simulation (degree variance; peak resilience):

| Topology | Degree variance | Peak $\mathcal R_{\mathrm{net}}$ |
|---|---|---|
| Erdos-Renyi | 5.74 | 0.021 |
| Small-world | 0.52 | 0.011 |
| Scale-free | 16.61 | 0.042 |

## Dependencies

Python 3.12, NumPy, SciPy, Matplotlib, NetworkX.

## Quick start

```bash
python3 analytical_model.py     # writes figures/fig1..fig5 and results/analytical_results.json
python3 network_simulation.py   # writes figures/fig6..fig10 and results/network_results.json
```

Run `analytical_model.py` first; `network_simulation.py` imports its parameter set.

## Citation

If you use this code, please cite the paper. The reference list is verified against
Crossref (see `results/ref_verification*.json`).
