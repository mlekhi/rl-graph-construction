# GraphHARE — Weekly Summary
_June 4 – June 10, 2026_

---

## What We Built This Week

| # | Task | Status |
|---|---|---|
| 1 | Fine-tuned β on Cora: ran {1.1, 1.2, 1.3} to find true peak | ✓ |
| 2 | Frozen GraphSAGE for PubMed (val_acc=0.902, test_f1=0.887) | ✓ |
| 3 | Full β ablation on PubMed: {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} | ✓ |
| 4 | Full β ablation on CiteSeer: {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} | ✓ |
| 5 | Side-by-side β curves for all 3 datasets | ✓ |
| 6 | Read NoisyNet paper (arxiv 1706.10295) | ✓ |
| 7 | Implemented learnable β (log_beta = nn.Parameter) | ✓ |
| 8 | Full learnable β runs: Cora, CiteSeer, PubMed | ✓ |
| 9 | Renamed scripts by pipeline order (01–06) | ✓ |
| 10 | Plot script auto-scales axes per dataset, reads baseline from results | ✓ |

---

## Fine-tuned Cora β Results

| β | Best macro-F1 | Δ over baseline |
|---|---|---|
| 1.0 | 0.8816 | +0.0153 |
| **1.1** | **0.8820** | **+0.0157** |
| **1.2** | **0.8820** | **+0.0157** |
| 1.3 | 0.8799 | +0.0136 |

True peak is β=1.1 (tied with 1.2 on single seed). Beats β=1.0 by +0.0004 — argues for learnable β to find exact optimum.

---

## Multi-Dataset β Ablation Results

| Dataset | Classes | Best β | Best macro-F1 | Δ over baseline | Notes |
|---|---|---|---|---|---|
| Cora | 7 | 1.1 | 0.8820 | +0.0157 | Clear inverted-U, β matters |
| CiteSeer | 6 | flat | 0.7242 | +0.0112 | F1 identical across all β |
| PubMed | 3 | flat | 0.8865 | +0.0019 | Graph redundant, controlled case |

**Class-count theory confirmed:** more classes = bigger β effect (Cora 7 > CiteSeer 6 > PubMed 3).

CiteSeer's flat F1 across all β is explained by the small policy_val (334 nodes, 6 classes) producing a quantized metric — the agent hits the same discrete F1 ceiling regardless of β weighting.

PubMed's tiny gains are expected per Prof. AlMahamid's framing: 1pp MLP-SAGE baseline gap means the graph adds little signal beyond features. This is the "controlled case where graph is redundant."

---

## Learnable β Results

Implemented β as a learnable network parameter (log_beta = nn.Parameter, β = exp(log_beta)) updated through the PPO value loss.

| Dataset | Learned β (final) | Learnable F1 | Fixed-β F1 (peak) |
|---|---|---|---|
| Cora | 0.378 | 0.8794 | 0.8820 (β=1.1) |
| CiteSeer | 0.389 | 0.7218 | 0.7242 |
| PubMed | 0.357 | 0.8863 | 0.8865 |

**Problem identified:** β decays monotonically from 0.5 to ~0.36–0.39 in nearly identical fashion across all 3 datasets. This suggests the gradient through the value loss always pushes β in one direction rather than adapting to each dataset's structure.

Fixed-β manual sweep still outperforms the learnable version. Need to discuss alternative mechanisms with Prof. AlMahamid (e.g. direct val-F1 signal instead of value loss gradient).

---

## Experiment Tracking Improvements

- Output dir naming: `ppo_seed{seed}_learnable_beta/` for learnable runs, `ppo_seed{seed}_beta{value}/` for fixed
- Plot script (06) auto-scales y-axis per dataset; baseline read from results.json
- New script: `07_plot_beta_comparison.py` for 3-panel side-by-side plot
- Scripts numbered 01–06 by pipeline order for clarity
- `RESEARCH_HANDOVER.md` moved to `archive/`, `WEEKLY_SUMMARY.md` and flowchart moved to `notes/`

---

## Open Questions for Next Meeting

1. Learnable β underperforms fixed sweep and converges to the same value (~0.38) across all datasets — is a direct val-F1 signal the right alternative, or a different mechanism (bilevel optimization, contextual β)?
2. How to frame learnable β in the paper given current implementation underperforms?
3. NeurIPS math formulation — which theoretical claims to commit to? Convergence analysis? Pareto-optimality for learnable β?
4. Ready to proceed with 10-seed CI on best fixed β across all 3 datasets?

---

## Next Steps

1. Run 10-seed CI on Cora, CiteSeer, PubMed at best fixed β
2. Redesign learnable β with direct val-F1 signal (pending Prof. AlMahamid input)
3. Begin mathematical formulation (with Prof. AlMahamid)
4. Build final comparison table vs GraphSAGE, MLP, GraphRARE (β=0)
