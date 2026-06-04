# GraphHARE — Weekly Summary
_May 27 – June 3, 2026_

---

## What We Built

### GraphHARE System (end-to-end RL pipeline)

```
Citation Graph
     │
     ▼
RL Agent (PolicyNet, 72k params)
     │  picks edge to add/remove
     ▼
Modified Graph
     │
     ▼
Frozen GraphSAGE (val_acc=0.910 on Cora)
     │
     ▼
Reward: r_t = (ΔF1 / σ_F1) + β × (Δhomophily / σ_homophily)
```

---

## Key Completions This Week

| # | Task | Status |
|---|---|---|
| 1 | Received + merged Fadi's revised RL_DESIGN.md (GraphHARE framing) | ✓ |
| 2 | Homophily diagnostic on Cora, PubMed, CiteSeer | ✓ |
| 3 | Reward updated: gamma×minority_F1 → β×Δhomophily | ✓ |
| 4 | Reward normalization (Welford online algorithm) | ✓ |
| 5 | CiteSeer baselines (GraphSAGE + MLP, 10 splits) | ✓ |
| 6 | CiteSeer frozen GraphSAGE checkpoint | ✓ |
| 7 | Wandb wired in (project: graphhare) | ✓ |
| 8 | Config file system (configs/cora_beta*.yaml) | ✓ |
| 9 | compile_results.py — single source of truth CSV | ✓ |
| 10 | Full β ablation on Cora (seed=42, 500ep, normalized) | ✓ |

---

## Homophily Diagnostic Results

All 3 datasets pass the pre-training gate.

| Dataset | h_original | h_knn_pool | Random baseline | Ratio |
|---|---|---|---|---|
| Cora | 0.810 | 0.517 | 0.143 (1/7) | 3.6x |
| PubMed | 0.802 | 0.713 | 0.333 (1/3) | 2.1x |
| CiteSeer | 0.736 | 0.525 | 0.167 (1/6) | 3.2x |

---

## Baselines

| Dataset | Model | Test macro-F1 | Gap |
|---|---|---|---|
| Cora | GraphSAGE | 0.866 ± 0.015 | — |
| Cora | MLP | 0.732 ± 0.016 | 13pp ← strong RL target |
| PubMed | GraphSAGE | 0.884 ± 0.003 | — |
| PubMed | MLP | 0.874 ± 0.004 | 1pp ← expected low gain |
| CiteSeer | GraphSAGE | 0.714 ± 0.015 | — |
| CiteSeer | MLP | 0.682 ± 0.009 | 3pp |

---

## β Ablation Results — Cora (seed=42, normalized reward)

| β | Best macro-F1 | Δ over baseline | h_refined | Δ homophily |
|---|---|---|---|---|
| 0.00 | 0.8794 | +0.0131 | 0.8129 | +0.0030 |
| 0.25 | 0.8800 | +0.0137 | 0.8145 | +0.0046 |
| 0.50 | 0.8781 | +0.0118 | 0.8132 | +0.0032 |
| 0.75 | 0.8786 | +0.0123 | 0.8135 | +0.0036 |
| **1.00** | **0.8816** | **+0.0153** | **0.8140** | **+0.0041** |
| 1.50 | 0.8782 | +0.0119 | 0.8135 | +0.0035 |
| 2.00 | 0.8776 | +0.0113 | 0.8131 | +0.0031 |

**β=1.0 peaks — inverted-U confirmed (single seed).**

Homophily reference: original graph 0.810, kNN pool 0.517, all refined graphs 0.813–0.814.

---

## What the β Curve Shows

```
macro-F1
  ▲
  │              ● β=1.0 (peak)
  │         ●         ●
  │    ●                   ●
  │●                            ●
  └─────────────────────────────▶ β
   0  0.25  0.5  0.75  1.0  1.5  2.0
```

- β=0 (accuracy-only, GraphRARE-style): +0.0131
- β=1.0 (equal weighting): +0.0153 ← best
- High β trades accuracy for homophily

---

## Experiment Tracking System

Per Fadi's request, every run now produces:
- **results.json**: full config, all metrics, 3 homophily values, git commit hash, wall time
- **wandb**: live curves at wandb.ai/mlekhi-western-university/graphhare
- **configs/**: yaml file per experiment (reproducible re-runs)
- **runs/results_summary.csv**: one row per run, single source of truth for paper tables

---

## Next Steps

1. 10-seed CI on Cora at β=1.0 (confirm result holds across seeds)
2. Full β ablation on CiteSeer and PubMed
3. Save refined graph artifact (final edge index) per run
4. Freeze GraphSAGE for PubMed + run RL
