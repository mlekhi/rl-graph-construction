# GraphHARE System Flowchart

## Current Setup (Fixed β)

```
EPISODE START
     │
     ▼
Original Citation Graph
     │
     ▼
┌─────────────────────────────────┐
│  for step in range(100):        │
│                                 │
│  1. Sample node by entropy      │
│  2. Get ~25 candidate edges     │
│  3. PolicyNet scores candidates │
│     → picks best edge action    │
│  4. Apply edit to graph         │
│                                 │
│  every 5 steps:                 │
│  5. Frozen SAGE forward pass    │
│  6. Compute reward:             │
│     r = (ΔF1/σ_F1)             │
│       + β × (Δhomophily/σ_hom) │
│     β = FIXED (e.g. 1.1)        │
└─────────────────────────────────┘
     │
     ▼
PPO Update (policy weights)
     │
     ▼
Repeat for 500 episodes
     │
     ▼
Best refined graph saved
```

---

## With Learnable β (NoisyNet-inspired)

```
EPISODE START
     │
     ▼
Original Citation Graph
     │
     ▼
┌─────────────────────────────────┐
│  for step in range(100):        │
│                                 │
│  1. Sample node by entropy      │
│  2. Get ~25 candidate edges     │
│  3. PolicyNet scores candidates │
│     → picks best edge action    │
│     β = exp(log_beta)           │ ← β is now a LEARNED PARAMETER
│       (starts ~1.0, adapts)     │   inside PolicyNet
│  4. Apply edit to graph         │
│                                 │
│  every 5 steps:                 │
│  5. Frozen SAGE forward pass    │
│  6. Compute reward:             │
│     r = (ΔF1/σ_F1)             │
│       + β × (Δhomophily/σ_hom) │
│     β learned, not fixed        │
└─────────────────────────────────┘
     │
     ▼
PPO Update
├── policy weights updated (as before)
└── log_beta updated ← NEW: β also learns from reward signal
     │
     ▼
Repeat for 500 episodes
     │
     ▼
β converges to optimal value
(e.g. 1.08 on Cora, lower on PubMed)
     │
     ▼
Best refined graph saved
β value saved in checkpoint
```

---

## What Changes

| | Fixed β | Learnable β |
|---|---|---|
| β source | you set it manually | network learns it |
| exploration | entropy coefficient | NoisyNet-style noise in β |
| result | β=1.1 from sweep | β≈1.1 found automatically |
| paper claim | "optimal β is 1.1" | "β self-tunes to 1.1 on Cora" |
| per-dataset | same β for all | different β per dataset |
| NeurIPS angle | ablation study | principled adaptive reward |
