# 🦟 flymj — A Fruit Fly Brain vs. Japanese Mahjong

A fun project: take the **male fruit fly whole-brain connectome** released by
Google Research / HHMI Janelia in 2026 (166,000 neurons, 125 million synapses),
extract the topology of its mushroom body (PN→KC→MBON, the fly's associative
learning circuit), use it as a fixed hidden-layer wiring pattern in a neural
network trained to predict Japanese mahjong discard/call decisions — then
wrap the trained model in a Streamlit app you can actually sit down and play
a full game against, with a live visualization of "which part of the fly
brain is thinking."

## This is not another "fly brain plays Doom" demo

There have been viral clips of connectomes "playing Doom / Mario". What
those demos actually do is use the connectome's topology as the fixed edge
set of a sparse recurrent network, while the **weights are still trained
from scratch** — the connectome itself only gives you "which neuron connects
to which," not synaptic strength or any dynamic activity. It's not really
"the fly brain thinking."

This project asks a narrower, cleaner question instead:

> Does using the real fly mushroom body's **sparse connectivity** (which KC
> connects to which PN) as a fixed neural-network structure actually
> outperform a **degree-matched but randomly wired** structure, on the task
> of predicting mahjong discards?

## TL;DR result

| Arm | Hidden-layer structure | Mean test accuracy (5–20 seeds) |
|---|---|---|
| **A** Real connectome | Sparse, 387→4063→97 (real wiring) | **65.73%** |
| **B** Degree-matched random | Sparse, 387→4063→97 (random wiring, same degree) | 65.61% |
| **C** Dense (parameter-matched) | Fully connected, 387→118→97 | 65.29% |

- **A and B both clearly beat C** (p<0.01, large effect size) — building the
  hidden layer as a large, sparsely-expanded structure has a real advantage,
  independent of whether the wiring is copied from biology.
- **A vs. B shows no significant difference** (t-test p=0.15, permutation
  p=0.07) — at this task and scale, there's no evidence that the fly's
  *specific* PN–KC wiring pattern beats a randomly wired pattern with the
  same degree distribution.

In short: **what wins is the "sparse expansion" shape, not the fly's
particular wiring diagram.** Full methodology/statistics/limitations are in
[`data/doc/writeup.md`](data/doc/writeup.md) (Cantonese).

## Two playable apps

The Arm A (real connectome) model is wrapped into two local Streamlit apps:

### 1. Single-decision analyzer (`src/app.py`)

Give it a hand + game situation, pick the exact scenario you want to ask
about, and see how the model ranks each option (which tile to discard, win,
pon, chii, riichi, ...), plus a 3D visualization of which PN/KC/MBON neurons
are "firing" hardest. Includes a basic genbutsu (safe-tile) defense hint.

```bash
uv run streamlit run src/app.py
```

### 2. A full, continuous mahjong table (`src/game_app.py`)

An actual game: you (seat 0) vs. three `jansou`-provided
`SmartEfficiencyAgent` bots. A full east+south round with dealer rotation
and scoring, all driven by `jansou.game.environment.Environment` — not a
hand-rolled fake engine. On your turn, just tap the tile in your hand to
discard it; the model marks its top pick with 🌟, and there's a live fly-brain
activation view plus defense hints.

```bash
uv run streamlit run src/game_app.py --server.port 8502
```

> ⚠️ This is strictly a local, for-fun toy. It is **not**, and should
> **not** be, hooked up to any online mahjong platform (Mahjong Soul,
> Tenhou, etc.) — those platforms explicitly forbid third-party real-time
> assistance.

## Rebuilding the pipeline from scratch

This project started as an 8-step research pipeline. The entire `data/`
directory (raw connectome + mahjong logs + training artifacts, several GB)
is gitignored and needs to be regenerated locally:

| # | Script | Produces | Needs |
|---|---|---|---|
| 1–2 | `src/build_masks.py` | `data/processed/masks.npz` | `data/raw/annotations.feather` + connectome edge weights feather ([Google/HHMI male CNS v1.0](https://www.janelia.org/)) |
| 3 | `src/build_discard_dataset.py` | `data/processed/discard_dataset.parquet` | Tenhou houou-room MJAI logs (see [`NikkeTryHard/tenhou-to-mjai`](https://github.com/NikkeTryHard/tenhou-to-mjai) releases) |
| 4 | `src/build_features.py` | `data/processed/features.npz` | the two above |
| 5–7 | `src/run_experiment.py` | `data/processed/experiment_results.csv` | 30 runs (arms A/B/C) |
| 8 | `src/plot_experiment.py` | `artifacts/arm_comparison.png` | the results above |
| — | `src/train_and_save_model.py` | `data/processed/model_arm_a.pt` (used by `app.py`) | masks + features |
| — | `src/validate_replay.py` | validates the `jansou` replay engine reproduces historical logs 100% | mahjong logs |
| — | `src/build_action_dataset.py` → `build_action_features.py` → `train_action_model.py` | `data/processed/action_model_arm_a.pt` (both apps use this for call/riichi/tsumo/defense suggestions) | mahjong logs + masks |

For the full file index, what each step does, and why it's designed this
way, see [`data/doc/writeup.md`](data/doc/writeup.md) (the research
write-up) and [`data/doc/task.md`](data/doc/task.md) (the original task
brief) — both in Cantonese.

## Setup

```bash
uv sync                 # installs everything in pyproject.toml (Python 3.13+)
uv run python main.py   # sanity check: reads annotations.feather
```

Key dependencies: `jansou` (an open-source Japanese mahjong rules engine +
MJAI parser), PyTorch, Streamlit, Polars/PyArrow (pandas is deliberately
avoided in the core pipeline, and PyArrow is used to work around a Polars
dictionary-encoding bug when reading the raw feather files).

## Data sources & usage note

- **Connectome**: Google Research / HHMI Janelia's 2026 male CNS v1.0
  release, CC-BY.
- **Mahjong logs**: Tenhou houou-room games, redistributed as MJAI-format
  logs by a third party (`tenhou-to-mjai`). Tenhou's own ToS asks that
  "general mahjong applications" contact them for permission; that inquiry
  was sent and is pending a reply. Development continued in the meantime
  using this third-party dataset — a known, accepted gray-area risk, used
  non-commercially, not redistributing the raw logs, for personal
  research/demo purposes only.

## Known limitations

- Arm A (real connectome) has only one mask and 5 seeds; Arm B (random) has
  20 different masks — the statistical power isn't symmetric between them
  (adding more Arm A seeds would strengthen confidence in "A vs. B, no
  difference").
- The degree-matched random mask only matches KC in-degree, not PN
  fan-out distribution — a deliberate design choice to keep the control
  clean, not an oversight.
- The mahjong logs cover only 2009, so meta/strategy shifts across other
  eras aren't represented.
- The 65–66% absolute accuracy is in a reasonable range versus published
  baselines (Bakuuchi 62.1%, CNN-based methods 68–70%, Suphx 76.7%), but the
  model itself hasn't been tuned and shouldn't be compared against
  state-of-the-art on absolute performance.

See the "Limitations" section of [`data/doc/writeup.md`](data/doc/writeup.md)
for the full version.
