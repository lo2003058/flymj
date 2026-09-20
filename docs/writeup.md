# Fly Connectome Topology vs. Random Topology: A Controlled Experiment on Japanese Mahjong Discard Prediction

## Research Question

In 2026, Google Research / HHMI Janelia open-sourced the complete male fruit fly whole-brain connectome (166,000 neurons, 125 million synapses). Viral clips of "fly brains playing Doom/Mario" appeared online, but what those demos actually do is use the connectome's topology as the fixed edge set of a sparse recurrent network, while the weights are still trained from scratch — the connectome itself provides no synaptic strength and no dynamic activity of any kind.

This experiment sets out to seriously answer one question:

> Does using the real fly brain's mushroom body (PN→KC→MBON) sparse connectivity as a neural network's fixed hidden-layer structure actually outperform a structure with the same degree distribution but randomly wired connections, on the task of predicting Japanese mahjong discards?

The mushroom body was chosen because it's biologically an associative-learning circuit: PN input → KC does sparse expansion coding → MBON reads out. This shape maps sensibly onto the "hand state → which tile to discard" decision, not as a forced analogy.

## Methodology

### 1. Connectome mask extraction (Steps 1-2)

Three layers were selected from `annotations.feather` (211,577 neurons) using the following definitions:

| Layer | Definition | Count |
|---|---|---|
| PN | `class == 'ALPN'` and `type` not starting with `M_` (excluding multiglomerular) | 387 |
| KC | `type` starts with `KC` | 4063 |
| MBON | `class == 'MBON'` | 97 |

(The PN count once diverged from the 100-200 originally expected; investigation concluded that expectation came from older single-hemisphere literature, while this male CNS dataset covers both hemispheres — 387 is roughly 190 per side x2. Confirmed with the user to proceed with 387.)

From `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (125 million edges), filtering `weight >= 5` extracted two binary masks: PN→KC (19,832 nonzero edges) and KC→MBON (33,496 edges). Mean KC in-degree was 4.88 (close to the expected 5-8).

Degree-matched random mask: for each KC, count how many PN inputs it has in the real mask (call it k), then draw k PNs at random (without replacement) from the PN pool; same for KC→MBON. 20 independent seeds (`np.random.default_rng(seed)`).

### 2. Mahjong discard-decision dataset (Steps 3-6)

**Data source**: Tenhou houou-room (high-rank) MJAI-format logs for 2009, 6897 matches, distributed via GitHub Releases by `NikkeTryHard/tenhou-to-mjai`. Parsed with `jansou` (an open-source Python mahjong library)'s `parse_mjai()`, replaying each event to reconstruct every seat's concealed hand, melds, and discard record.

> **Data usage note**: Tenhou's official ToS states that "general mahjong applications" should contact support@c-egg.com. This research has sent that inquiry and is awaiting a reply; development continued in the meantime using a third-party curated dataset — a known, accepted gray-area risk, used non-commercially, not redistributing the raw logs, for personal research use only.

3000 matches were processed (MAX_FILES=3000), yielding **1,556,465 discard decisions** (DISCARD + RIICHI_DISCARD), split by `match_id` into train (2400 matches) / val (300 matches) / test (300 matches), 80/10/10, ensuring a single match is never split across two sides.

### 3. Feature encoding (Step 4)

Each decision is encoded as a `(32, 34)` binary tensor (32 channels, 34 = number of mahjong tile types):

- Channels 0-3: own hand count thermometer (>=1/2/3/4)
- Channel 4: whether own hand has a red five
- Channels 5-8: own melded tiles count thermometer
- Channels 9-24: own + the 3 opponents' (in turn order) discard pile count thermometer (4 players x 4 planes)
- Channel 25: currently active dora tile
- Channels 26-27: round wind, seat wind (one-hot)
- Channels 28-31: all 4 players' riichi status (broadcast)

### 4. Model architecture

```
mahjong feature (32 x 34)
  -> Conv1d front-end (2 layers, 64 channels, ReLU)
  -> Linear -> PN layer (387)
  -> MaskedLinear(PN -> KC)
  -> ReLU
  -> MaskedLinear(KC -> MBON)
  -> Linear -> 34 logits
```

`MaskedLinear` is a plain `nn.Linear` whose weight is element-wise multiplied by a mask buffer that is never trained; masked-out positions are always 0.

### 5. Three arms

| Arm | Hidden-layer structure | Effective parameters (nonzero weights + biases) |
|---|---|---|
| A | Real connectome mask (387→4063→97, sparse) | 57,488 |
| B | Degree-matched random mask, 20 seeds | 57,488 (same as A) |
| C | Dense (no mask), hidden dim shrunk to 118 (387→118→97, fully connected) | 57,488 (same as A/B) |

The three arms' "effective parameter count" is deliberately matched, so any accuracy difference can be attributed to "how the connections are wired," not "which arm simply has more parameters." Arm C's hidden dim of 118 was solved from the equation `485H + 97 = 57488`, not chosen by trial and error.

### 6. Training

- Batch size 1024, Adam, lr=1e-3
- Up to 20 epochs, early stop after 3 epochs without a new best val accuracy; the best epoch's weights (by val_acc) are used to evaluate the test set
- Arm A: 5 seeds (the real mask never changes; the 5 runs only differ in model init/data order)
- Arm B: 20 seeds (each seed uses its corresponding degree-matched random mask, and that same seed also drives model init/data order)
- Arm C: 5 seeds (dense architecture; the 5 runs only differ in model init/data order)

## Results

![Arm comparison](../artifacts/arm_comparison.png)

| Arm | n (seeds) | mean test_acc | std | min | max |
|---|---|---|---|---|---|
| A (real connectome) | 5 | 65.73% | 0.15% | 65.60% | 65.92% |
| B (degree-matched random) | 20 | 65.61% | 0.12% | 65.41% | 65.83% |
| C (dense, param-matched) | 5 | 65.29% | 0.11% | 65.12% | 65.41% |

### Statistical tests (Welch's t-test + permutation test, 100,000 resamples)

| Comparison | Difference | Welch t | p (t-test) | p (permutation) | Cohen's d |
|---|---|---|---|---|---|
| A vs B | +0.12pp | 1.68 | 0.149 | 0.073 | 0.88 |
| A vs C | +0.44pp | 5.35 | 0.001 | 0.006 | 3.38 |
| B vs C | +0.32pp | 5.76 | 0.0007 | <0.0001 | 2.75 |

(For external reference: on discard-prediction tasks like this, published baselines include Bakuuchi at 62.1%, CNN-based methods at 68.8-70.44%, and Suphx at 76.7%. This experiment's 65-66% falls within a reasonable range, showing the pipeline itself is healthy — but that absolute number isn't the point of this study. All three arms use exactly the same architecture/data/training procedure, so the comparison is purely internal/relative.)

## Conclusion

The research question splits into two layers, each with a clear answer:

1. **Does the "sparse expansion structure" itself help?** — Yes. Arm A and Arm B (both 387→4063→97 sparse structures) clearly beat Arm C (the 387→118→97 dense structure), both comparisons at p<0.01 with a large effect size (d>2.7). In other words, expanding the hidden layer into thousands of sparsely-connected units has a genuine benefit in its own right, regardless of whether those connections copy a real biological wiring diagram.

2. **Does the fly's specific wiring pattern itself help?** — No evidence found. The gap between Arm A and Arm B (0.12 percentage points) is not statistically significant (t-test p=0.149, permutation p=0.073). In other words, at this task and this scale, **there is no evidence that the real fly brain's specific PN-KC wiring pattern outperforms a randomly wired pattern with the same degree distribution.**

This result is consistent with the study's initial expectation ("the three arms are about the same" would itself count as an answer), but more precise than "no difference at all": the difference lies in the *shape* (sparse expansion), not the *pattern* (specific wiring).

## Follow-up Experiment: Dataset Scaling

The A/B/C comparison above used a single year (2009), 3000 files, 1.55 million decisions. After completing it, a separate question was opened: **does simply scaling up training data (without changing the connection structure) make Arm A meaningfully better?** — This is independent of "which connection structure is better," and was run with its own separate scripts/files (the `*_scaled` series), without overwriting the original dataset/results used for the A/B/C comparison.

**Method**: downloaded 9 more years (2010-2018) of Tenhou houou-room logs (same source as 2009), taking the first 3000 files per year (the same per-year cap as before), combining all 10 years so train/val/test each have representation from every year (avoiding a hidden meta-drift leak where old eras are only in train and new eras only in test). This yields 15,470,316 decisions, 10x the original. Only Arm A (real connectome) was trained, 5 seeds, compared against the 5 Arm A seeds already in `experiment_results.csv`.

**Results**:

| Dataset | Decisions | mean test_acc | std | n (seeds) |
|---|---|---|---|---|
| Original (2009, 3000 files) | 1,556,465 | 65.73% | 0.15% | 5 |
| Scaled (2009-2018, 30000 files) | 15,470,316 | **69.72%** | 0.04% | 5 |

Difference = **+3.99 percentage points**, Welch t = 58.16 (p≈0), permutation test (100,000 resamples) p = 0.0024, Cohen's d = 36.78 (an order of magnitude larger than the already-"large" A vs C effect size). The std across the 5 seeds also shrank (0.15%→0.04%), a very consistent result.

**Conclusion**: on this task, the effect of "scaling up training data" on accuracy is far larger than "which connection structure is used" (+4pp vs. the non-significant 0.12pp A-B gap). Connection topology sets the "starting point," but dataset scale is the real bottleneck — consistent with general deep learning intuition, now verified first-hand on this specific task via a clean controlled experiment.

Scripts/outputs are in the `*_scaled` rows of the file index below. (This discard-only Arm A model once had a deployment checkpoint, `model_arm_a.pt`, used by a standalone "single-decision analyzer" app; that app and this checkpoint were later removed — see the "Removed" section below.)

**The action model got the same treatment**: `game_app.py` (the full playable game)'s call/riichi/tsumo/defense suggestions actually come from a different model (`action_model_arm_a.pt`), trained on data different from the discard-only dataset above (it also includes pon/chii/kan/riichi/ron/pass). Using the same 2009-2018 logs, `build_action_dataset_scaled.py` (replay+oracle, see `validate_replay.py`) built 19,656,702 raw decisions, 17,069,986 after a 20% PASS subsample, and retrained:

| Head | Original (2009) | Scaled (2009-2018) |
|---|---|---|
| self_type | – | 98.74% |
| discard | – | 68.93% |
| reaction | – | 88.22% |
| **overall** | – | **84.10%** |

(The original deployed checkpoint never had its own recorded test accuracy, so there's no direct percentage to compare against here — but it uses exactly the same pipeline/config, with dataset size the only difference, so in principle it should follow the same direction seen with the discard-only model above: more data, better model.) This checkpoint is now deployed.

**A pitfall hit while running both scaling jobs**: building 15-20 million rows, each carrying nested-list columns (hand_counts/meld_counts/discard_counts, etc.), into one big Python `list[dict]` before converting to a polars DataFrame copies the whole dataset in memory several times over (one copy each for filter/concat/sort) — this got OOM-killed twice on a 32GB-RAM machine. Fix: (1) build and write each year to disk as its own cache file immediately (so the script can resume after an interruption instead of starting over), and (2) do the final subsample step as a single filter using numpy-picked indices, instead of filtering twice then concatenating then sorting.

## Removed: the single-decision analyzer (`app.py`)

Phase 1/2 once had a standalone "single-decision analyzer" Streamlit app (`src/app.py` — given one hand + game situation, it showed how the model ranked each option), which needed its own separately-deployed discard-only checkpoint (`train_and_save_model.py` → `model_arm_a.pt`). It was later decided to keep only `game_app.py` (the actual playable game); the single-decision analyzer's purpose overlapped and became redundant, so `app.py`, `inference.py` (its dedicated model wrapper), `call_options.py` (its dedicated manual pon/chii/kan combo enumeration — `game_app.py` uses jansou's real legal actions instead, so this wasn't needed), and `train_and_save_model.py`/`model_arm_a.pt` were all removed. Verified beforehand that none of these files were used by `game_app.py` or any other pipeline script, so removing them doesn't affect any research result (neither the A/B/C comparison nor the scaling experiment relies on this checkpoint).

## Limitations

- **The A vs B comparison is asymmetric**: A has only one real mask, so its 5 runs only vary model init/data order; B has 20 different masks. For more confidence in the "A vs B, no significant difference" conclusion, A's seed count should be increased (e.g. to 20 as well), to make the statistical power symmetric on both sides.
- **The degree-matched random mask only matches "how many PN inputs each KC has" (in-degree)**, not which specific PNs happen to fan out a lot. In the real connectome, different PNs' out-degree distribution may be quite uneven; the random mask scrambles this heterogeneity. This is a deliberate part of the experimental design (to keep the control clean), not an oversight, but worth keeping in mind when interpreting the conclusion.
- **Mahjong log data**: the A/B/C wiring comparison used only 2009 logs, not covering meta/strategy shifts across other eras (the "Follow-up Experiment" above adds 2010-2018, but only to train Arm A, not to redo the A vs B topology comparison); the data source's ToS status has no official written confirmation (see the "Data usage note" above).
- **Absolute accuracy**: 65-66% falls within a reasonable range versus published literature, but the model itself (a Conv1d front-end + 32 hand-crafted feature channels) hasn't been heavily tuned and shouldn't be used to compare absolute performance against SOTA models.

## File Index

| Script | Produces | Corresponding Step |
|---|---|---|
| `src/io_utils.py` | feather loader | 1 |
| `src/labels.py` | KC/MBON/PN definitions (single source of truth) | 1-2 |
| `src/explore_labels.py` | label exploration/verification | 1 |
| `src/build_masks.py` | `data/processed/masks.npz` | 2 |
| `src/plot_degrees.py` | `artifacts/degree_dist.png` | 2 |
| `src/explore_paifu.py` | downloads logs, verifies `jansou` parsing | 3 |
| `src/paifu_replay.py` | event replay (label + full-state versions) | 3-4 |
| `src/build_discard_dataset.py` | `data/processed/discard_dataset.parquet` | 3/6 |
| `src/features.py` | (32,34) tensor encoding logic | 4 |
| `src/build_features.py` | `data/processed/features.npz` | 4/6 |
| `src/splits.py` | train/val/test split by match | 6 |
| `src/model.py` | `MaskedLinear` + `MahjongNet` | 5/7 |
| `src/arms.py` | the three arms' mask construction | 7 |
| `src/train.py` | reusable training logic (minibatch + early stopping) | 5/7 |
| `src/train_overfit.py` | 10,000-sample overfit validation | 5 |
| `src/pilot_compare.py` | pilot (1 A + 1 B) | 7 |
| `src/run_experiment.py` | the full 30-run set, `data/processed/experiment_results.csv`/`experiment_history.parquet` | 7 |
| `src/plot_experiment.py` | `artifacts/arm_comparison.png` | 8 |
| `src/download_paifu_years.py` | downloads 2010-2018 logs (follow-up) | scaling |
| `src/build_discard_dataset_scaled.py` | `data/processed/discard_dataset_scaled.parquet` (10 years) | scaling |
| `src/build_features_scaled.py` | `data/processed/features_scaled.npz` (10 years) | scaling |
| `src/train_scaling_experiment.py` | `data/processed/scaling_results.csv`, Arm A scaled vs. original statistical comparison | scaling |
| `src/build_action_dataset_scaled.py` | `data/processed/action_dataset_scaled.parquet` (10 years, cached per-year) | scaling |
| `src/build_action_features_scaled.py` | `data/processed/action_features_scaled.npz` (10 years) | scaling |
| `src/train_action_model.py` | `data/processed/action_model_arm_a.pt` (deployed; now uses the scaled dataset) | scaling/app |
