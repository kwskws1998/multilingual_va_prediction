# Multilingual ET + GMM + VA Gaze Concat Plan

## Goal

Replace the current ET predictor 2 with a new multilingual ET predictor trained on multilingual eye-tracking data, then test whether gaze concat helps the VA model more than the current English-leaning ET setup.

This is a side project focused on engineering usefulness first:

- build a multilingual ET predictor
- integrate it into the existing VA `gaze concat` pipeline
- compare against:
  - no gaze
  - old ET predictor 2
  - new multilingual ET predictor
- optionally add a Gaussian Mixture Model component to make the gaze representation distribution-aware instead of pure point estimates


## Why This Matters

The current ET predictor 2 is not an ideal cognitive feature source for multilingual VA prediction because it is effectively tied to an English-centered setup. Even if it works, the signal may be noisy or misaligned for non-English text.

The intended replacement is:

- multilingual encoder backbone
- multilingual eye-tracking supervision
- token/word-aligned fixation feature prediction
- direct use inside the current VA gaze concat architecture

The core question is not just "can multilingual ET be predicted?" but:

`Does a multilingual ET-derived gaze representation improve valence/arousal prediction more reliably than the current ET2 pipeline?`


## Current Repo Integration Points

Relevant files in the current codebase:

- `et2_wrapper.py`
- `models.py`
- `train_model.py`
- `fold_runner.py`
- `data_loader.py`
- `prepare_english_data.py`
- `compute_overall_metrics.py`

Current gaze concat flow:

1. `train_model.py` parses `--use-gaze-concat`
2. `fold_runner.py` builds `GazeConcatForSequenceRegression`
3. `models.py` loads the ET wrapper and calls `_compute_mapped_fixations(...)`
4. ET features are projected and concatenated with text embeddings
5. encoder output is used for final VA regression

This means the cleanest swap is:

- keep `models.py` concat logic mostly intact
- add a new multilingual ET wrapper with the same interface as the old ET2 wrapper
- make the VA model select old ET2 or new multilingual ET through config/CLI


## Proposed Architecture

### Stage A: Multilingual ET Predictor

Backbone candidates:

- `xlm-roberta-base`
- `xlm-roberta-large`
- `bert-base-multilingual-cased`

Preferred first pass:

- `xlm-roberta-base`

Reason:

- multilingual
- strong enough
- more realistic to train/debug than `large`

Output targets per token/word:

- `nFix`
- `FFD`
- `GPT`
- `TRT`
- `fixProp`

First version:

- standard regression head
- predict one value per ET feature
- keep outputs compatible with current `features_used` logic


### Stage B: GMM-Augmented Gaze Representation

The GMM component should be added to the gaze representation path, not necessarily to the ET predictor loss on day one.

Recommended first version:

1. multilingual ET predictor outputs raw gaze features
2. fit a GMM over token-level gaze feature vectors
3. compute per-token mixture-aware features such as:
   - posterior probabilities over `K` components
   - entropy / uncertainty
   - optional distance-to-component summaries
4. concatenate:
   - raw gaze features
   - GMM posterior features
   - optional uncertainty features
5. feed this into the existing fixation projector before text concat

This keeps the system practical while still adding distribution-aware information.


## Recommended Implementation Order

### Phase 1: Multilingual ET Predictor Only

Deliverable:

- new wrapper compatible with existing `GazeConcatForSequenceRegression`

Suggested new files:

- `et_multilingual_wrapper.py`
- optional ET training script such as `train_et_multilingual.py`
- optional ET dataset prep script such as `prepare_et_multilingual_data.py`

Requirements:

- same or similar interface as `FixationsPredictor_2`
- expose `_compute_mapped_fixations(input_ids, attention_mask)`
- return mapped token-level feature matrix and attention mask


### Phase 2: Plug Into VA

Deliverable:

- CLI/config option to choose ET source

Suggested flags:

- `--gaze-source et2`
- `--gaze-source multilingual-et`
- `--gaze-source multilingual-et-gmm`
- `--et-multilingual-checkpoint ...`

Likely files to modify:

- `train_model.py`
- `fold_runner.py`
- `models.py`


### Phase 3: Add GMM Features

Deliverable:

- GMM-aware gaze features in the concat path

Possible design:

- fit GMM offline and save parameters
- load GMM parameters in wrapper or model
- convert raw gaze vector to:
  - raw features
  - posterior responsibilities
  - uncertainty scalar(s)


## Recommended Experimental Setup

Minimum comparison:

1. `No gaze`
2. `Old ET2 + gaze concat`
3. `New multilingual ET + gaze concat`
4. `New multilingual ET + GMM + gaze concat`

Keep fixed:

- VA model backbone
- loss function
- learning rate
- batch size
- maxlen
- seed
- fold files

Primary metrics:

- `pearson_corr_valence`
- `pearson_corr_arousal`
- `mse_valence`
- `mse_arousal`
- `mae_valence`
- `mae_arousal`

Use final out-of-fold overall metrics, not just single-fold metrics.

Files already prepared for this:

- `compute_overall_metrics.py`
- `overall_metrics.csv`
- `dataset_metrics.csv`


## Design Choices To Resolve Later

### 1. ET Supervision Granularity

Need to decide:

- word-level supervision only
- token-level supervision derived from word-level labels

Likely path:

- train at word level
- map to tokenizer pieces during inference


### 2. Feature Normalization

Need a consistent policy for ET targets:

- raw values
- z-score per language
- min-max scaling
- log transform for skewed durations

This matters a lot because ET duration features are often heavy-tailed.


### 3. GMM Placement

Three options:

1. GMM over raw ET labels/features as post-processing
2. GMM over hidden ET representation
3. full mixture-density ET predictor

Recommended first pass:

- option 1

Reason:

- easiest to test
- lowest engineering risk
- easiest ablation against raw gaze features


### 4. Global vs Language-Specific GMM

Choices:

- one global GMM for all languages
- one GMM per language
- language-conditioned shared GMM

Recommended first pass:

- one global GMM

Reason:

- simplest
- avoids low-resource instability
- easier to debug


## Practical Build Strategy

### First concrete target

Build:

- multilingual ET predictor wrapper
- no GMM yet
- compatible with current `gaze concat`

Then verify:

- training runs end-to-end
- output shapes match current ET2 path
- VA training does not break

Only after that:

- add GMM features
- rerun ablations


## What To Ask For Next Time

When returning to this project later, ask for:

`Implement the multilingual ET predictor integration from MULTILINGUAL_ET_GMM_PLAN.md.`

Good follow-up requests:

- `Create the multilingual ET wrapper and hook it into gaze concat.`
- `Add CLI options for old ET2 vs new multilingual ET.`
- `Add the GMM feature pipeline on top of multilingual ET outputs.`
- `Prepare the ablation experiment commands and output comparison script.`


## Short Version

The intended final system is:

`text -> multilingual ET predictor -> raw gaze features -> GMM-aware gaze representation -> projector -> gaze concat with text embeddings -> VA regression`

The correct development order is:

1. multilingual ET predictor
2. VA integration
3. GMM augmentation
4. ablation against old ET2 and no-gaze baselines
