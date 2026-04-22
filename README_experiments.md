# VA + Gaze Experiment Runbook

This file is a practical guide to run combinations of:

- model (`distilbert`, `xlmroberta-base`, `xlmroberta-large`)
- loss (`mse`, `ccc`, `robust`, `mse+ccc`, `robust+ccc`)
- gaze fusion (off/on)
- ET features (`nFix,FFD,GPT,TRT,fixProp`)
- batch size
- optimizer and main optimization hyperparameters

---

## 1) Default VA hyperparameters (unchanged)

If you run with only required args, you keep the original VA defaults:

```bash
python train_model.py xlmroberta-base mse+ccc
```

Default training settings:

- batch size (all models): `16`
- learning rate: `6e-6`
- epochs: `10`
- weight decay: `0.01`
- warmup ratio: `0.1`
- optimizer: `adamw_torch`
- gradient accumulation steps: `1`
- seed: `42`
- maxlen: `200`

---

## 2) Full CLI options you can combine

```bash
python train_model.py <model> <loss> \
  [--use-gaze-concat] \
  [--et2-checkpoint <path>] \
  [--features-used <f1,f2,f3,f4,f5>] \
  [--fp-dropout <p1,p2>] \
  [--batch-size <int>] \
  [--batch-size-distil <int>] \
  [--batch-size-xlmrb <int>] \
  [--batch-size-xlmrl <int>] \
  [--learning-rate <float>] \
  [--train-epochs <int>] \
  [--weight-decay <float>] \
  [--warmup-ratio <float>] \
  [--optim <name>] \
  [--gradient-accumulation-steps <int>] \
  [--seed <int>] \
  [--maxlen <int>]
```

Notes:

- `--features-used` order is always: `nFix,FFD,GPT,TRT,fixProp`
- if `--use-gaze-concat` is not set, gaze features are ignored
- `--batch-size` sets all three models unless a model-specific batch size override is provided
- with `--use-gaze-concat`, use `--maxlen <= 255` to stay within encoder positional limits

---

## 3) Single-run templates

### A. Text-only VA (no gaze)

```bash
python train_model.py xlmroberta-base mse+ccc \
  --batch-size 16 \
  --learning-rate 6e-6 \
  --optim adamw_torch
```

### B. VA + GazeConcat (ET2, fcomb2.2 = FFD+TRT)

```bash
python train_model.py xlmroberta-base mse+ccc \
  --use-gaze-concat \
  --et2-checkpoint ./checkpoints/et_predictor2_seed123 \
  --features-used 0,1,0,1,0 \
  --fp-dropout 0.1,0.3 \
  --batch-size 8 \
  --optim adamw_torch
```

### C. Different optimizer/batch size

```bash
python train_model.py distilbert robust \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --optim adafactor \
  --train-epochs 8
```

---

## 4) Feature combinations

Each feature flag is binary (`0/1`) in order:

1. `nFix`
2. `FFD`
3. `GPT`
4. `TRT`
5. `fixProp`

Examples:

- all features: `1,1,1,1,1`
- fcomb2.2 (FFD+TRT): `0,1,0,1,0`
- TRT only: `0,0,0,1,0`
- FFD only: `0,1,0,0,0`

Total non-empty feature combos: `31`.

---

## 5) Grid search examples

### A. Sweep models x losses (text-only)

```bash
MODELS=(distilbert xlmroberta-base xlmroberta-large)
LOSSES=(mse ccc robust mse+ccc robust+ccc)

for MODEL in "${MODELS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    python train_model.py "$MODEL" "$LOSS" \
      --batch-size 16 \
      --learning-rate 6e-6 \
      --optim adamw_torch
  done
done
```

### B. Sweep gaze feature sets x optimizer x batch size

```bash
MODEL=xlmroberta-base
LOSS=mse+ccc
ET2=./checkpoints/et_predictor2_seed123

FEATURES=(
  "1,1,1,1,1"
  "0,1,0,1,0"
  "0,1,0,0,0"
  "0,0,0,1,0"
)
BATCHES=(8 16)
OPTIMS=(adamw_torch adamw_hf adafactor)

for FEAT in "${FEATURES[@]}"; do
  for BS in "${BATCHES[@]}"; do
    for OPT in "${OPTIMS[@]}"; do
      python train_model.py "$MODEL" "$LOSS" \
        --use-gaze-concat \
        --et2-checkpoint "$ET2" \
        --features-used "$FEAT" \
        --fp-dropout 0.1,0.3 \
        --batch-size "$BS" \
        --optim "$OPT"
    done
  done
done
```

### C. Exhaustive full sweep skeleton

Use carefully (this can be very large):

```bash
MODELS=(distilbert xlmroberta-base xlmroberta-large)
LOSSES=(mse ccc robust mse+ccc robust+ccc)
GAZE=(off on)
FEATURES=("1,1,1,1,1" "0,1,0,1,0" "0,1,0,0,0" "0,0,0,1,0")
BATCHES=(8 16)
OPTIMS=(adamw_torch adafactor)
ET2=./checkpoints/et_predictor2_seed123

for MODEL in "${MODELS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    for G in "${GAZE[@]}"; do
      for BS in "${BATCHES[@]}"; do
        for OPT in "${OPTIMS[@]}"; do
          if [ "$G" = "off" ]; then
            python train_model.py "$MODEL" "$LOSS" \
              --batch-size "$BS" \
              --optim "$OPT"
          else
            for FEAT in "${FEATURES[@]}"; do
              python train_model.py "$MODEL" "$LOSS" \
                --use-gaze-concat \
                --et2-checkpoint "$ET2" \
                --features-used "$FEAT" \
                --batch-size "$BS" \
                --optim "$OPT"
            done
          fi
        done
      done
    done
  done
done
```

---

## 6) Recommended first pass

1. Keep original VA defaults first (sanity baseline).
2. Add gaze with `0,1,0,1,0` and same optimizer/lr.
3. Sweep only one axis at a time (batch size, then optimizer, then features).
4. Use same seed for direct comparisons, then multi-seed for final reporting.
