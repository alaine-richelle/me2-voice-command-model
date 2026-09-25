# ME2 — Tiny Voice Command Model (VCM)

A **tiny** neural model that understands the most common spoken commands people
give their smart devices (Alexa / Google Home style): play music, control
lights, dim/color, set timers & alarms, adjust the thermostat, ask about
weather/time, media control, reminders, and calls/messages.

Built with an agent (OnIt) — initialized, trained, and committed end-to-end.

## Approach (v6): per-character CTC + grammar (trie) decoding

Instead of classifying the whole command in one shot, the model:

1. **Acoustic model** — a tiny 3-conv-block CNN + BiGRU over a 40×128 log-mel
   spectrogram emits **one probability distribution over the character
   alphabet (+ CTC blank) per time-step** (32 frames).
2. **Grammar decoding** — a **trie** built over the legal command transcripts
   is searched with a **blank-aware CTC beam search** that walks the trie one
   character at a time, so the decoder can *never* emit a non-command string.
   This is the "predict per character/phoneme, then run a search on a grammar
   data structure (trie) to iteratively form the command" approach.

**Decoder fix (the key improvement this iteration).** The v5 beam search
allowed a beam to repeat a character on a consecutive frame *without* a blank in
between — illegal in CTC. That double-counted repeats and drifted the beam onto
short high-probability leaves (`STOP`, `TIME`), collapsing long commands. The v6
decoder uses the correct previous-symbol state machine (a repeat is only legal
across a blank; every non-blank advances the trie). On the v5 weights alone this
lifted test intent accuracy **0.389 → 0.852**; after retraining it reaches
**0.866**.

## Result

- **Held-out test (10 unseen speakers):**
  - intent accuracy (beam, trie-constrained): **0.866**
  - exact-transcript accuracy (beam): **0.741**
- Model: **~3.65M params**, input 40×128 log-mel, 19 intents / 31 slotted
  classes. Still tiny for an on-device VCM.
- Clearly-distinct commands (lights on/off, play/pause/stop, weather/time,
  volume up/down, brightness) are separated reliably; the residual errors are
  concentrated in near-duplicate *slotted* variants and short vs. long commands
  (see `results_v6/error_analysis.txt`).

## Dataset

[Option B spoken-command dataset](https://github.com/markandrian30/AI231/tree/main/MEX2/OptionB)
— 17,986 recordings, **100 speakers** (84 foreign + 16 Filipino-English),
**31 command classes** spanning the 10 most common smart-device command
categories, in two acoustic conditions (clean / light-noise).

| Rank | Command category | Intents |
|---|---|---|
| 1 | Play music | `PLAY_MUSIC` |
| 2 | Ask a question / search | `WEATHER`, `TIME` |
| 3 | Control lights (IoT) | `LIGHT_ON`, `LIGHT_OFF` |
| 4 | Dim / color lights | `BRIGHTNESS`, `COLOR` |
| 5 | Set a timer | `TIMER` |
| 6 | Set an alarm | `ALARM` |
| 7 | Adjust thermostat | `TEMPERATURE` |
| 8 | Media control | `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN` |
| 9 | Reminders and lists | `CREATE_REMINDER`, `LIST_REMINDERS` |
| 10 | Calls and messaging | `CALL`, `MESSAGE` |

**Split** (official, speaker-based): train = 80 speakers, val = 10, test = 10
(the test speakers were never seen in training).

## Data preprocessing & augmentation

Raw wavs are resampled to 16 kHz and turned into **40-band × 128-frame log-mel**
spectrograms (the compact representation chosen for a tiny on-device model),
then normalized `(x − 0.1307) / 0.3081`.

**Waveform augmentation (train-only, cached in `features_v2.npz`)** mimics the
"real speech" variations the spec asks for — speed, pitch, volume, noise, and
word-breaks — so the model tolerates different voices and pronunciation:

- **speed** — time-stretch (pitch-preserving), factor ∈ [0.82, 1.18]
- **pitch** — resample-shift, factor ∈ [0.85, 1.15] (higher/lower voice)
- **volume** — random gain ∈ [0.6, 1.4]
- **noise** — white + pink noise at SNR ∈ [10, 30] dB (can mask consonants)
- **breaks** — short silence gaps at random word boundaries (hesitations)

**Online feature augmentation (train-only, every batch)** — SpecAugment-style
time/frequency masking + random time-shift, applied to the log-mel. Val/test
stay clean (no leakage; the split is fixed before any augmentation).

## Pipeline

```
data/OptionB/            (raw wavs + manifest.csv)  [not committed]
   │  build_features.py  16 kHz -> 40x128 log-mel + waveform augmentation
   ▼
features_v2.npz          X_clean, X_aug, per-char targets, split, intent
   │  train_v6.py        tiny CNN+BiGRU, CTC loss, greedy-intent model selection
   ▼
model.pkl + metrics_v6.json   canonical model + per-epoch history + test metrics
   │  evaluate_v6.py     full metric suite + confusion matrix + error analysis
   ▼
results_v6/              metrics.json, confusion_matrix.png, predictions_*.csv,
                         error_analysis.txt
```

## Reproduce

```bash
pip install torch numpy pandas scipy librosa soundfile scikit-learn matplotlib

# 1. build features (once; ~a few minutes)  -> features_v2.npz
python build_features.py

# 2. train (CPU, ~30 min for 15 epochs)     -> model.pkl, metrics_v6.json
python train_v6.py

# 3. full evaluation + error analysis        -> results_v6/
python evaluate_v6.py

# 4. inference
python inference.py --file data/OptionB/<some>.wav
python inference.py --microphone
```

## Microphone inference & adding recordings

```bash
python inference.py --microphone
```

1. Asks whether to enable microphone input.
2. Records for a configurable duration (`--duration`, default 3 s).
3. Applies the **exact same** preprocessing as training (16 kHz → 40×128
   log-mel, same normalization).
4. Runs `model.pkl` and prints the predicted command + confidence, flagging
   low-confidence (< 0.35) predictions as uncertain.
5. Asks whether to **save the recording as a new dataset sample**; if yes it
   asks for the **correct** label (it never auto-accepts the model's prediction)
   and stores it under `user_recordings/<label>/` with a metadata JSON
   (filename, command, timestamp, sample rate, duration, source).

Recordings are stored separately and are **not** automatically added to
training — incorporate them deliberately later.

> Note: microphone capture needs `sounddevice` (PortAudio). In a headless
> sandbox there is no mic, so `--file <wav>` is the portable path; the mic path
> degrades gracefully with a clear message.

## Model artifact

The single canonical model is **`model.pkl`** (Python `pickle`). It contains
everything needed for inference: model state, architecture name, label/intent
mapping, charset, blank index, preprocessing config, and the trie transcripts +
intent mapping. Obsolete checkpoints are not kept.

## Files

| File | Purpose |
|---|---|
| `build_features.py` | wavs → 16 kHz → 40×128 log-mel + waveform augmentation (`features_v2.npz`) |
| `augment.py` | waveform augmentation (speed/pitch/volume/noise/breaks) |
| `grammar_v4.py` | character alphabet, text normalization, trie (the grammar) |
| `decoder_v6.py` | correct blank-aware CTC trie beam search (reference, scalar) |
| `decoder_v6_fast.py` | vectorized blank-aware CTC trie beam search (used at runtime) |
| `data_v4.py` | feature loading + train/val/test index helpers |
| `train_v6.py` | tiny CNN+BiGRU + CTC training + greedy-intent selection + final beam eval |
| `evaluate_v6.py` | accuracy/precision/recall/F1/ROC-AUC + confusion matrix + error analysis |
| `inference.py` | file / microphone inference + confirmed-recording storage |
| `model.pkl` | **canonical** model (pickle) |
| `metrics_v6.json` | per-epoch history + test metrics + per-intent + confused pairs |
| `results_v6/` | metrics.json, confusion_matrix.png, predictions_*.csv, error_analysis.txt |
| `vcm_notebook.ipynb` | executed, explained notebook (primary deliverable) |
| `README.md` | this file |
