# ME2 — Tiny Voice Command Model (VCM)

A **tiny** neural model that understands the most common spoken commands people
give their smart devices (Alexa / Google Home style): play music, control
lights, dim/color, set timers & alarms, adjust the thermostat, ask about
weather/time, media control, reminders, and calls/messages.

Built with an agent (OnIt) — initialized, trained, and committed end-to-end.

## Result

- **Held-out test accuracy: 0.652** (best epoch 0.659) over 10 unseen speakers.
- Model: 2-conv-block CNN over 40×128 log-mel spectrograms, **~1.3M params**.
- The 31 classes include near-duplicate *slotted* variants (e.g. *brightness
  20 / 60 / 100*, *timer 10s / 30s / 1m*) that sound almost identical — that is
  what caps accuracy for a tiny model. Clearly-distinct commands (lights on/off,
  play/pause/stop, weather/time, volume up/down) are separated reliably.

## Dataset

[Option B spoken-command dataset](https://github.com/markandrian30/AI231/tree/main/MEX2/OptionB)
— 17,986 recordings, **100 speakers** (84 foreign + 16 Filipino-English),
**31 command classes** spanning the 10 most common smart-device command
categories, in two acoustic conditions (clean / light-noise).

The 31 classes map onto the ranked command list:

| Rank | Command category | Classes |
|---|---|---|
| 1 | Play music | `PLAY_MUSIC` |
| 2 | Ask a question / search | `WEATHER`, `TIME` |
| 3 | Control lights (IoT) | `LIGHT_ON`, `LIGHT_OFF` |
| 4 | Dim / color lights | `BRIGHTNESS_20/60/100`, `COLOR_RED/BLUE/GREEN` |
| 5 | Set a timer | `TIMER_10s/30s/1m` |
| 6 | Set an alarm | `ALARM_6_00AM/8_00AM/9_00PM` |
| 7 | Adjust thermostat | `TEMPERATURE_18/22/26` |
| 8 | Media control | `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP/DOWN` |
| 9 | Reminders and lists | `CREATE_REMINDER_*`, `LIST_REMINDERS` |
| 10 | Calls and messaging | `CALL`, `MESSAGE` |

**Split** (official, speaker-based): train = 80 speakers, val = 10, test = 10
(the test speakers were never seen in training).

## Data preprocessing — speed augmentation

Per the spec, we **modify the speed** of the training audio to mimic the
different speaking voices/tempos of humans. For every train clip we synthesize
one **time-stretched** copy: the waveform is resampled along the time axis by a
random factor in **[0.82, 1.18]** (factor > 1 = slower, < 1 = faster). Because
only the time axis is resampled, **pitch is preserved** — exactly the
"different speaking voice" effect wanted. The train set becomes *clean +
stretched* (2× the clips); val and test stay clean.

## Pipeline

```
data/OptionB/            (raw wavs + manifest.csv)  [not committed]
   │  preprocess.py      resample 16 kHz -> 40-band log-mel -> 128 frames
   ▼
features.npz             X[N,40,128], y, split, speaker, transcript, classes
   │  train.py           tiny CNN, clean + speed-stretched train set, 20 epochs
   ▼
ckpt.pt + history.json   weights + per-epoch train/val/test accuracy
   │  visualize.py       confusion matrix, 4x4 grid, training curve
   ▼
figures/*.png + results.json
```

## Reproduce

```bash
pip install torch numpy pandas scipy librosa soundfile scikit-learn matplotlib
python preprocess.py     # ~2 min  -> features.npz
python train.py          # ~8 min  -> ckpt.pt, history.json
python visualize.py      # ~30 s   -> figures/, results.json
```

Or run [`vcm_notebook.ipynb`](vcm_notebook.ipynb) end-to-end (every cell is
explained in the markdown above it).

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | wavs → 16 kHz → log-mel → fixed-length features (`features.npz`) |
| `train.py` | tiny CNN model + training (clean + speed-stretched) + eval |
| `visualize.py` | confusion matrix, 4×4 prediction grid, training curve |
| `build_notebook.py` | regenerates the executed notebook with outputs |
| `vcm_notebook.ipynb` | the full, executed, explained notebook (primary deliverable) |
| `README.md` | this file |
