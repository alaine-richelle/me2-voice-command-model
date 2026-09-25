"""
ME2 v2 - Feature extraction + augmentation pipeline.

Reads the OptionB wavs, resamples to 16 kHz, applies realistic augmentation
to TRAIN clips (speed / pitch / volume / noise / word-breaks), computes
40x128 log-mel spectrograms, and caches everything to features_v2.npz.

Output arrays (aligned by row index):
  X_clean   [N, 40, 128]     clean log-mel for every clip
  X_aug     [Ntr, 40, 128]   augmented log-mel for every TRAIN clip (aligned
                             to the train rows of X_clean)
  y_char    [N, T_max]       per-character target indices (CHARSET idx, no
                             blank) for CTC, right-padded with -1
  y_len     [N]              true length of each target sequence
  intent    [N]              intent string
  transcript_raw [N]         original transcript (for display)
  transcript_norm [N]        normalized [a-z ] transcript (ground truth)
  split     [N]              train/val/test
  speaker   [N]
  n_words   [N]              word count (for break placement)
  classes   [31]             the 31 command labels (kept for reference)
"""
import os
import json
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.signal import resample_poly
import librosa

from grammar import normalize_text, CH2IDX
from augment import augment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "OptionB")
OUT = os.path.join(HERE, "features_v2.npz")

SR = 16000
N_MELS = 40
N_FFT = 512
HOP = 160
FMAX = 8000
T_FRAMES = 128
MAX_LEN = 40          # max command characters to model (~2.5s at 128 frames)
SEED = 0


def load_wav_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def logmel(x):
    mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20, fmax=FMAX)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    L = librosa.amplitude_to_db(mel @ S, ref=np.max)
    L = (L + 80.0) / 80.0
    return L


def frame(L, T=T_FRAMES):
    t = L.shape[1]
    if t >= T:
        return L[:, :T]
    pad = np.zeros((N_MELS, T - t), dtype=L.dtype)
    return np.concatenate([L, pad], axis=1)


def char_targets(text):
    """Normalized transcript -> list of CHARSET indices (no blank)."""
    return [CH2IDX[c] for c in text if c in CH2IDX]


def main():
    rng = np.random.default_rng(SEED)
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    print("manifest rows:", len(man))
    print("splits:", man["split"].value_counts().to_dict())

    classes = sorted(man["label"].unique().tolist())
    print("num classes:", len(classes))

    X_clean, y_char, y_len = [], [], []
    intent_l, tr_raw, tr_norm, split_l, spk_l, nwords_l = [], [], [], [], [], []
    tr_idx = []
    bad = 0
    for i, row in man.iterrows():
        p = os.path.join(DATA, row["path"])
        try:
            x = load_wav_16k(p)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print("  BAD", row["path"], repr(e))
            continue
        L = frame(logmel(x))
        X_clean.append(L)
        norm = normalize_text(row["transcript"])
        tgts = char_targets(norm)
        y_char.append(tgts)
        y_len.append(len(tgts))
        intent_l.append(row["intent"])
        tr_raw.append(row["transcript"])
        tr_norm.append(norm)
        split_l.append(row["split"])
        spk_l.append(row["speaker"])
        nwords_l.append(len(norm.split()))
        if row["split"] == "train":
            tr_idx.append(len(X_clean) - 1)
        if (i + 1) % 2000 == 0:
            print("  processed", i + 1, "/", len(man))

    X_clean = np.stack(X_clean).astype(np.float32)
    N = len(X_clean)
    y_char_p = np.full((N, MAX_LEN), -1, dtype=np.int64)
    for i, t in enumerate(y_char):
        y_char_p[i, :min(len(t), MAX_LEN)] = t[:MAX_LEN]
    y_len = np.array(y_len, dtype=np.int64)

    # ---- augment TRAIN clips (one augmented copy each, aligned to train rows)
    print("augmenting train clips ...")
    X_aug_list = []
    for gi in tr_idx:
        row = man.iloc[gi]
        p = os.path.join(DATA, row["path"])
        x = load_wav_16k(p)
        xa = augment(x, int(nwords_l[gi]), rng)
        X_aug_list.append(frame(logmel(xa)))
        if len(X_aug_list) % 2000 == 0:
            print("  augmented", len(X_aug_list), "/", len(tr_idx))
    X_aug = np.stack(X_aug_list).astype(np.float32)   # [Ntr, 40, 128]

    np.savez_compressed(
        OUT,
        X_clean=X_clean, X_aug=X_aug,
        y_char=y_char_p, y_len=y_len,
        intent=np.array(intent_l), transcript_raw=np.array(tr_raw),
        transcript_norm=np.array(tr_norm), split=np.array(split_l),
        speaker=np.array(spk_l), n_words=np.array(nwords_l, dtype=np.int64),
        classes=np.array(classes),
        meta=np.array(json.dumps({
            "sr": SR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP,
            "fmax": FMAX, "T_frames": T_FRAMES, "max_len": MAX_LEN,
            "bad": bad, "n_train_aug": len(X_aug_list),
        }), dtype=object),
    )
    print("X_clean:", X_clean.shape, "X_aug:", X_aug.shape)
    print("saved ->", OUT)
    for s in ["train", "val", "test"]:
        m = np.array(split_l) == s
        print(f"  {s:5s}: n={m.sum():5d}  speakers={np.unique(np.array(spk_l)[m]).size}")
    print("done")


if __name__ == "__main__":
    main()
