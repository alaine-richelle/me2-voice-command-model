"""
ME2 - Tiny Voice Command Model (VCM)
Preprocessing: read OptionB wavs, resample to 16 kHz, compute log-mel spectrograms,
pad/truncate to a fixed length, cache to features.npz.

Uses the official speaker-based split from manifest.csv:
  train: s1-s67, s68-s80   (80 speakers)
  val:   s81-s88, s89-s90  (10 speakers)
  test:  s91-s99, s100     (10 speakers)
"""
import os, sys, json, glob
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data", "OptionB")
OUT  = os.path.join(os.path.dirname(__file__), "features.npz")

SR      = 16000      # target sample rate
N_MELS  = 40
N_FFT   = 512
HOP     = 160
FMAX    = 8000
T_FRAMES= 128        # ~2.048 s at hop 160

def load_wav_16k(path):
    """Load any wav, convert to mono float32, resample to 16 kHz."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        # resample_poly with gcd-based downsampling
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)

def logmel(x):
    """Float32 mono 16k waveform -> log-mel [N_MELS, T]."""
    # librosa 1.x: frequency limits live on the mel filterbank, not melspectrogram
    mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20, fmax=FMAX)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    S = mel @ S
    L = librosa.amplitude_to_db(S, ref=np.max)
    # normalize to roughly [-1, 1] for stable nets
    L = (L + 80.0) / 80.0
    return L

def frame(L, T=T_FRAMES):
    """Pad (right, zeros) or truncate to exactly T frames."""
    t = L.shape[1]
    if t >= T:
        return L[:, :T]
    pad = np.zeros((N_MELS, T - t), dtype=L.dtype)
    return np.concatenate([L, pad], axis=1)

def main():
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    print("manifest rows:", len(man))
    print("split counts:\n", man["split"].value_counts().to_dict())

    classes = sorted(man["label"].unique().tolist())
    cls2idx = {c: i for i, c in enumerate(classes)}
    print("num classes:", len(classes))

    X_list, y_list, split_list, spk_list, tr_list = [], [], [], [], []
    bad = 0
    for i, row in man.iterrows():
        p = os.path.join(DATA, row["path"])
        try:
            x = load_wav_16k(p)
            L = frame(logmel(x))
        except Exception as e:
            bad += 1
            if bad <= 3:
                print("  BAD", row["path"], repr(e))
            continue
        X_list.append(L)
        y_list.append(cls2idx[row["label"]])
        split_list.append(row["split"])
        spk_list.append(row["speaker"])
        tr_list.append(row["transcript"])
        if (i + 1) % 2000 == 0:
            print("  processed", i + 1, "/", len(man))

    X = np.stack(X_list).astype(np.float32)          # [N, N_MELS, T]
    y = np.array(y_list, dtype=np.int64)
    splits = np.array(split_list)
    speakers = np.array(spk_list)
    transcripts = np.array(tr_list)

    np.savez_compressed(OUT,
                        X=X, y=y, split=splits, speaker=speakers,
                        transcript=transcripts, classes=np.array(classes),
                        meta=np.array(json.dumps({
                            "sr": SR, "n_mels": N_MELS, "n_fft": N_FFT,
                            "hop": HOP, "fmax": FMAX, "T_frames": T_FRAMES,
                            "bad": bad,
                        }), dtype=object))

    print("X shape:", X.shape, "dtype", X.dtype)
    print("saved ->", OUT)
    for s in ["train", "val", "test"]:
        m = splits == s
        print(f"  {s:5s}: n={m.sum():5d}  speakers={np.unique(speakers[m]).size}")
    print("done")

if __name__ == "__main__":
    main()
