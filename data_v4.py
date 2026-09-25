"""ME2 v4 - Data loading + train/val/test index helpers.

Reuses the precomputed features_v2.npz (X_clean + X_aug log-mel spectrograms,
already augmented at the waveform level: speed/pitch/volume/noise/breaks).
This avoids re-reading 18k wavs on every run.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "features_v2.npz")


def load_features():
    """Load features_v2.npz and return a dict of numpy arrays + meta."""
    z = np.load(FEAT, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {
        "X_clean": z["X_clean"],            # [N, 40, 128] f32
        "X_aug": z["X_aug"],                # [Ntr, 40, 128] f32
        "y_char": z["y_char"],              # [N, 40] int64 (-1 pad)
        "y_len": z["y_len"].astype(np.int64),
        "intent": z["intent"],
        "transcript_raw": z["transcript_raw"],
        "transcript_norm": z["transcript_norm"],
        "split": z["split"],
        "speaker": z["speaker"],
        "n_words": z["n_words"].astype(np.int64),
        "classes": list(z["classes"]),
        "meta": meta,
    }


def split_indices(d):
    split = d["split"]
    return {
        "train": np.where(split == "train")[0],
        "val": np.where(split == "val")[0],
        "test": np.where(split == "test")[0],
    }


def train_union_arrays(d):
    """Return (X, y_char, y_len) for the TRAIN set = clean + augmented union.

    X_aug rows are aligned to the train rows of X_clean (same order), so the
    labels are simply duplicated.  This is the 2x training set.
    """
    tr = np.where(d["split"] == "train")[0]
    X_clean_tr = d["X_clean"][tr]
    X_aug = d["X_aug"]
    X = np.concatenate([X_clean_tr, X_aug], axis=0)
    yc = d["y_char"][tr]
    ylen = d["y_len"][tr]
    y_char = np.concatenate([yc, yc], axis=0)
    y_len = np.concatenate([ylen, ylen], axis=0)
    return X.astype(np.float32), y_char, y_len


def make_target_tensors(y_char, y_len, idx, max_len):
    """Gather target char indices for index array idx; clip to max_len.

    Returns (y_char [N, max_len] with -1->0 for padding, y_len [N]).
    Padding positions beyond y_len are ignored by ctc_loss (uses y_len).
    """
    yc = y_char[idx].copy()
    yl = y_len[idx].astype(np.int64)
    yc = np.minimum(yc, max_len - 1)
    yc[yc < 0] = 0
    return yc, yl
