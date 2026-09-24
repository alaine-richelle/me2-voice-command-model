"""
ME2 - Tiny Voice Command Model (VCM)
Model + training + evaluation.

Tiny CNN over log-mel spectrograms (40 x 128), 31 command classes.

Speed augmentation (spec: "modify the speed of the audio files to mimic the
different speaking voice of humans"): for every TRAIN clip we synthesize one
time-stretched copy (factor drawn from [0.82, 1.18]). Time-stretching via
linear interpolation of the waveform changes tempo/duration while preserving
pitch -- exactly the "different speaking voice" effect requested. The training
set is the UNION of clean + stretched clips (2x), so the model sees each
command at multiple speaking speeds. Val/test stay clean.
"""
import os, json
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.signal import resample_poly
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(__file__)
FEAT = os.path.join(HERE, "features.npz")
DATA = os.path.join(HERE, "data", "OptionB")
CKPT = os.path.join(HERE, "ckpt.pt")

N_CLASSES = 31
EPOCHS    = 20
BATCH     = 256
LR        = 1e-3
SEED      = 0
AUG_LO, AUG_HI = 0.82, 1.18
SR        = 16000


class TinyVCM(nn.Module):
    """Small 2-conv-block CNN classifier over a log-mel spectrogram.

    Input : [B, 1, 40, 128]
    Output: [B, 31] logits
    """
    def __init__(self, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flat = 32 * 10 * 32
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flat, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)


def load_features():
    z = np.load(FEAT, allow_pickle=True)
    X = torch.from_numpy(z["X"]).unsqueeze(1).float()
    y = torch.from_numpy(z["y"]).long()
    split = z["split"]
    classes = list(z["classes"])
    meta = json.loads(str(z["meta"]))
    return X, y, split, classes, meta


def mel_of(x):
    mel = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40, fmin=20, fmax=8000)
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=160)) ** 2
    L = librosa.amplitude_to_db(mel @ S, ref=np.max)
    L = (L + 80.0) / 80.0
    if L.shape[1] >= 128:
        L = L[:, :128]
    else:
        L = np.concatenate([L, np.zeros((40, 128 - L.shape[1]), L.dtype)], 1)
    return L.astype(np.float32)


def build_stretched_train():
    """Return [Ntr,40,128] stretched mels, one per train clip, aligned to split order."""
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man["split"] == "train"].reset_index(drop=True)
    out = []
    for i, row in man.iterrows():
        x, sr = sf.read(os.path.join(DATA, row["path"]), dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g)
        f = np.random.uniform(AUG_LO, AUG_HI)
        n_new = int(round(len(x) * f))
        idx_t = np.linspace(0, len(x) - 1, n_new)
        xs = np.interp(idx_t, np.arange(len(x)), x).astype(np.float32)
        out.append(mel_of(xs))
        if (i + 1) % 2000 == 0:
            print("  stretched", i + 1, "/", len(man))
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        tot += yb.numel()
    return correct / tot


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    X, y, split, classes, meta = load_features()
    tr = (split == "train").nonzero()[0]
    va = (split == "val").nonzero()[0]
    te = (split == "test").nonzero()[0]

    print("building speed-augmented (time-stretched) train copies ...")
    stretched = torch.from_numpy(build_stretched_train()).unsqueeze(1)   # [Ntr,1,40,128]

    # train set = clean + stretched (union), labels aligned
    Xt = torch.cat([X[tr], stretched], 0)
    yt = torch.cat([y[tr], y[tr]], 0)
    print("train samples:", Xt.size(0), "(clean + stretched)")

    dl_tr = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(TensorDataset(X[va], y[va]), batch_size=BATCH)
    dl_te = DataLoader(TensorDataset(X[te], y[te]), batch_size=BATCH)

    model = TinyVCM().to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_acc = n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * yb.size(0)
            run_acc += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
        sched.step()
        tr_acc = run_acc / n
        va_acc = evaluate(model, dl_va, device)
        te_acc = evaluate(model, dl_te, device)
        history.append({"epoch": ep, "train_loss": run_loss / n,
                        "train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc})
        print(f"ep {ep:2d}  loss {run_loss/n:.4f}  train {tr_acc:.4f}  "
              f"val {va_acc:.4f}  test {te_acc:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "classes": classes,
                "history": history, "meta": meta}, CKPT)
    print("final test acc:", history[-1]["test_acc"])
    with open(os.path.join(HERE, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("saved ckpt ->", CKPT)


if __name__ == "__main__":
    main()
