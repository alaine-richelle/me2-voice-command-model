"""ME2 v6 - per-char CTC + CORRECT blank-aware trie beam search (final, fast).

Design (why this is fast and correct)
-------------------------------------
* Acoustic model: tiny 3-conv + BiGRU, emits a per-frame char distribution
  (CTC head).  Trained with standard CTC loss.
* Selection: each epoch we pick the best model by FAST greedy-CTC intent
  accuracy on the validation set (greedy collapse is vectorizable -> seconds,
  not minutes).  This avoids running the slow trie beam search every epoch.
* Decoding: the CORRECT blank-aware trie beam search (decoder_v6_fast.py) is
  run ONCE at the end on the best model, for the final val + test metrics.
  A repeat is only legal across a blank; every non-blank advances the trie.
  This fixes v5's decoder bug (v5 allowed repeat-without-blank), lifting test
  intent acc 0.389 -> ~0.85 on the v5 weights alone.
* Augmentation: existing waveform augmentation (speed/pitch/volume/noise/
  breaks, cached in features_v2.npz) + online SpecAugment-style time/freq
  masking + time-shift on every TRAIN batch (val/test stay clean).
"""
import os, sys, json, time, logging, pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grammar_v4 import (N_OUT, BLANK, build_trie, normalize_text,
                        CH2IDX, IDX2CH)
from data_v4 import load_features, split_indices, train_union_arrays
from decoder_v6_fast import build_flat, beam_search_vec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS = 15
BATCH = 256
LR = 3e-3
SEED = 0
T_FRAMES = 32
CLIP = 30              # <= T_FRAMES so every target is CTC-feasible
BEAM = 4               # validated best (intent 0.852 on v5 weights)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUG = {
    "time_mask_n": 2, "time_mask_w": 8,
    "freq_mask_n": 2, "freq_mask_w": 8,
    "time_shift": 8, "prob": 1.0,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(HERE, "train_v6.log"), mode="w"),
                              logging.StreamHandler()])
log = logging.getLogger("v6")

import os as _os
torch.set_num_threads(min(64, _os.cpu_count() or 8))
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PerCharCNN(nn.Module):
    """input [B,1,40,128] -> [B,32,N_OUT]  (~3.65M params, tiny)."""

    def __init__(self, n_mels=40, n_out=N_OUT, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128 * 10, 256, num_layers=2, bidirectional=True,
                          dropout=dropout, batch_first=True)
        self.head = nn.Linear(512, n_out)

    def forward(self, x):
        h = self.cnn(x)
        h = h.permute(0, 3, 1, 2)
        h = h.reshape(h.size(0), h.size(1), -1)
        h, _ = self.gru(h)
        return self.head(h)


def ctc_loss(logits, targets, target_lengths, n_out=N_OUT):
    log_probs = F.log_softmax(logits, dim=-1)
    B, T, C = logits.shape
    lp = log_probs.permute(1, 0, 2).contiguous()
    input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
    return F.ctc_loss(lp, targets, input_lengths, target_lengths.long(),
                      blank=(n_out - 1), zero_infinity=False)


# ---------------------------------------------------------------------------
# Online feature augmentation (TRAIN only)
# ---------------------------------------------------------------------------
def feat_aug(x):
    """Vectorized SpecAugment-style masking + time shift (tensor ops, no sync).

    Time masking: shift each sample's spectrogram by a random offset and zero
    a random-width band (equivalent to masking a random interval).
    Freq masking: zero a random-width band of mel bins.
    """
    if np.random.rand() > AUG["prob"]:
        return x
    B, _, Fq, T = x.shape
    out = x.clone()
    dev = out.device
    # time masks
    for _ in range(AUG["time_mask_n"]):
        w = torch.randint(1, AUG["time_mask_w"] + 1, (1,)).item()
        t0 = torch.randint(0, max(1, T - w), (B, 1, 1, 1)).to(dev)
        ar = torch.arange(T, device=dev).view(1, 1, 1, T)
        band = (ar >= t0) & (ar < t0 + w)
        out = out.masked_fill(band, 0.0)
    # freq masks
    for _ in range(AUG["freq_mask_n"]):
        w = torch.randint(1, AUG["freq_mask_w"] + 1, (1,)).item()
        f0 = torch.randint(0, max(1, Fq - w), (B, 1, 1, 1)).to(dev)
        arf = torch.arange(Fq, device=dev).view(1, 1, Fq, 1)
        bandf = (arf >= f0) & (arf < f0 + w)
        out = out.masked_fill(bandf, 0.0)
    if AUG["time_shift"] > 0:
        sshift = np.random.randint(-AUG["time_shift"], AUG["time_shift"] + 1)
        if sshift != 0:
            out = torch.roll(out, shifts=sshift, dims=-1)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(x):
    return (x - 0.1307) / 0.3081


def _intent_of(trie, text):
    if not text:
        return None
    node = trie
    for ch in text:
        node = node.children.get(ch)
        if node is None:
            return None
    return node.terminals[0][1] if node.terminals else None


@torch.no_grad()
def greedy_eval(model, X, intents, norm_texts, trie, batch=512):
    """FAST selection metric: greedy CTC -> intent accuracy (vectorizable)."""
    model.eval()
    n = X.shape[0]
    ok = 0
    for s in range(0, n, batch):
        xb = torch.from_numpy(_norm(X[s:s+batch]).astype(np.float32))[:, None].to(DEVICE)
        logits = model(xb)
        lp = F.log_softmax(logits, dim=-1).cpu().numpy()
        for j in range(lp.shape[0]):
            path, prev = [], None
            for t in range(lp[j].shape[0]):
                c = int(np.argmax(lp[j][t]))
                if c != prev:
                    path.append(c)
                prev = c
            gt = "".join(IDX2CH[c] for c in path if c != BLANK)
            ok += int(_intent_of(trie, gt) == intents[s + j])
    return ok / n


@torch.no_grad()
def beam_eval(model, X, intents, norm_texts, trie, es, ed, eci, terminals,
              batch=256):
    """SLOW final metric: correct trie beam search.  Returns records + counts."""
    model.eval()
    n = X.shape[0]
    b_exact = b_intent = 0
    records = []
    per = defaultdict(lambda: [0, 0])
    confused = defaultdict(int)
    for s in range(0, n, batch):
        xb = torch.from_numpy(_norm(X[s:s+batch]).astype(np.float32))[:, None].to(DEVICE)
        logits = model(xb)
        lp = F.log_softmax(logits, dim=-1).cpu().numpy()
        for j in range(lp.shape[0]):
            gi = s + j
            bt, bn, bs = beam_search_vec(lp[j], es, ed, eci, terminals, beam_width=BEAM)
            bi = _intent_of(trie, bt) if bt else None
            gold_i = intents[gi]
            b_exact += int(bt == norm_texts[gi])
            ok = int(bi == gold_i)
            b_intent += ok
            per[gold_i][0] += 1
            per[gold_i][1] += ok
            if not ok:
                confused[(gold_i, bi)] += 1
            records.append({"idx": int(gi), "gold": gold_i, "pred": bi,
                            "gold_text": norm_texts[gi], "pred_text": bt,
                            "conf": float(np.exp(max(bs, -30))) if bt else 0.0})
    return {"beam_exact": b_exact / n, "beam_intent": b_intent / n,
            "records": records, "per": dict(per), "confused": confused}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    d = load_features()
    sp = split_indices(d)
    Xtr, ytr, yltr = train_union_arrays(d)
    Xv = d["X_clean"][sp["val"]]; iv = sp["val"]
    Xt = d["X_clean"][sp["test"]]; it = sp["test"]
    log.info("train=%d (clean+aug) val=%d test=%d", Xtr.shape[0], Xv.shape[0], Xt.shape[0])

    # targets: clip content AND length to CLIP (<= T_FRAMES) => CTC-feasible
    ytr_c = np.minimum(ytr, CLIP - 1); ytr_c[ytr_c < 0] = 0
    ytr_l = np.minimum(yltr, CLIP)

    t2i = {}
    for t, i in zip(d["transcript_norm"], d["intent"]):
        t2i.setdefault(t, i)
    trie = build_trie(list(t2i.items()))
    es, ed, eci, terminals, nnodes = build_flat(trie)
    log.info("trie nodes=%d terminals=%d", nnodes, len(terminals))

    model = PerCharCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params=%d", n_params)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    ds = TensorDataset(torch.from_numpy(_norm(Xtr).astype(np.float32))[:, None],
                       torch.from_numpy(ytr_c), torch.from_numpy(ytr_l.astype(np.int64)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)

    best = (-1.0, None)
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, yc, yl in dl:
            xb = feat_aug(xb.to(DEVICE, non_blocking=True))
            opt.zero_grad()
            logits = model(xb)
            loss = ctc_loss(logits, yc.to(DEVICE), yl.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * xb.size(0)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        gv = greedy_eval(model, Xv, d["intent"][iv], d["transcript_norm"][iv], trie)
        history.append({"epoch": ep, "train_loss": tot / Xtr.shape[0], "lr": lr,
                        "val_greedy_intent": gv})
        log.info("ep %2d loss=%.4f lr=%.5f | val_greedy_intent=%.4f (%.0fs)",
                 ep, tot / Xtr.shape[0], lr, gv, time.time() - t0)
        if gv > best[0] + 1e-4:
            best = (gv, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    log.info("best val greedy_intent=%.4f -> running final beam decode", best[0])

    vev = beam_eval(model, Xv, d["intent"][iv], d["transcript_norm"][iv],
                    trie, es, ed, eci, terminals)
    tev = beam_eval(model, Xt, d["intent"][it], d["transcript_norm"][it],
                    trie, es, ed, eci, terminals)
    log.info("VAL  beam_exact=%.4f beam_intent=%.4f", vev["beam_exact"], vev["beam_intent"])
    log.info("TEST beam_exact=%.4f beam_intent=%.4f", tev["beam_exact"], tev["beam_intent"])

    per_intent = []
    for k in sorted(tev["per"].keys()):
        totc, cor = tev["per"][k]
        per_intent.append([k, totc, cor, round(cor / totc, 4) if totc else 0.0])
    per_intent.sort(key=lambda r: r[3])
    top_conf = [(f"{a}->{b}", c) for (a, b), c in sorted(tev["confused"].items(),
                                                         key=lambda kv: -kv[1])[:15]]

    def dump(recs, path):
        import csv
        with open(os.path.join(HERE, path), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gold_intent", "pred_intent", "gold_text",
                        "pred_text", "confidence", "correct"])
            for r in recs:
                w.writerow([r["idx"], r["gold"], r["pred"], r["gold_text"],
                            r["pred_text"], f"{r['conf']:.4f}",
                            int(r["pred"] == r["gold"])])
    dump(vev["records"], "predictions_val.csv")
    dump(tev["records"], "predictions_test.csv")

    metrics = {
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
                   "beam": BEAM, "clip": CLIP, "t_frames": T_FRAMES,
                   "n_params": n_params, "feature": "40x128 log-mel",
                   "sr": 16000, "version": "v6", "online_aug": AUG},
        "history": history,
        "best_val_greedy_intent": best[0],
        "val": {"beam_exact": vev["beam_exact"], "beam_intent": vev["beam_intent"]},
        "test": {"beam_exact": tev["beam_exact"], "beam_intent_acc": tev["beam_intent"]},
        "per_intent": per_intent,
        "confused": top_conf,
    }
    with open(os.path.join(HERE, "metrics_v6.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    bundle = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": "PerCharCNN",
        "config": metrics["config"],
        "intents": sorted(set(d["intent"].tolist())),
        "classes": list(d["classes"]),
        "charset": list(IDX2CH.values()),
        "blank": BLANK,
        "best_val_greedy_intent": best[0],
        "test": metrics["test"],
        "history": history,
        "trie_transcripts": list(t2i.keys()),
        "trie_intents": t2i,
    }
    with open(os.path.join(HERE, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    log.info("saved model.pkl + metrics_v6.json  (total %.0fs)", time.time() - t0)


if __name__ == "__main__":
    main()
