"""ME2 v6 - Full evaluation + error analysis from the canonical model.pkl.

Loads model.pkl (model_state + trie), reconstructs the model + grammar, and on
the VALIDATION and TEST splits computes:
  * accuracy, macro/weighted precision, recall, F1
  * multiclass one-vs-rest ROC-AUC (from probabilities, NOT hard predictions)
  * confusion matrix (saved as PNG + printed)
  * per-class precision/recall/F1/support
  * per-sample prediction log (CSV): input, gold, pred, confidence, correct
  * most-confused command pairs

Writes:
  results_v6/metrics.json
  results_v6/confusion_matrix.png
  results_v6/predictions_val.csv
  results_v6/predictions_test.csv
  results_v6/error_analysis.txt
"""
import os, sys, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_v6")
os.makedirs(OUT, exist_ok=True)

from grammar_v4 import BLANK, IDX2CH, build_trie, N_OUT
from data_v4 import load_features, split_indices
from decoder_v6 import prepare_trie, ctc_trie_beam
from decoder_v6_fast import build_flat, beam_candidates, beam_search_vec
import pickle

NEG = -1e30
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BEAM = 4


def norm(x):
    return (x - 0.1307) / 0.3081


class PerCharCNN(nn.Module):
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
def run_split(model, X, intents, texts, trie, children, terminals, es, ed, eci, label2idx, n_classes):
    model.eval()
    n = X.shape[0]
    preds = np.zeros(n, dtype=int)
    golds = np.zeros(n, dtype=int)
    confs = np.zeros(n)
    records = []
    # collect per-class posterior (avg over frames) for ROC-AUC
    # We approximate the command posterior from the trie beam: build a
    # one-vs-rest score = P(best terminal of that class) via beam scores.
    # Simpler & correct: use the beam's per-terminal log-scores aggregated
    # by intent.  We run the beam and keep top terminal scores.
    proba = np.zeros((n, n_classes))
    for i in range(0, n, 256):
        xb = torch.from_numpy(norm(X[i:i+256]).astype(np.float32))[:, None].to(DEV)
        logits = model(xb)
        lp = F.log_softmax(logits, dim=-1).cpu().numpy()
        for j in range(lp.shape[0]):
            gi = i + j
            bt, bn, bs = ctc_trie_beam(lp[j], children, terminals, beam_width=BEAM)
            bi = _intent_of(trie, bt) if bt else None
            p = label2idx.get(bi, 0)
            preds[gi] = p
            golds[gi] = label2idx[intents[gi]]
            confs[gi] = float(np.exp(max(bs, -30))) if bt else 0.0
            # proper per-class posterior: aggregate beam candidate scores by
            # intent, then softmax over the 19 intents (one-vs-rest for AUC).
            cands = beam_candidates(lp[j], es, ed, eci, terminals,
                                    beam_width=BEAM, topk=20)
            agg = {}
            for (sc, tx, nd) in cands:
                it_ = _intent_of(trie, tx) if tx else None
                if it_ is None:
                    continue
                agg[it_] = max(agg.get(it_, NEG), sc)
            if agg:
                vals = np.array([agg.get(k, NEG) for k in label2idx])
                vals = vals - vals.max()
                e = np.exp(vals)
                proba[gi] = e / (e.sum() + 1e-12)
            else:
                proba[gi, p] = 1.0
            records.append((gi, intents[gi], bi, confs[gi],
                            int(bi == intents[gi])))
    return preds, golds, confs, proba, records


def metrics_from(preds, golds, proba, label2idx, intents):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score, confusion_matrix)
    acc = accuracy_score(golds, preds)
    prec, rec, f1, sup = precision_recall_fscore_support(
        golds, preds, average=None, zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        golds, preds, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        golds, preds, average="weighted", zero_division=0)
    # multiclass one-vs-rest ROC-AUC (needs >=2 classes present)
    try:
        auc_m = roc_auc_score(golds, proba, multi_class="ovr", average="macro")
        auc_w = roc_auc_score(golds, proba, multi_class="ovr", average="weighted")
    except ValueError as e:
        auc_m = auc_w = float("nan")
    cm = confusion_matrix(golds, preds, labels=list(range(len(label2idx))))
    # per-class table
    rows = []
    for pos, lab in enumerate(label2idx):
        rows.append({"intent": lab, "support": int(sup[pos]),
                     "precision": round(float(prec[pos]), 4),
                     "recall": round(float(rec[pos]), 4),
                     "f1": round(float(f1[pos]), 4)})
    rows.sort(key=lambda r: r["f1"])
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "roc_auc_macro": round(float(auc_m), 4),
        "roc_auc_weighted": round(float(auc_w), 4),
        "per_class": rows,
        "confusion": cm.tolist(),
    }


def main():
    with open(os.path.join(HERE, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = PerCharCNN()
    model.load_state_dict(bundle["model_state"])
    model.to(DEV).eval()

    t2i = bundle["trie_intents"]
    trie = build_trie(list(t2i.items()))
    children, terminals = prepare_trie(trie)
    es, ed, eci, terminals2, _ = build_flat(trie)
    terminals = terminals2

    d = load_features()
    sp = split_indices(d)
    intents_all = d["intent"]
    label2idx = {lab: i for i, lab in enumerate(sorted(set(intents_all.tolist())))}
    n_classes = len(label2idx)

    results = {"config": bundle["config"], "n_classes": n_classes}
    for split in ["val", "test"]:
        idx = sp[split]
        X = d["X_clean"][idx]
        ints = intents_all[idx]
        texts = d["transcript_norm"][idx]
        preds, golds, confs, proba, records = run_split(
            model, X, ints, texts, trie, children, terminals, es, ed, eci, label2idx, n_classes)
        m = metrics_from(preds, golds, proba, label2idx, ints)
        results[split] = m
        # per-sample CSV
        with open(os.path.join(OUT, f"predictions_{split}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gold_intent", "pred_intent", "confidence", "correct"])
            for (gi, g, p, c, ok) in records:
                w.writerow([int(gi), g, p, f"{c:.4f}", int(ok)])
        print(f"\n===== {split.upper()} ({len(idx)} samples) =====")
        print(f"Accuracy : {m['accuracy']:.4f}")
        print(f"Precision: macro {m['precision_macro']:.4f}  weighted {m['precision_weighted']:.4f}")
        print(f"Recall   : macro {m['recall_macro']:.4f}  weighted {m['recall_weighted']:.4f}")
        print(f"F1       : macro {m['f1_macro']:.4f}  weighted {m['f1_weighted']:.4f}")
        print(f"ROC-AUC  : macro {m['roc_auc_macro']:.4f}  weighted {m['roc_auc_weighted']:.4f}")

    # confusion matrix figure (test)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(results["test"]["confusion"])
        labs = list(label2idx.keys())
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=90, fontsize=7)
        ax.set_yticklabels(labs, fontsize=7)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        ax.set_title("ME2 v6 test confusion matrix (intent)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=110)
        print("\nwrote confusion_matrix.png")
    except Exception as e:
        print("confusion figure skipped:", repr(e))

    # error analysis: confused pairs (test)
    idx = sp["test"]
    ints = intents_all[idx]
    # recompute confused pairs from records
    confused = Counter()
    with open(os.path.join(OUT, "predictions_test.csv")) as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row["correct"] == "0":
                confused[(row["gold_intent"], row["pred_intent"])] += 1
    with open(os.path.join(OUT, "error_analysis.txt"), "w") as f:
        f.write("ME2 v6 - ERROR ANALYSIS (test split)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Worst-recognized intents (by F1):\n")
        for r in results["test"]["per_class"][:8]:
            f.write(f"  {r['intent']:18s} support={r['support']:4d}  "
                    f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}\n")
        f.write("\nBest-recognized intents (by F1):\n")
        for r in results["test"]["per_class"][-8:]:
            f.write(f"  {r['intent']:18s} support={r['support']:4d}  "
                    f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}\n")
        f.write("\nMost-confused pairs (gold -> pred : count):\n")
        for (g, p), c in confused.most_common(20):
            f.write(f"  {g:18s} -> {str(p):18s} : {c}\n")
    print("wrote error_analysis.txt")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote results_v6/metrics.json")


if __name__ == "__main__":
    main()
