"""
ME2 - Tiny Voice Command Model (VCM)
Evaluation + visualization helpers (confusion matrix, 4x4 grid of test clips,
training curve). Produces PNG figures + a results dict.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from train import TinyVCM, make_loaders, evaluate

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "ckpt.pt")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyVCM(n_classes=len(ck["classes"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["classes"], ck["history"], ck["meta"]


@torch.no_grad()
def predict_all(model, loader, device="cpu"):
    ys, preds = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def make_confusion(model, device="cpu"):
    dl, idx, classes = make_loaders()
    y, p = predict_all(model, dl["test"], device)
    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    acc = np.trace(cm) / cm.sum()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Test confusion matrix  (acc = {acc:.4f})")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=4,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "confusion.png"), dpi=130)
    plt.close(fig)
    return acc, y, p, classes


def make_grid(model, device="cpu", n=16, seed=0):
    """Pick n test clips (spread across classes), draw 4x4 mel grids w/ GT vs pred."""
    dl, idx, classes = make_loaders()
    X, y, split, classes_arr, meta = None, None, None, None, None
    import numpy as _np
    z = _np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)
    X = z["X"]; y = z["y"]; split = z["split"]
    test_idx = (split == "test").nonzero()[0]
    rng = _np.random.default_rng(seed)
    # pick ~1-2 per class, spread
    picks = []
    for c in range(len(classes_arr)):
        cand = test_idx[y[test_idx] == c]
        if len(cand):
            picks.append(rng.choice(cand))
    picks = picks[:n]
    while len(picks) < n:
        picks.append(rng.choice(test_idx))
    picks = picks[:n]

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[picks]).unsqueeze(1).float()).argmax(1).numpy()

    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for ax, pi, pr in zip(axes.ravel(), picks, logits):
        mel = X[pi]
        im = ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        ok = pr == y[pi]
        ax.set_title(f"GT:{classes_arr[y[pi]]}  PRED:{classes_arr[pr]}",
                     fontsize=7, color=("green" if ok else "red"))
        ax.set_facecolor("black")
    fig.suptitle("16 held-out test clips  (green = correct, red = wrong)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGDIR, "grid16.png"), dpi=130)
    plt.close(fig)
    return picks, y, logits, classes_arr


def make_curve(history):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    eps = [h["epoch"] for h in history]
    ax1.plot(eps, [h["train_loss"] for h in history], "o-", label="train loss", color="tab:red")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(eps, [h["train_acc"] for h in history], "s-", label="train acc", color="tab:blue")
    ax2.plot(eps, [h["val_acc"] for h in history], "^-", label="val acc", color="tab:green")
    ax2.plot(eps, [h["test_acc"] for h in history], "D-", label="test acc", color="tab:purple")
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 1.02)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "curve.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    model, classes, history, meta = load_model()
    acc, y, p, cl = make_confusion(model)
    picks, gy, gp, ga = make_grid(model)
    make_curve(history)
    results = {
        "test_accuracy": float(acc),
        "n_test": int(len(y)),
        "n_classes": int(len(cl)),
        "classes": list(cl),
        "history": history,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("TEST ACCURACY:", round(float(acc), 4))
    print("figures:", os.listdir(FIGDIR))
