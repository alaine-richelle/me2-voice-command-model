"""ME2 v6 - Inference for the Tiny Voice Command Model.

Uses the EXACT same preprocessing as training (16 kHz mono -> 40x128 log-mel,
normalized (x-0.1307)/0.3081) and the CORRECT blank-aware trie beam search
(decoder_v6_fast) to turn the per-char CTC output into a command.

Modes
-----
  python inference.py --microphone          # record from the mic, then predict
  python inference.py --file some.wav       # predict on an existing wav
  python inference.py --file some.wav --save-as turn_on
                                            # also store the recording as a
                                            # user dataset sample (label confirmed)

Microphone workflow (per spec):
  1. Ask whether to enable microphone input.
  2. Request permission / record for a configurable duration.
  3. Save the recording temporarily.
  4. Apply the SAME preprocessing used during training.
  5. Run model.pkl.
  6. Show predicted command + confidence; flag low confidence.
  7. Ask whether to save the recording as a NEW dataset sample; if yes, ask for
     the CORRECT label (never auto-accept the model's prediction) and store it
     under user_recordings/<label>/ with metadata.
"""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grammar_v4 import BLANK, IDX2CH, build_trie, N_OUT
from decoder_v6_fast import build_flat, beam_search_vec
import pickle

DEV = "cpu"  # model is tiny; CPU inference is fine and portable
SR = 16000
N_MELS = 40
N_FFT = 512
HOP = 160
FMAX = 8000
T_FRAMES = 128
MEAN, STD = 0.1307, 0.3081
BEAM = 4
LOW_CONF = 0.35          # below this we flag the prediction as uncertain
USER_REC = os.path.join(HERE, "user_recordings")


# ---------------------------------------------------------------------------
# Model (identical to train_v6.PerCharCNN)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Preprocessing (MUST match build_features.py / training)
# ---------------------------------------------------------------------------
def load_wav_16k(path):
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def logmel(x):
    import librosa
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


def wav_to_feature(x):
    """16 kHz mono waveform -> [1, 40, 128] normalized log-mel (training format)."""
    L = frame(logmel(x))
    L = (L - MEAN) / STD
    return L.astype(np.float32)[None, None]   # [1,1,40,128]


# ---------------------------------------------------------------------------
# Bundle load
# ---------------------------------------------------------------------------
def load_bundle():
    with open(os.path.join(HERE, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = PerCharCNN()
    model.load_state_dict(bundle["model_state"])
    model.eval()
    t2i = bundle["trie_intents"]
    trie = build_trie(list(t2i.items()))
    flat = build_flat(trie)
    return bundle, model, trie, flat


@torch.no_grad()
def predict(model, x, flat, trie):
    feat = wav_to_feature(x)
    logits = model(torch.from_numpy(feat))
    lp = F.log_softmax(logits, dim=-1).cpu().numpy()[0]
    es, ed, eci, terminals, _ = flat
    text, node, score = beam_search_vec(lp, es, ed, eci, terminals, beam_width=BEAM)
    intent = None
    if text:
        node_t = trie
        for ch in text:
            node_t = node_t.children.get(ch)
            if node_t is None:
                break
        if node_t is not None and node_t.terminals:
            intent = node_t.terminals[0][1]
    conf = float(np.exp(max(score, -30))) if text else 0.0
    return intent, text, conf


# ---------------------------------------------------------------------------
# Microphone capture (graceful: sounddevice if available, else file fallback)
# ---------------------------------------------------------------------------
def record_microphone(duration=3.0):
    """Record `duration` seconds from the default mic.  Returns (wav, path)."""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed.  Install it with `pip install "
            "sounddevice` (and a PortAudio backend) to use microphone mode, "
            "or use --file <wav> instead.")
    print(f"Recording {duration}s from microphone ... speak now.")
    audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    wav = audio[:, 0].astype(np.float32)
    tmp = os.path.join(HERE, "_tmp_recording.wav")
    sf.write(tmp, wav, SR)
    return wav, tmp


def save_user_recording(wav, label, src_path=None):
    """Store a CONFIRMED recording under user_recordings/<label>/ with metadata."""
    label = label.strip().lower().replace(" ", "_")
    d = os.path.join(USER_REC, label)
    os.makedirs(d, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"rec_{ts}.wav"
    out = os.path.join(d, fn)
    sf.write(out, wav, SR)
    meta = {
        "filename": fn,
        "command": label,
        "timestamp": ts,
        "sample_rate": SR,
        "duration": round(len(wav) / SR, 3),
        "source": "user_microphone",
        "src_path": src_path,
    }
    with open(os.path.join(d, f"rec_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ME2 v6 Tiny VCM inference")
    ap.add_argument("--microphone", action="store_true", help="record from mic")
    ap.add_argument("--file", type=str, default=None, help="predict on a wav file")
    ap.add_argument("--duration", type=float, default=3.0, help="mic recording seconds")
    ap.add_argument("--save-as", type=str, default=None,
                    help="also save the recording as a user sample with this label")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the interactive 'enable mic?' prompt")
    args = ap.parse_args()

    bundle, model, trie, flat = load_bundle()
    print("Loaded model.pkl  (params=%d, %d intents)"
          % (bundle["config"]["n_params"], len(bundle["intents"])))

    wav = None
    src = None
    if args.microphone:
        if not args.no_confirm:
            ans = input("Enable microphone input? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Microphone disabled. Exiting.")
                return
        try:
            wav, src = record_microphone(args.duration)
        except RuntimeError as e:
            print("MICROPHONE UNAVAILABLE:", e)
            print("Falling back: please provide a wav with --file.")
            if not args.file:
                return
    if args.file:
        wav = load_wav_16k(args.file)
        src = args.file

    if wav is None:
        ap.print_help()
        return

    intent, text, conf = predict(model, wav, flat, trie)
    print("\n--- RESULT ---")
    print(f"Predicted command : {intent}")
    print(f"Decoded text      : {text!r}")
    print(f"Confidence        : {conf:.3f}")
    if conf < LOW_CONF:
        print(f"NOTE: confidence {conf:.3f} < {LOW_CONF} -> prediction is UNCERTAIN.")

    # optionally save as a user dataset sample (label must be confirmed)
    save_label = None
    if args.save_as:
        save_label = args.save_as
    else:
        ans = input("\nSave this recording as a new dataset sample? [y/n] ").strip().lower()
        if ans in ("y", "yes"):
            save_label = input("Enter the CORRECT command label: ").strip()

    if save_label:
        out = save_user_recording(wav, save_label, src)
        print(f"Saved user recording -> {out}")
        print("(Stored under user_recordings/ for later use; NOT auto-added to training.)")

    # cleanup temp mic file
    if args.microphone and src and src.startswith("_tmp_"):
        try:
            os.remove(src)
        except OSError:
            pass


if __name__ == "__main__":
    main()
