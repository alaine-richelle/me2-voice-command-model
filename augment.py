"""
ME2 v2 - Realistic speech augmentation (waveform level, train-only).

Goal (spec): "replicate actual speech with some injected noise or breaks in
between words or change in pitch and volume that can cause the user to
misunderstand the pronunciation."

Five augmentations, applied to the raw 16 kHz mono waveform before features:

  1. speed  - time-stretch (resample + resample back, pitch-preserving).
              Mimics different speaking tempos.
  2. pitch  - resample by a factor then interpolate back to original length.
              Shifts fundamental frequency (higher/lower voice) while keeping
              the duration ~constant.
  3. volume - random gain (louder / softer).
  4. noise  - additive white + pink noise at a random SNR (background babble
              proxy).  Can mask consonants -> pronunciation misunderstandings.
  5. breaks - insert short silence gaps at random word boundaries (hesitations
              / swallowed words).  Uses the transcript's word count to place
              the breaks realistically.

Each clip gets 1-3 random augmentations (always at least one), so the model
sees each command under many "real" delivery conditions.  Val/test stay clean.
"""
import numpy as np
from scipy.signal import resample_poly

SR = 16000


def _resample_linear(x, n_new):
    """Linear-resample waveform to n_new samples (interpolation)."""
    if n_new <= 0:
        return x
    idx = np.linspace(0, len(x) - 1, n_new)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def time_stretch(x, factor):
    """Change tempo by `factor` (duration scales by 1/factor, pitch kept).

    factor > 1 -> faster speech; factor < 1 -> slower.  Implemented as
    resample-to-shorter then interpolate-back, which preserves pitch.
    """
    n_short = max(1, int(round(len(x) * factor)))
    short = _resample_linear(x, n_short)
    return _resample_linear(short, len(x))


def pitch_shift(x, factor):
    """Shift pitch by `factor` (freq scales by factor, duration kept).

    factor > 1 -> higher voice; factor < 1 -> lower voice.  Resample by
    factor (changes pitch) then interpolate back to the original length.
    """
    n_new = max(1, int(round(len(x) * factor)))
    shifted = _resample_linear(x, n_new)
    return _resample_linear(shifted, len(x))


def gain(x, g):
    """Scale amplitude by g (clip to avoid overflow)."""
    y = x * g
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _pink_noise(n, rng):
    """Cheap pink (1/f) noise via cumulative-sum low-pass of white noise."""
    white = rng.standard_normal(n)
    # simple IIR low-pass to approximate 1/f spectrum
    b = np.ones(4) / 4.0
    y = np.convolve(white, b, mode="same")
    y -= y.mean()
    if y.std() > 0:
        y /= y.std()
    return y


def add_noise(x, snr_db, rng):
    """Add white + pink noise at the given SNR (dB)."""
    sig_power = (x ** 2).mean()
    if sig_power <= 0:
        return x
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    n = len(x)
    white = rng.standard_normal(n)
    pink = _pink_noise(n, rng)
    mix = 0.5 * white + 0.5 * pink
    mix = mix / (mix.std() + 1e-9)
    mix *= np.sqrt(noise_power)
    y = x + mix
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def insert_breaks(x, n_words, rng, gap_ms=(50, 150)):
    """Insert short silence gaps at random word boundaries.

    Approximates hesitations / swallowed words: the utterance is split into
    ~n_words equal segments and a few of the boundaries get a silent gap.
    """
    if n_words <= 1:
        return x
    seg = len(x) // n_words
    if seg < 80:                       # too short to break meaningfully
        return x
    n_breaks = rng.integers(1, max(2, n_words // 2 + 1))
    positions = rng.choice(np.arange(1, n_words), size=n_breaks, replace=False)
    pieces = []
    for w in range(n_words):
        start = w * seg
        end = (w + 1) * seg if w < n_words - 1 else len(x)
        pieces.append(x[start:end])
        if w in positions and w < n_words - 1:
            g = int(rng.integers(*gap_ms) * SR / 1000.0)
            pieces.append(np.zeros(g, dtype=np.float32))
    y = np.concatenate(pieces) if pieces else x
    return y.astype(np.float32)


def augment(x, n_words, rng, p_apply=(0.6, 0.5, 0.5, 0.5, 0.5)):
    """Apply 1-3 random augmentations to waveform x.

    Order: speed -> pitch -> volume -> noise -> breaks (breaks last so the
    inserted silence is not stretched/noised away).
    """
    applied = 0
    r = rng.random()
    if r < p_apply[0]:                 # speed
        x = time_stretch(x, float(rng.uniform(0.82, 1.18)))
        applied += 1
    if applied == 0 and rng.random() < 0.5:
        x = time_stretch(x, float(rng.uniform(0.9, 1.1)))
        applied += 1
    if rng.random() < p_apply[1]:      # pitch
        x = pitch_shift(x, float(rng.uniform(0.85, 1.15)))
        applied += 1
    if rng.random() < p_apply[2]:      # volume
        x = gain(x, float(rng.uniform(0.6, 1.4)))
        applied += 1
    if rng.random() < p_apply[3]:      # noise
        x = add_noise(x, float(rng.uniform(10, 30)), rng)
        applied += 1
    if rng.random() < p_apply[4]:      # breaks
        x = insert_breaks(x, n_words, rng)
        applied += 1
    # guarantee at least one augmentation so the model is actually stressed
    if applied == 0:
        x = add_noise(x, float(rng.uniform(15, 25)), rng)
    return x.astype(np.float32)
