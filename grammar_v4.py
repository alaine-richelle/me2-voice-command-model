"""ME2 v4 - Grammar-based decoding for the Voice Command Model.

Instead of classifying the whole command in one shot, the acoustic model
emits ONE symbol per time-step (a character, or a CTC "blank").  A grammar
built as a TRIE over the known command transcripts then drives an iterative,
beam search that walks the trie one character at a time, scoring each path
with the acoustic model's per-character probabilities.

  * beam width = 1  -> greedy, strictly one character at a time ("iterative")
  * beam width > 1  -> grammar-guided beam search (focus on grammar)

The trie IS the grammar: it encodes exactly which character sequences are
legal commands, so the decoder can never emit a non-command string.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------
# Alphabet
# ----------------------------------------------------------------------------
CHARSET = list("abcdefghijklmnopqrstuvwxyz ")          # 27 symbols
BLANK   = len(CHARSET)                                 # 27 -> CTC blank
N_OUT   = len(CHARSET) + 1                             # 28 output classes
IDX2CH  = {i: c for i, c in enumerate(CHARSET)}
CH2IDX  = {c: i for i, c in enumerate(CHARSET)}


def normalize_text(t):
    """Lower-case, strip accents, keep only [a-z ] (collapse spaces).

    Digits and punctuation are removed so the acoustic model only has to
    produce letters/spaces.  e.g. "What's the weather?" -> "what s the weather"
    (apostrophe removed, letters kept).
    """
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^a-z]+", " ", t)        # any non-letter run -> single space
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------------------------------------------------------------------
# Trie (the grammar)
# ----------------------------------------------------------------------------
class Trie:
    """A prefix tree over legal command strings.

    Each node maps child-char -> node.  ``count`` is how many times that edge
    was seen in the corpus (prior smoothing).  Terminal nodes carry the full
    transcript + its intent.
    """

    def __init__(self):
        self.children = {}
        self.count = 0
        self.terminals = []          # list of (transcript, intent)

    def add(self, text, intent):
        node = self
        node.count += 1
        for ch in text:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = Trie()
                node.children[ch] = nxt
            node = nxt
            node.count += 1
        node.terminals.append((text, intent))

    def __len__(self):
        return self.count


def build_trie(texts_intents):
    """texts_intents: iterable of (normalized_text, intent)."""
    trie = Trie()
    for text, intent in texts_intents:
        if text:
            trie.add(text, intent)
    return trie


# ----------------------------------------------------------------------------
# Greedy CTC collapse (no grammar) - baseline for reporting
# ----------------------------------------------------------------------------
def ctc_collapse(logits):
    """Collapse a [T, N_OUT] log-prob matrix to a string (greedy, no grammar)."""
    import numpy as np
    logits = np.asarray(logits)
    path = []
    prev = None
    for t in range(logits.shape[0]):
        c = int(np.argmax(logits[t]))
        if c != prev:
            path.append(c)
        prev = c
    text = "".join(IDX2CH[c] for c in path if c != BLANK)
    return text, path


# ----------------------------------------------------------------------------
# Grammar-guided beam search over the trie
# ----------------------------------------------------------------------------
def beam_search(logits, trie, beam_width=4, max_len=None):
    """Walk the trie one character at a time, scoring paths with the
    per-character acoustic log-probs.  This is the "search on a grammar data
    structure (trie) to iteratively form the command".

    Args:
        logits: [T, N_OUT] log-probs (log_softmax output).  Timesteps are
                consumed one-per-emitted-char (a standard, cheap CTC
                approximation that pairs naturally with a trie: each emitted
                character consumes the next timestep).
        trie:   Trie built from the command grammar.
        beam_width: number of partial paths to keep (1 = greedy/iterative).
        max_len: stop after this many chars (default = T).

    Returns:
        (best_text, best_intent, score, ranked) where ranked is a list of
        (score, text, intent) for the surviving beams.
    """
    import numpy as np
    logits = np.asarray(logits)
    T = logits.shape[0]
    if max_len is None:
        max_len = T

    beams = [(0.0, trie, "")]
    for step in range(max_len):
        if step >= T:
            break
        col = logits[step]
        new_beams = []
        for score, node, text in beams:
            for ch, nxt in node.children.items():
                c = CH2IDX.get(ch)
                if c is None:
                    continue
                ns = score + col[c]
                new_beams.append((ns, nxt, text + ch))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    def beam_key(b):
        score, node, text = b
        term_bonus = 1.0 if node.terminals else 0.0
        return (term_bonus, score / (len(text) + 1e-9))

    if not beams:
        return "", None, float("-inf"), []
    beams.sort(key=beam_key, reverse=True)
    best_score, best_node, best_text = beams[0]
    intent = best_node.terminals[0][1] if best_node.terminals else None
    ranked = [(s, t, (n.terminals[0][1] if n.terminals else None))
              for s, n, t in beams]
    return best_text, intent, best_score, ranked


# ----------------------------------------------------------------------------
# Map an arbitrary decoded string to the nearest grammar command (fallback)
# ----------------------------------------------------------------------------
def nearest_command(text, trie, max_edit=3):
    """Return (transcript, intent) of the closest terminal in the trie, or
    (None, None) if nothing is within max_edit edits.  Used only when beam
    search fails to reach a terminal (robustness for noisy decodes)."""
    import itertools
    best = (float("inf"), None, None)
    stack = [(trie, "", 0)]
    terminals = []
    # collect terminals via DFS
    def dfs(node, prefix):
        if node.terminals:
            terminals.append((prefix, node.terminals[0][1]))
        for ch, nxt in node.children.items():
            dfs(nxt, prefix + ch)
    dfs(trie, "")
    for term, intent in terminals:
        d = _lev(text, term)
        if d < best[0]:
            best = (d, term, intent)
    d, term, intent = best
    if d <= max_edit:
        return term, intent
    return None, None


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
