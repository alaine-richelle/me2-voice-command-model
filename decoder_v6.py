"""ME2 v6 - CORRECT blank-aware CTC beam search constrained to a trie grammar.

Why this replaces the v5 decoder
--------------------------------
The v5 DP (in train_v5.py) allowed a beam to emit a character equal to its
``last`` symbol at a *consecutive* frame without a blank in between.  In CTC
that is illegal: two identical symbols on adjacent frames must come from the
same emission, so a "repeat" is only legal when the previous frame was a blank.
That bug double-counted repeated letters and, worse, let the search drift onto
short high-probability leaves (STOP, TIME), which is exactly why the v5 beam
collapsed long commands (PLAY_MUSIC, TIMER, COLOR) to STOP/TIME and scored
31% exact / 39% intent while greedy CTC reached 73.6% exact.

This module implements the standard CTC lattice DP with the correct
previous-symbol state machine:

  State at frame t: (trie_node, prev_symbol, text)
    prev_symbol in {BLANK, c1, c2, ...}  (the symbol emitted at frame t-1)

  Transitions at frame t with symbol s (probability p_s):
    * s == BLANK   : stay on node, prev becomes BLANK.  (free, no char)
    * s != BLANK and s != prev and node has child edge s:
        move to that child, append s, prev becomes s.
    * s != BLANK and s == prev : illegal (repeat must be preceded by blank).
    * s != BLANK and no child edge s : illegal (every non-blank advances grammar).

  A terminal node is a candidate at every frame; we track the best-scoring
  terminal path seen so far.  We prune to the top ``beam`` (node, prev) states
  each frame (keeping the best text per (node, prev)).

This is O(T * beam * alphabet) and is exact w.r.t. the beam width.
"""
import numpy as np

NEG = -1e30


def prepare_trie(trie):
    """Flatten a grammar_v4.Trie into (children, terminal_node_ids).

    children: dict node_id -> list[(child_node_id, char_idx)]
    """
    from grammar_v4 import CH2IDX
    nodes = []
    id_of = {}
    children = {}
    terminals = set()

    def get_id(node):
        k = id(node)
        if k in id_of:
            return id_of[k]
        nid = len(nodes)
        nodes.append(node)
        id_of[k] = nid
        return nid

    get_id(trie)
    stack = [trie]
    while stack:
        node = stack.pop()
        nid = get_id(node)
        kids = []
        for ch, nxt in node.children.items():
            ci = CH2IDX.get(ch)
            if ci is None:
                continue
            cid = get_id(nxt)
            kids.append((cid, ci))
            stack.append(nxt)
        children[nid] = kids
        if node.terminals:
            terminals.add(nid)
    return children, terminals


def ctc_trie_beam(lp, children, terminals, beam_width=8, terminal_bonus=0.0):
    """Blank-aware CTC beam search over a flattened trie.

    Args:
        lp: [T, C] log-probs (log_softmax output), C = N_OUT (chars + blank).
        children: dict node_id -> list[(child_node_id, char_idx)].
        terminals: set of terminal node ids.
        beam_width: number of (node, prev) states to keep per frame.
        terminal_bonus: additive log bonus for terminating (0 = off).

    Returns:
        (best_text, best_node, best_score) or ("", None, -inf).
    """
    from grammar_v4 import BLANK, IDX2CH

    T, C = lp.shape
    child_map = {}
    for nid, lst in children.items():
        child_map[nid] = {ci: cid for cid, ci in lst}

    root = 0
    # states: (node, prev) -> (score, text)
    states = {(root, BLANK): (0.0, "")}
    best = (NEG, "", None)

    for t in range(T):
        row = lp[t]
        new = {}
        for (nid, prev), (score, text) in states.items():
            # 1) blank transition: stay on node, prev -> BLANK
            sb = score + row[BLANK]
            key = (nid, BLANK)
            cur = new.get(key)
            if cur is None or sb > cur[0]:
                new[key] = (sb, text)
            # 2) emit a character: must differ from prev, must be a trie edge
            cm = child_map.get(nid)
            if cm:
                for ci, cid in cm.items():
                    if ci == prev:
                        continue  # repeat without intervening blank: illegal
                    ns = score + row[ci]
                    nkey = (cid, ci)
                    cur = new.get(nkey)
                    if cur is None or ns > cur[0]:
                        new[nkey] = (ns, text + IDX2CH[ci])
        if not new:
            break
        items = sorted(new.items(), key=lambda kv: kv[1][0], reverse=True)
        states = dict(items[:beam_width])
        for (nid, prev), (score, text) in states.items():
            if nid in terminals:
                eff = score + terminal_bonus
                if eff > best[0]:
                    best = (eff, text, nid)
    return best[1], best[2], best[0]


def decode_lp(lp, trie, beam_width=8, terminal_bonus=0.0):
    """Convenience: prepare + decode.  Returns (text, node_id, score)."""
    children, terminals = prepare_trie(trie)
    return ctc_trie_beam(lp, children, terminals, beam_width, terminal_bonus)
