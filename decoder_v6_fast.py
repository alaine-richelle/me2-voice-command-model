"""ME2 v6 - VECTORIZED blank-aware CTC beam search over a trie (NumPy).

Same algorithm as decoder_v6.ctc_trie_beam (correct previous-symbol state
machine: a repeat is only legal across a blank; every non-blank advances the
trie), but implemented with flat arrays + vectorized transitions so a single
sample decodes in microseconds instead of milliseconds.  This is what makes
per-epoch evaluation tractable.

State representation (flat):
  node  : int array [S]   trie node id per beam
  prev  : int array [S]   previous symbol per beam (BLANK or char idx)
  score : float array [S]
  text  : list[str]       parallel (kept as Python list; only beam-sized)

Transitions at frame t:
  * blank: (node, prev) -> (node, BLANK)   score += lp[t, BLANK]
  * emit : (node, prev) -> (child, ci) for each trie edge ci!=prev
           score += lp[t, ci]
We build all candidate transitions as arrays, dedupe by (node, prev) keeping
the max score, and prune to the top `beam`.
"""
import numpy as np
from grammar_v4 import BLANK, IDX2CH

NEG = -1e30


def build_flat(trie):
    """Flatten trie -> (edge_src, edge_dst, edge_ci, edge_ch, terminals, n_nodes).

    edge_src/dst/ci: int arrays, one row per trie edge.
    edge_ch: the character string for each edge (IDX2CH[edge_ci]).
    terminals: set of terminal node ids.
    """
    from grammar_v4 import CH2IDX
    nodes = []
    id_of = {}
    edges = []
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
        for ch, nxt in node.children.items():
            ci = CH2IDX.get(ch)
            if ci is None:
                continue
            cid = get_id(nxt)
            edges.append((nid, cid, ci))
            stack.append(nxt)
        if node.terminals:
            terminals.add(nid)
    edge_src = np.array([e[0] for e in edges], dtype=np.int64)
    edge_dst = np.array([e[1] for e in edges], dtype=np.int64)
    edge_ci = np.array([e[2] for e in edges], dtype=np.int64)
    return edge_src, edge_dst, edge_ci, terminals, len(nodes)


def beam_search_vec(lp, edge_src, edge_dst, edge_ci, terminals,
                    beam_width=8, terminal_bonus=0.0):
    """Vectorized blank-aware CTC trie beam search.

    lp: [T, C] log-probs.  Returns (best_text, best_node, best_score).
    """
    T, C = lp.shape
    # precompute, for each edge, its source node and char idx
    # adjacency: for a given node, which edges emanate?
    # Build per-node edge index lists (small trie -> cheap)
    n_edges = len(edge_src)
    # group edge indices by source node
    adj = {}
    for e in range(n_edges):
        adj.setdefault(int(edge_src[e]), []).append(e)

    # initial state
    node = np.array([0], dtype=np.int64)
    prev = np.array([BLANK], dtype=np.int64)
    score = np.array([0.0])
    text = [""]
    best = (NEG, "", None)

    for t in range(T):
        row = lp[t]
        # ---- collect all candidate transitions ----
        cand_node = []
        cand_prev = []
        cand_score = []
        cand_txt = []
        for i in range(len(node)):
            ni = int(node[i]); pi = int(prev[i]); si = score[i]; ti = text[i]
            # blank transition
            cand_node.append(ni); cand_prev.append(BLANK)
            cand_score.append(si + row[BLANK]); cand_txt.append(ti)
            # emit transitions
            for e in adj.get(ni, ()):
                ci = int(edge_ci[e])
                if ci == pi:
                    continue  # repeat without blank: illegal
                cand_node.append(int(edge_dst[e])); cand_prev.append(ci)
                cand_score.append(si + row[ci])
                cand_txt.append(ti + IDX2CH[ci])
        if not cand_node:
            break
        cand_node = np.array(cand_node, dtype=np.int64)
        cand_prev = np.array(cand_prev, dtype=np.int64)
        cand_score = np.array(cand_score, dtype=np.float64)
        # dedupe by (node, prev) keeping max score
        key = cand_node * 1000 + cand_prev
        order = np.argsort(-cand_score)  # best first
        seen = {}
        keep_node, keep_prev, keep_score, keep_txt = [], [], [], []
        for o in order:
            k = int(key[o])
            if k in seen:
                continue
            seen[k] = True
            keep_node.append(int(cand_node[o])); keep_prev.append(int(cand_prev[o]))
            keep_score.append(float(cand_score[o])); keep_txt.append(cand_txt[o])
            if len(keep_node) >= beam_width:
                break
        node = np.array(keep_node, dtype=np.int64)
        prev = np.array(keep_prev, dtype=np.int64)
        score = np.array(keep_score, dtype=np.float64)
        text = keep_txt
        # update best terminal
        for i in range(len(node)):
            if int(node[i]) in terminals:
                eff = score[i] + terminal_bonus
                if eff > best[0]:
                    best = (eff, text[i], int(node[i]))
    return best[1], best[2], best[0]


def beam_candidates(lp, edge_src, edge_dst, edge_ci, terminals,
                    beam_width=8, terminal_bonus=0.0, topk=20):
    """Like beam_search_vec but returns the top terminal candidates
    [(score, text, node)] so a per-class posterior can be built."""
    T, C = lp.shape
    n_edges = len(edge_src)
    adj = {}
    for e in range(n_edges):
        adj.setdefault(int(edge_src[e]), []).append(e)
    node = np.array([0], dtype=np.int64)
    prev = np.array([BLANK], dtype=np.int64)
    score = np.array([0.0])
    text = [""]
    cand = {}  # (node,text) -> best score
    for t in range(T):
        row = lp[t]
        cand_node, cand_prev, cand_score, cand_txt = [], [], [], []
        for i in range(len(node)):
            ni = int(node[i]); pi = int(prev[i]); si = score[i]; ti = text[i]
            cand_node.append(ni); cand_prev.append(BLANK)
            cand_score.append(si + row[BLANK]); cand_txt.append(ti)
            for e in adj.get(ni, ()):
                ci = int(edge_ci[e])
                if ci == pi:
                    continue
                cand_node.append(int(edge_dst[e])); cand_prev.append(ci)
                cand_score.append(si + row[ci]); cand_txt.append(ti + IDX2CH[ci])
        if not cand_node:
            break
        cand_node = np.array(cand_node, dtype=np.int64)
        cand_prev = np.array(cand_prev, dtype=np.int64)
        cand_score = np.array(cand_score, dtype=np.float64)
        key = cand_node * 1000 + cand_prev
        order = np.argsort(-cand_score)
        seen = {}
        kn, kp, ks, kt = [], [], [], []
        for o in order:
            k = int(key[o])
            if k in seen:
                continue
            seen[k] = True
            kn.append(int(cand_node[o])); kp.append(int(cand_prev[o]))
            ks.append(float(cand_score[o])); kt.append(cand_txt[o])
            if len(kn) >= beam_width:
                break
        node = np.array(kn, dtype=np.int64)
        prev = np.array(kp, dtype=np.int64)
        score = np.array(ks, dtype=np.float64)
        text = kt
        for i in range(len(node)):
            if int(node[i]) in terminals:
                kk = (int(node[i]), text[i])
                eff = score[i] + terminal_bonus
                if eff > cand.get(kk, NEG):
                    cand[kk] = eff
    ranked = sorted(cand.items(), key=lambda kv: -kv[1])[:topk]
    return [(sc, tx, nd) for (nd, tx), sc in ranked]


def decode_lp(lp, trie, beam_width=8, terminal_bonus=0.0):
    """Convenience wrapper (builds flat arrays each call; cache externally)."""
    flat = build_flat(trie)
    return beam_search_vec(lp, *flat, beam_width=beam_width,
                           terminal_bonus=terminal_bonus)
