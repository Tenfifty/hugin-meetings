"""Sortformer diarization with voice-based speaker identity.

Streaming Sortformer v2.1 finds speaker turns as well as NeMo's MSDD at about
a fifth of the GPU time and without MSDD's memory growth on long parts (MSDD
OOMs on a 63 min part on the 8 GB card). But it decides *who* is speaking
from a speaker cache of ~15 s of audio shared between four output slots, and
both limits show up in meetings:

- The four slots are fixed by the model's output layer (not a setting), so in
  a room with five to seven people several of them share one label.
- A speaker who is silent for a few minutes can come back in a new slot, and
  keeps alternating between the two for the rest of the meeting.

So Sortformer's labels are only taken as turns. Identity is re-decided here
from TitaNet voice embeddings of the single-speaker speech: a label whose
pieces form two clearly different voices is split, and labels with the same
voice are merged. The thresholds come from 14 meetings (Sep 2026) where MSDD's
clusters served as reference: one person's first and second half of a
meeting are >= 0.83 cosine apart (median 0.95), two different people <= 0.62
(median 0.27). On 16 other meetings this took Sortformer from 14.6 % to 7.6 %
of words labelled differently from MSDD, with no voice left on two labels.

Enrolled names are applied afterwards by the usual pyannote step, which is
deliberately a different embedding model from the one clustering here.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
EMBEDDING_MODEL = "titanet_large"

# The model card's "very high latency" configuration, in 80 ms frames; the
# right one for offline files. A larger speaker cache (376) mended one
# meeting and broke two others: the model was trained with 188.
CHUNK_LEN = 340
CHUNK_RIGHT_CONTEXT = 40
FIFO_LEN = 40
SPKCACHE_UPDATE_PERIOD = 300
SPKCACHE_LEN = 188

PIECE_S = 3.0         # embedding window; TitaNet is reliable from ~1.5 s
MIN_PIECE_S = 1.5
MERGE_COS = 0.75      # same person: >= 0.83 across halves of one meeting
SPLIT_COS = 0.70      # different people: <= 0.62
MIN_SPLIT_PIECES = 8  # ~20 s of speech on each side before a split is believed
MAX_SPLIT_DEPTH = 3
MAX_OVERLAP_S = 10.0  # two labels talking at once this long are two people ...
SURE_SAME_COS = 0.90  # ... unless their voices are this close: Sortformer can
                      # light two slots for one voice at the same time
EMBED_BATCH = 64

Segment = tuple[float, float, str]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def single_speaker_intervals(segments: list[Segment]) -> list[Segment]:
    """Each segment minus the time any other label is active."""
    out = []
    for a, b, label in segments:
        parts = [(a, b)]
        for c, d, other in segments:
            if other == label or d <= a or c >= b:
                continue
            nxt = []
            for x, y in parts:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            parts = nxt
        out += [(x, y, label) for x, y in parts]
    return out


def embedding_pieces(segments: list[Segment]) -> list[Segment]:
    """Cut single-speaker speech into PIECE_S windows; a short tail joins the last one."""
    pieces = []
    for a, b, label in single_speaker_intervals(segments):
        n = int((b - a) // PIECE_S)
        bounds = [a + k * PIECE_S for k in range(n + 1)]
        if (b - a) - n * PIECE_S >= MIN_PIECE_S:
            bounds.append(b)
        elif n >= 1:
            bounds[-1] = b
        pieces += [(bounds[k], bounds[k + 1], label) for k in range(len(bounds) - 1)]
    return sorted(pieces)


def _split(idx: np.ndarray, emb: np.ndarray, depth: int = 0) -> list[np.ndarray]:
    """Bisect a set of pieces while the two halves are clearly different voices."""
    from sklearn.cluster import KMeans

    if depth >= MAX_SPLIT_DEPTH or len(idx) < 2 * MIN_SPLIT_PIECES:
        return [idx]
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(emb[idx])
    parts = [idx[km.labels_ == k] for k in (0, 1)]
    if min(len(p) for p in parts) < MIN_SPLIT_PIECES:
        return [idx]
    c0, c1 = (_unit(emb[p].mean(0)) for p in parts)
    if float(c0 @ c1) >= SPLIT_COS:
        return [idx]
    return _split(parts[0], emb, depth + 1) + _split(parts[1], emb, depth + 1)


def _overlap_s(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(y, w) - max(x, v)) for x, y in a for v, w in b)


def refine_speakers(
    segments: list[Segment], pieces: list[Segment], embeddings: np.ndarray
) -> list[Segment]:
    """Re-decide speaker identity of ``segments`` from voice embeddings of ``pieces``.

    ``pieces`` are ``embedding_pieces(segments)`` and ``embeddings`` their
    vectors, row for row. Returns segments labelled ``speaker_N``.
    """
    if not pieces:
        return list(segments)
    emb = _unit(np.asarray(embeddings, dtype=np.float64))
    starts = np.array([p[0] for p in pieces])
    ends = np.array([p[1] for p in pieces])
    orig = np.array([p[2] for p in pieces], dtype=object)
    cluster = np.empty(len(pieces), dtype=object)
    k = 0
    for label in sorted(set(orig)):
        for part in _split(np.where(orig == label)[0], emb):
            cluster[part] = f"c{k}"
            k += 1

    # Relabel at piece granularity so a turn change inside one Sortformer
    # segment (two people sharing a slot) is kept. Time no piece covers
    # (overlap, short tails, short segments) takes the nearest piece of the
    # same Sortformer label.
    by_label = defaultdict(list)
    for i, label in enumerate(orig):
        by_label[label].append(i)
    out = []
    for a, b, label in segments:
        idx = by_label.get(label)
        if not idx:
            out.append([a, b, f"u:{label}"])
            continue
        inside = [i for i in idx if starts[i] < b and ends[i] > a]
        cuts = sorted({a, b, *(max(a, starts[i]) for i in inside), *(min(b, ends[i]) for i in inside)})
        for x, y in zip(cuts, cuts[1:]):
            mid = (x + y) / 2
            hit = [i for i in inside if starts[i] <= mid < ends[i]]
            i = hit[0] if hit else min(idx, key=lambda j: min(abs(starts[j] - mid), abs(ends[j] - mid)))
            out.append([x, y, cluster[i]])

    # Merge clusters with the same voice, most similar pair first.
    while True:
        labels = sorted({c for c in cluster})
        centroid = {c: _unit(emb[cluster == c].mean(0)) for c in labels}
        best = None
        for i, p in enumerate(labels):
            for q in labels[i + 1:]:
                cos = float(centroid[p] @ centroid[q])
                if cos < MERGE_COS or (best is not None and cos <= best[0]):
                    continue
                if cos < SURE_SAME_COS:
                    tp = [(a, b) for a, b, c in out if c == p]
                    tq = [(a, b) for a, b, c in out if c == q]
                    if _overlap_s(tp, tq) > MAX_OVERLAP_S:
                        continue
                best = (cos, p, q)
        if best is None:
            break
        _, keep, drop = best
        cluster[cluster == drop] = keep
        out = [[a, b, keep if c == drop else c] for a, b, c in out]

    out.sort(key=lambda s: (s[0], s[1]))
    joined = []
    for a, b, c in out:
        if joined and joined[-1][2] == c and a - joined[-1][1] < 0.05:
            joined[-1][1] = max(joined[-1][1], b)
        else:
            joined.append([a, b, c])
    names = {}
    for *_, c in joined:
        names.setdefault(c, f"speaker_{len(names)}")
    return [(a, b, names[c]) for a, b, c in joined]


class SortformerDiarizer:
    """Wav path in, ``(start, end, label)`` list out; ``[]`` when there is no speech."""

    def __init__(self, device: str):
        from nemo.collections.asr.models import EncDecSpeakerLabelModel, SortformerEncLabelModel

        self.device = device
        self.model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL, map_location=device).eval()
        modules = self.model.sortformer_modules
        modules.chunk_len = CHUNK_LEN
        modules.chunk_right_context = CHUNK_RIGHT_CONTEXT
        modules.fifo_len = FIFO_LEN
        modules.spkcache_update_period = SPKCACHE_UPDATE_PERIOD
        modules.spkcache_len = SPKCACHE_LEN
        modules._check_streaming_parameters()
        self.embedder = EncDecSpeakerLabelModel.from_pretrained(EMBEDDING_MODEL, map_location=device).eval()

    def __call__(self, wav_path: str | Path) -> list[Segment]:
        import torch

        with torch.inference_mode():
            lines = self.model.diarize(audio=[str(wav_path)], batch_size=1)[0]
        segments = []
        for line in lines:
            start, end, label = str(line).split()
            segments.append((float(start), float(end), label))
        pieces = embedding_pieces(segments)
        if not pieces:
            return segments
        return refine_speakers(segments, pieces, self._embed(wav_path, pieces))

    def _embed(self, wav_path: str | Path, pieces: list[Segment]) -> np.ndarray:
        import soundfile as sf
        import torch

        audio, sr = sf.read(str(wav_path), dtype="float32")
        out = []
        for k in range(0, len(pieces), EMBED_BATCH):
            chunks = [audio[int(a * sr):int(b * sr)] for a, b, _ in pieces[k:k + EMBED_BATCH]]
            batch = np.zeros((len(chunks), max(len(c) for c in chunks)), np.float32)
            for i, c in enumerate(chunks):
                batch[i, :len(c)] = c
            with torch.inference_mode():
                _, emb = self.embedder.forward(
                    input_signal=torch.from_numpy(batch).to(self.device),
                    input_signal_length=torch.tensor([len(c) for c in chunks], device=self.device),
                )
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out)
