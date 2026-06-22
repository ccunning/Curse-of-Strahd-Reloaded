#!/usr/bin/env python3
"""
Phase 2 — resolve.py

Groups canonical_hints from records.jsonl into canonical entities by
resolving aliases that refer to the same NPC, place, item, or faction.

Resolution pipeline (in confidence order):
  1. Proper-name filter        — drop lowercase canonical_hints (generic creatures)
  2. Wikilink anchor grouping  — guaranteed same entity (shared #anchor)
  3. Exact normalisation       — case / article / punctuation variants
  4. Prefix containment        — "Strahd" is a word-boundary prefix of "Strahd von Zarovich"
  5. Embedding cosine ≥ 0.92   — nomic-embed-text auto-merge (very similar names)
  6. Embedding cosine 0.70–0.91 — queued for LLM adjudication
  7. Token Jaccard ≥ 0.30      — also queued for LLM (catches string overlaps embeddings miss)
  8. LLM adjudication          — gemma4:26b with /think, conservative NO-default prompt

Outputs:
  pipeline/aliases.json       — machine-readable cluster map
  pipeline/aliases_review.txt — human-readable for DM review

*** REQUIRED GATE: review aliases_review.txt before running assemble.py ***
"""

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

PIPELINE_DIR = Path(__file__).parent
RECORDS_FILE = PIPELINE_DIR / "records.jsonl"
ALIASES_FILE = PIPELINE_DIR / "aliases.json"
REVIEW_FILE  = PIPELINE_DIR / "aliases_review.txt"

OLLAMA_HOST    = os.environ.get("OLLAMA_HOST",    "http://snowpeak:11434")
EMBED_MODEL    = os.environ.get("EMBED_MODEL",    "nomic-embed-text")
RESOLVE_MODEL  = os.environ.get("RESOLVE_MODEL",  "qwen3-coder:30b")

EMBED_MERGE_SIM = float(os.environ.get("EMBED_MERGE_SIM", "0.92"))
EMBED_LLM_SIM   = float(os.environ.get("EMBED_LLM_SIM",  "0.70"))
JACCARD_LLM     = float(os.environ.get("JACCARD_LLM",    "0.30"))

VALID_TYPES = {"NPC", "place", "item", "faction"}


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, elements: list):
        self.parent: dict = {e: e for e in elements}
        self.rank:   dict = {e: 0  for e in elements}
        self._ev:    dict = {e: set() for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y, evidence: str) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        ev_type = evidence.split(":")[0]
        self._ev[rx] = self._ev[rx] | self._ev[ry] | {ev_type}
        return True

    def clusters(self) -> dict:
        groups: dict = defaultdict(list)
        for e in self.parent:
            groups[self.find(e)].append(e)
        return dict(groups)

    def evidence_types(self, root) -> set:
        return self._ev[self.find(root)]


# ── String helpers ────────────────────────────────────────────────────────────

def normalise(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"['''""]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def token_set(name: str) -> frozenset:
    return frozenset(t for t in re.findall(r"\b\w+\b", normalise(name)) if len(t) >= 3)


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_prefix_match(a: str, b: str) -> bool:
    """
    True if normalise(a) is a word-boundary prefix of normalise(b), or vice versa.
    Prefix-only avoids false positives: "saint markovia" is a suffix of
    "abbot of saint markovia" but they're different entities.
    """
    na, nb = normalise(a), normalise(b)
    if len(na) < 5 or len(nb) < 5:
        return False

    def _is_prefix(short: str, long: str) -> bool:
        if not long.startswith(short):
            return False
        return len(long) == len(short) or long[len(short)] == " "

    return _is_prefix(na, nb) or _is_prefix(nb, na)


def extract_anchor(wikilink: str) -> str | None:
    m = re.search(r"#([^\]|]+)", wikilink)
    return m.group(1).strip().lower() if m else None


def anchor_matches_entity(anchor: str, canonical: str) -> bool:
    """
    Guard against Phase 1 wikilink contamination: only use an anchor for
    grouping if it's a plausible name form of this entity's canonical_hint.
    """
    na, nc = normalise(anchor), normalise(canonical)
    if na == nc:
        return True
    if nc in na and len(nc) >= 5 and len(nc) / max(len(na), 1) >= 0.40:
        return True
    if na in nc and len(na) >= 5:
        return True
    return jaccard(anchor, canonical) >= 0.45


# ── Embeddings ────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return dot / norm if norm else 0.0


def batch_embed(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    """Return {text: embedding} for all texts. Batches requests to avoid timeouts."""
    result: dict[str, list[float]] = {}
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            for item, text in zip(resp.data, chunk):
                result[text] = item.embedding
        except Exception as exc:
            print(f"  Embedding batch {i//BATCH + 1} error: {exc}", file=sys.stderr)
    return result


# ── LLM adjudication ─────────────────────────────────────────────────────────

# Explicit D&D false-positive examples to make the model conservative
_LLM_SYSTEM = """\
You are a strict alias resolver for the "Curse of Strahd: Reloaded" D&D campaign guide.
Your job is to decide whether two entity names refer to the SAME individual entity.

Be CONSERVATIVE — when in doubt, answer NO.
Different characters can share a surname (father ≠ son).
Different locations can share a word ("Mount X" ≠ "Mount Y", "Castle X" ≠ a room inside it).
A place named after a person is NOT the same entity as that person.
A faction is NOT the same as a location it occupies.

Correct NO examples:
- Baron Vargas Vallakovich vs Victor Vallakovich  (father vs son)
- Mount Baratok vs Mount Ghakis  (two different mountains)
- Castle Ravenloft vs Castle Crypts  (whole castle vs a room inside it)
- House Wachter vs Death House  (different entities, share word "house")
- Saint Markovia vs Abbey of Saint Markovia  (person vs place named after them)
- Fanes of Barovia vs Barovia  (sacred sites vs the whole land)
"""


def llm_adjudicate(
    pairs: list[tuple[str, str, str]],
    client: OpenAI,
) -> list[bool | None]:
    """
    Batch YES/NO for (name_a, name_b, entity_type).
    Returns list[bool|None] aligned with pairs (None = parse failure).
    """
    if not pairs:
        return []

    numbered = "\n".join(
        f'{i+1}. [{etype}] "{a}"  vs  "{b}"'
        for i, (a, b, etype) in enumerate(pairs)
    )
    prompt = (
        "/think\n"
        "For each numbered pair, reply with the number and YES (same entity) "
        "or NO (different entities).\n"
        "One answer per line. No explanations after the YES/NO.\n\n"
        + numbered
    )

    try:
        resp = client.chat.completions.create(
            model=RESOLVE_MODEL,
            max_tokens=len(pairs) * 20 + 100,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    except Exception as exc:
        print(f"  LLM error: {exc}", file=sys.stderr)
        return [None] * len(pairs)

    results: list[bool | None] = [None] * len(pairs)
    for line in raw.strip().splitlines():
        m = re.match(r"^\s*(\d+)[.):\s]+(.+)", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(pairs):
            ans = m.group(2).strip().upper()
            if ans.startswith("YES"):
                results[idx] = True
            elif ans.startswith("NO"):
                results[idx] = False
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main(run_llm: bool = True) -> None:
    client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

    # ── Load & filter records ─────────────────────────────────────────────
    print("Loading records.jsonl ...")
    raw: list[dict] = []
    skipped_lowercase = 0
    with open(RECORDS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            etype = rec.get("entity_type", "NPC")
            if etype not in VALID_TYPES:
                continue
            hint = rec.get("canonical_hint", "")
            # Proper-name filter: English proper nouns are always capitalised.
            # Generic creature types (zombie, wight, phase spider…) are lowercase.
            if not hint or not hint[0].isupper():
                skipped_lowercase += 1
                continue
            raw.append(rec)

    print(f"  {len(raw)} records kept, {skipped_lowercase} lowercase hints dropped")

    # Aggregate per (canonical_hint, entity_type)
    agg: dict[tuple, dict] = {}
    for rec in raw:
        etype = rec["entity_type"]
        key   = (rec["canonical_hint"], etype)
        if key not in agg:
            agg[key] = {
                "canonical_hint": rec["canonical_hint"],
                "entity_type":    etype,
                "surface_forms":  set(rec.get("surface_forms", [])),
                "wikilink_seeds": set(rec.get("wikilink_seeds", [])),
                "source_files":   set(),
                "record_count":   0,
            }
        agg[key]["surface_forms"].update(rec.get("surface_forms", []))
        agg[key]["wikilink_seeds"].update(rec.get("wikilink_seeds", []))
        agg[key]["source_files"].add(rec["source_file"])
        agg[key]["record_count"] += 1

    all_keys = list(agg.keys())
    print(f"  {len(all_keys)} unique (canonical_hint, entity_type) pairs\n")

    by_type: dict[str, list] = defaultdict(list)
    for k in all_keys:
        by_type[k[1]].append(k)

    uf = UnionFind(all_keys)
    counters = {"wikilink": 0, "exact": 0, "prefix": 0, "embedding": 0, "llm": 0}

    # ── Step 1: Wikilink anchor grouping ─────────────────────────────────
    print("[1/6] Wikilink anchor grouping ...")
    anchor_map: dict[str, list] = defaultdict(list)
    for key, data in agg.items():
        for seed in data["wikilink_seeds"]:
            anc = extract_anchor(seed)
            if anc and anchor_matches_entity(anc, key[0]):
                anchor_map[anc].append(key)

    for anc, keys in anchor_map.items():
        for i in range(1, len(keys)):
            if uf.union(keys[0], keys[i], f"wikilink:{anc}"):
                counters["wikilink"] += 1
    print(f"  {counters['wikilink']} merges")

    # ── Step 2: Exact normalisation ───────────────────────────────────────
    print("[2/6] Exact normalisation ...")
    for etype, keys in by_type.items():
        norm_map: dict[str, tuple] = {}
        for key in keys:
            n = normalise(key[0])
            if n in norm_map:
                if uf.union(norm_map[n], key, "exact"):
                    counters["exact"] += 1
            else:
                norm_map[n] = key
    print(f"  {counters['exact']} merges")

    # ── Step 3: Prefix containment ────────────────────────────────────────
    print("[3/6] Prefix containment ...")
    for etype, keys in by_type.items():
        for i, ka in enumerate(keys):
            for kb in keys[i+1:]:
                if uf.find(ka) == uf.find(kb):
                    continue
                if is_prefix_match(ka[0], kb[0]):
                    if uf.union(ka, kb, "prefix"):
                        counters["prefix"] += 1
    print(f"  {counters['prefix']} merges")

    # ── Step 4: Embeddings ────────────────────────────────────────────────
    print(f"[4/6] Embeddings via {EMBED_MODEL} ...")
    texts = list({k[0] for k in all_keys})
    print(f"  Embedding {len(texts)} unique names ...", flush=True)
    embeddings = batch_embed(texts, client)
    print(f"  Got {len(embeddings)}/{len(texts)} embeddings")

    uncertain: list[tuple[tuple, tuple, float, str]] = []  # (ka, kb, score, method)

    embed_pairs = 0
    for etype, keys in by_type.items():
        emb_keys = [k for k in keys if k[0] in embeddings]
        for i, ka in enumerate(emb_keys):
            for kb in emb_keys[i+1:]:
                if uf.find(ka) == uf.find(kb):
                    continue
                sim = cosine(embeddings[ka[0]], embeddings[kb[0]])
                embed_pairs += 1
                if sim >= EMBED_MERGE_SIM:
                    if uf.union(ka, kb, f"embedding:{sim:.3f}"):
                        counters["embedding"] += 1
                elif sim >= EMBED_LLM_SIM:
                    uncertain.append((ka, kb, sim, "embed"))

    print(f"  {counters['embedding']} auto-merges, {len(uncertain)} pairs queued (embed sim {EMBED_LLM_SIM:.0%}–{EMBED_MERGE_SIM:.0%})")

    # ── Step 5: Token Jaccard — supplement embedding queue ───────────────
    print(f"[5/6] Token Jaccard ≥ {JACCARD_LLM} (supplement LLM queue) ...")
    jac_added = 0
    already_queued = {(ka, kb) for ka, kb, _, _ in uncertain} | {(kb, ka) for ka, kb, _, _ in uncertain}
    for etype, keys in by_type.items():
        for i, ka in enumerate(keys):
            for kb in keys[i+1:]:
                if uf.find(ka) == uf.find(kb):
                    continue
                if (ka, kb) in already_queued:
                    continue
                j = jaccard(ka[0], kb[0])
                if j >= JACCARD_LLM:
                    uncertain.append((ka, kb, j, "jaccard"))
                    already_queued.add((ka, kb))
                    jac_added += 1
    print(f"  {jac_added} additional pairs → total LLM queue: {len(uncertain)}")

    # ── Step 6: LLM adjudication ──────────────────────────────────────────
    if uncertain and run_llm:
        to_send = [
            (ka, kb, score, method)
            for ka, kb, score, method in sorted(uncertain, key=lambda x: -x[2])
            if uf.find(ka) != uf.find(kb)
        ]
        print(f"[6/6] LLM adjudication: {len(to_send)} pairs via {RESOLVE_MODEL} ...")
        BATCH = 20
        total_batches = math.ceil(len(to_send) / BATCH)
        for bi in range(total_batches):
            batch = to_send[bi * BATCH : (bi + 1) * BATCH]
            print(f"  batch {bi+1}/{total_batches} ({len(batch)} pairs) ...",
                  end=" ", flush=True)
            inputs = [(ka[0], kb[0], ka[1]) for ka, kb, _, _ in batch]
            answers = llm_adjudicate(inputs, client)
            merged = 0
            for (ka, kb, _, _), yes in zip(batch, answers):
                if yes is True and uf.find(ka) != uf.find(kb):
                    if uf.union(ka, kb, "llm_yes"):
                        counters["llm"] += 1
                        merged += 1
            print(f"{merged} merges")
    elif uncertain:
        print(f"[6/6] LLM adjudication skipped (--no-llm). {len(uncertain)} pairs unresolved.")

    # ── Build alias map ───────────────────────────────────────────────────
    print("\nBuilding alias map ...")
    raw_clusters = uf.clusters()

    alias_map: dict[str, dict] = {}
    for root, members in sorted(raw_clusters.items(), key=lambda x: x[0][0]):
        if not members:
            continue

        def score(key: tuple) -> tuple:
            d = agg[key]
            return (len(d["wikilink_seeds"]), d["record_count"], len(key[0]))

        best      = max(members, key=score)
        canonical = best[0]
        etype     = best[1]

        all_aliases: set[str] = set()
        all_sources: set[str] = set()
        all_seeds:   set[str] = set()
        total_recs  = 0
        for m in members:
            all_aliases.add(m[0])
            all_aliases.update(agg[m]["surface_forms"])
            all_sources.update(agg[m]["source_files"])
            all_seeds.update(agg[m]["wikilink_seeds"])
            total_recs += agg[m]["record_count"]

        ev = uf.evidence_types(root)
        for conf in ["wikilink", "exact", "prefix", "embedding", "llm_yes"]:
            if conf in ev:
                confidence = conf
                break
        else:
            confidence = "singleton"

        alias_map[canonical] = {
            "entity_type":    etype,
            "aliases":        sorted(all_aliases - {canonical}),
            "all_hints":      sorted({m[0] for m in members}),
            "source_files":   sorted(all_sources),
            "wikilink_seeds": sorted(all_seeds),
            "record_count":   total_recs,
            "confidence":     confidence,
        }

    # Remove aliases that are canonical names of other clusters (Phase 1 contamination)
    all_canonicals_lower = {c.lower() for c in alias_map}
    removed_cross = 0
    for canonical, data in alias_map.items():
        canon_lower = canonical.lower()
        cleaned = [
            a for a in data["aliases"]
            if not (a.lower() in all_canonicals_lower and a.lower() != canon_lower)
        ]
        removed_cross += len(data["aliases"]) - len(cleaned)
        data["aliases"] = cleaned
    if removed_cross:
        print(f"  Removed {removed_cross} cross-entity alias contaminations")

    with open(ALIASES_FILE, "w") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)
    print(f"  {len(alias_map)} canonical entities → {ALIASES_FILE.name}")

    # ── Write review file ─────────────────────────────────────────────────
    conf_order  = ["wikilink", "exact", "prefix", "embedding", "llm_yes", "singleton"]
    conf_labels = {
        "wikilink":  "WIKILINK — high confidence (shared source anchors)",
        "exact":     "EXACT MATCH — high confidence",
        "prefix":    "PREFIX/SUBSTRING — review recommended",
        "embedding": "EMBEDDING — review recommended",
        "llm_yes":   "LLM ADJUDICATED — review recommended",
        "singleton": "SINGLETONS (no alias merges)",
    }
    by_conf: dict[str, list] = defaultdict(list)
    for canonical, data in alias_map.items():
        by_conf[data["confidence"]].append((canonical, data))

    lines = [
        "=" * 72,
        "PHASE 2 — ALIAS CLUSTER REVIEW",
        "Curse of Strahd: Reloaded entity pipeline",
        "=" * 72,
        "",
        "How to correct this output:",
        "  SPLIT bad merge  → remove aliases from aliases.json entry; add new key.",
        "  MERGE missing    → add the canonical_hint to an existing entry's aliases.",
        "  RENAME canonical → change the top-level key in aliases.json.",
        "  BAD HINT (★)    → canonical_hint looks like a description; rename the key.",
        "",
        "Pipeline summary:",
        f"  Lowercase dropped    : {skipped_lowercase}  (generic creatures, not proper names)",
        f"  Wikilink merges      : {counters['wikilink']}",
        f"  Exact merges         : {counters['exact']}",
        f"  Prefix merges        : {counters['prefix']}",
        f"  Embedding merges     : {counters['embedding']}",
        f"  LLM merges           : {counters['llm']}",
        f"  Total entities       : {len(alias_map)}",
        "",
        "Breakdown by confidence:",
    ]
    for conf in conf_order:
        n = len(by_conf[conf])
        if n:
            lines.append(f"  {conf:12s} : {n}")
    lines.append("")

    for conf in conf_order:
        group = sorted(by_conf[conf], key=lambda x: -x[1]["record_count"])
        if not group:
            continue
        lines.append("─" * 72)
        lines.append(f"## {conf_labels[conf]}  [{len(group)}]")
        lines.append("")

        if conf == "singleton":
            for etype in ("NPC", "place", "item", "faction"):
                sings = [(c, d) for c, d in group if d["entity_type"] == etype]
                if not sings:
                    continue
                lines.append(f"  [{etype}] ({len(sings)} entities)")
                for canonical, data in sings:
                    src = data["source_files"][0] if data["source_files"] else "?"
                    bad = len(canonical) > 60 or "," in canonical
                    marker = " ★ BAD HINT" if bad else ""
                    lines.append(f"    {canonical}  ← {Path(src).name}{marker}")
                lines.append("")
        else:
            for canonical, data in group:
                n_src = len(data["source_files"])
                other_hints = [h for h in data["all_hints"] if h != canonical]
                hint_note = ""
                if other_hints:
                    shown = ", ".join(other_hints[:3])
                    hint_note = f"  [merged: {shown}{'…' if len(other_hints) > 3 else ''}]"
                lines.append(
                    f"  [{data['entity_type']}] {canonical}"
                    f"  ({n_src} file{'s' if n_src != 1 else ''}, {data['record_count']} records)"
                    + hint_note
                )
                for alias in sorted(data["aliases"]):
                    if alias.lower() != canonical.lower():
                        lines.append(f"      = {alias}")
                lines.append("")

    lines += [
        "=" * 72,
        "END — edit aliases.json, then run: python3 pipeline/assemble.py",
        "=" * 72,
    ]

    with open(REVIEW_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Review summary → {REVIEW_FILE.name}")

    print()
    print("=" * 60)
    print("PHASE 2 COMPLETE — REVIEW REQUIRED BEFORE PROCEEDING")
    print("=" * 60)
    print(f"  cat {REVIEW_FILE}")
    print(f"  edit {ALIASES_FILE}")
    print()
    print("  Then: python3 pipeline/assemble.py")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 2: entity alias resolution")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM adjudication (heuristics + embeddings only)")
    args = ap.parse_args()
    main(run_llm=not args.no_llm)
