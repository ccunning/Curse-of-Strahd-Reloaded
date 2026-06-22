#!/usr/bin/env python3
"""
Phase 1 — extract.py

Loop over every source markdown file, call the LLM once per chunk, and write
one JSONL record per (entity, source_file) pair to records.jsonl.

Large files are split by top-level # headings so each chunk fits within the
model's usable context window. Entities from multiple chunks of the same file
are merged before writing.

Resumable: re-running skips files whose source_file already appears in records.jsonl.

Environment variables:
  OLLAMA_HOST    Ollama base URL (default: http://snowpeak:11434)
  EXTRACT_MODEL  Model to use (default: gemma4:26b)
  CHUNK_CHARS    Max characters per chunk (default: 80000 ≈ 20K tokens)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

VAULT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent
OUTPUT_FILE = PIPELINE_DIR / "records.jsonl"

OLLAMA_HOST   = os.environ.get("OLLAMA_HOST",   "http://snowpeak:11434")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "qwen3-coder:30b")
CHUNK_CHARS   = int(os.environ.get("CHUNK_CHARS", "80000"))
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS",  "12000"))

# ── Skip lists ───────────────────────────────────────────────────────────────

SKIP_FILES: set[str] = {
    "Act I - Into the Mists/Act I Summary.md",
    "Act II - The Shadowed Town/Act II Summary.md",
    "Act III - The Broken Land/Act III Summary.md",
    "Act IV - Secrets of the Ancient/Act IV Summary.md",
    "Appendices/Bestiary.md",
    "Appendices/Glossary.md",
    "_other/templates/combat.md",
    "Introduction/Changelog.md",
}

SKIP_DIRS: set[str] = {
    ".git", ".obsidian", ".goose", "images",
    "NPCs", "Places", "Items",
    ".trash", "pipeline",
}

IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def extract_wikilinks(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for inner in re.findall(r"\[\[([^\]]+)\]\]", text):
        target = inner.split("|")[0].strip()
        if Path(target).suffix.lower() in IMAGE_EXTS:
            continue
        wl = f"[[{inner}]]"
        if wl not in seen:
            seen.add(wl)
            result.append(wl)
    return result


def match_wikilinks(all_wikilinks: list[str], surface_forms: list[str]) -> list[str]:
    """Return wikilinks from all_wikilinks that refer to this entity."""
    sf_lower = [sf.lower() for sf in surface_forms]
    matched: list[str] = []
    for wl in all_wikilinks:
        inner = wl[2:-2]
        target_raw  = inner.split("|")[0].strip()
        display_raw = inner.split("|")[1].strip().lower() if "|" in inner else ""
        file_part   = target_raw.split("#")[0].lower()
        anchor      = target_raw.split("#")[1].lower() if "#" in target_raw else ""
        for sf in sf_lower:
            # Exact anchor match (highest confidence)
            if anchor and sf == anchor:
                matched.append(wl); break
            # Display alias match
            if display_raw and (sf == display_raw or sf in display_raw or display_raw in sf):
                matched.append(wl); break
            # Anchor substring match
            if anchor and (sf in anchor or anchor in sf):
                matched.append(wl); break
            # Bare target match (e.g. [[Vistani]] or [[Strahd von Zarovich]])
            if file_part and sf == file_part:
                matched.append(wl); break
    return matched


def _split_and_group(text: str, pattern: str, max_chars: int) -> list[str]:
    """
    Split text on regex pattern (lookahead), group parts into chunks ≤ max_chars.
    Parts larger than max_chars are emitted alone (caller handles further splitting).
    """
    parts = re.split(pattern, text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        if not part:
            continue
        if current_len + len(part) > max_chars and current:
            chunks.append("".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += len(part)

    if current:
        chunks.append("".join(current))

    return chunks


def chunk_by_headings(text: str, max_chars: int) -> list[str]:
    """
    Split text by headings so no chunk exceeds max_chars.
    Pass 1: split on top-level '# ' headings.
    Pass 2: any section still over max_chars is re-split on '## ' headings.
    Pass 3: any section still over max_chars is kept as-is (the model will
            have to handle it; Ollama will surface an error if it can't).
    """
    level1 = _split_and_group(text, r"(?=\n# )", max_chars)

    result: list[str] = []
    for chunk in level1:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            # Second pass: split on ## headings
            level2 = _split_and_group(chunk, r"(?=\n## )", max_chars)
            result.extend(level2)

    return result


def merge_chunk_entities(all_chunk_entities: list[list[dict]]) -> list[dict]:
    """
    Merge entity lists from multiple chunks of the same file.
    Deduplicates by canonical_hint (case-insensitive), unioning lists fields.
    """
    merged: dict[str, dict] = {}
    for chunk_ents in all_chunk_entities:
        for ent in chunk_ents:
            key = ent.get("canonical_hint", "").lower()
            if not key:
                continue
            sf  = ent.get("surface_forms", [])
            bul = ent.get("bullets", [])
            fore = ent.get("foreshadowing", [])
            if key not in merged:
                merged[key] = {**ent}
                merged[key]["surface_forms"]  = list(sf)
                merged[key]["wikilink_seeds"] = list(ent.get("wikilink_seeds", []))
                merged[key]["bullets"]        = list(bul)
                merged[key]["foreshadowing"]  = list(fore)
            else:
                ex = merged[key]
                ex["surface_forms"] = list(dict.fromkeys(ex["surface_forms"] + sf))
                ex["wikilink_seeds"] = list(dict.fromkeys(
                    ex["wikilink_seeds"] + ent.get("wikilink_seeds", [])
                ))
                for b in bul:
                    if b not in ex["bullets"]:
                        ex["bullets"].append(b)
                for f in fore:
                    if f not in ex["foreshadowing"]:
                        ex["foreshadowing"].append(f)
    return list(merged.values())


def get_source_files() -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(VAULT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            if not fname.endswith(".md"):
                continue
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(VAULT_ROOT))
            if rel in SKIP_FILES:
                continue
            files.append(fpath)
    return files


def load_processed() -> set[str]:
    processed: set[str] = set()
    if not OUTPUT_FILE.exists():
        return processed
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                processed.add(rec["source_file"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


# Expected JSON shape (not used for constrained decoding — included in the prompt).
# qwen3-coder reasons via /think then outputs free-form JSON, which we parse directly.
# This is more reliable than grammar-based constrained decoding for complex schemas.
ENTITY_JSON_SHAPE = """{
  "entities": [
    {
      "surface_forms": ["all name variants used in this text"],
      "canonical_hint": "Full canonical name (prefer wikilink anchor text)",
      "entity_type": "NPC|place|item|faction",
      "source_anchor": "nearest ## or ### heading, or empty string",
      "bullets": ["1–5 factual bullets: what this text reveals about the entity"],
      "foreshadowing": ["DM prep items / future events, or empty array"]
    }
  ]
}"""

SYSTEM = """\
/think
You are analysing a Dungeons & Dragons campaign guide ("Curse of Strahd: Reloaded") \
to extract a structured entity index for the Dungeon Master.

Extract every named entity the text meaningfully discusses — NPCs (named people/creatures), \
places (locations, buildings, regions), items (named objects, weapons, artifacts, relics), \
and factions (named groups, orders, cults, noble houses).

Do NOT extract:
- Generic creature types ("vampire spawn", "werewolf", "night hag") unless the text \
  gives them a personal name.
- Game mechanics, dice rolls, or chapter/arc cross-references.
- The player characters themselves (tracked separately).
- Entities mentioned only in passing with no descriptive content.

Consolidate: if the same entity appears under multiple headings in this chunk, \
produce ONE record with bullets covering all appearances.

After reasoning, output ONLY a valid JSON object with no markdown fences or extra text.\
"""


def build_user_message(rel_path: str, chunk_text: str,
                        all_wikilinks: list[str], chunk_index: int, total_chunks: int) -> str:
    wl_block = "\n".join(f"  {w}" for w in all_wikilinks[:100]) or "  (none found)"
    chunk_note = (
        f" (chunk {chunk_index}/{total_chunks})" if total_chunks > 1 else ""
    )
    return (
        f"FILE: {rel_path}{chunk_note}\n\n"
        f"WIKILINKS IN THE FULL FILE (high-confidence entity seeds — "
        f"prefer anchor text as canonical_hint):\n"
        f"{wl_block}\n\n"
        f"TEXT TO ANALYSE:\n---\n{chunk_text}\n---\n\n"
        f"Return ONLY this JSON structure, no markdown fences:\n{ENTITY_JSON_SHAPE}"
    )


# ── Core extraction ──────────────────────────────────────────────────────────

def call_llm(client: OpenAI, rel_path: str, chunk_text: str,
             all_wikilinks: list[str], chunk_index: int, total_chunks: int) -> list[dict]:
    """Call the model on one chunk; return raw entity dicts."""
    user_msg = build_user_message(rel_path, chunk_text, all_wikilinks, chunk_index, total_chunks)
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )

    content = (response.choices[0].message.content or "").strip()

    # Strip <think>...</think> blocks if the model outputs them explicitly
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if not content:
        print(f"  WARNING: empty response for {rel_path} chunk {chunk_index}", file=sys.stderr)
        return []

    # Direct JSON parse
    try:
        return json.loads(content).get("entities", [])
    except json.JSONDecodeError:
        pass

    # Fallback: extract from a ```json ... ``` code block if the model added one
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)).get("entities", [])
        except json.JSONDecodeError:
            pass

    print(f"  WARNING: JSON parse error for {rel_path} chunk {chunk_index}: "
          f"content[:200]={content[:200]!r}", file=sys.stderr)
    return []


def process_file(client: OpenAI, fpath: Path) -> list[dict]:
    rel = str(fpath.relative_to(VAULT_ROOT))
    text = fpath.read_text(encoding="utf-8")
    all_wikilinks = extract_wikilinks(text)

    chunks = chunk_by_headings(text, CHUNK_CHARS)
    total = len(chunks)
    file_slug = slugify(rel.replace("/", "--").replace(".md", ""))
    now = datetime.now(timezone.utc).isoformat()

    all_chunk_entities: list[list[dict]] = []
    for i, chunk in enumerate(chunks, 1):
        if total > 1:
            print(f"    chunk {i}/{total} ({len(chunk)//1000}K chars) ...", end=" ", flush=True)
        raw_ents = call_llm(client, rel, chunk, all_wikilinks, i, total)
        if total > 1:
            print(f"{len(raw_ents)} entities")
        all_chunk_entities.append(raw_ents)

    merged = merge_chunk_entities(all_chunk_entities)

    # Augment with Python-derived fields
    seen_canonical: set[str] = set()
    records: list[dict] = []
    for ent in merged:
        canonical: str = ent.get("canonical_hint", "").strip()
        if not canonical:
            continue
        canon_key = canonical.lower()
        if canon_key in seen_canonical:
            continue
        seen_canonical.add(canon_key)

        surface_forms: list[str] = ent.get("surface_forms", [canonical])
        seeds = match_wikilinks(all_wikilinks, surface_forms)

        records.append({
            "record_id":      f"{file_slug}--{slugify(canonical)}",
            "source_file":    rel,
            "source_anchor":  ent.get("source_anchor", ""),
            "entity_type":    ent.get("entity_type", "NPC"),
            "surface_forms":  surface_forms,
            "canonical_hint": canonical,
            "wikilink_seeds": seeds,
            "bullets":        ent.get("bullets", []),
            "foreshadowing":  ent.get("foreshadowing", []),
            "extracted_at":   now,
        })

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

    # Quick connectivity check
    try:
        client.models.list()
    except Exception as exc:
        print(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}", file=sys.stderr)
        sys.exit(1)

    source_files = get_source_files()
    processed    = load_processed()
    todo         = [f for f in source_files
                    if str(f.relative_to(VAULT_ROOT)) not in processed]

    print(f"Model:        {EXTRACT_MODEL}")
    print(f"Ollama host:  {OLLAMA_HOST}")
    print(f"Chunk limit:  {CHUNK_CHARS:,} chars")
    print(f"Source files: {len(source_files)} total, "
          f"{len(processed)} done, {len(todo)} to process\n")

    if not todo:
        print("Nothing to do.")
        return

    total_entities = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, fpath in enumerate(todo, 1):
            rel      = str(fpath.relative_to(VAULT_ROOT))
            size_kb  = fpath.stat().st_size // 1024
            n_chunks = len(chunk_by_headings(fpath.read_text(), CHUNK_CHARS))
            chunk_note = f", {n_chunks} chunks" if n_chunks > 1 else ""
            print(f"[{i}/{len(todo)}] {rel} ({size_kb} KB{chunk_note}) ...",
                  end="\n" if n_chunks > 1 else " ", flush=True)
            try:
                records = process_file(client, fpath)
                for rec in records:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                total_entities += len(records)
                if n_chunks == 1:
                    print(f"{len(records)} entities")
                else:
                    print(f"  → {len(records)} entities total")
            except Exception as exc:
                print(f"  ERROR — {exc}", file=sys.stderr)
                raise

    print(f"\nDone. {total_entities} records written to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
