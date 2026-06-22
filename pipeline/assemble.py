#!/usr/bin/env python3
"""
Phase 3 — assemble.py

Creates one Obsidian markdown file per canonical entity by combining the
alias clusters from aliases.json with the extracted bullets from records.jsonl.

For each entity:
  - Collects all records whose canonical_hint is in the entity's all_hints
  - Deduplicates bullets (exact first, then drops substrings of longer bullets)
  - Writes an Obsidian-formatted .md file to NPCs/, Places/, Items/, or Factions/
  - File includes YAML frontmatter (aliases, tags), bullet summary, foreshadowing,
    and wikilinks back to every source file where the entity was found

Safe to re-run: skips existing files unless --overwrite is passed.

Usage:
  python3 pipeline/assemble.py            # skip existing files
  python3 pipeline/assemble.py --overwrite # regenerate everything
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

VAULT_ROOT   = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent
ALIASES_FILE = PIPELINE_DIR / "aliases.json"
RECORDS_FILE = PIPELINE_DIR / "records.jsonl"

# Output directories (created if absent)
TYPE_DIRS = {
    "NPC":     VAULT_ROOT / "NPCs",
    "place":   VAULT_ROOT / "Places",
    "item":    VAULT_ROOT / "Items",
    "faction": VAULT_ROOT / "Factions",
}


# ── Filename helpers ──────────────────────────────────────────────────────────

_UNSAFE = re.compile(r'[\\/:*?"<>|#\[\]]')

def safe_filename(name: str) -> str:
    """Convert a canonical name to a safe Obsidian filename (no extension)."""
    name = _UNSAFE.sub("-", name)
    name = re.sub(r"-{2,}", "-", name).strip("- ")
    return name or "unnamed"


def entity_path(canonical: str, entity_type: str) -> Path:
    out_dir = TYPE_DIRS.get(entity_type, VAULT_ROOT / "Factions")
    return out_dir / f"{safe_filename(canonical)}.md"


# ── Bullet deduplication ──────────────────────────────────────────────────────

def _norm_bullet(b: str) -> str:
    return re.sub(r"\s+", " ", b.lower().strip().rstrip("."))


def dedup_bullets(raw: list[str]) -> list[str]:
    """
    1. Exact-normalised dedup (case/whitespace-insensitive).
    2. Remove any bullet whose normalised text is a substring of a longer bullet.
    Returns bullets sorted longest-first (most informative first).
    """
    # Step 1: exact dedup, keeping first occurrence
    seen_norm: dict[str, str] = {}
    for b in raw:
        n = _norm_bullet(b)
        if n and n not in seen_norm:
            seen_norm[n] = b

    bullets = list(seen_norm.values())
    norms   = [_norm_bullet(b) for b in bullets]

    # Step 2: drop bullets whose normalised form is contained in a longer one
    keep = []
    for i, b in enumerate(bullets):
        ni = norms[i]
        dominated = any(
            ni != norms[j] and ni in norms[j]
            for j in range(len(bullets))
        )
        if not dominated:
            keep.append(b)

    # Sort longest first so the most detailed facts lead
    return sorted(keep, key=lambda x: -len(x))


# ── Wikilink formatting ───────────────────────────────────────────────────────

def source_wikilink(source_file: str) -> str:
    """
    Return an Obsidian wikilink for a source file.
    Display name = filename without extension, stripping leading sort prefix
    like "Arc A - " so it reads naturally.
    """
    path = Path(source_file)
    stem = path.stem  # e.g. "Arc A - Escape From Death House"
    # Strip leading "Arc X - " prefix for cleaner display
    display = re.sub(r"^Arc [A-Z\d]+ - ", "", stem)
    # Vault-relative path without extension
    link_target = str(path.with_suffix(""))
    return f"[[{link_target}|{display}]]"


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_entity(canonical: str, data: dict, bullets: list[str],
                  foreshadowing: list[str]) -> str:
    etype       = data["entity_type"]
    aliases     = data.get("aliases", [])
    source_files = data.get("source_files", [])
    n_sources   = len(source_files)

    # ── YAML frontmatter ──
    lines = ["---"]
    lines.append(f"entity_type: {etype}")
    if aliases:
        lines.append("aliases:")
        for a in sorted(set(aliases)):
            # YAML-escape aliases that contain colons or quotes
            safe = a.replace('"', '\\"')
            lines.append(f'  - "{safe}"')
    lines.append("tags:")
    lines.append(f"  - {etype}")
    lines.append("---")
    lines.append("")

    # ── Title ──
    lines.append(f"# {canonical}")
    lines.append("")
    src_note = f"{n_sources} source file{'s' if n_sources != 1 else ''}"
    lines.append(f"> **{etype}** · {src_note}")
    lines.append("")

    # ── Key Facts ──
    if bullets:
        lines.append("## Key Facts")
        lines.append("")
        for b in bullets:
            # Ensure bullet text ends with a period for consistency
            b = b.rstrip()
            if b and b[-1] not in ".!?":
                b += "."
            lines.append(f"- {b}")
        lines.append("")

    # ── DM Notes / Foreshadowing ──
    if foreshadowing:
        lines.append("## DM Notes")
        lines.append("")
        for f in foreshadowing:
            f = f.rstrip()
            if f and f[-1] not in ".!?":
                f += "."
            lines.append(f"- {f}")
        lines.append("")

    # ── Appears In ──
    if source_files:
        lines.append("## Appears In")
        lines.append("")
        for sf in sorted(source_files):
            lines.append(f"- {source_wikilink(sf)}")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(overwrite: bool = False) -> None:
    # Create output directories
    for d in TYPE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load alias map
    print("Loading aliases.json ...")
    with open(ALIASES_FILE) as f:
        alias_map: dict[str, dict] = json.load(f)
    print(f"  {len(alias_map)} canonical entities")

    # Build lookup: canonical_hint → canonical name
    # (one canonical_hint can belong to at most one canonical entity)
    hint_to_canonical: dict[str, str] = {}
    for canonical, data in alias_map.items():
        for hint in data.get("all_hints", [canonical]):
            hint_to_canonical[hint] = canonical

    # Load and bucket records by canonical entity
    print("Loading records.jsonl ...")
    buckets: dict[str, list[dict]] = defaultdict(list)
    total_records = 0
    with open(RECORDS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hint = rec.get("canonical_hint", "")
            canon = hint_to_canonical.get(hint)
            if canon:
                buckets[canon].append(rec)
                total_records += 1
    print(f"  {total_records} records matched to {len(buckets)} entities")

    # Write entity files
    written = skipped = errors = 0
    type_counts: dict[str, int] = defaultdict(int)

    for canonical, data in sorted(alias_map.items()):
        etype = data.get("entity_type", "NPC")
        out_path = entity_path(canonical, etype)

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        # Aggregate bullets and foreshadowing from all matched records
        raw_bullets: list[str] = []
        raw_foreshadowing: list[str] = []
        for rec in buckets.get(canonical, []):
            raw_bullets.extend(rec.get("bullets", []))
            raw_foreshadowing.extend(rec.get("foreshadowing", []))

        bullets       = dedup_bullets([b for b in raw_bullets if b.strip()])
        foreshadowing = dedup_bullets([f for f in raw_foreshadowing if f.strip()])

        try:
            content = render_entity(canonical, data, bullets, foreshadowing)
            out_path.write_text(content, encoding="utf-8")
            written += 1
            type_counts[etype] += 1
        except Exception as exc:
            print(f"  ERROR writing {out_path.name}: {exc}", file=sys.stderr)
            errors += 1

    print()
    print(f"Written : {written}")
    for etype, count in sorted(type_counts.items()):
        dir_name = TYPE_DIRS.get(etype, VAULT_ROOT / "Factions").name
        print(f"  {etype:8s}: {count:4d}  → {dir_name}/")
    if skipped:
        print(f"Skipped : {skipped}  (already exist; use --overwrite to regenerate)")
    if errors:
        print(f"Errors  : {errors}", file=sys.stderr)

    print()
    print("Phase 3 complete.")
    print("Review a few files in NPCs/, Places/, Items/, Factions/")
    print("Then run: python3 pipeline/rewrite.py")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 3: assemble entity files")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing entity files")
    args = ap.parse_args()
    main(overwrite=args.overwrite)
