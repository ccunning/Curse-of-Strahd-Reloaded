#!/usr/bin/env python3
"""
Phase 4 — rewrite.py

Scans markdown files in the vault and inserts Obsidian wikilinks wherever entity
names appear as plain text. Reads aliases.json to know each entity's canonical
name, safe aliases, and target file path.

Rules:
  - Processes all .md files except pipeline/, images/, PCs/, _other/
  - Skips YAML frontmatter, fenced code blocks, markdown headers
  - Protects existing [[wikilinks]] and inline `code` from modification
  - Links each entity FIRST OCCURRENCE per file only (Obsidian convention)
  - Longest-alias-first matching: "Strahd von Zarovich" before "Strahd"
  - Alias safety filter: alias must contain the first key word of the canonical name
    (prevents Phase 2 contamination like "Rahadin" being linked to Strahd)

Default: dry-run — prints a summary of what would change without touching files.
Use --write to actually modify files.

Usage:
  python3 pipeline/rewrite.py           # dry run
  python3 pipeline/rewrite.py --write   # rewrite in place
  python3 pipeline/rewrite.py --write --verbose  # show each change
"""

import json
import re
import sys
from pathlib import Path

VAULT_ROOT   = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent
ALIASES_FILE = PIPELINE_DIR / "aliases.json"

# Directories never processed (relative to VAULT_ROOT)
SKIP_DIRS = {"pipeline", "images", "PCs", "_other", ".git", ".obsidian"}

# Vault-relative prefix → entity file root
ENTITY_ROOTS = {"NPCs", "Places", "Items", "Factions"}

# ── Alias safety filter ───────────────────────────────────────────────────────

# Words treated as honorifics/titles/articles when determining the "first proper
# name word" of a canonical or alias.  A string whose first non-title word matches
# the canonical’s first non-title word is considered a safe short-form alias.
_TITLE_WORDS = {
    # Articles / prepositions
    "the", "a", "an", "of", "in", "at", "by", "and", "or", "for", "from",
    "with", "to",
    # Noble / clerical titles
    "count", "countess", "baron", "baroness", "lady", "lord", "sir",
    "dr", "doctor", "saint", "king", "queen", "prince", "princess",
    "father", "mother", "captain", "master", "mistress",
    # Name particles
    "von", "van", "de", "du", "del", "der", "el", "al",
    # Slavic honorifics (baba = crone/grandmother, used for Baba Lysaga)
    "baba", "old", "elder",
}


def _first_proper(name: str) -> str:
    """
    Return the first word of *name* that is NOT a title / article (len >= 3).
    This is used to compare whether an alias and a canonical refer to the same
    named entity.

    Examples:
      "Strahd von Zarovich"   → "strahd"
      "Count Strahd"          → "strahd"   (skips "count")
      "King Barov von ..."    → "barov"    (skips "king")
      "Baba Lysaga"           → "lysaga"   (skips "baba")
      "Lysaga"                → "lysaga"
    """
    for word in name.lower().split():
        if word not in _TITLE_WORDS and len(word) >= 3:
            return word
    parts = name.lower().split()
    return parts[0] if parts else ""


def is_safe_alias(alias: str, canonical: str) -> bool:
    """
    Return True if *alias* is safe to auto-link to *canonical*.

    Safety rules (all must pass):
      1. No apostrophes  — rejects possessives like "Kasimir’s spellbook"
      2. Starts with uppercase  — rejects "the cousin", "the girl"
      3. Not a bare "The/A/An + single word"  — rejects ultra-generic two-word phrases
      4. The alias’s first proper word == the canonical’s first proper word
         — rejects Phase-2 contamination where unrelated characters share a surname
           e.g. "Sergei von Zarovich" alias on "Strahd von Zarovich"
    """
    # Rule 1: no possessives
    if "’" in alias or "’" in alias:
        return False

    # Rule 2: must start uppercase
    if not alias[:1].isupper():
        return False

    # Rule 3: bare "The/A/An + one word" is too generic
    words = alias.split()
    if len(words) == 2 and words[0].lower() in {"the", "a", "an"}:
        return False

    # Rule 4: first proper word must match
    return _first_proper(alias) == _first_proper(canonical)


# ── Build entity lookup ───────────────────────────────────────────────────────

def _safe_stem(name: str) -> str:
    """Same logic as assemble.py safe_filename (no extension)."""
    name = re.sub(r'[\\/:*?"<>|#\[\]]', "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("- ")
    return name or "unnamed"


def build_entity_table(alias_map: dict) -> list[tuple[re.Pattern, str, str]]:
    """
    Return a list of (compiled_pattern, vault_relative_link_path, canonical_name),
    sorted longest-alias-first so longer matches win over shorter prefixes.
    """
    entries: list[tuple[str, str, str]] = []  # (alias_text, link_path, canonical)

    for canonical, data in alias_map.items():
        etype = data.get("entity_type", "NPC")
        root_map = {"NPC": "NPCs", "place": "Places", "item": "Items", "faction": "Factions"}
        root = root_map.get(etype, "Factions")
        link_path = f"{root}/{_safe_stem(canonical)}"

        # Always include the canonical name itself
        entries.append((canonical, link_path, canonical))

        # Include safe aliases
        for alias in data.get("aliases", []):
            if alias == canonical:
                continue
            if is_safe_alias(alias, canonical):
                entries.append((alias, link_path, canonical))

    # Sort longest alias first (prevents short aliases swallowing long matches)
    entries.sort(key=lambda x: -len(x[0]))

    # Compile regex patterns
    result = []
    for alias_text, link_path, canonical in entries:
        try:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(alias_text) + r"(?!\w)",
                re.IGNORECASE,
            )
            result.append((pattern, link_path, canonical))
        except re.error:
            pass  # skip malformed aliases

    return result


# ── Line-level processing ─────────────────────────────────────────────────────

# Matches existing [[wikilinks]] or inline `code`
_PROTECTED_RE = re.compile(r"`[^`\n]+`|\[\[.*?\]\]")


def _split_protected(line: str) -> list[tuple[bool, str]]:
    """Split *line* into alternating (is_protected, text) segments."""
    segments: list[tuple[bool, str]] = []
    last = 0
    for m in _PROTECTED_RE.finditer(line):
        if m.start() > last:
            segments.append((False, line[last:m.start()]))
        segments.append((True, m.group()))
        last = m.end()
    if last < len(line):
        segments.append((False, line[last:]))
    return segments


def _relink_segment(
    text: str,
    entries: list[tuple[re.Pattern, str, str]],
    already_linked: set[str],
    newly_linked: set[str],
) -> str:
    """
    Apply entity-link substitutions to a text segment.

    After each substitution we re-split for protected spans so that text we
    just inserted (e.g. [[Places/Barovia|Barovia]]) is treated as protected
    in subsequent iterations, preventing nested wikilinks.
    """
    while True:
        # Re-split every iteration so freshly-inserted [[...]] spans are protected.
        sub_segments = _split_protected(text)
        best: tuple[int, int, str, str, str] | None = None
        offset = 0

        for is_protected, seg in sub_segments:
            if not is_protected:
                for pattern, link_path, canonical in entries:
                    if canonical in already_linked or canonical in newly_linked:
                        continue
                    m = pattern.search(seg)
                    if m:
                        abs_start = offset + m.start()
                        if best is None or abs_start < best[0]:
                            best = (abs_start, offset + m.end(), canonical, link_path, m.group())
            offset += len(seg)

        if best is None:
            break

        start, end, canonical, link_path, orig_text = best
        wikilink = f"[[{link_path}|{orig_text}]]"
        text = text[:start] + wikilink + text[end:]
        newly_linked.add(canonical)

    return text


def relink_line(
    line: str,
    entries: list[tuple[re.Pattern, str, str]],
    already_linked: set[str],
) -> tuple[str, set[str]]:
    """
    Replace first-occurrence entity mentions in *line* with Obsidian wikilinks.
    Returns (new_line, set_of_newly_linked_canonicals).
    Existing [[wikilinks]] and inline `code` are never modified.
    """
    segments = _split_protected(line)
    newly_linked: set[str] = set()
    parts: list[str] = []

    for is_protected, text in segments:
        if is_protected:
            parts.append(text)
        else:
            parts.append(_relink_segment(text, entries, already_linked, newly_linked))

    return "".join(parts), newly_linked


# ── File-level processing ─────────────────────────────────────────────────────

def process_file(
    path: Path,
    entries: list[tuple[re.Pattern, str, str]],
    *,
    write: bool,
    verbose: bool,
) -> tuple[bool, int]:
    """
    Scan *path* and optionally rewrite it with entity wikilinks.
    Returns (was_changed, num_links_added).
    """
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"  SKIP (read error): {path}  — {exc}", file=sys.stderr)
        return False, 0

    lines = original.splitlines(keepends=True)
    new_lines: list[str] = []
    already_linked: set[str] = set()
    total_links = 0

    in_frontmatter = False
    frontmatter_done = False
    in_code_block = False

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n").rstrip("\r")
        nl = raw_line[len(line):]  # preserve original line ending

        # ── Frontmatter (YAML between opening and closing ---) ──
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            new_lines.append(raw_line)
            continue
        if in_frontmatter:
            if line.strip() == "---":
                in_frontmatter = False
                frontmatter_done = True
            new_lines.append(raw_line)
            continue

        # ── Fenced code blocks ──
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            new_lines.append(raw_line)
            continue
        if in_code_block:
            new_lines.append(raw_line)
            continue

        # ── Markdown headers — skip linking inside headings ──
        if stripped.startswith("#"):
            new_lines.append(raw_line)
            continue

        # ── Normal line — apply relinking ──
        new_line, newly_linked = relink_line(line, entries, already_linked)
        already_linked.update(newly_linked)
        total_links += len(newly_linked)

        if newly_linked and verbose:
            for canon in sorted(newly_linked):
                print(f"    + {canon}")

        new_lines.append(new_line + nl)

    new_text = "".join(new_lines)
    changed = new_text != original

    if changed and write:
        try:
            path.write_text(new_text, encoding="utf-8")
        except Exception as exc:
            print(f"  ERROR writing {path}: {exc}", file=sys.stderr)
            return False, 0

    return changed, total_links


# ── File collection ───────────────────────────────────────────────────────────

def collect_files() -> list[Path]:
    """Return all .md files to process (vault-wide, excluding skip dirs)."""
    result: list[Path] = []
    for md in sorted(VAULT_ROOT.rglob("*.md")):
        # Check if any ancestor directory is in SKIP_DIRS
        rel = md.relative_to(VAULT_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        result.append(md)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(write: bool = False, verbose: bool = False) -> None:
    print("Loading aliases.json ...")
    with open(ALIASES_FILE) as f:
        alias_map: dict = json.load(f)
    print(f"  {len(alias_map)} canonical entities")

    print("Building entity patterns ...")
    entries = build_entity_table(alias_map)
    print(f"  {len(entries)} linkable aliases/names")

    files = collect_files()
    print(f"  {len(files)} markdown files to process")
    print()

    if not write:
        print("DRY RUN — no files will be modified. Pass --write to apply changes.")
        print()

    total_files_changed = 0
    total_links = 0

    for path in files:
        rel = path.relative_to(VAULT_ROOT)
        if verbose:
            print(f"  {rel}")
        changed, n = process_file(path, entries, write=write, verbose=verbose)
        if changed:
            total_files_changed += 1
            total_links += n
            if not verbose:
                print(f"  {'Wrote' if write else 'Would change'}: {rel}  (+{n} links)")

    print()
    print(f"{'Modified' if write else 'Would modify'}: {total_files_changed} files")
    print(f"Links {'added' if write else 'to add'}: {total_links}")
    if not write:
        print()
        print("Run with --write to apply.")
    else:
        print()
        print("Phase 4 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 4: rewrite source files with entity wikilinks")
    ap.add_argument("--write",   action="store_true", help="Actually modify files (default: dry run)")
    ap.add_argument("--verbose", action="store_true", help="Show each linked entity per file")
    args = ap.parse_args()
    main(write=args.write, verbose=args.verbose)
