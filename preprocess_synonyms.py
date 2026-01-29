#!/usr/bin/env python3
"""preprocess_synonyms.py

Build synonym sets from a TSV-like synonym table for synonym-substitution steganography,
*without* transitive merging (no Union-Find/DSU).

Input format (per line):
<lemma>\t<syn1>\t<syn2>\t...

Key differences vs preprocess_synonyms.py:
- Each line becomes an independent candidate synset (after filtering + dedupe).
- We then make synsets DISJOINT so that each normalized token appears in at most
  one synset (required for bijective token_to_entry).
- Optional: drop or truncate very large synsets to keep substitutions semantically tighter.

Outputs (same structure as before):
- synsets.json: list[list[str]]
- token_to_entry.json: mapping normalized_token -> [synset_id, index_in_synset]
- report.json: stats

Usage:
  python preprocess_synonyms_disjoint.py --in_tsv synonyms.tsv --out_dir out_syn

Recommended knobs for better semantic quality:
  --max_synset_size 32   (or 64)
  --sort_policy small_first

"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")


def norm_token(s: str) -> str:
    """Normalization used for matching / identity."""
    s = s.strip()
    s = ZERO_WIDTH_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = SPACE_RE.sub(" ", s)
    s = s.casefold()
    return s


def is_single_word_surface(s: str) -> bool:
    """Keep only single-word surfaces (no spaces after trimming/collapsing)."""
    s = SPACE_RE.sub(" ", s.strip())
    return (" " not in s) and (s != "")


@dataclass
class DropStats:
    dropped_multiword: int = 0
    dropped_empty: int = 0
    dropped_lines_too_short: int = 0
    dropped_synsets_singleton_after_disjoint: int = 0
    dropped_synsets_over_max_size: int = 0
    truncated_synsets_over_max_size: int = 0
    kept_unique_tokens_total: int = 0


def parse_tsv_lines(path: Path, stats: DropStats) -> Iterable[List[Tuple[str, str]]]:
    """Yield per line a list of (norm, surface) tokens, single-word only.

    - Dedupe within a line by normalized form.
    - Keeps first surface seen for a normalized token within that line.
    """
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            cols = raw.split("\t")
            cols = [c for c in cols if c.strip() != ""]
            if len(cols) < 2:
                stats.dropped_lines_too_short += 1
                continue

            seen_norm: Set[str] = set()
            toks: List[Tuple[str, str]] = []
            for c in cols:
                surf = SPACE_RE.sub(" ", c.strip())
                if surf == "":
                    stats.dropped_empty += 1
                    continue
                if not is_single_word_surface(surf):
                    stats.dropped_multiword += 1
                    continue
                n = norm_token(surf)
                if n in seen_norm:
                    continue
                seen_norm.add(n)
                toks.append((n, surf))

            if len(toks) < 2:
                stats.dropped_lines_too_short += 1
                continue

            yield toks


def build_synsets_disjoint(
    path: Path,
    max_synset_size: Optional[int],
    oversize_policy: str,
    sort_policy: str,
) -> Tuple[List[List[str]], Dict[str, Tuple[int, int]], dict]:
    """Build disjoint synsets.

    oversize_policy:
      - 'drop': drop synsets larger than max_synset_size
      - 'truncate': keep only the first max_synset_size items (deterministic)

    sort_policy determines which candidate synsets get to "claim" shared tokens first:
      - 'small_first' (recommended): keeps tighter semantic clusters
      - 'large_first': maximizes capacity but can worsen semantics
    """

    stats = DropStats()

    # Global stable representative surface for each normalized token: first seen in file wins.
    norm_to_surface: Dict[str, str] = {}

    # Candidate synsets as lists of normalized tokens
    candidates: List[List[str]] = []

    for toks in parse_tsv_lines(path, stats):
        norms = [n for n, _ in toks]
        # record surfaces
        for n, surf in toks:
            if n not in norm_to_surface:
                norm_to_surface[n] = surf
                stats.kept_unique_tokens_total += 1

        # sort deterministically (by normalized token)
        norms = sorted(norms)

        if max_synset_size is not None and len(norms) > max_synset_size:
            if oversize_policy == "drop":
                stats.dropped_synsets_over_max_size += 1
                continue
            elif oversize_policy == "truncate":
                stats.truncated_synsets_over_max_size += 1
                norms = norms[:max_synset_size]
            else:
                raise ValueError("oversize_policy must be drop or truncate")

        if len(norms) >= 2:
            candidates.append(norms)

    # Decide ordering for disjointization
    if sort_policy == "small_first":
        candidates.sort(key=lambda ss: (len(ss), ss))
    elif sort_policy == "large_first":
        candidates.sort(key=lambda ss: (-len(ss), ss))
    else:
        raise ValueError("sort_policy must be small_first or large_first")

    claimed: Set[str] = set()
    synsets_norm: List[List[str]] = []

    for ss in candidates:
        # keep only tokens not already used elsewhere
        filtered = [n for n in ss if n not in claimed]
        if len(filtered) < 2:
            stats.dropped_synsets_singleton_after_disjoint += 1
            continue
        for n in filtered:
            claimed.add(n)
        synsets_norm.append(filtered)

    # deterministic ordering of synsets themselves (already deterministic from candidates sort,
    # but keep stable rule anyway)
    synsets_norm.sort(key=lambda ss: (len(ss), ss))

    # Convert to surfaces
    synsets: List[List[str]] = []
    for ss_norm in synsets_norm:
        synsets.append([norm_to_surface[n] for n in ss_norm])

    # Build index: norm token -> (synset_id, index)
    token_to_entry: Dict[str, Tuple[int, int]] = {}
    for sid, ss_norm in enumerate(synsets_norm):
        for idx, n in enumerate(ss_norm):
            token_to_entry[n] = (sid, idx)

    report = {
        "input_file": str(path),
        "synset_count": len(synsets),
        "token_count": sum(len(s) for s in synsets),
        "unique_tokens_seen_kept_singleword": stats.kept_unique_tokens_total,
        "dropped_multiword": stats.dropped_multiword,
        "dropped_empty": stats.dropped_empty,
        "dropped_lines_too_short": stats.dropped_lines_too_short,
        "dropped_synsets_over_max_size": stats.dropped_synsets_over_max_size,
        "truncated_synsets_over_max_size": stats.truncated_synsets_over_max_size,
        "dropped_synsets_singleton_after_disjoint": stats.dropped_synsets_singleton_after_disjoint,
        "max_synset_size": max_synset_size,
        "oversize_policy": oversize_policy,
        "sort_policy": sort_policy,
        "notes": [
            "Each TSV line becomes a candidate synset (no transitive DSU merging).",
            "Synsets are made disjoint: each normalized token appears in at most one synset.",
            "Tokens are normalized with NFKC + whitespace collapse + casefold (same as previous preprocess).",
        ],
    }

    return synsets, token_to_entry, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input TSV (tab-separated) synonyms file")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument(
        "--max_synset_size",
        type=int,
        default=None,
        help="If set, apply an upper bound to synset size (recommended 32 or 64).",
    )
    ap.add_argument(
        "--oversize_policy",
        choices=["drop", "truncate"],
        default="truncate",
        help="What to do with synsets larger than --max_synset_size.",
    )
    ap.add_argument(
        "--sort_policy",
        choices=["small_first", "large_first"],
        default="small_first",
        help="Which synsets get to claim shared tokens first during disjointization.",
    )

    args = ap.parse_args()

    in_path = Path(args.in_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synsets, token_to_entry, report = build_synsets_disjoint(
        in_path,
        max_synset_size=args.max_synset_size,
        oversize_policy=args.oversize_policy,
        sort_policy=args.sort_policy,
    )

    (out_dir / "synsets.json").write_text(
        json.dumps(synsets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # JSON wants lists, not tuples
    token_to_entry_json = {k: [v[0], v[1]] for k, v in token_to_entry.items()}
    (out_dir / "token_to_entry.json").write_text(
        json.dumps(token_to_entry_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"OK: wrote {len(synsets)} synsets to {out_dir / 'synsets.json'}")
    print(f"OK: wrote token index to {out_dir / 'token_to_entry.json'}")
    print(f"OK: wrote report to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
