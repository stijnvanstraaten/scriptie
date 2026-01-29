"""
preprocess_synonyms_pos.py

Second-stage preprocessing:
- Reads synsets.json from stage 1 (single-word only).
- Uses spaCy to POS-tag each token (in a neutral context sentence).
- For each synset, keeps only tokens that match the synset's majority POS.
- Drops synsets that end up with < 2 tokens.
- Writes new synsets + index + POS metadata.

Usage:
  python preprocess_synonyms_pos.py \
    --in_synsets out_syn/synsets.json \
    --out_dir out_syn_pos \
    --spacy_model nl_core_news_sm

Notes:
- POS tagging of single tokens is noisy; we tag in context: "Ik zie <word> ."
- Deterministic ordering is preserved (sorted by normalized form).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

SPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")


def norm_token(s: str) -> str:
    s = s.strip()
    s = ZERO_WIDTH_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = SPACE_RE.sub(" ", s)
    s = s.casefold()
    return s


@dataclass
class PosReport:
    input_synset_count: int = 0
    output_synset_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    dropped_synsets_small_after_pos: int = 0
    pruned_tokens_count: int = 0


def tag_token_pos(nlp, surface: str) -> Tuple[str, str]:
    """
    Returns (pos, morph_string) for a token.
    We tag in context for better POS stability.
    """
    # Neutral-ish context; period makes it a complete sentence
    doc = nlp(f"Ik zie {surface}.")
    # Find the token that matches surface best.
    # spaCy may tokenize punctuation separately, so we search for token with same text or normalized match.
    target_norm = norm_token(surface)
    best = None
    for t in doc:
        # skip punctuation tokens
        if t.is_space or t.is_punct:
            continue
        if norm_token(t.text) == target_norm:
            best = t
            break
    if best is None:
        # fallback: take the last non-punct token
        for t in reversed(doc):
            if not (t.is_space or t.is_punct):
                best = t
                break

    if best is None:
        return ("X", "")
    return (best.pos_, str(best.morph) if best.morph else "")


def majority_pos(pos_list: List[str]) -> str:
    """
    Choose a majority POS. Ties broken deterministically by POS string.
    """
    c = Counter(pos_list)
    # sort by count desc, then pos asc
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_synsets(path: Path) -> List[List[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_synsets", required=True, help="Path to synsets.json from stage 1.")
    ap.add_argument("--out_dir", required=True, help="Output directory for POS-filtered artifacts.")
    ap.add_argument("--spacy_model", default="nl_core_news_sm", help="spaCy model name (Dutch).")
    ap.add_argument("--keep_morph", action="store_true", help="Include morph strings in pos_lexicon.json.")
    args = ap.parse_args()

    in_path = Path(args.in_synsets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synsets = load_synsets(in_path)

    report = PosReport()
    report.input_synset_count = len(synsets)
    report.input_token_count = sum(len(ss) for ss in synsets)

    # Lazy import so the script can be imported without spaCy installed
    import spacy  # type: ignore

    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy model '{args.spacy_model}'.\n"
            f"Install it first, e.g.:\n"
            f"  python -m spacy download {args.spacy_model}\n"
        ) from e

    # Build POS lexicon: norm -> {surface, pos, morph}
    pos_lex: Dict[str, Dict[str, str]] = {}

    # Tag all unique tokens once
    all_tokens: Dict[str, str] = {}  # norm -> surface representative
    for ss in synsets:
        for surf in ss:
            n = norm_token(surf)
            if n not in all_tokens:
                all_tokens[n] = surf

    for n, surf in all_tokens.items():
        pos, morph = tag_token_pos(nlp, surf)
        rec = {"surface": surf, "pos": pos}
        if args.keep_morph:
            rec["morph"] = morph
        pos_lex[n] = rec

    # POS-filter synsets
    new_synsets: List[List[str]] = []
    synset_pos: List[str] = []

    for ss in synsets:
        # Determine majority POS within this synset
        poss = []
        ss_norm = []
        for surf in ss:
            n = norm_token(surf)
            ss_norm.append(n)
            poss.append(pos_lex.get(n, {}).get("pos", "X"))

        maj = majority_pos(poss)

        # Keep only tokens whose POS == maj
        kept_norm = [n for n, p in zip(ss_norm, poss) if p == maj]
        dropped_here = len(ss_norm) - len(kept_norm)
        report.pruned_tokens_count += dropped_here

        if len(kept_norm) < 2:
            report.dropped_synsets_small_after_pos += 1
            continue

        # Deterministic ordering inside synset: sort by normalized form
        kept_norm = sorted(set(kept_norm))

        # Output surfaces: we keep the stored representative surface from pos_lex
        kept_surfaces = [pos_lex[n]["surface"] for n in kept_norm]
        new_synsets.append(kept_surfaces)
        synset_pos.append(maj)

    # Deterministic ordering of synsets themselves: by size then lexicographically (on normalized forms)
    def synset_sort_key(ss_surfaces: List[str]):
        norms = [norm_token(x) for x in ss_surfaces]
        norms.sort()
        return (len(ss_surfaces), norms)

    # We must reorder synset_pos accordingly, so we sort with indices
    order = list(range(len(new_synsets)))
    order.sort(key=lambda i: synset_sort_key(new_synsets[i]))

    new_synsets = [new_synsets[i] for i in order]
    synset_pos = [synset_pos[i] for i in order]

    report.output_synset_count = len(new_synsets)
    report.output_token_count = sum(len(ss) for ss in new_synsets)

    # Build token_to_entry_pos: norm -> [sid, idx]
    token_to_entry_pos: Dict[str, List[int]] = {}
    for sid, ss in enumerate(new_synsets):
        # sort inside synset already deterministic but re-ensure index mapping is on that list
        for idx, surf in enumerate(ss):
            token_to_entry_pos[norm_token(surf)] = [sid, idx]

    # Write outputs
    save_json(out_dir / "synsets_pos.json", new_synsets)
    save_json(out_dir / "token_to_entry_pos.json", token_to_entry_pos)
    save_json(out_dir / "synset_pos.json", synset_pos)
    save_json(out_dir / "pos_lexicon.json", pos_lex)
    save_json(out_dir / "report_pos.json", report.__dict__)

    print(f"Wrote POS-filtered synsets: {out_dir / 'synsets_pos.json'}")
    print(f"Wrote POS token index:      {out_dir / 'token_to_entry_pos.json'}")
    print(f"Wrote synset POS labels:    {out_dir / 'synset_pos.json'}")
    print(f"Wrote POS lexicon:          {out_dir / 'pos_lexicon.json'}")
    print(f"Wrote report:               {out_dir / 'report_pos.json'}")


if __name__ == "__main__":
    main()