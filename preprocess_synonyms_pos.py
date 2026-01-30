#!/usr/bin/env python3
"""
preprocess_synonyms_pos.py

Tweede preprocessingstap: POS-filtering van synsets.

- Leest synsets.json uit stap 1.
- Bepaalt POS per token met spaCy (in context).
- Houdt per synset alleen tokens met de meerderheids-POS.
- Verwijdert synsets met < 2 tokens na filtering.
- Schrijft POS-gefilterde synsets en bijbehorende indexbestanden.
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
ALLOWED_POS = {"NOUN", "VERB", "ADJ"}


def norm_token(s: str) -> str:
    """Normaliseert een token voor consistente matching."""
    s = s.strip()
    s = ZERO_WIDTH_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = SPACE_RE.sub(" ", s)
    s = s.casefold()
    return s


@dataclass
class PosReport:
    """Statistieken over POS-filtering."""
    input_synset_count: int = 0
    output_synset_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    dropped_synsets_small_after_pos: int = 0
    pruned_tokens_count: int = 0


def tag_token_pos(nlp, surface: str) -> Tuple[str, str]:
    """
    Bepaalt POS (en optioneel morfologie) voor een token in context.

    Tokens worden getagd in een neutrale zin om POS-stabiliteit te verbeteren.
    """
    doc = nlp(f"Ik zie {surface}.")
    target_norm = norm_token(surface)
    best = None

    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if norm_token(t.text) == target_norm:
            best = t
            break

    if best is None:
        for t in reversed(doc):
            if not (t.is_space or t.is_punct):
                best = t
                break

    if best is None:
        return ("X", "")

    return (best.pos_, str(best.morph) if best.morph else "")


def majority_pos(pos_list: List[str]) -> str:
    """Bepaalt de meerderheids-POS (deterministisch bij gelijke aantallen)."""
    c = Counter(pos_list)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_synsets(path: Path) -> List[List[str]]:
    """Laadt synsets.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    """Schrijft object als JSON-bestand."""
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

    # Lazy import zodat script ook zonder spaCy geïmporteerd kan worden
    import spacy  # type: ignore

    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy model '{args.spacy_model}'.\n"
            f"Install it first, e.g.:\n"
            f"  python -m spacy download {args.spacy_model}\n"
        ) from e

    # POS-lexicon: norm -> {surface, pos, morph}
    pos_lex: Dict[str, Dict[str, str]] = {}

    # Verzamel unieke tokens
    all_tokens: Dict[str, str] = {}
    for ss in synsets:
        for surf in ss:
            n = norm_token(surf)
            if n not in all_tokens:
                all_tokens[n] = surf

    # Tag elk token één keer
    for n, surf in all_tokens.items():
        pos, morph = tag_token_pos(nlp, surf)
        rec = {"surface": surf, "pos": pos}
        if args.keep_morph:
            rec["morph"] = morph
        pos_lex[n] = rec

    # POS-filtering per synset
    new_synsets: List[List[str]] = []
    synset_pos: List[str] = []

    for ss in synsets:
        poss = []
        ss_norm = []

        for surf in ss:
            n = norm_token(surf)
            ss_norm.append(n)
            poss.append(pos_lex.get(n, {}).get("pos", "X"))

        maj = majority_pos(poss)

        # Verwijder synsets met niet-toegestane meerderheids-POS
        if maj not in ALLOWED_POS:
            report.dropped_synsets_small_after_pos += 1
            continue

        kept_norm = [n for n, p in zip(ss_norm, poss) if p == maj]
        report.pruned_tokens_count += len(ss_norm) - len(kept_norm)

        if len(kept_norm) < 2:
            report.dropped_synsets_small_after_pos += 1
            continue

        kept_norm = sorted(set(kept_norm))
        kept_surfaces = [pos_lex[n]["surface"] for n in kept_norm]

        new_synsets.append(kept_surfaces)
        synset_pos.append(maj)

    # Deterministische sortering van synsets
    def synset_sort_key(ss_surfaces: List[str]):
        norms = [norm_token(x) for x in ss_surfaces]
        norms.sort()
        return (len(ss_surfaces), norms)

    order = list(range(len(new_synsets)))
    order.sort(key=lambda i: synset_sort_key(new_synsets[i]))

    new_synsets = [new_synsets[i] for i in order]
    synset_pos = [synset_pos[i] for i in order]

    report.output_synset_count = len(new_synsets)
    report.output_token_count = sum(len(ss) for ss in new_synsets)

    # Index: norm token -> [synset_id, index]
    token_to_entry_pos: Dict[str, List[int]] = {}
    for sid, ss in enumerate(new_synsets):
        for idx, surf in enumerate(ss):
            token_to_entry_pos[norm_token(surf)] = [sid, idx]

    # Outputs
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
