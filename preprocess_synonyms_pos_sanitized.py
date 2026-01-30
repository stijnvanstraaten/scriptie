#!/usr/bin/env python3
"""
preprocess_synonyms_pos_sanitized.py

Derde preprocessingstap: sanitizing + macht-van-twee + deterministische
ambiguïteitsafhandeling.

- Filtert tokens die geen "woordtoken" zijn.
- Dedupliceert per synset op genormaliseerde vorm.
- Truncateert synsets naar groottes als macht van twee (voor vaste bitmapping).
- Lost globale ambiguïteit op door elk genormaliseerd token toe te wijzen aan
  precies één "winner" synset (deterministisch).

Outputs:
- synsets_pos.sanitized.json
- synset_pos.sanitized.json
- token_to_entry_pos.sanitized.json
- sanitize_report.json
"""
import argparse, json, math, os, re, unicodedata
from collections import defaultdict
from typing import List, Dict

WORD_RE = re.compile(r"^[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*$")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
SPACE_RE = re.compile(r"\s+")

def is_word_token(tok: str) -> bool:
    """Controleert of een token voldoet aan de woordtoken-regel."""
    return bool(WORD_RE.fullmatch(tok))

def norm_token(s: str) -> str:
    """Normaliseert een token voor consistente matching."""
    s = s.strip()
    s = ZERO_WIDTH_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = SPACE_RE.sub(" ", s)
    s = s.casefold()
    return s

def bits_per_synset(k: int) -> int:
    """Aantal bits dat past bij synsetgrootte k (log2, afgerond naar beneden)."""
    return int(math.floor(math.log2(k))) if k >= 2 else 0

def read_json(path: str):
    """Leest JSON-bestand."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj):
    """Schrijft JSON-bestand."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synsets_in", required=True)
    ap.add_argument("--synset_pos_in", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    synsets: List[List[str]] = read_json(args.synsets_in)
    synset_pos = read_json(args.synset_pos_in)

    os.makedirs(args.out_dir, exist_ok=True)

    report = {
        "synsets_total": len(synsets),
        "items_total_before": sum(len(ss) for ss in synsets),

        "removed_not_wordtoken": 0,
        "removed_dedup_norm_in_synset": 0,
        "capped_due_to_power_of_two_limit": 0,
        "synsets_dropped_len_lt_2": 0,

        "global_ambiguous_norm_tokens_total": 0,
        "global_ambiguous_norm_tokens_removed_nonwinner": 0,
        "global_ambiguous_norm_tokens_kept_winner": 0,

        "examples": {
            "not_wordtoken": [],
            "dedup_norm": [],
            "ambiguous_norm_sample": [],
        },
    }

    stage_synsets: List[List[str]] = []
    stage_pos: List[str] = []

    def get_pos(old_sid: int):
        """Haalt POS-label op, compatibel met list of dict input."""
        if isinstance(synset_pos, list):
            return synset_pos[old_sid]
        return synset_pos[str(old_sid)]

    # 1-4) per-synset filter/dedupe/truncate
    for old_sid, ss in enumerate(synsets):
        filtered = []
        for w in ss:
            if not is_word_token(w):
                report["removed_not_wordtoken"] += 1
                if len(report["examples"]["not_wordtoken"]) < 20:
                    report["examples"]["not_wordtoken"].append(w)
                continue
            filtered.append(w)
        if len(filtered) < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue

        seen = set()
        deduped = []
        for w in filtered:
            nt = norm_token(w)
            if nt in seen:
                report["removed_dedup_norm_in_synset"] += 1
                if len(report["examples"]["dedup_norm"]) < 20:
                    report["examples"]["dedup_norm"].append(w)
                continue
            seen.add(nt)
            deduped.append(w)
        if len(deduped) < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue

        b = bits_per_synset(len(deduped))
        limit = 1 << b
        if limit < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue
        if len(deduped) > limit:
            report["capped_due_to_power_of_two_limit"] += (len(deduped) - limit)
            deduped = deduped[:limit]

        if len(deduped) < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue

        stage_synsets.append(deduped)
        stage_pos.append(get_pos(old_sid))

    # 5) globale ambiguïteit: norm_token -> winner synset
    occ = defaultdict(list)
    for sid, ss in enumerate(stage_synsets):
        for w in ss:
            occ[norm_token(w)].append(sid)

    winner = {}
    for nt, sids in occ.items():
        if len(sids) <= 1:
            continue
        report["global_ambiguous_norm_tokens_total"] += 1
        # winner: grootste synset; ties -> laagste sid
        best_sid = min(sids, key=lambda s: (-len(stage_synsets[s]), s))
        winner[nt] = best_sid
        if len(report["examples"]["ambiguous_norm_sample"]) < 20:
            report["examples"]["ambiguous_norm_sample"].append(
                {"nt": nt, "sids": sids, "winner": best_sid}
            )

    final_synsets: List[List[str]] = []
    final_pos: List[str] = []

    for sid, ss in enumerate(stage_synsets):
        out = []
        for w in ss:
            nt = norm_token(w)
            if nt in winner:
                if winner[nt] == sid:
                    out.append(w)
                    report["global_ambiguous_norm_tokens_kept_winner"] += 1
                else:
                    report["global_ambiguous_norm_tokens_removed_nonwinner"] += 1
            else:
                out.append(w)

        if len(out) < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue

        # opnieuw-truncate na winner-filtering
        b = bits_per_synset(len(out))
        limit = 1 << b
        if len(out) > limit:
            report["capped_due_to_power_of_two_limit"] += (len(out) - limit)
            out = out[:limit]
        if len(out) < 2:
            report["synsets_dropped_len_lt_2"] += 1
            continue

        final_synsets.append(out)
        final_pos.append(stage_pos[sid])

    # Index: norm token -> [synset_id, index]
    token_to_entry: Dict[str, List[int]] = {}
    for sid, ss in enumerate(final_synsets):
        for idx, w in enumerate(ss):
            token_to_entry[norm_token(w)] = [sid, idx]

    report["synsets_total_after"] = len(final_synsets)
    report["items_total_after"] = sum(len(ss) for ss in final_synsets)

    out_synsets = os.path.join(args.out_dir, "synsets_pos.sanitized.json")
    out_synset_pos = os.path.join(args.out_dir, "synset_pos.sanitized.json")
    out_index = os.path.join(args.out_dir, "token_to_entry_pos.sanitized.json")
    out_report = os.path.join(args.out_dir, "sanitize_report.json")

    write_json(out_synsets, final_synsets)
    write_json(out_synset_pos, final_pos)
    write_json(out_index, token_to_entry)
    write_json(out_report, report)

    print("OK wrote:")
    print(" ", out_synsets)
    print(" ", out_synset_pos)
    print(" ", out_index)
    print(" ", out_report)
    print("Stats:", json.dumps({k: report[k] for k in [
        "synsets_total", "synsets_total_after",
        "items_total_before", "items_total_after",
        "removed_not_wordtoken", "removed_dedup_norm_in_synset",
        "global_ambiguous_norm_tokens_total",
        "global_ambiguous_norm_tokens_removed_nonwinner",
        "global_ambiguous_norm_tokens_kept_winner",
        "capped_due_to_power_of_two_limit",
        "synsets_dropped_len_lt_2"
    ]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
