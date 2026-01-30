#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import unicodedata
import zlib
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

# --- Tokenization/word definition (consistent met stego-constraints) ---
TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*|[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+"
)
WORD_RE = re.compile(r"^[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*$")

PREFERRED_TEXT_COLS = ("text", "content", "body", "title")

def is_word_token(tok: str) -> bool:
    return bool(WORD_RE.fullmatch(tok))

def norm_token(tok: str) -> str:
    # Normalisatie conform encode/decode-pipeline
    t = unicodedata.normalize("NFKC", tok).casefold()
    t = t.replace("’", "'")
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    t = " ".join(t.split())
    return t


# --- Optionele lemma-lookup (voor matchen van verbogen vormen) ---
# Indien ingeschakeld (--use_lemma_lookup):
#   1) surface lookup
#   2) lemma(surface) lookup via spaCy
#
# Decoder heeft dit niet nodig: geschreven stegotokens zijn synset-items.
def make_lemmatizer(spacy_model: str):
    """Maak een gecachte lemma(surface)->lemma functie via spaCy."""
    try:
        import spacy  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "spaCy is required for --use_lemma_lookup. Install with: pip install spacy\n"
            "and download a Dutch model, e.g.: python -m spacy download nl_core_news_sm"
        ) from e

    try:
        nlp = spacy.load(spacy_model)
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy model '{spacy_model}'. Install it with:\n"
            f"  python -m spacy download {spacy_model}\n"
        ) from e

    @lru_cache(maxsize=50000)
    def lemma(surface: str) -> str:
        # Tag in korte context voor stabielere POS/lemma
        doc = nlp(f"Ik zie {surface}.")
        target = norm_token(surface)
        best = None
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            if norm_token(t.text) == target:
                best = t
                break
        if best is None:
            # fallback: laatste niet-interpunctietoken
            for t in reversed(doc):
                if not (t.is_space or t.is_punct):
                    best = t
                    break
        if best is None:
            return surface
        # Sommige modellen geven '-PRON-'; behoud dan surface
        lem = best.lemma_
        if not lem or lem == "-PRON-":
            return surface
        return lem

    return lemma

def bits_per_synset(k: int) -> int:
    return int(math.floor(math.log2(k))) if k >= 2 else 0

def u32_to_bits(x: int) -> str:
    # Big-endian 32-bit
    return "".join("1" if (x >> (31 - i)) & 1 else "0" for i in range(32))

def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)

def match_case_like(src: str, repl: str) -> str:
    # Eenvoudige case-matching
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl[:1].upper() + repl[1:]
    return repl

def load_synset_pos(path: str) -> Dict[int, str]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    if isinstance(data, list):
        return {i: v for i, v in enumerate(data)}
    raise ValueError("synset_pos must be a list or dict")

def choose_text_column(fieldnames: List[str], user_text_col: Optional[str]) -> str:
    if user_text_col:
        if user_text_col not in fieldnames:
            raise ValueError(f"--text_col '{user_text_col}' not in CSV columns: {fieldnames}")
        return user_text_col
    for c in PREFERRED_TEXT_COLS:
        if c in fieldnames:
            return c
    # fallback: eerste kolom
    return fieldnames[0]

def row_capacity_bits(
    text: str,
    synsets: List[List[str]],
    token_to_entry: Dict[str, Tuple[int, int]],
    synset_pos: Dict[int, str],
    pos_mode: str
,
    lemmatize: Optional[callable] = None
) -> int:
    cap = 0
    for tok in TOKEN_RE.findall(text):
        if not is_word_token(tok):
            continue
        nt = norm_token(tok)
        entry = token_to_entry.get(nt)
        if entry is None and lemmatize is not None:
            lem = lemmatize(tok)
            if lem:
                entry = token_to_entry.get(norm_token(lem))
        if entry is None:
            continue
        sid, idx = entry
        if sid < 0 or sid >= len(synsets):
            continue
        if pos_mode and pos_mode != "ANY":
            p = synset_pos.get(sid)
            if p is not None and p != pos_mode:
                continue
        ss = synsets[sid]
        b = bits_per_synset(len(ss))
        if b <= 0:
            continue
        # Alleen eerste 2^b indices zijn encodable
        limit = 1 << b
        if idx < 0 or idx >= limit:
            continue
        cap += b
    return cap

def embed_message_in_row(
    text: str,
    bitstring: str,
    synsets: List[List[str]],
    token_to_entry: Dict[str, Tuple[int, int]],
    synset_pos: Dict[int, str],
    pos_mode: str
,
    lemmatize: Optional[callable] = None
) -> Optional[str]:
    """
    Embed volledige bitstring in één tekst.
    Geeft stegotekst bij succes, anders None.
    """
    tokens = TOKEN_RE.findall(text)
    pos = 0  # bitpositie

    for i, tok in enumerate(tokens):
        if not is_word_token(tok):
            continue
        nt = norm_token(tok)
        entry = token_to_entry.get(nt)
        if entry is None and lemmatize is not None:
            lem = lemmatize(tok)
            if lem:
                entry = token_to_entry.get(norm_token(lem))
        if entry is None:
            continue
        sid, idx_current = entry
        if sid < 0 or sid >= len(synsets):
            continue

        if pos_mode and pos_mode != "ANY":
            p = synset_pos.get(sid)
            if p is not None and p != pos_mode:
                continue

        ss = synsets[sid]
        b = bits_per_synset(len(ss))
        if b <= 0:
            continue
        limit = 1 << b

        # Token niet encodable binnen window
        if idx_current < 0 or idx_current >= limit:
            continue

        remaining = len(bitstring) - pos
        if remaining <= 0:
            break

        if remaining < b:
            chunk = bitstring[pos:] + ("0" * (b - remaining))
            v = int(chunk, 2)
            if v >= limit:
                v = 0
            tokens[i] = match_case_like(tok, ss[v])
            pos = len(bitstring)
            break
        else:
            v = int(bitstring[pos:pos + b], 2)
            pos += b
            if v >= limit:
                v = 0
            tokens[i] = match_case_like(tok, ss[v])

            if pos >= len(bitstring):
                break

    if pos < len(bitstring):
        return None  # onvoldoende capaciteit
    return "".join(tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--synsets", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--synset_pos", required=True)
    ap.add_argument("--pos", default="ANY", help="NOUN/VERB/ADJ/ADV/ANY")
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--message", required=True)
    ap.add_argument("--text_col", default=None, help="Name of column to embed into (default: auto)")
    ap.add_argument("--use_lemma_lookup", action="store_true", help="Gebruik lemma-lookup bij missende surface-vormen (vereist spaCy).")
    ap.add_argument("--spacy_model", default="nl_core_news_sm", help="spaCy-model voor lemma-lookup.")
    ap.add_argument("--print_stats", action="store_true")
    args = ap.parse_args()

    synsets: List[List[str]] = json.load(open(args.synsets, "r", encoding="utf-8"))
    idx_raw = json.load(open(args.index, "r", encoding="utf-8"))
    token_to_entry: Dict[str, Tuple[int, int]] = {k: (v[0], v[1]) for k, v in idx_raw.items()}
    synset_pos = load_synset_pos(args.synset_pos)

    lemmatizer = make_lemmatizer(args.spacy_model) if args.use_lemma_lookup else None

    payload = args.message.encode("utf-8")
    length = len(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    bitstring = u32_to_bits(length) + u32_to_bits(crc) + bytes_to_bits(payload)
    needed_bits = len(bitstring)

    attempted = 0
    succeeded = 0
    skipped = 0

    with open(args.in_csv, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row / no columns.")
        text_col = choose_text_column(reader.fieldnames, args.text_col)

        with open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
            OUT_COLS = ["datetime", "title", "content", "stego", "category", "url"]

            writer = csv.DictWriter(fout, fieldnames=OUT_COLS)
            writer.writeheader()

            for row in reader:
                if attempted >= args.max_rows:
                    break
                attempted += 1

                text = row.get(text_col, "") or ""
                original_text = text
                cap = row_capacity_bits(text, synsets, token_to_entry, synset_pos, args.pos, lemmatize=lemmatizer)

                if cap < needed_bits:
                    skipped += 1
                    continue  # onvoldoende capaciteit voor deze rij

                stego_text = embed_message_in_row(text, bitstring, synsets, token_to_entry, synset_pos, args.pos, lemmatize=lemmatizer)
                if stego_text is None:
                    # Verwacht zeldzaam bij correcte capaciteitscheck
                    skipped += 1
                    continue
                out_row = {
                    "datetime": row.get("datetime", "") or "",
                    "title": row.get("title", "") or "",
                    "content": original_text,            # behoud origineel
                    "stego": stego_text,                 # schrijf stego
                    "category": row.get("category", "") or "",
                    "url": row.get("url", "") or "",
                }
                writer.writerow(out_row)
                succeeded += 1

    print(f"Message bytes: {length}")
    print(f"Needed bits (LEN+CRC+payload): {needed_bits}")
    print(f"Attempted rows: {attempted}")
    print(f"Succeeded rows written: {succeeded}")
    print(f"Skipped rows (insufficient capacity): {skipped}")
    print(f"OK: wrote stego CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
