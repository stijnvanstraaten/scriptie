#!/usr/bin/env python3
"""
decode.py

Decodeert een embedded payload uit stegotekst door synset-indexen terug te lezen.

Het script gebruikt dezelfde preprocessing-artefacten als de encoder:
- synsets.json (of afgeleide variant)
- token_to_entry.json
- synset_pos.json

De decoder extraheert per encodable token een bitblok met lengte b=floor(log2(|synset|)).
Vervolgens wordt een header (LEN + CRC32) gelezen en wordt de payload gevalideerd
met CRC32. Alleen CRC-correcte berichten worden gerapporteerd.

Input:
- Een CSV met een kolom 'stego' (of een gekozen tekstkolom).

Output:
- Gevonden berichten (bytes of UTF-8) naar stdout, inclusief LEN en CRC32.
"""

import argparse
import csv
import json
import math
import re
import unicodedata
import zlib
from typing import Dict, List, Tuple, Optional


TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*|[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+"
)
WORD_RE = re.compile(r"^[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*$")

PREFERRED_TEXT_COLS = ("text", "content", "body", "title")


def is_word_token(tok: str) -> bool:
    """True als tok een woordtoken is onder de gebruikte constraints."""
    return bool(WORD_RE.fullmatch(tok))


def norm_token(tok: str) -> str:
    """Normalisatie voor lookup (Unicode + casefold + apostrof + opschoning)."""
    t = unicodedata.normalize("NFKC", tok).casefold()
    t = t.replace("’", "'")
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    t = " ".join(t.split())
    return t


def bits_per_synset(k: int) -> int:
    """Aantal encodable bits op basis van synsetgrootte (floor(log2))."""
    return int(math.floor(math.log2(k))) if k >= 2 else 0


def u32_from_bits(bits: List[int]) -> int:
    """Decodeert een 32-bit big-endian integer uit bits."""
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def bits_to_bytes(bits: List[int]) -> bytes:
    """Zet een bitlijst om naar bytes (8 bits per byte)."""
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)


def load_synset_pos(path: str) -> Dict[int, str]:
    """Laadt synset_pos als dict[int->str], ongeacht list/dict input."""
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    if isinstance(data, list):
        return {i: v for i, v in enumerate(data)}
    raise ValueError("synset_pos must be a list or dict")


def choose_text_column(fieldnames: List[str], user_text_col: Optional[str]) -> str:
    """Kiest een tekstkolom (user override, anders heuristiek)."""
    if user_text_col:
        if user_text_col not in fieldnames:
            raise ValueError(f"--text_col '{user_text_col}' not in CSV columns: {fieldnames}")
        return user_text_col
    for c in PREFERRED_TEXT_COLS:
        if c in fieldnames:
            return c
    return fieldnames[0]


def extract_bits_from_text(
    text: str,
    synsets: List[List[str]],
    token_to_entry: Dict[str, Tuple[int, int]],
    synset_pos: Dict[int, str],
    pos_mode: str
) -> List[int]:
    """Haalt bits uit encodable tokens in de gegeven tekst."""
    bits: List[int] = []
    for tok in TOKEN_RE.findall(text):
        if not is_word_token(tok):
            continue
        nt = norm_token(tok)
        entry = token_to_entry.get(nt)
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

        limit = 1 << b
        if idx < 0 or idx >= limit:
            continue

        for shift in range(b - 1, -1, -1):
            bits.append((idx >> shift) & 1)

    return bits


def try_decode_from_bits(bits: List[int], max_len: int) -> Optional[Tuple[int, int, bytes]]:
    """Probeert LEN+CRC+payload te reconstrueren uit een bitstream."""
    if len(bits) < 64:
        return None

    header = bits[:64]
    length = u32_from_bits(header[:32])
    crc_expected = u32_from_bits(header[32:64])

    if length <= 0 or length > max_len:
        return None

    need = 64 + length * 8
    if len(bits) < need:
        return None

    payload_bits = bits[64:need]
    payload = bits_to_bytes(payload_bits)
    crc_got = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_got != crc_expected:
        return None

    return (length, crc_expected, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--synsets", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--synset_pos", required=True)
    ap.add_argument("--pos", default="ANY")
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_hits", type=int, default=0,
                    help="0 = geen limiet, >0 = stop na dit aantal hits")
    ap.add_argument("--print_utf8", action="store_true")
    ap.add_argument("--print_row_index", action="store_true")
    args = ap.parse_args()

    synsets: List[List[str]] = json.load(open(args.synsets, "r", encoding="utf-8"))
    idx_raw = json.load(open(args.index, "r", encoding="utf-8"))
    token_to_entry: Dict[str, Tuple[int, int]] = {k: (v[0], v[1]) for k, v in idx_raw.items()}
    synset_pos = load_synset_pos(args.synset_pos)

    hits = 0
    checked = 0
    unlimited = (args.max_hits == 0)

    with open(args.in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row / no columns.")
        text_col = choose_text_column(reader.fieldnames, args.text_col)

        for row_i, row in enumerate(reader):
            if checked >= args.max_rows:
                break
            checked += 1

            # Decodeer uit stego-kolom
            text = (row.get("stego") or "").strip()
            if not text:
                continue

            bits = extract_bits_from_text(text, synsets, token_to_entry, synset_pos, args.pos)
            decoded = try_decode_from_bits(bits, args.max_len)
            if decoded is None:
                continue

            length, crc_expected, payload = decoded
            hits += 1

            if args.print_row_index:
                print(f"\n=== HIT #{hits} (row={row_i}) ===")
            else:
                print(f"\n=== HIT #{hits} ===")

            print(f"LEN: {length}")
            print(f"CRC32: {crc_expected:08x}")

            if args.print_utf8:
                try:
                    print(payload.decode("utf-8", errors="strict"))
                except UnicodeDecodeError:
                    print(payload.decode("utf-8", errors="replace"))
            else:
                print(payload)

            if (not unlimited) and hits >= args.max_hits:
                break

    if hits == 0:
        print("No valid messages found (no row had a CRC-correct payload).")
    else:
        print(f"\nDone. Checked {checked} rows, found {hits} valid message(s).")


if __name__ == "__main__":
    main()
