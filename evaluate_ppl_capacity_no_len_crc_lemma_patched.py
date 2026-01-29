#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Tokenization / normalization (must match your stego pipeline)
# -------------------------
TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*|[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+"
)
WORD_RE = re.compile(r"^[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*$")

def is_word_token(tok: str) -> bool:
    return bool(WORD_RE.fullmatch(tok))

def norm_token(tok: str) -> str:
    t = unicodedata.normalize("NFKC", tok).casefold()
    t = t.replace("’", "'")
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    t = " ".join(t.split())
    return t

def bits_per_synset(k: int) -> int:
    return int(math.floor(math.log2(k))) if k >= 2 else 0

def u32_to_bits(x: int) -> str:
    return "".join("1" if (x >> (31 - i)) & 1 else "0" for i in range(32))

def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)

def match_case_like(src: str, repl: str) -> str:
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

def normalize_pos_arg(pos: str) -> str:
    p = (pos or "").strip().upper()
    if p in ("ALL", "ANY", ""):
        return "ANY"
    return p



def build_lemma_map(text: str, nlp) -> Dict[str, str]:
    """Map norm_token(surface) -> norm_token(lemma) using spaCy once per row."""
    m: Dict[str, str] = {}
    doc = nlp(text)
    for t in doc:
        if not is_word_token(t.text):
            continue
        surf = norm_token(t.text)
        lem = t.lemma_ if t.lemma_ else t.text
        m.setdefault(surf, norm_token(lem))
    return m

def lookup_entry(nt: str, token_to_entry: Dict[str, Tuple[int,int]], lemma_map: Optional[Dict[str,str]]) -> Optional[Tuple[int,int]]:
    e = token_to_entry.get(nt)
    if e is not None:
        return e
    if lemma_map is None:
        return None
    lem = lemma_map.get(nt)
    if not lem:
        return None
    return token_to_entry.get(lem)

# -------------------------
# Stego capacity + embedding per article
# -------------------------
def row_capacity_bits(
    text: str,
    synsets: List[List[str]],
    token_to_entry: Dict[str, Tuple[int, int]],
    synset_pos: Dict[int, str],
    pos_mode: str,
    lemma_map: Optional[Dict[str,str]] = None,
) -> int:
    cap = 0
    for tok in TOKEN_RE.findall(text):
        if not is_word_token(tok):
            continue
        nt = norm_token(tok)
        entry = lookup_entry(nt, token_to_entry, lemma_map)
        if entry is None:
            continue
        sid, idx = entry
        if sid < 0 or sid >= len(synsets):
            continue

        if pos_mode != "ANY":
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
        cap += b
    return cap

@dataclass
class EmbedResult:
    stego: Optional[str]
    substitutions: int
    success: int

def embed_message_in_row(
    text: str,
    bitstring: str,
    synsets: List[List[str]],
    token_to_entry: Dict[str, Tuple[int, int]],
    synset_pos: Dict[int, str],
    pos_mode: str,
    lemma_map: Optional[Dict[str,str]] = None,
) -> EmbedResult:
    """
    Embed entire bitstring in this row's text (one-message-per-row).
    Returns (stego_text or None), substitutions count, success flag.
    """
    tokens = TOKEN_RE.findall(text)
    pos = 0
    subs = 0

    for i, tok in enumerate(tokens):
        if not is_word_token(tok):
            continue
        nt = norm_token(tok)
        entry = lookup_entry(nt, token_to_entry, lemma_map)
        if entry is None:
            continue
        sid, idx_current = entry
        if sid < 0 or sid >= len(synsets):
            continue

        if pos_mode != "ANY":
            p = synset_pos.get(sid)
            if p is not None and p != pos_mode:
                continue

        ss = synsets[sid]
        b = bits_per_synset(len(ss))
        if b <= 0:
            continue
        limit = 1 << b
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
            new_tok = match_case_like(tok, ss[v])
            if new_tok != tok:
                subs += 1
            tokens[i] = new_tok
            pos = len(bitstring)
            break
        else:
            v = int(bitstring[pos:pos + b], 2)
            pos += b
            if v >= limit:
                v = 0
            new_tok = match_case_like(tok, ss[v])
            if new_tok != tok:
                subs += 1
            tokens[i] = new_tok

            if pos >= len(bitstring):
                break

    if pos < len(bitstring):
        return EmbedResult(None, subs, 0)
    return EmbedResult("".join(tokens), subs, 1)


# -------------------------
# Perplexity (causal LM) with sliding window
# -------------------------
@torch.no_grad()
def perplexity(text, model, tokenizer, device, max_length=1024, stride=512):
    text = (text or "").strip()
    if not text:
        return None

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)
    if input_ids.numel() < 2:
        return None

    model.eval()
    nlls = []
    seq_len = input_ids.size(0)

    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # only score newly added tokens

            input_ids_slice = input_ids[begin_loc:end_loc]
            target_ids = input_ids_slice.clone()
            target_ids[:-trg_len] = -100  # mask all but the new tokens

            outputs = model(input_ids_slice.unsqueeze(0), labels=target_ids.unsqueeze(0))
            nlls.append(outputs.loss * trg_len)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len).item()
    return float(ppl)



# -------------------------
# Main experiment
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV (dutch_news.csv or news_stego.csv)")
    ap.add_argument("--out_csv", default="results_perplexity.csv", help="Results CSV")
    ap.add_argument("--text_col", default="content", help="Original text column")
    ap.add_argument("--stego_col", default="stego", help="Stego text column (if present)")
    ap.add_argument("--id_cols", default="url,datetime,title", help="Comma-separated identifier cols to copy into results")

    ap.add_argument("--synsets", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--synset_pos", required=True)

    ap.add_argument("--modes", default="NOUN,VERB,ADJ,ANY", help="Comma-separated modes")
    ap.add_argument("--max_rows", type=int, default=1000)

    ap.add_argument("--message", default="test", help="Message to embed when generating stego")
    ap.add_argument("--omit_len_crc", action="store_true",
                    help="If set: embed ONLY the payload bits (no 32-bit LEN and no 32-bit CRC). This reduces needed bits by 64, increasing success rate, but removes integrity checking.")
    ap.add_argument("--use_lemma_lookup", action="store_true",
                    help="If set: when a surface token is not in the index, try spaCy lemma (normalized) for lookup. Greatly increases capacity with inflected forms.")
    ap.add_argument("--spacy_model", default="nl_core_news_sm",
                    help="spaCy Dutch model to use for lemmatization when --use_lemma_lookup is set.")
    ap.add_argument("--generate_stego", action="store_true",
                    help="If set: generate stego per mode from content. If not: read stego from stego_col (only mode ANY makes sense then).")

    ap.add_argument("--model_name", default="GroNLP/gpt2-small-dutch")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    modes = [normalize_pos_arg(m.strip()) for m in args.modes.split(",") if m.strip()]
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    # Load stego artifacts
    synsets: List[List[str]] = json.load(open(args.synsets, "r", encoding="utf-8"))
    idx_raw = json.load(open(args.index, "r", encoding="utf-8"))
    token_to_entry: Dict[str, Tuple[int, int]] = {k: (v[0], v[1]) for k, v in idx_raw.items()}
    synset_pos = load_synset_pos(args.synset_pos)

    # spaCy lemmatizer (optional)
    nlp = None
    if args.use_lemma_lookup:
        nlp = spacy.load(args.spacy_model)

    # Build payload bitstring
    payload = args.message.encode("utf-8")
    if args.omit_len_crc:
        # Payload-only mode (no header). Higher capacity/success, but no integrity check.
        bitstring = bytes_to_bits(payload)
        needed_bits = len(bitstring)
        crc = None
    else:
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        bitstring = u32_to_bits(len(payload)) + u32_to_bits(crc) + bytes_to_bits(payload)
        needed_bits = len(bitstring)
    # Load model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    out_fields = (
        id_cols
        + ["row_index", "mode", "header_mode", "needed_bits", "capacity_bits", "fits", "success", "substitutions",
           "ppl_content", "ppl_stego", "delta_ppl", "ratio_ppl"]
    )
    with open(args.in_csv, "r", encoding="utf-8", newline="") as fin, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header/columns.")

        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row_i, row in enumerate(reader):
            if row_i >= args.max_rows:
                break

            content = (row.get(args.text_col) or "").strip()
            if not content:
                continue

            lemma_map = build_lemma_map(content, nlp) if nlp is not None else None

            # PPL original once per article (reuse across modes)
            ppl_content = perplexity(
                content, model, tokenizer, device,
                max_length=args.max_length, stride=args.stride
            )

            for mode in modes:
                cap = row_capacity_bits(content, synsets, token_to_entry, synset_pos, mode, lemma_map=lemma_map)
                fits = 1 if cap >= needed_bits else 0

                stego_text = None
                subs = 0
                success = 0

                if args.generate_stego:
                    if fits:
                        er = embed_message_in_row(content, bitstring, synsets, token_to_entry, synset_pos, mode, lemma_map=lemma_map)
                        stego_text = er.stego
                        subs = er.substitutions
                        success = er.success
                else:
                    # Evaluate-only: just read stego from CSV
                    # (In this mode, you probably want modes="ANY")
                    stego_text = (row.get(args.stego_col) or "").strip()
                    success = 1 if stego_text else 0
                    subs = 0  # unknown unless you compute diff; left 0 by default

                ppl_stego = None
                delta = None
                ratio = None

                if stego_text:
                    ppl_stego = perplexity(
                        stego_text, model, tokenizer, device,
                        max_length=args.max_length, stride=args.stride
                    )
                    if ppl_content is not None and ppl_stego is not None:
                        delta = ppl_stego - ppl_content
                        ratio = ppl_stego / ppl_content if ppl_content != 0 else None

                out_row = {k: (row.get(k, "") or "") for k in id_cols}
                out_row.update({
                    "row_index": row_i,
                    "mode": mode,
                    "header_mode": ("payload_only" if args.omit_len_crc else "len_crc"),
                    "needed_bits": needed_bits,
                    "capacity_bits": cap,
                    "fits": fits,
                    "success": success,
                    "substitutions": subs,
                    "ppl_content": ppl_content,
                    "ppl_stego": ppl_stego,
                    "delta_ppl": delta,
                    "ratio_ppl": ratio,
                })
                writer.writerow(out_row)

    print(f"OK: wrote results to {args.out_csv}")
    print(f"Model: {args.model_name}  device={device}")
    if args.omit_len_crc:
        print(f"needed_bits={needed_bits} (payload-only; message={args.message!r})")
    else:
        print(f"needed_bits={needed_bits} (LEN+CRC+payload; message={args.message!r}, crc32={crc:08x})")

if __name__ == "__main__":
    main()