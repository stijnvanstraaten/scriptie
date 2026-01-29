#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gutenberg_to_csv.py

Doel:
- Laadt HuggingFace dataset: ChocoLlama/gutenberg-dutch
- Schrijft CSV met kolommen: article_id, content, stego
- content:
  - geen newlines (alles in 1 regel)
  - boilerplate/frontmatter zoveel mogelijk weggefilterd
  - netjes afkappen rond max_chars (niet midden in een woord; liefst op paragraaf-einde)
  - mag iets boven max_chars gaan om netjes af te ronden (soft_overflow)

PowerShell usage (LET OP: PowerShell gebruikt ` (backtick) voor line continuation, niet \ ):
  python .\gutenberg_to_csv.py `
    --out_csv gutenberg_dutch.csv `
    --max_chars 6000 `
    --soft_overflow 300 `
    --min_chars 500

CMD usage (1 regel):
  python gutenberg_to_csv.py --out_csv gutenberg_dutch.csv --max_chars 6000 --soft_overflow 300 --min_chars 500

Extra:
  --limit N      : alleen eerste N voorbeelden per split (handig testen)
  --progress K   : print elke K items status

Opmerking:
- Dit is allemaal heuristisch. Gutenberg-teksten zijn rommelig; 100% perfecte boilerplate removal bestaat niet.
"""

import argparse
import re
from datasets import load_dataset
import pandas as pd

PARA_TOKEN = " <P> "  # tijdelijke paragraafmarker, later vervangen door spatie


# --- Gutenberg START/END markers (best-effort) ---
GB_START_PATTERNS = [
    r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
]
GB_END_PATTERNS = [
    r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
]


# --- Frontmatter / productie-regels die vaak weg kunnen ---
FRONTMATTER_PATTERNS = [
    r"^produced by\b.*",
    r"^online distributed proofreading\b.*",
    r"\bproject gutenberg\b",
    r"\bpgdp\b",
    r"http[s]?://\S+",
    r"www\.\S+",
    r"\be-book\b",
    r"\bebook\b",
]

# --- TOC / index achtige regels (heuristiek) ---
TOC_LINE_PATTERNS = [
    r"^\s*(inhoud|inhoudsopgave|contents)\s*$",
    r"^\s*(blz\.|bldz\.|bladz\.|bladzijde|pagina)\s*$",
    r"^\s*deel\s+[ivxlcdm]+\b.*\b(blz\.|bldz\.|bladz\.)\b.*$",
    r"^\s*[ivxlcdm]+\.\s+.+\s+\d+\s*$",
    r"^\s*hoofdstuk\s+\w+.*\s+\d+\s*$",
]

INLINE_TOC_SEGMENT_RE = re.compile(
    r"(?is)\bdeel\s+[ivxlcdm]+\s+bladz?\.?\b.*?(?=(\bdeel\s+[ivxlcdm]+\s+bladz?\.?\b)|$)"
)


def extract_text_field(ex: dict) -> str | None:
    """
    Dataset kan verschillende sleutel-namen hebben. We proberen een paar logische.
    """
    for key in ["text", "content", "book", "body"]:
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            return ex[key]
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    """
    Knip tussen START/END markers als ze aanwezig zijn.
    """
    text = raw
    for pat in GB_START_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[m.end():]
            break
    for pat in GB_END_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[:m.start()]
            break
    return text


def strip_ascii_box_blocks(text: str) -> str:
    """
    SNEL: verwijder ASCII-‘dozen’ zoals:

    +------------------+
    | tekst ...        |
    +------------------+

    Dit haalt o.a. veel van die Gutenberg-e-book disclaimers weg (zoals jouw voorbeelden).
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")

    border_re = re.compile(r"^\s*\+[=\-]{5,}\+\s*$")
    pipe_re = re.compile(r"^\s*\|.*\|\s*$")

    out = []
    i = 0
    n = len(lines)

    while i < n:
        if border_re.match(lines[i]):
            j = i + 1
            saw_pipe = False
            while j < n and pipe_re.match(lines[j]):
                saw_pipe = True
                j += 1
            if saw_pipe and j < n and border_re.match(lines[j]):
                # volledige box: skip
                i = j + 1
                continue
            # niet echt een box -> behoud regel
            out.append(lines[i])
            i += 1
        else:
            out.append(lines[i])
            i += 1

    return "\n".join(out)


def strip_inline_markers_and_toc(text: str) -> str:
    """
    Verwijdert o.a. [Illustratie: ...], en TOC-achtige blokken/segmenten.
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # bracketed illustraties e.d.
    t = re.sub(r"\[\s*Illustratie\s*:[^\]]*\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\s*Decoratieve\s+illustratie[^\]]*\]", " ", t, flags=re.IGNORECASE)

    # expliciete editorial-notices (regel-gebaseerd)
    t = re.sub(r"(?im)^\s*de tekst in dit bestand.*$", "", t)
    t = re.sub(r"(?im)^\s*bladzijde-?nummering.*$", "", t)
    t = re.sub(r"(?im)^\s*overduidelijke druk- en spelfouten.*$", "", t)
    t = re.sub(r"(?im)^\s*variaties in spelling.*$", "", t)
    t = re.sub(r"(?im)^\s*voetnoten.*$", "", t)
    t = re.sub(r"(?im)^\s*afgebroken woorden.*$", "", t)

    # inline TOC segment zoals: "Deel I Bladz. ... Deel II Bladz. ..."
    t = re.sub(INLINE_TOC_SEGMENT_RE, " ", t)

    # line-based TOC verwijderen: runs van TOC-ish regels skippen
    lines = t.split("\n")
    cleaned = []
    toc_run = 0

    for ln in lines:
        s = ln.strip()
        low = s.lower()

        is_toc = any(re.search(p, low) for p in TOC_LINE_PATTERNS)

        # extra heuristiek: korte regel eindigend op paginanummer (maar geen echte zin)
        if not is_toc and re.match(r"^.{0,80}\s+\d{1,4}\s*$", s) and not re.search(r"[\.!?]$", s):
            is_toc = True

        if is_toc:
            toc_run += 1
            if toc_run >= 2:
                continue
            cleaned.append(ln)
        else:
            toc_run = 0
            cleaned.append(ln)

    return "\n".join(cleaned)


def strip_frontmatter_lines(text: str) -> str:
    """
    Verwijder losse frontmatter/metadata regels (Produced by, URLs, all-caps titels, etc.)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned = []
    for ln in lines:
        ln_strip = ln.strip()
        if not ln_strip:
            cleaned.append("")
            continue

        lower = ln_strip.lower()
        drop = False
        for pat in FRONTMATTER_PATTERNS:
            if re.search(pat, lower):
                drop = True
                break
        if drop:
            continue

        # veel ALL CAPS korte regels zijn vaak titel/druk-info
        if ln_strip.isupper() and len(ln_strip) <= 160:
            continue

        # regels met alleen caps/cijfers/punctuatie => vaak metadata
        if re.match(r"^[A-ZÀ-ÖØ-Ý0-9][A-ZÀ-ÖØ-Ý0-9\s\-\.\,\:\;\/]{10,}$", ln_strip) and not re.search(r"[a-zà-ÿ]", ln_strip):
            continue

        cleaned.append(ln_strip)

    return "\n".join(cleaned)


def trim_to_first_real_paragraph(text: str) -> str:
    """
    Vaak staat er nog titel/voorwoordachtige rommel. We proberen te starten bij de eerste 'echte' paragraaf.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n+", text)

    for i, p in enumerate(paras):
        p_stripped = p.strip()
        if len(p_stripped) < 250:
            continue
        if not re.search(r"[a-zà-ÿ]", p_stripped):
            continue
        upper_ratio = sum(c.isupper() for c in p_stripped) / max(1, len(p_stripped))
        if upper_ratio > 0.5:
            continue
        return "\n\n".join(paras[i:]).strip()

    return text.strip()


def normalize_keep_paragraphs(text: str) -> str:
    """
    Normaliseer whitespace, maar bewaar paragraafscheidingen als PARA_TOKEN.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    lines = [re.sub(r"[ \u00A0]+", " ", ln).strip() for ln in text.split("\n")]

    out = []
    empty_seen = False
    for ln in lines:
        if ln == "":
            empty_seen = True
            continue
        if empty_seen and out:
            out.append(PARA_TOKEN)
        empty_seen = False
        out.append(ln)

    joined = " ".join(out)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def smart_truncate_to_paragraph(text: str, max_chars: int, soft_overflow: int) -> str:
    """
    Knip netjes:
      1) liefst op paragraaf-einde (PARA_TOKEN)
      2) anders op zins-einde
      3) anders op woordgrens
    Mag iets over max_chars om netjes af te ronden.
    """
    if len(text) <= max_chars:
        return text.replace(PARA_TOKEN, " ").strip()

    hard_limit = max_chars + max(0, soft_overflow)
    candidate = text[: min(len(text), hard_limit)]

    # 1) paragraafgrens
    p_idx = candidate.rfind(PARA_TOKEN)
    if p_idx != -1 and p_idx >= int(0.5 * max_chars):
        out = candidate[:p_idx].strip()
        return out.replace(PARA_TOKEN, " ").strip()

    # 2) zinsgrens in de laatste ~800 chars
    tail_start = max(0, len(candidate) - 800)
    tail = candidate[tail_start:]
    matches = list(re.finditer(r"[\.!?](?:\"|\'|\)|\]|\s|$)", tail))
    if matches:
        end_pos = tail_start + matches[-1].end()
        out = candidate[:end_pos].strip()
        if len(out) >= int(0.6 * max_chars):
            return out.replace(PARA_TOKEN, " ").strip()

    # 3) woordgrens
    space_idx = candidate.rfind(" ")
    if space_idx != -1:
        candidate = candidate[:space_idx].strip()

    return candidate.replace(PARA_TOKEN, " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=str, default="gutenberg_dutch.csv", help="Output CSV file")
    parser.add_argument("--max_chars", type=int, default=6000, help="Soft target length (characters)")
    parser.add_argument("--soft_overflow", type=int, default=300, help="Allowed overflow for neat ending")
    parser.add_argument("--min_chars", type=int, default=500, help="Skip texts shorter than this after cleaning")
    parser.add_argument("--limit", type=int, default=0, help="If >0, process only first N examples per split")
    parser.add_argument("--progress", type=int, default=200, help="Print progress every K processed examples per split")
    args = parser.parse_args()

    print("Loading dataset: ChocoLlama/gutenberg-dutch ...")
    ds = load_dataset("ChocoLlama/gutenberg-dutch")

    rows = []
    article_counter = 0

    for split in ds.keys():
        print(f"Processing split: {split}")
        processed = 0
        kept = 0

        for ex in ds[split]:
            if args.limit and processed >= args.limit:
                break

            raw = extract_text_field(ex)
            processed += 1

            if args.progress and processed % args.progress == 0:
                print(f"  {split}: processed={processed}, kept={kept}, total_out={article_counter}")

            if raw is None:
                continue

            # pipeline
            raw = strip_gutenberg_boilerplate(raw)
            raw = strip_ascii_box_blocks(raw)
            raw = strip_inline_markers_and_toc(raw)
            raw = strip_frontmatter_lines(raw)
            raw = trim_to_first_real_paragraph(raw)

            norm = normalize_keep_paragraphs(raw)
            final = smart_truncate_to_paragraph(norm, args.max_chars, args.soft_overflow)
            final = re.sub(r"\s+", " ", final).strip()  # single line

            if len(final) < args.min_chars:
                continue

            rows.append({"article_id": f"gutenberg_{article_counter}", "content": final, "stego": ""})
            article_counter += 1
            kept += 1

        print(f"  {split} done: processed={processed}, kept={kept}")

    df = pd.DataFrame(rows, columns=["article_id", "content", "stego"])
    print(f"Total articles written: {len(df)}")
    print(f"Writing CSV to: {args.out_csv}")
    df.to_csv(args.out_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
