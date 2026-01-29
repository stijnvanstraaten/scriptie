# Scriptie – Synoniem-gebaseerde tekststeganografie (NL)

Deze repository bevat de code en artefacten die zijn gebruikt voor mijn scriptie over synoniem-gebaseerde tekststeganografie in het Nederlands. Het systeem verbergt een bericht in een covertekst door woorden te vervangen door synoniemen uit een vooraf verwerkte synoniemenbron, en kan het bericht vervolgens weer decoderen met behulp van dezelfde synsets en indexering.

De repository bevat GEEN ruwe tekstdata of lexica. Gebruikers worden geacht de benodigde datasets zelf te downloaden via de hieronder beschreven stappen.

Deze pipeline is getest op Windows (PowerShell). Alle voorbeeldcommando’s zijn geschreven voor PowerShell.

---

## Overzicht (high level)

- Data: Nederlandse coverteksten uit Project Gutenberg (via HuggingFace)
- Synoniemenbron: Open Dutch WordNet (ODWN)
- Preprocessing: synsets bouwen, POS-filtering en sanitizing
- Encoding: encode_lemma_lookup.py (met optionele lemma-lookup via spaCy)
- Decoding: decode.py (met LEN + CRC32)
- Evaluatie: evaluate_ppl_capacity_no_len_crc_lemma_patched.py (capaciteit + perplexity)
- Analyse: analyze_results.py

---

## Installatie

### Python

Aanbevolen: Python 3.10+ (Windows).

Installeer afhankelijkheden:

pip install pandas datasets spacy torch transformers

### spaCy model (Nederlands)

Voor POS-filtering en lemma-lookup:

python -m spacy download nl_core_news_sm

---

## Benodigde data (handmatig downloaden)

### Open Dutch WordNet (ODWN)

Download Open Dutch WordNet van:

https://github.com/MartenPostma/OpenDutchWordnet

Download het relevante TSV-bestand en plaats het in de root van deze repository als:

synonyms.tsv

Let op: dit bestand wordt niet meegeleverd in deze repo vanwege licentie- en distributieoverwegingen.

---

### Project Gutenberg (Nederlands)

Coverteksten worden automatisch gedownload via HuggingFace door:

https://huggingface.co/datasets/ChocoLlama/gutenberg-dutch

Je hoeft hiervoor geen bestanden handmatig te downloaden; dit gebeurt automatisch via gutenberg_to_csv.py.

---

## Quickstart (Windows / PowerShell)

### 1. Coverteksten maken

python gutenberg_to_csv.py --out_csv gutenberg_dutch.csv

---

### 2. Synoniemen preprocessen (3 stappen)

#### 2.1 Disjoint synsets

python preprocess_synonyms.py --in_tsv synonyms.tsv --out_dir out_syn

#### 2.2 POS-filtering (spaCy)

python preprocess_synonyms_pos.py --in_synsets out_syn/synsets.json --out_dir out_syn_pos --spacy_model nl_core_news_sm

#### 2.3 Sanitizing + macht-van-twee

python preprocess_synonyms_pos_sanitized.py --in_dir out_syn_pos --out_dir out_syn_pos_sanitized

---

### 3. Encoding (maak stegotekst)

python encode_lemma_lookup.py `
  --in_csv gutenberg_dutch.csv `
  --out_csv gutenberg_dutch_stego.csv `
  --synsets out_syn_pos_sanitized/synsets_pos.sanitized.json `
  --index out_syn_pos_sanitized/token_to_entry_pos.sanitized.json `
  --synset_pos out_syn_pos_sanitized/synset_pos.sanitized.json `
  --pos ANY `
  --message "test" `
  --use_lemma_lookup `
  --spacy_model nl_core_news_sm

---

### 4. Decoding (lees bericht terug)

python decode.py `
  --in_csv gutenberg_dutch_stego.csv `
  --synsets out_syn_pos_sanitized/synsets_pos.sanitized.json `
  --index out_syn_pos_sanitized/token_to_entry_pos.sanitized.json `
  --synset_pos out_syn_pos_sanitized/synset_pos.sanitized.json `
  --pos ANY `
  --print_utf8

---

### 5. Evaluatie (capaciteit + perplexity)

python evaluate_ppl_capacity_no_len_crc_lemma_patched.py `
  --in_csv gutenberg_dutch.csv `
  --out_csv results_perplexity_lang.csv `
  --synsets out_syn_pos_sanitized/synsets_pos.sanitized.json `
  --index out_syn_pos_sanitized/token_to_entry_pos.sanitized.json `
  --synset_pos out_syn_pos_sanitized/synset_pos.sanitized.json `
  --modes NOUN,VERB,ADJ,ANY `
  --max_rows 150 `
  --message "test" `
  --generate_stego `
  --use_lemma_lookup `
  --spacy_model nl_core_news_sm

---

## Pipeline in detail

### A – Coverteksten (gutenberg_to_csv.py)

Doel:  
Download en verwerk Nederlandse Gutenberg-teksten naar een CSV met coverteksten.

Output:
- gutenberg_dutch.csv
  - article_id
  - content

Wat gebeurt er:
- opschonen van Gutenberg boilerplate
- normalisatie van whitespace
- knippen naar een maximum lengte

---

### B – Synsets preprocessen

#### B1 – Disjoint synsets (preprocess_synonyms.py)

- normaliseren van tokens
- single-word filter
- disjointization: elk token komt in max. één synset

Output (out_syn/):
- synsets.json
- token_to_entry.json

---

#### B2 – POS-filtering (preprocess_synonyms_pos.py)

- spaCy POS-tagging
- verwijderen van tokens met verkeerde woordsoort
- verwijderen van synsets met <2 tokens

Output (out_syn_pos/):
- synsets_pos.json
- token_to_entry_pos.json
- synset_pos.json

---

#### B3 – Sanitizing + macht-van-twee (preprocess_synonyms_pos_sanitized.py)

- dedupe en filtering op woord-tokens
- oplossen van conflicten (token in meerdere synsets)
- truncate naar grootte 2^b

Output (out_syn_pos_sanitized/):
- synsets_pos.sanitized.json
- token_to_entry_pos.sanitized.json
- synset_pos.sanitized.json

---

### C – Encoding (encode_lemma_lookup.py)

- tokenisatie van covertekst
- lookup op surface en (optioneel) lemma via spaCy
- per synset wordt b = floor(log2(|synset|)) bits geschreven
- header met LEN + CRC32
- vervanging van woorden op basis van synset-index

Output:
- CSV met stego kolom

---

### D – Decoding (decode.py)

- tokenisatie van stegotekst
- lookup per woord → synset-id + index
- reconstructie van bits
- validatie met LEN + CRC32

Output:
- gedecodeerde payload (bij correcte CRC)

---

### E – Evaluatie (evaluate_ppl_capacity_no_len_crc_lemma_patched.py)

- berekent capaciteit per tekst
- controleert of payload past
- genereert stego in-memory
- meet perplexity (origineel vs stego)

Output:
- results_perplexity_lang.csv

---

## Belangrijke opties

### POS-modus

- NOUN, VERB, ADJ, ANY  
- bepaalt welke synsets gebruikt mogen worden

### Lemma lookup (spaCy)

- --use_lemma_lookup  
- koppelt vervoegde woordvormen in de tekst aan lemma-gebaseerde synsets

---

## Bestanden & directories (globaal)

Scripts:
- gutenberg_to_csv.py
- preprocess_synonyms.py
- preprocess_synonyms_pos.py
- preprocess_synonyms_pos_sanitized.py
- encode_lemma_lookup.py
- decode.py
- evaluate_ppl_capacity_no_len_crc_lemma_patched.py
- analyze_results.py

Gegenereerde data (niet in repo):
- synonyms.tsv (handmatig downloaden)
- out_syn/
- out_syn_pos/
- out_syn_pos_sanitized/
- gutenberg_dutch.csv
- gutenberg_dutch_stego.csv
- results_perplexity_lang.csv

---

## Troubleshooting

spaCy model niet gevonden:
python -m spacy download nl_core_news_sm

Decode vindt geen geldig bericht:
- dezelfde POS-modus bij encode en decode  
- dezelfde sanitized synsets/index  
- voldoende capaciteit in de covertekst  
- tekst niet aangepast na encoding (CRC faalt anders)

Lemma lookup lijkt geen effect te hebben:
- gebruik --use_lemma_lookup
- zorg dat spaCy correct is geïnstalleerd
- lemma-lookup helpt vooral bij vervoegde vormen

---

## Reproduceerbaarheid

Voor reproduceerbare resultaten:
- gebruik dezelfde synsets (out_syn_pos_sanitized/)
- gebruik dezelfde POS-modus
- fixeer --max_rows
- gebruik dezelfde payloadgrootte

---

## Licentie en data

### Open Dutch WordNet

This project uses data from Open Dutch WordNet (Postma et al., 2016), licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

Original source:
https://github.com/MartenPostma/OpenDutchWordnet

Any derived synset files generated by this pipeline must also be distributed under CC BY-SA 4.0.

---

### Project Gutenberg

This project uses Dutch public-domain texts from Project Gutenberg, accessed via the HuggingFace dataset ChocoLlama/gutenberg-dutch.

The original texts are in the public domain. However, Project Gutenberg distributes these texts under the Project Gutenberg License. This repository does not redistribute the full texts; users are expected to download the data themselves via the provided scripts.

Sources:
https://huggingface.co/datasets/ChocoLlama/gutenberg-dutch  
https://www.gutenberg.org/
