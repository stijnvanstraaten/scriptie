# Ontwerp en evaluatie van een synoniem-gebaseerd tekststeganografiesysteem voor het Nederlands

Deze repository bevat de code die is gebruikt voor de bachelor scriptie *“Ontwerp en evaluatie van een synoniem-gebaseerd tekststeganografiesysteem voor het Nederlands”*.  
Het systeem verbergt geheime informatie in bestaande Nederlandstalige teksten door woorden te vervangen door synoniemen, die gerelateerd zijn aan een bitwaarde. De verborgen informatie kan vervolgens weer worden teruggelezen met behulp van vaste synset-indexering (bitwaarde) en foutcontrole met LEN en CRC.

De focus ligt op het analyseren van de invloed van woordsoorten (NOUN, VERB, ADJ en ANY) op capaciteit, succesratio en tekstkwaliteit (gemeten met perplexity).

Deze repository bevat geen ruwe tekstdata of lexica. Gebruikers worden geacht de benodigde datasets zelf te downloaden zoals hieronder beschreven.

De pipeline is getest op Windows (PowerShell). Alle voorbeeldcommando’s zijn geschreven voor PowerShell.

---

## Overzicht

Het systeem bestaat uit de volgende hoofdonderdelen:

- Coverteksten uit Project Gutenberg (Nederlands)
- Synoniemenbron: Open Dutch WordNet
- Preprocessing van coverteksten en synoniemen
- Encoding van bits in tekst via synoniemvervanging
- Decoding van bits met foutcontrole (LEN + CRC)
- Evaluatie van capaciteit, succesratio en perplexity

---

## Installatie

### Python

Aanbevolen: Python 3.10 of hoger (Windows).

Installeer afhankelijkheden via requirements.txt:

pip install -r requirements.txt

### spaCy model (Nederlands)

Voor POS-tagging en lemma-lookup:

python -m spacy download nl_core_news_sm

---

## Benodigde data

### Open Dutch WordNet

Als synoniemenbron wordt gebruikgemaakt van Open Dutch WordNet.

Download de synoniemenlijst van:

https://github.com/MartenPostma/OpenDutchWordnet

Plaats het TSV-bestand in de root van deze repository als:

synonyms.tsv

De ruwe synoniemenlijst bevat tienduizenden synsets en wordt in meerdere stappen gepreprocessed voordat deze bruikbaar is voor het systeem.

---

### Project Gutenberg (Nederlands)

Voor coverteksten wordt gebruikgemaakt van het Nederlandstalige deel van Project Gutenberg, via de HuggingFace dataset:

https://huggingface.co/datasets/ChocoLlama/gutenberg-dutch

De teksten worden automatisch gedownload en gepreprocessed via het script `gutenberg_to_csv.py`. Je hoeft deze bestanden niet handmatig te downloaden.

---

## Quickstart (Windows / PowerShell)

### 1. Downloaden en preprocessen van coverteksten

python gutenberg_to_csv.py --out_csv gutenberg_dutch.csv

Tijdens deze stap worden de Gutenberg-teksten opgeschoond, genormaliseerd en omgezet naar tekstsegmenten van 6000 karakters. Elk segment wordt opgeslagen als één rij in een CSV-bestand.

---

### 2. Preprocessen van synoniemen

De preprocessing van synoniemen bestaat uit drie stappen.

#### 2.1 Disjoint synsets

python preprocess_synonyms.py --in_tsv synonyms.tsv --out_dir out_syn

Deze stap normaliseert woorden, verwijdert onbruikbare tokens en zorgt ervoor dat synsets disjoint worden gemaakt.

Output:
- synsets.json
- token_to_entry.json

---

#### 2.2 POS-filtering (spaCy)

python preprocess_synonyms_pos.py --in_synsets out_syn/synsets.json --out_dir out_syn_pos --spacy_model nl_core_news_sm

Deze stap bepaalt automatisch de woordsoort (NOUN, VERB, ADJ) met spaCy en splitst synsets per woordsoort.

Output:
- synsets_pos.json
- token_to_entry_pos.json
- synset_pos.json

---

#### 2.3 Sanitizing + macht-van-twee

python preprocess_synonyms_pos_sanitized.py --in_dir out_syn_pos --out_dir out_syn_pos_sanitized

Deze stap:
- verwijdert duplicaten,
- lost ambiguïteit op (elk woord in maximaal één synset),
- truncate synsets tot groottes die een macht van twee zijn.

Output:
- synsets_pos.sanitized.json
- token_to_entry_pos.sanitized.json
- synset_pos.sanitized.json

---

### 3. Encoding (cover → stego)

python encode_lemma_lookup.py `
  --in_csv gutenberg_dutch.csv `
  --out_csv gutenberg_dutch_stego.csv `
  --synsets out_syn_pos_sanitized/synsets_pos.sanitized.json `
  --index out_syn_pos_sanitized/token_to_entry_pos.sanitized.json `
  --synset_pos out_syn_pos_sanitized/synset_pos.sanitized.json `
  --pos ANY `
  --message "test" `
  --use_lemma_lookup `
  --max_rows 150 `
  --spacy_model nl_core_news_sm

Tijdens encoding worden woorden in de covertekst vervangen door synoniemen op basis van hun positie binnen een synset. De te embedden bitstream bestaat uit:

LEN ∥ CRC32 ∥ PAYLOAD

Als de beschikbare capaciteit onvoldoende is, faalt het encoderen.

---

### 4. Decoding (stego → bits → bericht)

python decode.py `
  --in_csv gutenberg_dutch_stego.csv `
  --synsets out_syn_pos_sanitized/synsets_pos.sanitized.json `
  --index out_syn_pos_sanitized/token_to_entry_pos.sanitized.json `
  --synset_pos out_syn_pos_sanitized/synset_pos.sanitized.json `
  --pos ANY `
  --max_rows 150 `
  --print_utf8

Tijdens decoding worden de gebruikte synoniemen terugvertaald naar bitwaarden. Het bericht wordt alleen geaccepteerd wanneer de CRC-controle succesvol is.

---

### 5. Evaluatie (capaciteit en perplexity)

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

Deze stap berekent per document:

- beschikbare capaciteit
- succesratio van embedding
- verandering in perplexity (origineel vs stego)

---

## Pipeline

De volledige pipeline bestaat uit de volgende stappen:

1. Downloaden en preprocessen van coverteksten uit de Gutenberg dataset.  
2. Preprocessen en opschonen van de synoniemenlijst (inclusief POS-filtering en sanitizing).  
3. Identificatie van vervangbare woorden in de covertekst op basis van synsets.  
4. Berekenen van de beschikbare capaciteit per document.  
5. Maken van de te embedden bitstream (inclusief LEN en CRC).  
6. Vervanging van woorden door synoniemen (encoding).  
7. Reconstructie van de bitstream en foutcontrole (decoding).  
8. Evaluatie van tekstkwaliteit met behulp van perplexity en statistische berekeningen.  

Alle experimenten worden uitgevoerd per document en per woordsoortmodus (NOUN, VERB, ADJ en ANY).

---

## Belangrijke ontwerpkeuzes

- Synsets worden getrunceerd tot groottes die een macht van twee zijn.
- Elk woord mag in maximaal één synset voorkomen (ambiguïteit wordt verwijderd).
- Er wordt gebruikgemaakt van LEN + CRC voor foutdetectie.
- Er wordt geen morfologische aanpassing toegepast; woorden worden vervangen in lemma-vorm.
- spaCy wordt gebruikt voor POS-tagging en lemma-lookup.

Deze keuzes zorgen voor stabiele en voorspelbare decoding, maar verlagen de maximale capaciteit.

---

## Bestanden en directories

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
- synonyms.tsv
- gutenberg_dutch.csv
- gutenberg_dutch_stego.csv
- out_syn/
- out_syn_pos/
- out_syn_pos_sanitized/
- results_perplexity_lang.csv

---

## Licentie en data

### Open Dutch WordNet

Dit project maakt gebruik van Open Dutch WordNet, gelicenseerd onder Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Afgeleide synsetbestanden vallen onder dezelfde licentie.

Bron:
https://github.com/MartenPostma/OpenDutchWordnet

---

### Project Gutenberg

Dit project maakt gebruik van publieke domein teksten van Project Gutenberg, via HuggingFace. Deze repository distribueert geen volledige teksten. Gebruikers downloaden de data zelf via de scripts.

Bronnen:
https://huggingface.co/datasets/ChocoLlama/gutenberg-dutch  
https://www.gutenberg.org/
