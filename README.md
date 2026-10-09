# hei20_article_indexer

Bygger vektorindeksene som HEI 2.0-chatboten (`hdir/hei20_chatbot_server`) bruker, fra artikkellisten i `startUrls.json`.

Koden er hentet fra `vectorindex_server_editor` (`utils/create_HEI_vectorindex`) og inneholder kun det som trengs for byggejobbene. Ingen Apify.

## Indekser

| Modus | Indeks | Innhold | Chatbot-modus |
|---|---|---|---|
| `legacy` | `hvaerinnafor` | Artiklene, delt i biter | Art |
| `legacy` | `hvaerinnafor_qa_bank` | Forhåndsgenererte spørsmål med svar, kort svar, kategori og alvorlighetsgrad | Relaterte spørsmål, `/examples` |
| `unified` | `hvaerinnafor_unified` | Hybrid: artikkelbiter + ekte ung.no-spørsmål/svar fra Azure SQL | Hyb |

Alt lagres under `./blobstorage/chatbot/` (kan endres med `--storage`).

### Slik bygges hybridindeksen (`--mode unified`)

1. **Artikkelbiter** (`node_type="article"`): artiklene deles i biter (3600 tegn), og LLM lager opptil 10 spørsmål per bit som legges i `aliases`. Spørsmålene caches i `hvaerinnafor_article_aliases.json` (nøkkel: SHA-256 av teksten), så uendrede biter ikke koster nye LLM-kall.
2. **Q&A-noder** (`node_type="qa"`): ung.no-spørsmål fra `dbo.qa` i Azure SQL for de siste `--months-back` månedene (kun rader med PSA/SSA-tema). Teksten er veilederens svar; spørsmålet ligger i `aliases`.
3. **Cross-pollination**: hvert ung.no-spørsmål søkes mot artikkelbitene, og legges som ekstra alias på de 3 beste treffene med likhet ≥ 0,55 (maks 50 aliases per bit).
4. Alt bygges som én indeks og sjekkes for konsistens før den lagres.

`aliases` er med i embeddingen, men holdes utenfor teksten LLM-en ser (`excluded_llm_metadata_keys`).

## Oppsett

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # fyll inn nøkler
```

`--mode unified` krever [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server) og `UNG_AZURESQL_ODBC` i `.env`.

## Kjøring

```bash
python build_index.py                              # legacy: hvaerinnafor + hvaerinnafor_qa_bank
python build_index.py --mode unified               # hvaerinnafor_unified
python build_index.py --mode both                  # alle tre
python build_index.py --mode both --from-scratch   # full ombygging
```

| Argument | Standard | Beskrivelse |
|---|---|---|
| `--mode` | `legacy` | `legacy`, `unified` eller `both` |
| `--months-back` | `12` | `unified`: hvor langt bakover ung.no-Q&A hentes |
| `--from-scratch` | – | Sletter indeksmappe(r) og manifester først. Alias-cachen beholdes; slett den manuelt for å generere nye aliases |
| `--skip-qa-bank` | – | `legacy`/`both`: bygg kun `hvaerinnafor` |
| `--start-urls` | `startUrls.json` | JSON-fil med artikkellisten |
| `--storage` | `./blobstorage/chatbot/` | Hvor indeksene lagres |

`legacy` er inkrementell: `hvaerinnafor_processed_docs.json` og `hvaerinnafor_qa_bank_processed_docs.json` holder styr på ferdige artikler, så en ny kjøring bare tar med nye. `unified` bygges alltid på nytt fra alle artikler.

Etter hentingen ligger `scraping/scraped_pages.json` og `scraping/documents.xlsx` under `--storage`, slik at du kan se hvilken tekst som ble hentet fra hver side.

## Filer

| Fil | Innhold |
|---|---|
| `build_index.py` | Startpunkt, valg av modus, inkrementell legacy-bygging |
| `scraping.py` | Henter sidene (ung.no, sexogsamfunn.no, helsenorge.no `selvtestConfig`) |
| `legacy_workflow.py` | LangGraph-workflows for `hvaerinnafor` og `hvaerinnafor_qa_bank` |
| `unified.py` | Bygging av `hvaerinnafor_unified` |
| `ungno_sql.py` | Tilkobling og spørring mot Azure SQL |
| `prompts.py` | Prompter |
| `psa_ssa_topics.json` | Kategorier brukt ved kategorisering av artikler |
| `startUrls.json` | Artikkellisten |
