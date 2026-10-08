# hei20_article_indexer

Script som bygger vektorindeksen **hvaerinnafor** fra artikkellisten i
`startUrls.json`.

Koden er hentet fra `vectorindex_server` og `digiung_lab` (artikkelhenting), men inneholder kun det som trengs for selve indekseringsjobben. Ingen Apify.

## Hva scriptet gjør

1. Henter hver artikkel i `startUrls.json` med `requests` og trekker ut teksten med BeautifulSoup. Menyer, cookie-bannere og chat-widgets fjernes. Duplikate URL-er hoppes over, og artikler som feiler logges og hoppes over.
2. Lager metadata per artikkel med LLM: spørsmål artikkelen besvarer og kategori.
3. Bygger og lagrer vektorindeksen for artiklene → `<storage>/hvaerinnafor/`.
4. Fjerner nesten like spørsmål, svarer på hvert spørsmål med kildebelagt svar og setter alvorlighetsgrad.
5. Bygger og lagrer en Q&A-indeks → `<storage>/hvaerinnafor_qa_bank/`.
6. Skriver loggen `combined_documents_for_hvaerinnafor.xlsx`.

## Oppsett

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # fyll inn nøkler
```

## Kjøring

```bash
python build_index.py
```

Valgfrie argumenter:

| Argument | Standard | Beskrivelse |
|---|---|---|
| `--name` | `hvaerinnafor` | Navn på indeksen |
| `--start-urls` | `startUrls.json` | JSON-fil med artikkellisten |
| `--storage` | `./blobstorage/chatbot/` | Hvor indeksene lagres |

## Endre artikkellisten

Rediger `startUrls.json`:

```json
[
  {"url": "https://www.ung.no/forelskelse/3220_Kyss.html", "method": "GET"}
]
```
