# hei20_articles

Script som bygger vektorindeksen **hvaerinnafor** fra artikkellisten i
`scraping/hvaerinnafor/startUrls.json`.

Koden er hentet fra `vectorindex_server`, men inneholder kun det som trengs for selve indekseringsjobben.

## Hva scriptet gjør

1. Crawler artiklene i `startUrls.json` med Apify (`apify/website-content-crawler`) og innstillingene i `config.json`.
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
| `--name` | `hvaerinnafor` | Mappe under `scraping/` med `config.json` og `startUrls.json` |
| `--storage` | `./blobstorage/chatbot/` | Hvor indeksene lagres |
| `--content-type` | `html_content` | `html_content` (ren tekst) eller `markdown_content` |

## Endre artikkellisten

Rediger `scraping/hvaerinnafor/startUrls.json`:

```json
[
  {"url": "https://www.ung.no/forelskelse/3220_Kyss.html", "method": "GET"}
]
```
