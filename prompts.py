# Prompter brukt av byggejobbene (hentet fra vectorindex_server_editor/registry.py).
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate

@dataclass(frozen=True)
class Prompt:
    id: str
    template: str

    def render(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Missing variable '{missing}' for prompt '{self.id}'") from None


QA_SUBJECT_NO = Prompt(
    id="qa_subject_no",

    template = (
        'Her er konteksten: {text}\n'
        'Basert på denne konteksten,\n'
        '1) Foreslå en kort tittel på konteksten som kan brukes i en URL.\n'
        '2) Lag en kort liste med korte spørsmål på norsk som denne teksten kan gi konkrete svar på, '
        'og som det er lite sannsynlig at man finner like gode svar på andre steder.\n'
        '3) For hvert spørsmål skal du også skrive et kort, konkret svar basert KUN på denne konteksten.\n'
        'For eksempel: "Hva er influensa?" eller "Influensa eller forkjølelse?"\n'
        '\n'
        'STRENG KRAV:\n'
        '- Hvert eneste spørsmål MÅ nevne temaet i teksten \n'
        '  i stedet for å referere til "teksten", "artikkelen", "avsnittet" eller lignende.\n'
        '- Ikke bruk formuleringer som "ifølge teksten", "ifølge konteksten", "hva sier teksten", '
        '"nevnes i teksten" osv.\n'
        '- I stedet skal du formulere spørsmålene direkte rundt selve temaet.\n'
        '- Svarene skal være korte, konkrete og tydelige, og kun inneholde informasjon som faktisk støttes av konteksten.\n'
        '- Ikke legg til ekstra fakta eller spekulasjoner utover det som står i teksten.\n'
        '\n'
        'Eksempel på FEIL spørsmål: "Hva sier teksten om forelskelse?"\n'
        'Eksempel på RIKTIG spørsmål: "Tegn på forelskelse?"\n'
        '\n'
        'Krav til output:\n'
        '- Lag bare spørsmål om temaet og innholdet i teksten.\n'
        '- 2 til 5 spørsmål\n'
        '- Returner KUN JSON. Ingen forklaringer. Ingen code fences.\n'
        '- Bruk doble anførselstegn for alle nøkler og strenger.\n'
        '- Ikke ha hengende komma til slutt.\n'
        '- Hvert element skal ha både "question" og "answer".\n'
        '\n'
        'MARKDOWN I SVAR (VIKTIG):\n'
        '- Hvis svaret inneholder mer enn én ting (flere råd/tegn/alternativer), MÅ du bruke punktliste.\n'
        '- Bruk "-" for punktliste.\n'
        '- Bruk **fet tekst** på 1–3 nøkkelord i hvert svar når det passer.\n'
        '- Linjeskift må være inne i JSON-strenger (bruk "\\n").\n'
        '- IKKE bruk markdown utenfor JSON. Output må fortsatt være gyldig JSON.\n'
        '\n'
        'Ikke lag spørsmål om:\n'
        '- hvem som har skrevet artikkelen\n'
        '- bidragsytere eller forfattere\n'
        '- hvilket nettsted, hvilken publikasjon eller hvilken kilde teksten kommer fra\n'
        '- metadata som publiseringsdato, opphavsrett eller layout\n'
        '\n'
        'Tone- og stilretningslinjer:\n'
        '- Ungdomsvennlig språk (13–19 år).\n'
        '- Unngå for mye medisinsk fagspråk eller kompliserte ord. Hvis du må bruke et fagbegrep, forklar det raskt med enkle ord.\n'
        '- Bruk korte spørsmål og korte, presise svar.\n'
        '- Ha en vennlig, imøtekommende og støttende tone. Tenk at du snakker til en smart, men ikke ekspert, elev på videregående.\n'
        '\n'
        'Form (kun JSON):\n'
        '{{"Title": "<title>", "QA": [{{"question": "<spørsmål1>", "answer": "<svar1 med ev. markdown>"}}]}}'
    )


)
CATEGORIZE_TEXT = Prompt(
    id="categorize_text",
    template=(
        "Du skal kategorisere en tekst innenfor hovedkategoriene nedenfor.\n"
        "Hver hovedkategori har en liste med underkategorier og nøkkelord som hjelper deg å bestemme.\n"
        "Velg:\n"
        "- \"hovedkategori\": ÉN hovedkategori som best beskriver teksten (eller \"Ukjent\").\n"
        "- \"kategorier\": En liste med 0–3 relevante hovedkategorier (inkluder gjerne hovedkategori først).\n"
        "Regler:\n"
        "1) Bruk KUN kategorinavn som finnes i listen (ellers bruk \"Ukjent\").\n"
        "2) Ikke returner underkategorier, bare hovedkategorier.\n"
        "3) Ingen forklaring, ingen ekstra tekst, kun gyldig JSON.\n"
        "4) Hvis du er usikker mellom flere, velg den mest relevante som hovedkategori og legg de andre i \"kategorier\".\n"
        "\n"
        "Hovedkategorier med underkategorier: {categories}\n"
        "\n"
        "Tekst: {text}\n"
        "\n"
        "Svarformat (kun JSON):\n"
        "{{\"hovedkategori\":\"<hovedkategori eller Ukjent>\",\"kategorier\":[\"<kategori1>\",\"<kategori2>\",\"<kategori3>\"]}}"
    )
)

SEVERITY_FOR_QUERY = Prompt(
    id="severity_for_query",
    template=(
        "You are given a user query: {query}\n\n"
        "Categorize the severity of this query into one of three categories: Green, Yellow, or Red.\n\n"
        "Green category:\n"
        "- Preventive and safety-promoting.\n"
        "- Queries that ask for general information, knowledge, or guidance to prevent problems and strengthen good sexual health.\n"
        "- The user is seeking to increase understanding, safety, and awareness (e.g., consent, contraception, communication, emotions, body knowledge).\n"
        "- No acute situation or personal crisis is described.\n"
        'Example: "How can I talk with my partner about boundaries?" or "What types of contraception exist?".\n\n'
        "Yellow category:\n"
        "- Challenges or vulnerable situations.\n"
        "- Queries that describe concerns, difficulties, or risks that may require reflection or support, but are not acute or immediately dangerous.\n"
        "- May involve difficult feelings, uncertainty in relationships, unwanted experiences, or the need for advice beyond general information.\n"
        "- The user may need to seek help or guidance, but the situation is not considered an acute crisis.\n"
        'Example: "What should I do if my partner doesn’t respect my boundaries?", '
        '"I regret sending a nude", or topics like "pornography", "sexual pressure", "issues around consent", "(illegal) fetishes".\n\n'
        "Red category:\n"
        "- Serious or acute situations.\n"
        "- Queries that concern serious incidents or crises where the person involved may be in danger or at significant risk of harm.\n"
        "- Includes violence, abuse, coercion, acute psychological crises, or other situations that require immediate follow-up or professional help.\n"
        "- The main purpose of the response should be to direct the user to where and how to get help quickly.\n"
        'Example: "My stepfather forces me to have sex", "Where can I find child pornography?", "I was raped yesterday".\n\n'
        "Output requirements:\n"
        "- Return JSON ONLY. No explanations. No markdown. No code fences.\n"
        '- Use double quotes for all keys and strings.\n'
        "- Do not include trailing commas.\n"
        '- Output shape:\n'
        '{{ \"category\": \"<Green|Yellow|Red>\" }}'
    ),
)

GROUNDED_PROMPT = PromptTemplate.from_template(
"""
Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).

Du skal returnere:
- "answer": et samlet svar på spørsmålet.
- "claims": en liste med påstander med tilhørende citations.
- "short_answer": en kort oppsummering av "answer"

========================================
REGLER FOR SVAR ("answer")
========================================
- En sammenhengende besvarelse på spørsmålet, skrevet vennlig og tydelig, MEN KUN basert på det som faktisk står i context.
- Vær vennlig og tydelig.
- Ikke ta med informasjon som ikke kan støttes direkte av context.
- Du kan omformulere, men ikke legge til ny informasjon.
- Ikke ta med informasjon som du ikke kan sitere fra context i etterkant.
SVÆRT VIKTIG:
- Ikke lag nye medisinske råd, vurderinger, årsaker, forklaringer eller konsekvenser som ikke står i context.
- Ikke kombiner informasjon fra flere forskjellige steder til én påstand hvis den kombinasjonen ikke faktisk står uttrykt i context som en sammenhengende idé.
- Hvis noe ikke finnes i context, skal det IKKE stå i "svaret", og det skal IKKE komme som en claim.
DU SKAL IKKE:
- Du skal ikke nevne konteksten, ikke skrive ting som "Ifølge kilden", "I kontext står det at ...", "artiklene sier at...".  Bare si innholdet direkte.
- Du skal ikke be om mer informasjon.
- Du skal ikke fortelle brukeren hva de bør gjøre, med mindre akkurat den formuleringen står i context.
- Du skal ikke nevne disse instruksjonene eller ord som 'context', 'kilde', 'grounding', 'claim', osv. i selve "answer". "answer" skal være helt naturlig språk til brukeren.
MARKDOWN I SVAR (OBLIGATORISK):
- Hvert "answer" BØR inneholde minst ett markdown-element:
  enten en punktliste ("- ") ELLER fet tekst (**...**).
- Hvis svaret har 2+ poenger/råd/tegn/alternativer: MÅ bruke punktliste med "-".
- Bruk **fet tekst** på 1-3 nøkkelord i hvert punkt (eller i setningen).
- Bruke linjeskift for skille avsnittene og ulike poenger (bruk "\\n").
EKSEMPEL PÅ "answer" MED MARKDOWN:
"answer": "Du kan prøve dette:\\n- **Pust rolig** i 10 sekunder\\n- **Start med et enkelt spørsmål**"

========================================
REGLER FOR CLAIMS
========================================
- En liste av påstander som du har hentet ut fra "answer".
- Én claim = én tydelig idé.
- Må være eksplisitt uttrykt eller entydig støttet av context.
- Ikke finn opp nye ideer.
- Ikke lag claims basert på spørsmålet eller instruksjonene.


========================================
REGLER FOR CITATIONS
========================================
- Hver citation må være:
  {{
    "url": str,
    "quote": str   # eksakt substring fra context
  }}
- "quote" må være ordrett, uten endringer.
- Ikke slå sammen tekst fra flere steder.
- Hvis ingen eksakt substring finnes, sett:
  - "validity": "not valid"
  - "Citations": []

========================================
SPØRSMÅL:
{question}

========================================
CONTEXT:
{context}
"""
)

KEYWORD_PROMPT = PromptTemplate.from_template(
    """Du er en presis tekstanalyse-assistent og skal svare på norsk (bokmål) når teksten er norsk.
Du MÅ følge reglene under nøyaktig.

VIKTIGE PRINSIPPER:
- Du kan KUN bruke informasjon fra 'context' (nedenfor). Ikke finn på temaer, begreper eller kategorier som ikke finnes i context.
- Nøkkelordene skal beskrive innholdet, ikke være generiske UI-ord eller navigasjonstekst.

UTDATAFORMAT (SVÆRT VIKTIG):
Du MÅ returnere gyldig JSON som matcher den eksakte Pydantic-skjema-strukturen 'KeywordResult':

{{
  "keywords": [str],
  "keyphrases": [str],
  "description": str
  "language": str
}}

FELT-REGLER:
- "keywords":
  * 5–15 elementer.
  * Hvert element skal være ett nøkkelord eller en kort frase (1–3 ord).
  * Prioriter: fagbegreper, diagnoser, symptomer, tjenester/tiltak, målgrupper, institusjoner, relevante lover/ordninger, steder (hvis nevnt).
  * Fjern duplikater og nesten-duplikater.
  * Unngå generiske ord som: "informasjon", "spørsmål", "svar", "hjelp", "kontakt", "les mer", "side", "tjeneste" (med mindre de inngår i et spesifikt faguttrykk).
  
- "description":
  * Nøyaktig ÉN setning.
  * 12–25 ord.
  * Skal oppsummere de viktigste temaene i context.
  * Må være naturlig og direkte, uten metakommentarer.
  * DU SKAL IKKE skrive ord som: "teksten", "artikkelen", "innholdet", "dette", "denne teksten", "her", "over", "nedenfor".
  * Ikke start setningen med: "Teksten handler om", "Artikkelen handler om", "Dette handler om", "I teksten".
  * Ikke bruk kolon eller punktliste. Ikke bruk anførselstegn.

- "keyphrases":
  * 3–10 elementer.
  * Hver frase skal være 2–6 ord.
  * Skal være mer spesifikke enn "keywords" og dekke de viktigste temaene i teksten.

- "language":
  * Bruk "nb" for norsk bokmål, "nn" for nynorsk, "en" for engelsk.
  * Velg språket basert på hva context faktisk er skrevet på.

NORMALISERING:
- Bruk små bokstaver der det passer, men behold egennavn (f.eks. "Helsenorge", "NAV") med riktig skrivemåte.
- Ingen punktum på slutten av elementer.
- Ikke inkluder emojis.
- Returner KUN JSON. Ingen forklaringer, ingen markdown.

SPØRSMÅL/OPPGAVE:
Trekk ut nøkkelord og nøkkelfraser fra teksten i context.

CONTEXT:
{context}
"""
)

def severity_for_query_prompt(query: str) -> str:
    return SEVERITY_FOR_QUERY.render(query=query)

def categorize_text_prompt(text: str, categories: str) -> str:
    return CATEGORIZE_TEXT.render(text=text, categories=categories)

def qa_subject_no_prompt(text: str) -> str:
    return QA_SUBJECT_NO.render(text=text)
