import os
import re
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from apify_client import ApifyClient
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from prompts import severity_for_query_prompt, qa_subject_no_prompt, categorize_text_prompt, GROUNDED_PROMPT

MAX_METADATA_LENGTH = 1000
_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id", "file_id", "url")

CATEGORIES = {
    "Eksen": ["eks", "eksen", "eks-kjæreste", "ex", "skal jeg gå tilbake til eksen", "følelser for eksen", "forelsket i eksen", "hvordan få eksen tilbake", "vinne tilbake eksen", "jeg savner eksen min", "sammen med eksen igjen", "tenker på eksen", "eksen savner meg", "når eksen angrer", "eksen vil ha meg tilbake", "venn med eksen", "hvordan komme over eksen"],
    "Kjæreste / Forhold": ["kjæreste", "kjærester", "par", "partner", "parforhold", "forhold", "sammen", "holde på", "problemer", "krangler", "sjalu", "sjalusi", "usikkerhet", "tillitsbrudd", "kommunikasjon", "forventning", "samtale", "praten", "avstandsforhold", "vold", "overgrep", "sint", "mistenksom", "svik", "tilgi", "skam", "skyldfølelse"],
    "Sex / Intimitet": ["sex", "samleie", "kåt", "lyst på sex", "lite lyst", "liten lyst", "frustrert", "frustrerende", "sex med en venn", "vennesex", "sex med en kompis", "sex med en kollega", "må man være kjærester for å ha sex?", "hvem kan jeg ha sex med?", "overtale noen til sex", "overnatting", "overnatte", "sove sammen", "ligge sammen", "sjekke opp", "hooke", "første kyss", "tungekyss", "kyssing"],
    "Forelskelse / Flørting": ["forelskelse", "forelsket", "forelska", "betatt", "elsker", "besatt", "oppslukt", "kjendisforelskelse", "justin bieber", "første forelskelse", "test forelsket", "tegn på forelskelse", "fysiske symptomer", "hvordan få noen til å like deg", "hvordan få kjæreste", "hvordan få henne/han interessert", "liker han meg", "liker hun meg", "flørt", "flørting", "flørteskole", "flørtetips", "blikkontakt", "initiativ", "første steg", "signaler", "snapchat-flørting", "meldinger", "bilder"],
    "Vennskap": ["venn", "venner", "venninner", "kamerater", "bestevenn", "vanskelig vennskap", "dårlige venner", "avslutte vennskap", "avsluttet vennskap", "si ifra til en venn", "vennen min hører ikke på meg", "vennen min er forelsket i meg", "forelska i en venn"],
    "Ekteskap": ["ekteskap", "ekteskapsloven", "gift", "vie", "bryllup", "kirkebryllup", "borgerlig vielse"],
    "Utroskap": ["utro", "utroskap", "usikkerhet", "tillitsbrudd", "svik", "skam", "skyldfølelse"],
    "Kjærlighetssorg": ["kjærlighetssorg", "hvordan komme seg over kjærlighetssorg", "hjelp mot kjærlighetssorg", "behandling av kjærlighetssorg", "tips mot kjærlighetssorg", "råd mot kjærlighetssorg", "redd for å bli avvist", "avvisning", "avvist"],
    "Ungdom / Sosiale tema": ["muslim", "muslimsk kjæreste", "strenge foreldre", "kultur", "barneloven", "kontroll", "foreldre", "nei", "regler", "alder", "aldersforskjell (ung/gammel/eldre)", "fest", "ferie", "tur", "syden", "alkohol", "drikker", "fantasi", "avtaler"],
    "Aktiviteter / Tips": ["daten", "kino", "middag", "aktivitet", "finne på", "crush", "tips", "råd", "hva skal vi gjøre", "snakke", "småprat"],
}


class Citation(BaseModel):
    url: str
    quote: str = Field(..., min_length=8)

class Claim(BaseModel):
    claim: str
    Citations: List[Citation]
    validity: Literal["valid", "not valid"]

class GroundedAnswer(BaseModel):
    answer: str
    claims: List[Claim]


class State_buildIndex(TypedDict, total=False):
    llm: Any
    index: VectorStoreIndex
    similarity_top_k: int
    similarity_cutoff: float
    content_type: Literal["markdown_content", "html_content"]
    name: str
    scraping_dir: str
    storage: str
    documents: List[Document]
    answered_questions: List[Document]

# === Helpers ===

def _embed_texts(texts: List[str], model, batch_size: int = 64) -> np.ndarray:
    if model is None:
        raise ValueError("embed model is None; set Settings.embed_model.")
    embs = []
    for i in range(0, len(texts), batch_size):
        embs.extend(model.get_text_embedding_batch(texts[i:i+batch_size]))
    return np.asarray(embs, dtype=np.float32)

def _deduplicate_semantic_docs_greedy(docs: List[Document], embed_model, threshold: float = 0.85) -> List[Document]:
    """
    Greedy semantic deduplication over a list of Document objects (uses doc.text).
    Keeps the first representative if max similarity < threshold.
    """
    if not docs:
        return []

    texts = [d.text or "" for d in docs]
    embeddings = _embed_texts(texts, model=embed_model)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    keep_idx: List[int] = []
    for i in range(len(texts)):
        if not keep_idx:
            keep_idx.append(i)
            continue
        sims = embeddings[keep_idx] @ embeddings[i]
        if float(np.max(sims)) < threshold:
            keep_idx.append(i)

    return [docs[i] for i in keep_idx]

def _persist_storage_for_item(name: str, storage: str, documents: List[Document]):
    logging.info(f'Loaded {len(documents)} documents for {name}')

    try:
        if getattr(Settings, "text_splitter", None) is None:
            Settings.text_splitter = SentenceSplitter(chunk_size=1200, chunk_overlap=150)

        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        logging.info('VectorStoreIndex created in memory')

        persist_dir = os.path.join(storage, name)
        storage_context.persist(persist_dir=persist_dir)
        logging.info(f'Index persisted to {persist_dir}')
        return index

    except Exception:
        logging.error("Failed to persist storage", exc_info=True)
        return None

def _iter_meta_rows(meta, base_path: str = ""):
    """
    Yield rows for a 'long' metadata table.
    - For dicts: recurse with dotted keys.
    - For lists: one row per element with meta_index and meta_path like 'answered_questions[0]'.
    - For scalars: single row.
    """
    if meta is None:
        return

    if isinstance(meta, dict):
        for k, v in meta.items():
            path = f"{base_path}.{k}" if base_path else k
            if isinstance(v, (dict, list)):
                yield from _iter_meta_rows(v, base_path=path)
            else:
                yield {
                    "meta_key": k,
                    "meta_index": None,
                    "meta_path": path,
                    "value": v,
                    "value_type": type(v).__name__,
                    "value_len": len(v) if isinstance(v, str) else None,
                }
    elif isinstance(meta, list):
        for i, v in enumerate(meta):
            path = f"{base_path}[{i}]" if base_path else f"[{i}]"
            if isinstance(v, (dict, list)):
                for row in _iter_meta_rows(v, base_path=path):
                    if row.get("meta_index") is None:
                        row["meta_index"] = i
                    yield row
            else:
                yield {
                    "meta_key": base_path.split(".")[-1] if base_path else "",
                    "meta_index": i,
                    "meta_path": path,
                    "value": v,
                    "value_type": type(v).__name__,
                    "value_len": len(v) if isinstance(v, str) else None,
                }
    else:
        yield {
            "meta_key": base_path or "",
            "meta_index": None,
            "meta_path": base_path or "",
            "value": meta,
            "value_type": type(meta).__name__,
            "value_len": len(meta) if isinstance(meta, str) else None,
        }

def _load_run_input(item: str, scraping_dir: str) -> Dict[str, Any]:
    """
    Reads <scraping_dir>/<item>/config.json and injects startUrls.json as 'startUrls'.
    """
    with open(os.path.join(scraping_dir, item, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(os.path.join(scraping_dir, item, 'startUrls.json'), 'r', encoding='utf-8') as f:
        start_urls = json.load(f)

    run_input = config.get(item, {})
    run_input['startUrls'] = start_urls
    return run_input

def _dataset_items(client: ApifyClient, dataset_id: str) -> List[Dict[str, Any]]:
    return list(client.dataset(dataset_id).iterate_items())

def _truncate_metadata(metadata: dict) -> dict:
    max_lengths = {
        "url": 300,
        "title": 200,
        "keywords": 300,
        "description": 300,
    }

    truncated = {key: str(metadata.get(key, ""))[:max_len] for key, max_len in max_lengths.items()}

    # Trim the least important field until the metadata fits
    while len(json.dumps(truncated, ensure_ascii=False)) > MAX_METADATA_LENGTH:
        for field in ["description", "keywords", "title", "url"]:
            if len(truncated[field]) > 50:
                truncated[field] = truncated[field][:-10]
                break
        else:
            break

    return truncated

def _preferred_display_id(node) -> str:
    meta = getattr(node, "metadata", {}) or {}
    ids = [str(meta[k]) for k in _POSSIBLE_META_IDS if meta.get(k)]
    chunk_id = getattr(node, "id_", None) or getattr(node, "node_id", None)
    if chunk_id:
        ids.append(str(chunk_id))
    return ids[0] if ids else "unknown"

def _node_text(n) -> str:
    t = getattr(n, "text", None)
    if t:
        return t
    get_content = getattr(n, "get_content", None)
    if callable(get_content):
        return get_content(metadata_mode="all") or ""
    return ""

def _format_context_from_nodes(nodes, max_chars_per_node=3000, max_nodes=100) -> str:
    parts = []
    for nws in nodes[:max_nodes]:
        node = getattr(nws, "node", nws)
        txt = _node_text(node).strip()
        if not txt:
            continue
        parts.append(f"[{_preferred_display_id(node)}]\n{textwrap.dedent(txt[:max_chars_per_node])}")
    return "\n\n".join(parts)

def _transform_dataset_item(doc_item, content_type) -> Document:
    meta = doc_item.get('metadata', {})
    title = meta.get('title', 'Untitled')
    keywords = meta.get('keywords', 'Untitled')
    description = meta.get('description', 'Untitled')

    markdown_content = doc_item.get("markdown", "") or ""
    if content_type == "markdown_content":
        text_content = markdown_content
    else:
        text_content = doc_item.get("text", "") or ""

    # remove all text after split_phrase
    result = text_content.partition("Fikk du svar på det du lurte på?")[0]

    # fix for getting correct title from Hva er innafor:
    if title == "Hva er innafor - Helsenorge":
        match = re.search(r"Spørsmål:\s*(.*)", markdown_content, re.DOTALL)
        if match:
            title = match.group(1).splitlines()[0]
            print(f'title fra spørsmål:{title}')
        else:
            print("No 'Spørsmål' section found.")

    metadata = _truncate_metadata({
        "url": doc_item.get("url", "Unknown URL"),
        "title": title,
        "keywords": keywords,
        "description": description,
    })
    metadata["category"] = ""
    metadata["severity"] = ""
    metadata["answered_questions"] = []

    return Document(text=result, metadata=metadata)

def _classify_relevancy(score: float, thresholds: dict[str, float]) -> str:
    """Returns "Strong", "Medium" or "Rejected"."""
    strong = float(thresholds.get("strong", 0.60))
    medium = float(thresholds.get("medium", 0.50))
    weak   = float(thresholds.get("weak",   0.35))
    strong, medium, weak = sorted([strong, medium, weak], reverse=True)

    s = float(score)
    if s >= strong:
        return "Strong"
    if s >= medium:
        return "Medium"
    return "Rejected"

# === Node functions ===

def apify_call_load_documents(state: State_buildIndex) -> dict:
    """
    Runs the 'apify/website-content-crawler' actor with config.json + startUrls.json,
    fetches the dataset items and turns them into Documents.
    """
    item = state["name"]
    run_input = _load_run_input(item, state["scraping_dir"])

    token = os.getenv('APIFY_TOKEN')
    if not token:
        raise RuntimeError("Missing APIFY_TOKEN in environment.")

    client = ApifyClient(token)
    run = client.actor("apify/website-content-crawler").call(run_input=run_input)
    dataset_id = (run or {}).get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("No defaultDatasetId found on apify run result.")

    raw_items = _dataset_items(client, dataset_id)
    documents = [_transform_dataset_item(it, content_type=state["content_type"]) for it in raw_items]

    print(f'------------ apify run OK: {len(documents)} documents -----------')
    return {"documents": documents}

def create_metadata_for_documents(state: State_buildIndex) -> dict:
    categories_json = json.dumps(CATEGORIES, ensure_ascii=False, indent=4)

    updated_docs = []
    for doc in state["documents"]:
        qa_raw = state["llm"].invoke(qa_subject_no_prompt(text=doc.text)).content
        category_raw = state["llm"].invoke(categorize_text_prompt(text=doc.text, categories=categories_json)).content
        print('category_raw: ', category_raw)

        try:
            questions = json.loads(qa_raw).get("Questions", [])
            print('Questions:', questions)
            if not isinstance(questions, list) or not all(isinstance(x, str) for x in questions):
                raise ValueError("'Questions' must be a list of strings")
            doc.metadata["answered_questions"] = questions
            doc.metadata["category"] = json.loads(category_raw).get("kategori")
            doc.metadata["valid"] = True
        except Exception as e:
            print(f"Failed to parse questions: {e} | raw={qa_raw!r}")
            doc.metadata["answered_questions"] = []

        updated_docs.append(doc)

    return {"documents": updated_docs}

def persist_storage_for_documents(state: State_buildIndex) -> dict:
    index = _persist_storage_for_item(name=state["name"], storage=state["storage"], documents=state["documents"])
    return {"index": index}

def create_answered_questions(state: State_buildIndex) -> dict:
    unique_answered_questions_with_answers: List[Document] = []

    try:
        index = state.get("index")
        if index is None:
            logging.warning("No index available; skipping answered-questions generation.")
            return {"answered_questions": []}

        retriever = index.as_retriever(
            similarity_top_k=state["similarity_top_k"],
            similarity_cutoff=state["similarity_cutoff"]
        )

        answered_questions: List[Document] = []
        for doc in state["documents"]:
            for q in doc.metadata.get("answered_questions", []) or []:
                new_doc = Document(text=q)
                new_doc.metadata["from_doc_id"] = getattr(doc, "doc_id", None)
                new_doc.metadata["url"] = doc.metadata.get("url", "")
                new_doc.metadata["category"] = doc.metadata.get("category", "")
                new_doc.metadata["valid"] = True
                print(f'newdoc: {new_doc.text}, {new_doc.metadata}\n')
                answered_questions.append(new_doc)
        print(f'Found {len(answered_questions)} answered questions')

        # remove the similar questions
        unique_answered_questions = _deduplicate_semantic_docs_greedy(answered_questions, embed_model=Settings.embed_model, threshold=0.80)
        print(f'Found {len(unique_answered_questions)} unique_answered questions')

        thresholds = state.get("relevancy_thresholds", {
            "strong": 0.60,
            "medium": 0.55,
            "weak":   0.35,
        })

        for doc in unique_answered_questions:
            nodes = retriever.retrieve(doc.text) or []
            if not nodes:
                continue

            best_score = float(max(n.score or 0.0 for n in nodes))
            if _classify_relevancy(best_score, thresholds) == "Rejected":
                continue

            ctx = _format_context_from_nodes(nodes)
            chain = (
                RunnableLambda(lambda _, q=doc.text, c=ctx: {"question": q, "context": c})
                | GROUNDED_PROMPT
                | state["llm"].with_structured_output(GroundedAnswer)
            )
            ga: GroundedAnswer = chain.invoke({})

            sev_resp = state["llm"].invoke(severity_for_query_prompt(query=doc.text))
            doc.metadata["severity"] = json.loads(sev_resp.content).get("category", "")
            doc.metadata["answer"] = ga.answer
            unique_answered_questions_with_answers.append(doc)

    except Exception:
        logging.error("create_answered_questions failed", exc_info=True)

    return {"answered_questions": unique_answered_questions_with_answers}

def persist_storage_for_answered_questions(state: State_buildIndex) -> dict:
    qa_docs = state.get("answered_questions", [])
    if qa_docs:
        _persist_storage_for_item(name=f"{state['name']}_qa_bank", storage=state["storage"], documents=qa_docs)
    else:
        logging.warning("No answered questions; skipping qa_bank index.")
    return {}

def create_log_excel(state: State_buildIndex) -> dict:
    """
    Writes combined_documents_for_<name>.xlsx with sheets:
      - 'metadata_long': per-metadata entry
      - 'text': document texts
      - 'answered_questions': one row per Q&A pair
    """
    item = state.get("name", "item")
    safe_item = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(item))
    out_path = Path(f"combined_documents_for_{safe_item}.xlsx")

    long_rows, text_rows, qa_rows = [], [], []

    for idx, doc in enumerate(state.get("documents", [])):
        text_rows.append({"row_index": idx, "item": item, "text": str(doc.text)})
        for row in _iter_meta_rows(doc.metadata or {}):
            long_rows.append({"row_index": idx, "item": item, **row})

    for qdoc in state.get("answered_questions", []):
        qa_rows.append({
            "question": qdoc.text,
            "answer": qdoc.metadata.get("answer", ""),
            "category": qdoc.metadata.get("category", ""),
            "severity": qdoc.metadata.get("severity", ""),
            "valid": True,
            "url": qdoc.metadata.get("url", ""),
        })

    wide_cols = ("value", "meta_path", "text", "question", "answer")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        wrap_fmt = writer.book.add_format({"text_wrap": True, "valign": "top"})
        for ws_name, rows in [("metadata_long", long_rows), ("text", text_rows), ("answered_questions", qa_rows)]:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=ws_name, index=False)
            ws = writer.sheets[ws_name]
            for col_idx, col_name in enumerate(df.columns):
                if col_name in wide_cols:
                    ws.set_column(col_idx, col_idx, 60, wrap_fmt)
                else:
                    ws.set_column(col_idx, col_idx, 20)

    print(f"✅ Excel written to: {out_path.resolve()}")
    return {}

# === Workflow ===
builder = StateGraph(State_buildIndex)

builder.add_node("apify_call_load_documents", apify_call_load_documents)
builder.add_node("create_metadata_for_documents", create_metadata_for_documents)
builder.add_node("persist_storage_for_documents", persist_storage_for_documents)
builder.add_node("create_answered_questions", create_answered_questions)
builder.add_node("persist_storage_for_answered_questions", persist_storage_for_answered_questions)
builder.add_node("create_log_excel", create_log_excel)

builder.add_edge(START, "apify_call_load_documents")
builder.add_edge("apify_call_load_documents", "create_metadata_for_documents")
builder.add_edge("create_metadata_for_documents", "persist_storage_for_documents")
builder.add_edge("persist_storage_for_documents", "create_answered_questions")
builder.add_edge("create_answered_questions", "persist_storage_for_answered_questions")
builder.add_edge("persist_storage_for_answered_questions", "create_log_excel")
builder.add_edge("create_log_excel", END)

build_hei_index_workflow = builder.compile()
