"""
Per-artikkel LangGraph-workflows for 'legacy'-indeksene:
  - build_hei_index_ingest_doc_workflow: metadata (spørsmål/svar, kategorier) + upsert i <name>
  - build_hei_index_qa_doc_workflow:     grounded svar per spørsmål + upsert i <name>_qa_bank
"""
import os
import time
import json
import logging
import textwrap
from pathlib import Path
from typing import Iterable, List, Literal, Optional
from typing_extensions import TypedDict

from openai import APIConnectionError
from pydantic import BaseModel, Field, ValidationError
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from prompts import severity_for_query_prompt, qa_subject_no_prompt, categorize_text_prompt, GROUNDED_PROMPT

_CATEGORIES_PATH = Path(__file__).parent / "psa_ssa_topics.json"
_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id", "file_id", "url")
_NO_ANSWER = "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten."


class Citation(BaseModel):
    url: str
    quote: str = Field(..., min_length=8)

class Claim(BaseModel):
    claim: str
    Citations: List[Citation]
    validity: Literal["valid", "not valid"]

class GroundedAnswer(BaseModel):
    answer: str
    short_answer: str
    claims: List[Claim]

class Reference(TypedDict):
    name: str
    url: str
    icon_url: str
    relevancy_index: float

class State_singleDoc(TypedDict):
    llm: any
    index: Optional[VectorStoreIndex]
    similarity_top_k: int
    similarity_cutoff: float
    name: str
    storage: str
    document: Document
    answered_questions: List[Document]
    relevancy_thresholds: dict[str, float]


# === Helpers ===

def ensure_text_splitter():
    if getattr(Settings, "text_splitter", None) is None:
        Settings.text_splitter = SentenceSplitter(chunk_size=3600, chunk_overlap=200)

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _parse_categories_from_llm_json(obj: dict, max_k: int = 3) -> list[str]:
    # Preferred new format: {"kategorier":[...]}
    if isinstance(obj.get("kategorier"), list):
        return _dedupe_preserve_order([str(x) for x in obj["kategorier"]])[:max_k]

    # Backward compatible old format: {"kategori":"..."}
    if obj.get("kategori"):
        return _dedupe_preserve_order([str(obj["kategori"])])[:max_k]

    return []

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

def _invoke_with_retry(runnable, inp, retries=5, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return runnable.invoke(inp)
        except (APIConnectionError, ValidationError) as e:
            logging.warning("Retryable error %s/%s: %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(delay)

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

def _format_context_from_nodes(nodes, max_chars_per_node=1000, max_nodes=20) -> str:
    parts = []
    for nws in nodes[:max_nodes]:
        node = getattr(nws, "node", nws)
        txt = _node_text(node).strip()
        if not txt:
            continue
        parts.append(f"[{_preferred_display_id(node)}]\n{textwrap.dedent(txt[:max_chars_per_node])}")
    return "\n\n".join(parts)

def _upsert_docs_into_index(name: str, storage: str, documents: Iterable[Document]) -> VectorStoreIndex:
    """
    Incrementally upsert Documents into a VectorStoreIndex persisted at <storage>/<name>.
    Creates the index if it doesn't exist; otherwise loads and inserts.
    Documents are stored in the docstore with doc_id=url so docstore.get_document(url) works.
    """
    persist_dir = os.path.join(storage, name)
    os.makedirs(storage, exist_ok=True)
    ensure_text_splitter()

    if os.path.isdir(persist_dir):
        logging.info("Loading existing index from %s", persist_dir)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        logging.info("Creating new index at %s", persist_dir)
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents([], storage_context=storage_context, show_progress=False)

    # Idempotent upsert: drop previously-ingested nodes for these docs so a
    # re-ingest REPLACES a document's chunks instead of stacking duplicates.
    for d in documents:
        ref_id = getattr(d, "doc_id", None) or getattr(d, "id_", None)
        if not ref_id:
            continue
        try:
            index.delete_ref_doc(ref_id, delete_from_docstore=True)
        except Exception:
            pass  # first-time ingest: nothing to delete

    try:
        storage_context.docstore.add_documents(documents, allow_update=True)
    except TypeError:
        storage_context.docstore.add_documents(documents)
    logging.info("Added/updated %d documents in docstore for index '%s'", len(documents), name)

    nodes = Settings.text_splitter.get_nodes_from_documents(documents)
    logging.info("Upserting %d nodes into index '%s'", len(nodes), name)
    index.insert_nodes(nodes)

    storage_context.persist(persist_dir=persist_dir)
    return index


# === Node functions ===

def create_metadata_for_document(state: State_singleDoc) -> dict:
    doc = state["document"]
    categories_json = _CATEGORIES_PATH.read_text(encoding="utf-8")

    qa_raw = state["llm"].invoke(qa_subject_no_prompt(text=doc.text)).content
    logging.info("qa raw: %s", qa_raw)

    category_raw = state["llm"].invoke(categorize_text_prompt(text=doc.text, categories=categories_json)).content
    logging.info("category_raw: %s", category_raw)

    questions: list[str] = []
    qa_pairs: list[dict] = []

    try:
        obj = json.loads(qa_raw)

        title_suggestion = obj.get("Title")
        if isinstance(title_suggestion, str) and title_suggestion.strip():
            doc.metadata["suggested_title"] = title_suggestion.strip()

        raw_qa_list = obj.get("QA", [])
        if not isinstance(raw_qa_list, list):
            raise ValueError('"QA" must be a list')

        for item in raw_qa_list:
            if not isinstance(item, dict):
                raise ValueError("Each QA item must be an object")
            q = item.get("question")
            a = item.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                raise ValueError('Each QA item must have string "question" and "answer"')
            q = q.strip()
            if not q:
                continue
            questions.append(q)
            qa_pairs.append({"question": q, "answer": a.strip()})

        doc.metadata["answered_questions"] = questions
        doc.metadata["qa_pairs"] = qa_pairs

        category_obj = json.loads(category_raw)
        cats = _parse_categories_from_llm_json(category_obj, max_k=3)

        hovedkategori = category_obj.get("hovedkategori")
        hovedkategori = hovedkategori.strip() if isinstance(hovedkategori, str) else ""
        if not hovedkategori and cats:
            hovedkategori = cats[0]
        if not hovedkategori:
            hovedkategori = "Ukjent"

        doc.metadata["categories"] = cats          # list for UI/analytics
        doc.metadata["category"] = hovedkategori
        doc.metadata["valid"] = True
    except Exception as e:
        logging.error("Failed to parse QA/category structure: %s | raw=%r", e, qa_raw)
        doc.metadata["answered_questions"] = []
        doc.metadata["qa_pairs"] = []
        doc.metadata["valid"] = False

    return {"document": doc}

def persist_storage_for_document(state: State_singleDoc) -> dict:
    index = _upsert_docs_into_index(name=state["name"], storage=state["storage"], documents=[state["document"]])
    return {"index": index}

def create_answered_questions_for_document(state: State_singleDoc) -> dict:
    answered_questions: List[Document] = []
    index = state.get("index")
    if index is None:
        logging.warning("No index available; skipping answered-questions generation.")
        return {"answered_questions": []}

    doc = state["document"]
    questions = doc.metadata.get("answered_questions", []) or []

    # Map question -> short_answer
    qa_map: dict[str, str] = {}
    for pair in doc.metadata.get("qa_pairs", []) or []:
        if isinstance(pair, dict):
            q = pair.get("question")
            a = pair.get("answer")
            if isinstance(q, str) and isinstance(a, str) and q.strip():
                qa_map[q.strip()] = a.strip()

    from_doc_id = getattr(doc, "doc_id", None)
    thresholds = state.get("relevancy_thresholds", {"strong": 0.60, "medium": 0.55, "weak": 0.35})

    retriever = index.as_retriever(
        similarity_top_k=state["similarity_top_k"],
        similarity_cutoff=state["similarity_cutoff"],
    )

    for q in questions:
        if not isinstance(q, str) or not q.strip():
            continue
        q_clean = q.strip()

        qdoc = Document(text=q_clean)
        qdoc.metadata["short_answer"] = qa_map.get(q_clean, "")
        qdoc.metadata["from_doc_id"] = from_doc_id
        qdoc.metadata["url"] = doc.metadata.get("url", "")

        # Categories / main category (hovedkategori)
        cats = doc.metadata.get("categories")
        if not isinstance(cats, list):
            cats = []
        hovedkategori = (doc.metadata.get("category") or "").strip()
        if "|" in hovedkategori:
            hovedkategori = hovedkategori.split("|", 1)[0].strip()
        if not hovedkategori and cats:
            hovedkategori = str(cats[0]).strip()
        if not hovedkategori:
            hovedkategori = "Ukjent"
        cats = [hovedkategori] + [c for c in cats if c and c != hovedkategori]

        qdoc.metadata["categories"] = cats[:3]
        qdoc.metadata["category"] = hovedkategori
        qdoc.metadata["valid"] = doc.metadata.get("valid", True)

        # Severity
        sev_resp = state["llm"].invoke(severity_for_query_prompt(query=q_clean))
        try:
            qdoc.metadata["severity"] = json.loads(sev_resp.content).get("category", "")
        except Exception as e:
            logging.error("Failed to parse severity JSON: %s | raw=%r", e, sev_resp.content)
            qdoc.metadata["severity"] = ""

        try:
            nodes = retriever.retrieve(q_clean) or []
        except Exception:
            logging.error("Retriever failed for question: %s", q_clean, exc_info=True)
            nodes = []

        if not nodes:
            qdoc.metadata["valid"] = False
            qdoc.metadata["answer"] = _NO_ANSWER
            answered_questions.append(qdoc)
            continue

        best_score = float(max(getattr(n, "score", 0.0) or 0.0 for n in nodes))
        if _classify_relevancy(best_score, thresholds) == "Rejected":
            qdoc.metadata["valid"] = False
            qdoc.metadata["answer"] = _NO_ANSWER
            answered_questions.append(qdoc)
            continue

        refs: List[Reference] = []
        seen_urls = set()
        for nws in nodes:
            meta = getattr(getattr(nws, "node", nws), "metadata", {}) or {}
            url = (meta.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            refs.append({
                "name": (meta.get("title") or "Ingen tittel").lstrip(),
                "url": url,
                "icon_url": meta.get("icon_url", "Ingen URL for ikon"),
                "relevancy_index": float(getattr(nws, "score", 0.0)),
            })
            if len(refs) >= 5:
                break

        ctx = _format_context_from_nodes(nodes)
        chain = (
            RunnableLambda(lambda _, q=q_clean, c=ctx: {"question": q, "context": c})
            | GROUNDED_PROMPT
            | state["llm"].with_structured_output(GroundedAnswer)
        )

        try:
            ga: GroundedAnswer = chain.invoke({})
            qdoc.metadata.update({
                "answer": ga.answer,
                "short_answer": ga.short_answer,
                "references": refs,
                "valid": True,
            })
        except Exception:
            logging.error("Grounded answer generation failed for question: %s", q_clean, exc_info=True)
            qdoc.metadata["valid"] = False
            qdoc.metadata["answer"] = qdoc.metadata.get("short_answer") or _NO_ANSWER

        answered_questions.append(qdoc)

    return {"answered_questions": answered_questions}

def persist_storage_for_answered_questions_single(state: State_singleDoc) -> dict:
    _upsert_docs_into_index(
        name=f"{state['name']}_qa_bank",
        storage=state["storage"],
        documents=state["answered_questions"],
    )
    return {}


builder_ingest = StateGraph(State_singleDoc)
builder_ingest.add_node("create_metadata_for_document", create_metadata_for_document)
builder_ingest.add_node("persist_storage_for_document", persist_storage_for_document)
builder_ingest.add_edge(START, "create_metadata_for_document")
builder_ingest.add_edge("create_metadata_for_document", "persist_storage_for_document")
builder_ingest.add_edge("persist_storage_for_document", END)
build_hei_index_ingest_doc_workflow = builder_ingest.compile()

builder_qa = StateGraph(State_singleDoc)
builder_qa.add_node("create_answered_questions_for_document", create_answered_questions_for_document)
builder_qa.add_node("persist_storage_for_answered_questions_single", persist_storage_for_answered_questions_single)
builder_qa.add_edge(START, "create_answered_questions_for_document")
builder_qa.add_edge("create_answered_questions_for_document", "persist_storage_for_answered_questions_single")
builder_qa.add_edge("persist_storage_for_answered_questions_single", END)
build_hei_index_qa_doc_workflow = builder_qa.compile()
