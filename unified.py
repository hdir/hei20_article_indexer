"""
Bygger hybridindeksen <unified_name> (standard: hvaerinnafor_unified):
  - artikkelbiter (node_type="article") med opptil 10 LLM-genererte spørsmål som 'aliases'
  - ung.no-spørsmål/svar fra Azure SQL (node_type="qa"), svaret som tekst og spørsmålet som alias
  - "cross-pollination": ekte ung.no-spørsmål legges til som aliases på artikkelbitene de ligner mest på
'aliases' påvirker kun embeddingen; feltet holdes utenfor det LLM-en ser.
"""
import os
import json
import hashlib
import logging
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document, TextNode

from prompts import qa_subject_no_prompt
from legacy_workflow import ensure_text_splitter
from ungno_sql import connect_mssql, fetch_questions


def _chunk_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _article_aliases_cache_path(storage: str, name: str) -> str:
    return os.path.join(storage, f"{name}_article_aliases.json")


def _load_article_aliases_cache(storage: str, name: str) -> dict:
    path = _article_aliases_cache_path(storage, name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logging.warning("Failed to read article aliases cache at %s; starting fresh.", path)
        return {}


def _save_article_aliases_cache(storage: str, name: str, cache: dict) -> None:
    path = _article_aliases_cache_path(storage, name)
    os.makedirs(storage, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _aliases_for_chunk(llm, chunk_text: str, max_aliases: int = 10) -> List[str]:
    if not chunk_text or not chunk_text.strip():
        return []
    qa_prompt = qa_subject_no_prompt(text=chunk_text)
    try:
        qa_resp = llm.invoke(qa_prompt)
        obj = json.loads(qa_resp.content)
    except Exception as e:
        logging.warning("Alias generation failed for chunk: %s", e)
        return []

    raw_qa_list = obj.get("QA", [])
    if not isinstance(raw_qa_list, list):
        return []

    aliases: List[str] = []
    for item in raw_qa_list:
        if isinstance(item, dict):
            q = item.get("question")
            if isinstance(q, str) and q.strip():
                aliases.append(q.strip())
        if len(aliases) >= max_aliases:
            break
    return aliases


def _build_article_nodes(
    llm,
    documents: List[Document],
    max_aliases: int = 10,
    storage: Optional[str] = None,
    name: Optional[str] = None,
) -> List[TextNode]:
    ensure_text_splitter()
    use_cache = bool(storage and name)
    cache = _load_article_aliases_cache(storage, name) if use_cache else {}
    cache_hits = 0
    cache_misses = 0

    article_nodes: List[TextNode] = []
    for doc in documents:
        chunks = Settings.text_splitter.get_nodes_from_documents([doc])
        for chunk in chunks:
            chunk.metadata = dict(getattr(chunk, "metadata", {}) or {})
            text_hash = _chunk_text_hash(chunk.text)
            cached = cache.get(text_hash) if use_cache else None
            if isinstance(cached, list):
                aliases = [a for a in cached if isinstance(a, str)][:max_aliases]
                cache_hits += 1
            else:
                aliases = _aliases_for_chunk(llm, chunk.text, max_aliases=max_aliases)
                if use_cache:
                    cache[text_hash] = aliases
                cache_misses += 1
            chunk.metadata["aliases"] = aliases
            chunk.metadata["node_type"] = "article"
            excluded = list(getattr(chunk, "excluded_llm_metadata_keys", []) or [])
            if "aliases" not in excluded:
                excluded.append("aliases")
            chunk.excluded_llm_metadata_keys = excluded
            article_nodes.append(chunk)

    if use_cache and cache_misses > 0:
        _save_article_aliases_cache(storage, name, cache)
    logging.info("Article alias cache: %d hits, %d misses", cache_hits, cache_misses)
    return article_nodes


def _build_psa_ssa_nodes(
    rows: List[dict],
    max_chars_question: int = 800,
    max_chars_answer: int = 1200,
) -> List[TextNode]:
    nodes: List[TextNode] = []
    for row in rows:
        q = (row.get("question") or "")[:max_chars_question].strip()
        a = (row.get("answer") or "")[:max_chars_answer].strip()
        if not a:
            continue
        node = TextNode(
            text=a,
            metadata={
                "url": row.get("url", "") or "",
                "title": row.get("title", "") or "",
                "category": row.get("psa_ssa_topic", "") or "",
                "parent_topic": row.get("parent_topic", "") or "",
                "severity": "",
                "icon_url": "",
                "node_type": "qa",
                "answered_date": row.get("answered_date", "") or "",
                "aliases": [q] if q else [],
            },
            id_=row.get("url") or None,
            excluded_llm_metadata_keys=["aliases"],
        )
        nodes.append(node)
    return nodes


def _cross_pollinate(
    article_nodes: List[TextNode],
    psa_ssa_nodes: List[TextNode],
    threshold: float = 0.55,
    top_k: int = 3,
    max_total_aliases_per_node: int = 50,
    max_chars_per_alias: int = 200,
) -> None:
    if not article_nodes or not psa_ssa_nodes:
        return

    article_by_id = {n.id_: n for n in article_nodes}
    logging.info("Building interim article-only index for cross-pollination (%d nodes)...", len(article_nodes))
    interim_index = VectorStoreIndex(article_nodes, show_progress=True)
    retriever = interim_index.as_retriever(similarity_top_k=top_k)

    candidates: List[tuple] = []
    for psa_node in psa_ssa_nodes:
        aliases = psa_node.metadata.get("aliases") or []
        if not aliases:
            continue
        question = aliases[0]
        if not question.strip():
            continue
        try:
            results = retriever.retrieve(question)
        except Exception:
            logging.error("Cross-pollination retrieval failed for: %s", question, exc_info=True)
            continue
        for nws in results:
            score = float(getattr(nws, "score", 0.0))
            if score < threshold:
                continue
            cid = getattr(nws.node, "id_", None)
            if cid is None or cid not in article_by_id:
                continue
            candidates.append((score, cid, question))

    candidates.sort(key=lambda t: t[0], reverse=True)

    attached = 0
    skipped_capped = 0
    skipped_dup = 0
    for score, cid, question in candidates:
        target_aliases = article_by_id[cid].metadata.setdefault("aliases", [])
        if len(target_aliases) >= max_total_aliases_per_node:
            skipped_capped += 1
            continue
        q_clipped = question[:max_chars_per_alias]
        if q_clipped in target_aliases:
            skipped_dup += 1
            continue
        target_aliases.append(q_clipped)
        attached += 1
    logging.info(
        "Cross-pollination attached %d question-aliases (skipped %d capped, %d duplicates, per-node cap=%d, max %d chars/alias)",
        attached, skipped_capped, skipped_dup, max_total_aliases_per_node, max_chars_per_alias,
    )


def _verify_persisted_consistency(persist_dir: str, *, strict: bool = True) -> None:
    """Verify the persisted vector store and docstore reference the same nodes.

    The fatal condition is an *orphan embedding*: a vector whose node ID is not
    in the docstore. When such a node lands in a query's top-k, LlamaIndex
    raises "Node ID ... not found in fetched nodes" and the whole answer is
    rejected. Reads the persisted JSON (exactly what the chatbot loads) so it
    also catches persist-time issues.

    Raises RuntimeError when ``strict`` and orphans exist, so a corrupt index
    is never uploaded/shipped. Docstore nodes without an embedding are only
    informational (they just can't be retrieved).
    """
    vs_path = os.path.join(persist_dir, "default__vector_store.json")
    ds_path = os.path.join(persist_dir, "docstore.json")

    with open(vs_path, encoding="utf-8") as f:
        vs = json.load(f)
    with open(ds_path, encoding="utf-8") as f:
        ds = json.load(f)

    emb_ids = set((vs.get("embedding_dict") or {}).keys())
    doc_ids = set((ds.get("docstore/data") or {}).keys())
    orphans = emb_ids - doc_ids
    no_emb = doc_ids - emb_ids

    summary = (
        f"Consistency check for {persist_dir}: {len(emb_ids)} embeddings, "
        f"{len(doc_ids)} docstore nodes, {len(orphans)} orphan embedding(s), "
        f"{len(no_emb)} node(s) without embedding."
    )

    if orphans:
        logging.error("%s INCONSISTENT — sample orphan IDs: %s", summary, list(orphans)[:5])
        if strict:
            raise RuntimeError(
                f"Index at {persist_dir} is inconsistent: {len(orphans)} embeddings reference "
                f"node IDs missing from the docstore. Aborting so a corrupt index is not shipped."
            )
    else:
        logging.info("%s OK", summary)


def _persist_unified_index(all_nodes: List[TextNode], storage: str, name: str) -> None:
    persist_dir = os.path.join(storage, name)
    os.makedirs(storage, exist_ok=True)

    logging.info("Building unified index over %d nodes...", len(all_nodes))
    index = VectorStoreIndex(all_nodes, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
    logging.info("Persisted unified index to %s", persist_dir)

    # Fail the build if the persisted vector store and docstore disagree, so we
    # never ship an index that crashes retrieval at query time.
    _verify_persisted_consistency(persist_dir)


def run_unified_build(
    llm,
    documents: List[Document],
    name: str = "hvaerinnafor",
    storage: str = "./blobstorage/chatbot/",
    months_back: int = 12,
    cross_pollinate_threshold: float = 0.55,
    cross_pollinate_top_k: int = 3,
    unified_name: str = "hvaerinnafor_unified",
    max_aliases_per_chunk: int = 10,
    max_total_aliases_per_node: int = 50,
):
    article_nodes = _build_article_nodes(
        llm,
        documents,
        max_aliases=max_aliases_per_chunk,
        storage=storage,
        name=name,
    )
    logging.info("Unified: built %d article chunk nodes", len(article_nodes))

    conn = connect_mssql()
    try:
        rows = fetch_questions(conn, months_back)
        logging.info("Unified: fetched %d psa_ssa rows (last %d months)", len(rows), months_back)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    psa_ssa_nodes = _build_psa_ssa_nodes(rows)
    logging.info("Unified: built %d psa_ssa nodes", len(psa_ssa_nodes))

    _cross_pollinate(
        article_nodes,
        psa_ssa_nodes,
        threshold=cross_pollinate_threshold,
        top_k=cross_pollinate_top_k,
        max_total_aliases_per_node=max_total_aliases_per_node,
    )

    _persist_unified_index(article_nodes + psa_ssa_nodes, storage=storage, name=unified_name)
