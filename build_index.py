"""
Bygger vektorindeksene for 'hvaerinnafor' fra artikkellisten i startUrls.json.

  --mode legacy   hvaerinnafor + hvaerinnafor_qa_bank (inkrementelt, én artikkel om gangen)
  --mode unified  hvaerinnafor_unified (artikler + ung.no-Q&A fra Azure SQL)
  --mode both     begge, for A/B-sammenligning

Indeksene lagres under --storage (standard ./blobstorage/chatbot/).
"""
import os
import copy
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from scraping import load_documents
from legacy_workflow import build_hei_index_ingest_doc_workflow, build_hei_index_qa_doc_workflow, State_singleDoc
from unified import run_unified_build, _article_aliases_cache_path

NAME = "hvaerinnafor"
UNIFIED_NAME = "hvaerinnafor_unified"
RELEVANCY_THRESHOLDS = {"strong": 0.60, "medium": 0.55, "weak": 0.35}


# === Manifests (which doc_ids are already processed) ===

def _get_manifest_path(storage: str, name: str) -> Path:
    storage_dir = Path(storage)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / f"{name}_processed_docs.json"

def _wipe_index(storage: str, name: str) -> None:
    """Delete a persisted index dir and its processed-docs manifest (used by --from-scratch)."""
    persist_dir = os.path.join(storage, name)
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        logging.info("[from-scratch] wiped index dir %s", persist_dir)
    manifest = _get_manifest_path(storage, name)
    if manifest.exists():
        manifest.unlink()
        logging.info("[from-scratch] wiped manifest %s", manifest)

def _load_processed_doc_ids(storage: str, name: str) -> set[str]:
    manifest_path = _get_manifest_path(storage, name)
    if not manifest_path.exists():
        return set()
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        logging.warning("Failed to read processed-doc manifest; starting fresh.")
        return set()

def _save_processed_doc_ids(storage: str, name: str, doc_ids: set[str]) -> None:
    with _get_manifest_path(storage, name).open("w", encoding="utf-8") as f:
        json.dump(sorted(doc_ids), f, ensure_ascii=False, indent=2)


# === Legacy: hvaerinnafor + hvaerinnafor_qa_bank ===

def run_incremental_build_for_all_documents(
    llm,
    documents: List[Document],
    name: str,
    storage: str,
    similarity_top_k: int = 5,
    similarity_cutoff: float = 0.75,
    build_qa: bool = True,
):
    qa_name = f"{name}_qa_bank"
    processed_main = _load_processed_doc_ids(storage, name)
    processed_qa = _load_processed_doc_ids(storage, qa_name)

    def _state(doc: Document, index=None) -> State_singleDoc:
        return {
            "llm": llm,
            "index": index,
            "similarity_top_k": similarity_top_k,
            "similarity_cutoff": similarity_cutoff,
            "name": name,
            "storage": storage,
            "document": doc,
            "answered_questions": [],
            "relevancy_thresholds": RELEVANCY_THRESHOLDS,
        }

    # PHASE A: ingest all docs into the main index
    ingested_this_run: List[Document] = []
    for idx, doc in enumerate(documents):
        doc_id = getattr(doc, "doc_id", None) or doc.metadata.get("url") or str(id(doc))
        print(f"[A] Ingest {idx+1}/{len(documents)}: {doc.metadata.get('url')}")

        if doc_id in processed_main:
            logging.info("[A] Skip already ingested doc_id=%s", doc_id)
            continue

        state = _state(doc)
        try:
            build_hei_index_ingest_doc_workflow.invoke(state)
            processed_main.add(doc_id)
            _save_processed_doc_ids(storage, name, processed_main)
            ingested_this_run.append(state["document"])  # includes created metadata
            logging.info("[A] Ingested doc_id=%s", doc_id)
        except Exception:
            logging.error("[A] Failed ingest doc_id=%s; will retry on next run", doc_id, exc_info=True)

    if not build_qa:
        logging.info("Skipping QA / qa_bank phase — main index only.")
        return

    # Load the main index AFTER all ingests, so QA sees all nodes
    persist_dir = os.path.join(storage, name)
    if not os.path.isdir(persist_dir):
        logging.warning("Main index dir missing, skipping QA phase.")
        return
    main_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))

    # PHASE B: generate QA only for docs ingested in THIS run
    for idx, doc in enumerate(ingested_this_run):
        doc_id = getattr(doc, "doc_id", None) or doc.metadata.get("url") or str(id(doc))
        print(f"[B] QA {idx+1}/{len(ingested_this_run)}: {doc.metadata.get('url')}")

        if doc_id in processed_qa:
            logging.info("[B] Skip already QA-processed doc_id=%s", doc_id)
            continue

        try:
            build_hei_index_qa_doc_workflow.invoke(_state(doc, index=main_index))
            processed_qa.add(doc_id)
            _save_processed_doc_ids(storage, qa_name, processed_qa)
            logging.info("[B] QA done doc_id=%s", doc_id)
        except Exception:
            logging.error("[B] Failed QA doc_id=%s; will retry on next run", doc_id, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Bygg vektorindekser fra startUrls.json")
    parser.add_argument("--mode", choices=["legacy", "unified", "both"], default="legacy",
                        help="legacy = hvaerinnafor + hvaerinnafor_qa_bank (standard). "
                             "unified = hvaerinnafor_unified. both = begge, for A/B.")
    parser.add_argument("--months-back", type=int, default=12,
                        help="unified: hvor mange måneder bakover ung.no-Q&A hentes fra Azure SQL")
    parser.add_argument("--from-scratch", action="store_true",
                        help="slett indeksmappe(r) og manifest før bygging (full ombygging)")
    parser.add_argument("--skip-qa-bank", action="store_true",
                        help="legacy/both: bygg kun hvaerinnafor, hopp over hvaerinnafor_qa_bank")
    parser.add_argument("--start-urls", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "startUrls.json"),
                        help="JSON-fil med artikkellisten")
    parser.add_argument("--storage", default="./blobstorage/chatbot/", help="Hvor indeksene lagres")
    args = parser.parse_args()

    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    for noisy in ("httpx", "httpcore", "openai", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        timeout=120,
        temperature=0.0,
    )

    Settings.embed_model = AzureOpenAIEmbedding(
        model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    )

    storage = args.storage
    if args.from_scratch:
        if args.mode in ("legacy", "both"):
            _wipe_index(storage, NAME)
            if not args.skip_qa_bank:
                _wipe_index(storage, f"{NAME}_qa_bank")
        if args.mode in ("unified", "both"):
            # The alias cache is kept on purpose: unchanged chunks reuse their aliases.
            _wipe_index(storage, UNIFIED_NAME)
            logging.info("[from-scratch] keeping alias cache %s", _article_aliases_cache_path(storage, NAME))

    documents = load_documents(llm, args.start_urls, output_dir=os.path.join(storage, "scraping"))
    logging.info("Scraped %d documents", len(documents))

    # The legacy build enriches the documents' metadata in place; the unified
    # build must start from the plain scraped documents.
    unified_documents = copy.deepcopy(documents) if args.mode == "both" else documents

    if args.mode in ("legacy", "both"):
        run_incremental_build_for_all_documents(
            llm=llm,
            documents=documents,
            name=NAME,
            storage=storage,
            similarity_top_k=5,
            similarity_cutoff=0.75,
            build_qa=not args.skip_qa_bank,
        )

    if args.mode in ("unified", "both"):
        run_unified_build(
            llm=llm,
            documents=unified_documents,
            name=NAME,
            storage=storage,
            months_back=args.months_back,
            unified_name=UNIFIED_NAME,
        )


if __name__ == "__main__":
    main()
