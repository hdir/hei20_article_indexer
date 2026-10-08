"""
Bygger vektorindeksen for 'hvaerinnafor' basert på artikkellisten i
scraping/hvaerinnafor/startUrls.json.

Resultat:
  <storage>/hvaerinnafor/          - indeks over artiklene
  <storage>/hvaerinnafor_qa_bank/  - indeks over genererte spørsmål med svar
  combined_documents_for_hvaerinnafor.xlsx - logg over dokumenter, metadata og Q&A
"""
import os
import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from workflow import build_hei_index_workflow, State_buildIndex


def main():
    parser = argparse.ArgumentParser(description="Bygg vektorindeks fra startUrls.json")
    parser.add_argument("--name", default="hvaerinnafor", help="Mappe under scraping/ med config.json og startUrls.json")
    parser.add_argument("--storage", default="./blobstorage/chatbot/", help="Hvor indeksene lagres")
    parser.add_argument("--content-type", default="html_content", choices=["html_content", "markdown_content"])
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        model_kwargs={"response_format": {"type": "json_object"}},
        timeout=120,
    )

    Settings.embed_model = AzureOpenAIEmbedding(
        model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    )

    init_state: State_buildIndex = {
        "llm": llm,
        "similarity_top_k": 10,
        "similarity_cutoff": 0.75,
        "content_type": args.content_type,
        "name": args.name,
        "scraping_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraping"),
        "storage": args.storage,
        "documents": [],
    }

    build_hei_index_workflow.invoke(init_state)


if __name__ == "__main__":
    main()
