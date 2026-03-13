"""
Managed RAG — Azure AI Search + LangGraph
==========================================

KEY CONCEPT — "Managed" vs "Local" RAG
───────────────────────────────────────
Local RAG  (demos 1-6):  YOU manage everything.
  • Load docs → chunk → embed → store in Chroma → query

Managed RAG (this demo): The cloud manages storage, chunking & vectorization.
  • Upload docs in the Azure Portal UI → Azure AI Search indexes them
  • Your app just sends a query and gets back relevant chunks

Your code no longer needs:  ChromaDB, embedding models, chunking logic.
Azure AI Search handles all of that as a managed service.

GRAPH
─────
  START → retrieve (Azure AI Search) → generate (LLM) → END

SETUP (Azure Portal)
──────────────────────

1. Create azure storage account
2. Creating a container in the storage account
3. Open the documents container. Open container, click Upload and select at least one file to upload (it should be less than 16MB in size).
4. Create an Azure AI Search resource, set "Free" pricing tier for testing.
5. Go to https://ai.azure.com/ create a new Azure AI Foundry project 
6. Go to your foundry project -> "Models + endpoints" -> "Deploy model" -> "Deploy base model" for embeddings and deploy it. You can use the "text-embedding-3-small" model for this demo.
7. Deploy a GPT-4o-mini or gpt-5-mini model for generation by following the same process as above. Note the deployment name you choose, you'll need it in the .env file.
8. Open your Azure AI Search resource -> "Import data (new)" → "Azure Blob Storage" → RAG -> connect to your storage account and select the container where you uploaded your documents.
8.1 Choose "Kind" as "Microsoft Foundry", select the correct project foundry and model, which you have created. Then default options for rest and "Create". 
9. After finishing 8.1 azure opens "Search explorer" view, where can test your search index. You can use the "Search" field to enter a query and see the retrieved chunks from your documents.
10. Collect values for Python code, listed below, and add them to your .env file.
11. Run the Python code in this demo to see managed RAG in action!

ENV (.env file)
──────────────
  AZURE_SEARCH_SERVICE_NAME=my-search-service   # just the name of your search service, not the full URL
  AZURE_SEARCH_INDEX_NAME=my-index        # the name you gave your index in the Azure Portal when connecting to your blob storage
  AZURE_SEARCH_API_KEY=...                # this you get from the Azure Portal → your Search resource → Settings → Keys -> "Admin key"
  AZURE_SEARCH_CONTENT_FIELD=content      # text field in your index (default: content)
  AZURE_SEARCH_TOP_K=3                    # number of chunks to retrieve (default: 3)

  AZURE_OPENAI_API_KEY=...                                 # this you get from the foundry generation model deployment page
  AZURE_OPENAI_ENDPOINT=https://my-hub.openai.azure.com/   # this you get from the foundry generation model deployment page
  AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini     # your deployment name in Azure AI Foundry
  AZURE_OPENAI_API_VERSION=2024-12-01-preview  # this is the latest API version as of June 2024, check Azure docs for updates (13.3.2026 no update)
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

AZURE_SERVICE    = os.environ["AZURE_SEARCH_SERVICE_NAME"]
AZURE_INDEX      = os.environ["AZURE_SEARCH_INDEX_NAME"]
AZURE_KEY        = os.environ["AZURE_SEARCH_API_KEY"]
CONTENT_FIELD    = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
TOP_K            = int(os.getenv("AZURE_SEARCH_TOP_K", "3"))

AOAI_ENDPOINT    = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY         = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_DEPLOYMENT  = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ─── Managed retriever ────────────────────────────────────────────────────────
#
# This is the whole point of "managed" RAG.
# Instead of a local Chroma vector store, we point at an Azure AI Search index.
# Azure handles embedding at query time if semantic/vector search is configured.
#
retriever = AzureAISearchRetriever(
    service_name=AZURE_SERVICE,
    index_name=AZURE_INDEX,
    api_key=AZURE_KEY,
    content_key=CONTENT_FIELD,   # the field in your index that contains the text
    top_k=TOP_K,
)

# ─── LLM (Azure OpenAI via Azure AI Foundry) ─────────────────────────────────
#
# AzureChatOpenAI connects to a model deployment in your Azure AI Foundry hub.
# The deployment name is set in the Portal — it's not the model name itself.
#
llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    azure_deployment=AOAI_DEPLOYMENT,
    api_version=AOAI_API_VERSION,
)

# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str           # user question
    context: list[str]   # chunks returned by Azure AI Search
    answer: str          # final LLM response


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: State) -> dict:
    """
    Query Azure AI Search.
    Azure handles the vector search / semantic ranking server-side.
    We receive ready-to-use text chunks — no local embedding required.
    """
    docs = retriever.invoke(state["query"])
    return {"context": [doc.page_content for doc in docs]}


def generate(state: State) -> dict:
    """Generate an answer grounded in the retrieved chunks."""
    context_block = "\n\n---\n\n".join(state["context"])

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the context doesn't contain enough information, say so clearly."
        )),
        HumanMessage(content=(
            f"Context:\n\n{context_block}\n\n"
            f"Question: {state['query']}"
        )),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


# ─── Graph ────────────────────────────────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = input("Ask a question about your documents: ").strip()

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("-" * 60)

    result = graph.invoke({"query": query})

    print(f"Retrieved {len(result['context'])} chunk(s) from Azure AI Search:\n")
    for i, chunk in enumerate(result["context"], 1):
        preview = chunk[:200].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    print(f"\nAnswer:\n{result['answer']}")
