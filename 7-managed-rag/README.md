# Managed RAG — Azure AI Search + LangGraph

A minimal demonstration of **Managed RAG**: using Azure cloud services to handle document storage, chunking, and vectorization, while LangGraph orchestrates the retrieval and generation pipeline.

## What is Managed RAG?

In the earlier demos (1–6) you managed the entire RAG pipeline yourself:

```
Load docs → chunk → embed → store in ChromaDB → query
```

In Managed RAG, Azure handles all of that:

```
Upload docs in the Portal → Azure AI Search indexes, chunks & vectorizes them
Your app just queries the index and gets back relevant text chunks
```

Your Python code no longer needs ChromaDB, an embedding model, or chunking logic. The cloud is the database.

## How it works

The LangGraph pipeline has two nodes:

```
START → retrieve → generate → END
```

| Node | What it does |
|---|---|
| `retrieve` | Sends the user query to Azure AI Search. Azure runs vector search server-side and returns the most relevant text chunks. |
| `generate` | Passes the retrieved chunks as context to an Azure OpenAI model, which produces a grounded answer. |

The key insight is in the `retrieve` node — there is no local embedding call. Azure AI Search does the vectorization using the embedding model you deployed in Azure AI Foundry.

## Setup

### 1. Create an Azure Storage Account

Create a storage account in the Azure Portal. Inside it, create a **container** and upload at least one document (PDF, DOCX, TXT — must be under 16 MB).

### 2. Create an Azure AI Search resource

In the Azure Portal, create an Azure AI Search resource. The **Free** pricing tier is sufficient for testing.

### 3. Create an Azure AI Foundry project

Go to [https://ai.azure.com/](https://ai.azure.com/) and create a new project.

### 4. Deploy an embedding model

In your Foundry project, go to **Models + endpoints** → **Deploy model** → **Deploy base model**.
Deploy `text-embedding-3-small` (or similar). This model will vectorize your documents and queries inside Azure AI Search.

### 5. Deploy a generation model

In the same section, deploy a chat model such as `gpt-4o-mini` or `gpt-5-mini`.
**Note the deployment name you choose** — you will need it in the `.env` file.

### 6. Connect your documents to Azure AI Search

In your Azure AI Search resource, click **Import data (new)** → **Azure Blob Storage** → **RAG**.
Connect to the storage account and container from step 1.

When prompted for "Kind", choose **Microsoft Foundry**, then select your Foundry project and the embedding model you deployed. Accept the remaining defaults and click **Create**.

### 7. Verify with Search Explorer

After the import finishes, Azure opens the **Search Explorer**. Enter a query to confirm that your documents were indexed and that relevant chunks are being returned.

### 8. Configure the Python project

Copy `.env.example` to `.env` and fill in your values:

```env
# Azure AI Search — Portal → your Search resource → Settings → Keys
AZURE_SEARCH_SERVICE_NAME=your-search-service-name
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_SEARCH_API_KEY=...                  # Settings → Keys → Admin key

# Text field name in your index schema (check the Portal, often "chunk")
AZURE_SEARCH_CONTENT_FIELD=chunk
AZURE_SEARCH_TOP_K=3

# Azure OpenAI — Foundry → your deployment → details page
AZURE_OPENAI_ENDPOINT=https://your-hub.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini       # the name you gave the deployment
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### 9. Install dependencies and run

```bash
pip install -r requirements.txt
python demo1-managed-rag.py
```

You will be prompted to enter a question about your documents.

## Project structure

```
7-managed-rag/
├── demo1-managed-rag.py   # main demo
├── .env.example           # configuration template
├── requirements.txt       # Python dependencies
└── README.md
```
