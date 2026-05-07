# n8n Product Knowledge Assistant

A source-grounded RAG assistant for answering questions about selected n8n product documentation.

**Live demo:** [n8n Docs Assistant](https://proknowledgeassistant.streamlit.app/)

> Portfolio demo built on selected n8n documentation. Not affiliated with n8n.

## Overview

This project is a Retrieval-Augmented Generation (RAG) demo built to answer product documentation questions using selected n8n docs as the knowledge base.

The assistant retrieves relevant documentation passages, ranks evidence using hybrid retrieval, and generates concise answers with visible source references. It is designed as a portfolio project to demonstrate practical applied AI engineering skills, including document preprocessing, retrieval, prompt design, LLM integration, streaming responses, and deployment.

## Key Features
- **Metadata-enriched embeddings** using document titles, heading paths, categories, and source paths to improve retrieval quality
- **Hybrid retrieval** using dense semantic search and sparse keyword-based retrieval
- **Source-grounded answers** with relevant documentation shown beside each response
- **Streaming generation** for a faster user experience
- **Follow-up question handling** for short context-dependent queries
- **Multiple LLM modes**:
  - OpenAI-compatible API backend
  - Local Ollama backend through HTTP API
  - Retrieval-only fallback mode
- **Lightweight domain-aware retrieval rules** to improve source selection for common n8n topics such as environment variables, installation, permissions, and AI features
- **Streamlit Cloud deployment** with Python 3.10

## Example Questions

You can try questions such as:

- How can I configure environment variables in n8n?
- How about database?
- How can users manage permissions?
- What AI features does n8n provide?
- How does source control work in n8n?

## Tech Stack

- Python 3.10
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- BM25 sparse retrieval
- OpenAI-compatible API
- Local Ollama support via HTTP API
- Git / GitHub

## Architecture

The application follows a modular RAG pipeline:

1. **Document preprocessing**  
   Selected n8n Markdown documentation is cleaned, chunked, and enriched with metadata.

2. **Metadata-enriched vector indexing**  
   Cleaned chunks are embedded together with document titles, heading paths, categories, and source paths. The original display content is stored separately for LLM context and source preview.

3. **Hybrid retrieval**  
   User questions are matched against the documentation using dense semantic retrieval and sparse keyword retrieval.

4. **Source prioritization**  
   Retrieved passages are filtered and reranked using lightweight domain-aware rules.

5. **Answer generation**  
   A source-grounded prompt is sent to the configured LLM backend.

6. **UI display**  
   The Streamlit app displays the answer and relevant sources side by side, with streaming output enabled.

## Project Structure

```text
.
├── app.py                         # Streamlit UI
├── rag_pipeline.py                # Retrieval, reranking, prompt construction, and response logic
├── llm_client.py                  # LLM backend abstraction
├── config.py                      # Project configuration
├── build_vectorstore.py           # Documentation preprocessing and vectorstore creation
├── retrieval/
│   ├── dense_retriever.py         # Chroma-based dense retrieval
│   ├── sparse_retriever.py        # BM25 sparse retrieval
│   ├── hybrid_retriever.py        # Hybrid retrieval combination
│   └── source_metadata.py         # Source type and scope metadata helpers
├── data/
│   ├── raw/
│   │   └── n8n_selected_docs/     # Curated subset of n8n documentation
│   └── processed/
│       └── n8n_chunks.json         # Cleaned chunks for inspection
├── vectorstore/                   # Local Chroma vector database
├── assets/                        # CSS and UI assets
└── requirements.txt
```

## Runtime

This project was developed and tested with Python 3.10.

## Environment Variables

For local development, create a `.env` file in the project root:

```env
LLM_BACKEND=openai_compatible
OPENAI_API_KEY=your_openai_api_key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_MODEL=gpt-4o-mini
```

Do not commit `.env` or API keys to GitHub.

The code also supports a local Ollama backend for development, configured through `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository includes a curated subset of n8n documentation and a prebuilt Chroma vector store. To rebuild the vector store, run:

```bash
python build_vectorstore.py
```

Run the app:

```bash
streamlit run app.py
```


## Evaluation and Quality Checks

This project includes several practical quality-control mechanisms:

- Retrieved sources are displayed beside generated answers for transparency.
- The assistant is instructed to answer only from retrieved documentation context.
- Lightweight domain-aware retrieval rules help reduce tangential sources for common documentation topics.
- Follow-up questions are rewritten into standalone retrieval queries when possible.
- The UI exposes technical diagnostics such as backend, retrieval weights, and response time.

## Project Purpose

This project is an independent applied AI portfolio demo built with selected n8n documentation. It demonstrates how product documentation can be transformed into a searchable knowledge assistant using RAG, source-aware retrieval, and LLM-based answer generation.

It also shows an end-to-end workflow covering data preparation, retrieval design, LLM integration, user-facing interface design, and cloud deployment.