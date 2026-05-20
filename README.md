# AnnualSight — AI-Powered Financial Document Analyser

A Retrieval-Augmented Generation (RAG) pipeline that enables natural language querying of annual reports. Built to explore how LLMs can be applied to financial document analysis.

## What it does

Upload any annual report (PDF) and ask questions in plain English:

- *"What are the biggest risk factors?"*
- *"What was the net profit in 2024?"*
- *"What is the strategy for climate risk?"*

The tool retrieves the most relevant sections from the document and uses an LLM to formulate a precise, source-grounded answer.

## How it works

1. **PDF ingestion** — extracts raw text using PyMuPDF
2. **Chunking** — splits text into 500-word segments with 50-word overlap
3. **Embeddings** — converts each chunk into a 1536-dimensional vector using OpenAI `text-embedding-3-small`
4. **Vector search** — stores and queries embeddings using FAISS (Facebook AI Similarity Search)
5. **Answer generation** — retrieves top 5 relevant chunks and passes them as context to `gpt-4o-mini`

## Tech stack

- Python
- OpenAI API (embeddings + chat completions)
- FAISS
- PyMuPDF
- NumPy

## Setup

```bash
git clone https://github.com/joelaanhane/financial-rag.git
cd financial-rag
python -m venv venv
venv\Scripts\activate
pip install openai pymupdf faiss-cpu numpy python-dotenv
```

Add your OpenAI API key to a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

Add an annual report PDF to the project folder as `annual_report.pdf`, then run:
```bash
python rag.py
```

## Example

```
AnnualSight ready. Type your question or 'quit' to exit.

Question: What are the biggest risk factors?

Answer: The biggest risk factors in 2024 for ING include:
1. Geopolitical risk
2. People risk
3. Cybercrime
4. Inflation risk
5. IT risk
6. Model risk

Sources: pages [159, 161, 214, 428]
```