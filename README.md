# AnnualSight — AI-Powered Financial Document Analyser

A Retrieval-Augmented Generation (RAG) application that lets you upload any annual report (PDF) and ask questions about it in plain English.

## What it does

Upload an annual report, then ask questions like:

- *"What was the net profit in 2024?"*
- *"What are the biggest risk factors?"*
- *"How did the CET1 ratio change compared to last year?"*

AnnualSight retrieves the most relevant sections from the document and uses an LLM to formulate a precise, source-grounded answer with page citations.

## Features

- **Upload any PDF** — not limited to specific banks or years
- **Multi-report analysis** — select multiple reports to compare across banks or years
- **Source transparency** — every answer shows the exact chunks used as context
- **Duplicate detection** — canonical name matching prevents the same bank being stored twice under different spellings

## Architecture

```
Upload PDF  →  Extract text (PyMuPDF)
            →  Chunk text (500 words, 50 overlap)
            →  Embed chunks (OpenAI text-embedding-3-small)
            →  Store to disk (embeddings + chunks + JSON metadata)

Ask question →  Embed question
             →  FAISS similarity search (top-k chunks)
             →  Generate answer (gpt-4o-mini, context-grounded)
             →  Return answer with page citations
```

## Project structure

```
financial-rag/
├── app.py              # Streamlit homepage
├── pages/
│   ├── 1_Upload.py     # Upload and process PDF reports
│   ├── 2_Library.py    # Browse stored reports
│   └── 3_Analyse.py    # Ask questions across reports
├── rag.py              # Core RAG pipeline
├── storage.py          # Report storage and retrieval layer
└── reports/            # Processed reports stored here
    └── {year}/
        └── {bank}/
            ├── chunks.pkl
            ├── embeddings.npy
            └── meta.json
```

## Tech stack

| Component | Technology |
|---|---|
| PDF extraction | PyMuPDF |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector search | FAISS (IndexFlatL2) |
| Answer generation | OpenAI `gpt-4o-mini` |
| UI | Streamlit |
| Storage | NumPy, pickle, JSON |

## Setup

```bash
git clone https://github.com/joelaanhane/financial-rag.git
cd financial-rag
python -m venv venv
venv\Scripts\activate
pip install streamlit openai pymupdf faiss-cpu numpy python-dotenv
```

Add your OpenAI API key to a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

Run the app:

```bash
streamlit run app.py
```

## Usage

1. Go to **Upload** — select a bank name, year, and PDF file, then click *Process and save*
2. Go to **Library** — view all stored reports with page and chunk counts
3. Go to **Analyse** — select one or more reports, type a question, click *Ask*

## Example

```
Question: What was Rabobank's net profit in 2024?

Answer: Rabobank reported a net profit of EUR 5,163 million in 2024.
        [Rabobank 2024, Page 42]

Source chunks used:
  Rabobank 2024 — Page 42
  Rabobank 2024 — Page 43
  Rabobank 2024 — Page 89
```
