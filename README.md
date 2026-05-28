# AnnualSight — AI-Powered Financial Document Analyser

A Retrieval-Augmented Generation (RAG) pipeline that enables natural language querying of annual reports. Built to explore how LLMs can be applied to financial document analysis.

## What it does

Upload any annual report (PDF) and ask questions in plain English:

- *"What are the biggest risk factors?"*
- *"What was the net profit in 2024?"*
- *"What is the strategy for climate risk?"*

The tool retrieves the most relevant sections from the document and uses an LLM to formulate a precise, source-grounded answer.

There are three modes of use:

| Mode | File | Description |
|---|---|---|
| Single-document RAG | `rag.py` | Query one annual report |
| Multi-document agent | `agent.py` | Query and compare ING, ABN AMRO, and Rabobank simultaneously |
| Evaluation pipeline | `evaluate.py` | Automatically assess answer quality with LLM-as-a-judge |

## How it works

### Single-document RAG (`rag.py`)

1. **PDF ingestion** — extracts raw text using PyMuPDF
2. **Chunking** — splits text into 500-word segments with 50-word overlap
3. **Embeddings** — converts each chunk into a 1536-dimensional vector using OpenAI `text-embedding-3-small`
4. **Vector search** — stores and queries embeddings using FAISS (Facebook AI Similarity Search)
5. **Answer generation** — retrieves top 5 relevant chunks and passes them as context to `gpt-4o-mini`

### Multi-document agent (`agent.py`)

Extends the RAG pipeline to support cross-document analysis using a LangChain agent:

1. **Multi-index loading** — builds separate FAISS indexes for ING, ABN AMRO, and Rabobank (with disk caching)
2. **Tool-equipped agent** — each bank's index is exposed as a LangChain tool (`search_ing`, `search_abnamro`, `search_rabobank`)
3. **Reasoning loop** — a `gpt-4o-mini` agent autonomously decides which banks to query and synthesises a final answer with source citations

### Evaluation pipeline (`evaluate.py`)

Assesses answer quality using LLM-as-a-judge on a fixed test set:

1. **Test cases** — predefined questions with known expected answers
2. **Grounding check** — an LLM judge determines whether each answer is supported by the retrieved context
3. **Correctness check** — a second LLM judge compares the answer against the expected answer semantically
4. **Summary report** — prints pass rates for both grounding and correctness

## Tech stack

- Python
- OpenAI API (embeddings + chat completions)
- LangChain (agent framework, tool use)
- FAISS
- PyMuPDF
- NumPy

## Setup

```bash
git clone https://github.com/joelaanhane/financial-rag.git
cd financial-rag
python -m venv venv
venv\Scripts\activate
pip install openai pymupdf faiss-cpu numpy python-dotenv langchain langchain-openai
```

Add your OpenAI API key to a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

### Single-document RAG

Add an annual report PDF to the project folder as `annual_report.pdf`, then run:
```bash
python rag.py
```

### Multi-document agent

Add the three Dutch bank annual reports as `ing_2024.pdf`, `abnamro_2024.pdf`, and `rabobank_2024.pdf`, then run:
```bash
python agent.py
```

### Evaluation

```bash
python evaluate.py
```

## Examples

### Single-document RAG

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

### Multi-document agent

```
AnnualSight Agent ready. Type your question or 'quit' to exit.

Question: How does ING's CET1 ratio compare to ABN AMRO's?

Answer: ING reported a CET1 ratio of 13.6% (page 45), while ABN AMRO reported
14.3% (page 38), putting ABN AMRO slightly above ING on this capital metric.
```

### Evaluation

```
Question: What was Rabobank's net profit in 2024?
Answer: Rabobank reported a net profit of EUR 5,163 million in 2024.
Grounded: GROUNDED
Contains expected: CORRECT
--------------------------------------------------

Results: 3/3 correct
Grounded: 3/3 grounded in source documents
```
