from dotenv import load_dotenv
import os
import fitz
import numpy as np
import faiss
from openai import OpenAI
import pickle

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_pdf(path):
    """Load a PDF file and extract text per page with page numbers.
    
    Args:
        path: Path to the PDF file.
    Returns:
        List of dicts with 'text' and 'page_num' keys.
    """
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({"text": text, "page_num": page_num + 1})
    return pages

def chunk_text(pages, chunk_size=500, overlap=50):
    """Split pages into overlapping chunks of chunk_size words.
    
    Args:
        pages: List of page dicts from load_pdf.
        chunk_size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.
    Returns:
        List of dicts with 'text' and 'page' keys.
    """
    chunks = []
    for page in pages:
        words = page["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append({"text": chunk, "page": page["page_num"]})
    return chunks

def get_embeddings(chunks):
    """Convert a list of text chunks into OpenAI embedding vectors.
    
    Args:
        chunks: List of chunk dicts with 'text' key.
    Returns:
        Numpy array of shape (n_chunks, 1536) with float32 embeddings.
    """
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings, dtype="float32")

def build_index(embeddings):
    """Build a FAISS index from a numpy array of embeddings.
    
    Args:
        embeddings: Numpy array of shape (n_chunks, n_dimensions).
    Returns:
        FAISS IndexFlatL2 index with all embeddings added.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, chunks, index, k=5):
    """Search the FAISS index for the k most relevant chunks.
    
    Args:
        query: Search query string.
        chunks: List of chunk dicts with 'text' and 'page' keys.
        index: FAISS index built from the same chunks.
        k: Number of results to return.
    Returns:
        List of k most relevant chunk dicts.
    """
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def answer_question(query, chunks, index):
    """Answer a question using RAG and return the answer with source pages.
    
    Args:
        query: Question string.
        chunks: List of chunk dicts.
        index: FAISS index.
    Returns:
        Answer string with source page numbers appended.
    """
    relevant_chunks = search(query, chunks, index)
    context = "\n\n".join([c["text"] for c in relevant_chunks])
    pages = sorted(set([c["page"] for c in relevant_chunks]))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer questions based only on the provided context. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message.content
    return f"{answer}\n\nSources: pages {pages}"

if __name__ == "__main__":
    EMBEDDINGS_FILE = "embeddings.npy"
    CHUNKS_FILE = "chunks.pkl"

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading saved embeddings...")
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks")
    else:
        print("Processing PDF...")
        pages = load_pdf("annual_report.pdf")
        chunks = chunk_text(pages)
        print(f"Number of chunks: {len(chunks)}")
        print("Creating embeddings... (this may take a while)")
        embeddings = get_embeddings(chunks)
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        print("Embeddings saved")

    index = build_index(embeddings)
    print("FAISS index built")

    print("\nAnnualSight ready. Type your question or 'quit' to exit.\n")

    while True:
        query = input("Question: ")
        if query.lower() == "quit":
            break
        answer = answer_question(query, chunks, index)
        print(f"\nAnswer: {answer}\n")