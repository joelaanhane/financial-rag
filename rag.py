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
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings, dtype="float32")

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, chunks, index, k=5):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def answer_question(query, chunks, index):
    relevant_chunks = search(query, chunks, index)
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer questions based only on the provided context. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

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
    text = load_pdf("annual_report.pdf")
    print(f"PDF loaded: {len(text)} characters")
    chunks = chunk_text(text)
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