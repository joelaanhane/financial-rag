from dotenv import load_dotenv
import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag import load_pdf, chunk_text, get_embeddings, build_index, search

load_dotenv()

DOCUMENTS = {
    "ing": "ing_2024.pdf",
    "abnamro": "abnamro_2024.pdf",
    "rabobank": "rabobank_2024.pdf"
}

def load_or_create_index(name, path):
    """Load saved embeddings and chunks from disk, or create them from a PDF.
    
    Args:
        name: Short name for the document, used as filename prefix.
        path: Path to the PDF file.
    Returns:
        Tuple of (chunks, faiss_index).
    """
    embeddings_file = f"{name}_embeddings.npy"
    chunks_file = f"{name}_chunks.pkl"
    
    if os.path.exists(embeddings_file) and os.path.exists(chunks_file):
        print(f"Loading saved embeddings for {name}...")
        embeddings = np.load(embeddings_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
    else:
        print(f"Processing {path}...")
        pages = load_pdf(path)
        chunks = chunk_text(pages)
        print(f"Creating embeddings for {name}... (this may take a while)")
        embeddings = get_embeddings(chunks)
        np.save(embeddings_file, embeddings)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
    
    index = build_index(embeddings)
    return chunks, index

print("Loading all documents...")
indices = {}
for name, path in DOCUMENTS.items():
    chunks, index = load_or_create_index(name, path)
    indices[name] = {"chunks": chunks, "index": index}
print("All documents loaded.\n")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def search_ing(query: str) -> str:
    """Search the ING 2024 annual report for information."""
    results = search(query, indices["ing"]["chunks"], indices["ing"]["index"])
    return "\n\n".join([f"Page {c['page']}: {c['text']}" for c in results])

@tool
def search_abnamro(query: str) -> str:
    """Search the ABN AMRO 2024 annual report for information."""
    results = search(query, indices["abnamro"]["chunks"], indices["abnamro"]["index"])
    return "\n\n".join([f"Page {c['page']}: {c['text']}" for c in results])

@tool
def search_rabobank(query: str) -> str:
    """Search the Rabobank 2024 annual report for information."""
    results = search(query, indices["rabobank"]["chunks"], indices["rabobank"]["index"])
    return "\n\n".join([f"Page {c['page']}: {c['text']}" for c in results])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_ing, search_abnamro, search_rabobank]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst with access to the 2024 annual reports of ING, ABN AMRO, and Rabobank. Use the search tools to find relevant information before answering. Always cite which bank and page number your information comes from."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("AnnualSight Agent ready. Type your question or 'quit' to exit.\n")

    while True:
        query = input("Question: ")
        if query.lower() == "quit":
            break
        result = executor.invoke({"input": query})
        print(f"\nAnswer: {result['output']}\n")