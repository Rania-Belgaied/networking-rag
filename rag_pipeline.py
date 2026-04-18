"""
Networking RAG Pipeline
=======================
A Retrieval-Augmented Generation system for networking knowledge.

Components:
- Embedding model : all-MiniLM-L6-v2 (sentence-transformers)
- Vector database : ChromaDB (persistent local storage)
- Reranker        : cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM             : Mistral (mistral-small-latest)
"""

import os
import requests
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS     = {
    "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
    "Content-Type":  "application/json"
}
RERANKER_THRESHOLD = -3      # chunks below this score are considered irrelevant
CHUNK_SIZE         = 200     # words per chunk
CHUNK_OVERLAP      = 50      # shared words between consecutive chunks
N_RETRIEVE         = 5       # candidates to retrieve from ChromaDB
N_FINAL            = 2       # chunks to keep after reranking

# ── Models ───────────────────────────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Vector Database ───────────────────────────────────────────────────────────

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="networking_docs")

# ── Document Loading and Chunking ─────────────────────────────────────────────

def load_and_chunk(filepath, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Load a text file and split it into overlapping word-based chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    words  = text.split()
    chunks = []
    i      = 0

    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def index_document(filepath):
    """
    Embed and store a document in ChromaDB.
    Skips indexing if the collection is already populated.
    """
    if collection.count() > 0:
        print(f"Knowledge base loaded from disk — {collection.count()} chunks")
        return

    chunks     = load_and_chunk(filepath)
    embeddings = embedder.encode(chunks).tolist()
    source     = os.path.basename(filepath)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": source, "chunk_index": i}
                   for i in range(len(chunks))]
    )
    print(f"Knowledge base created — {collection.count()} chunks indexed")


# ── Retrieval and Reranking ───────────────────────────────────────────────────

def retrieve_and_rerank(query, n_retrieve=N_RETRIEVE, n_final=N_FINAL):
    """
    Retrieve candidate chunks from ChromaDB then rerank them.

    1. Embed the query and fetch n_retrieve nearest chunks
    2. Score each (query, chunk) pair with the cross-encoder reranker
    3. Return the top n_final chunks sorted by reranker score
    """
    # Step 1 — vector search
    n_retrieve     = min(n_retrieve, collection.count())
    query_embedding = embedder.encode([query]).tolist()
    results         = collection.query(
        query_embeddings=query_embedding,
        n_results=n_retrieve
    )

    candidates = results["documents"][0]
    metadatas  = results["metadatas"][0]

    # Step 2 — rerank
    pairs  = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)

    # Step 3 — sort and truncate
    ranked = sorted(
        zip(scores, candidates, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    top_chunks    = [item[1] for item in ranked[:n_final]]
    top_metadatas = [item[2] for item in ranked[:n_final]]
    top_scores    = [float(item[0]) for item in ranked[:n_final]]

    return top_chunks, top_metadatas, top_scores


# ── RAG Answer ────────────────────────────────────────────────────────────────

def rag_answer(question):
    """
    Answer a question using the RAG pipeline.

    Returns:
        answer  (str)   : the model's response
        sources (list)  : chunk IDs used to generate the answer
        scores  (list)  : reranker scores for each source chunk
    """
    chunks, metadatas, scores = retrieve_and_rerank(question)

    # Refuse if no chunk is relevant enough
    if max(scores) < RERANKER_THRESHOLD:
        return "I don't have information on that topic.", [], scores

    context = "\n\n".join(chunks)
    system  = (
        "You are a networking assistant.\n"
        "Use the context provided below to answer the user's question.\n"
        "You may reason and infer from the context.\n"
        "If the topic is completely absent from the context, "
        "say: 'I don't have information on that topic.'\n"
        "Do not introduce facts not supported by the context.\n\n"
        "Context:\n" + context
    )

    response = requests.post(MISTRAL_URL, headers=HEADERS, json={
        "model":    "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question}
        ]
    })

    if response.status_code != 200:
        error_msg = response.json().get("message", "Unknown error")
        return f"API error {response.status_code}: {error_msg}", [], scores

    answer  = response.json()["choices"][0]["message"]["content"]
    sources = [f"chunk_{m['chunk_index']}" for m in metadatas]

    return answer, sources, scores


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Index the knowledge base (runs once, then loads from disk)
    index_document("data/networking_guide.txt")

    # Demo questions
    questions = [
        "What is the difference between OSPF and BGP?",
        "How does IPSec work?",
        "What happens at layer 2 of the OSI model?",
        "What is the difference between stateful and next-generation firewalls?",
        "What is the capital of France?",
    ]

    for question in questions:
        answer, sources, scores = rag_answer(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"   Sources : {sources}")
        print(f"   Scores  : {[round(s, 4) for s in scores]}")
        print()