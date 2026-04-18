# Networking RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that answers networking 
questions grounded in a local knowledge base — no hallucination, 
with source attribution.

## Architecture

User question
↓
Embedding model (all-MiniLM-L6-v2)
↓
ChromaDB vector search → top 5 candidate chunks
↓
Cross-encoder reranker → top 2 most relevant chunks
↓
Mistral LLM with injected context
↓
Grounded answer + source attribution

## Stack

| Component | Technology |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB (persistent local storage) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Mistral API (mistral-small-latest) |

## Setup

1. Clone the repository
2. Install dependencies : `pip install -r requirements.txt`
3. Create a `.env` file with your Mistral API key : 
MISTRAL_API_KEY=your_key_here
4. Run the pipeline : `python rag_pipeline.py`

The knowledge base is indexed automatically on first run and 
persisted to disk for all subsequent runs.

## Example Output

Q: What is the difference between OSPF and BGP?
A: OSPF is an interior gateway protocol used within a single
autonomous system. BGP is an exterior gateway protocol used
between autonomous systems on the internet...
Sources : ['chunk_2', 'chunk_5']
Scores  : [5.6066, -8.2376]
Q: What is the capital of France?
A: I don't have information on that topic.
Sources : []
Scores  : [-10.7774, -11.1081]

## Key Design Decisions

**Why RAG instead of fine-tuning?** RAG updates instantly when 
documents change, requires no retraining, and provides source 
attribution — fine-tuning would bake knowledge into the model 
with no traceability.

**Why a reranker?** Embedding similarity captures general semantic 
proximity but misses precise relevance. The cross-encoder reranker 
reads the query and chunk together, giving significantly more 
accurate relevance scores.

**Why a score threshold?** Prevents the model from hallucinating 
answers when no retrieved chunk is actually relevant to the question.
