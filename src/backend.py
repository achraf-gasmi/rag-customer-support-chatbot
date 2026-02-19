import chromadb
from openai import OpenAI
import pandas as pd

# ── Ollama client ─────────────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL      = "llama3"
SIMILARITY_THRESHOLD = 0.75

# ── ChromaDB setup ────────────────────────────────────────────────────
chroma_client     = chromadb.PersistentClient(path="data/chroma_db")
collection        = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def load_knowledge_base():
    """Embed and store knowledge base into ChromaDB if not already done."""
    if collection.count() > 0:
        print(f"ChromaDB already has {collection.count()} documents, skipping embedding.")
        return

    print("Embedding knowledge base into ChromaDB...")
    df = pd.read_csv("data/knowledge_base.csv")

    # Embed all documents in one call
    texts = df["query_text"].tolist()
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    embeddings = [e.embedding for e in response.data]

    # Store in ChromaDB
    collection.add(
        ids=[str(row["document_id"]) for _, row in df.iterrows()],
        embeddings=embeddings,
        documents=df["response_text"].tolist(),
        metadatas=[
            {"category": row["category"], "intent": row["intent"]}
            for _, row in df.iterrows()
        ]
    )
    print(f"Stored {collection.count()} documents in ChromaDB.")

def search_knowledge_base(query_text):
    """Search ChromaDB for the most relevant response."""
    # Embed the query
    response = client.embeddings.create(
        input=[query_text],
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding

    # Search ChromaDB for top 3 results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    top_response   = results["documents"][0][0]
    top_distance   = results["distances"][0][0]
    top_metadata   = results["metadatas"][0][0]

    # ChromaDB cosine distance is 0-2, convert to similarity 0-1
    confidence_score = round(1 - (top_distance / 2), 4)

    return {
        "response":         top_response,
        "confidence_score": confidence_score,
        "category":         top_metadata["category"],
        "intent":           top_metadata["intent"],
        "source":           "knowledge_base" if confidence_score >= SIMILARITY_THRESHOLD else "llm"
    }

def generate_llm_response(query_text, conversation_history):
    """Generate a response using Ollama when no good match is found."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant for ShopEase, "
                "an e-commerce company. Answer clearly and concisely."
            )
        }
    ]
    messages += conversation_history
    messages.append({"role": "user", "content": query_text})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def get_response(query_text, conversation_history):
    """Main function: search knowledge base, fall back to LLM if needed."""
    result = search_knowledge_base(query_text)

    if result["source"] == "knowledge_base":
        # Good match found in ChromaDB
        retrieved_response = result["response"]
    else:
        # No good match — generate with LLM
        retrieved_response = generate_llm_response(query_text, conversation_history)
        result["category"] = "UNKNOWN"
        result["intent"]   = "unknown"

    result["response"] = retrieved_response

    # Update conversation history
    conversation_history.append({"role": "user",      "content": query_text})
    conversation_history.append({"role": "assistant",  "content": retrieved_response})

    return result

if __name__ == "__main__":
    load_knowledge_base()

    # Quick test
    history = []
    test_queries = [
        "How do I cancel my order?",                      # should match knowledge base
        "When can I talk to someone from support?",       # paraphrased - semantic match
        "What is your return policy for damaged items?"   # might trigger LLM fallback
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        result = get_response(query, history)
        print(f"A: {result['response']}")
        print(f"Source: {result['source']} | Confidence: {result['confidence_score']} | Category: {result['category']}")

