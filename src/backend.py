import os
import networkx as nx
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableSequence

# ── Models ────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "qwen3-embedding:8b"
CHAT_MODEL           = "qwen3:8b"
SIMILARITY_THRESHOLD = 0.70
TOP_K                = 10
MAX_MEMORY_MESSAGES  = 6

# ── LangChain Ollama setup ────────────────────────────────────────────
llm = OllamaLLM(
    model=CHAT_MODEL,
    temperature=0.5
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# ── Simple conversation memory ────────────────────────────────────────
conversation_memory = []

# ── ChromaDB via LangChain ────────────────────────────────────────────
vectorstore = None

def load_knowledge_base():
    global vectorstore

    if os.path.exists("data/chroma_db") and len(os.listdir("data/chroma_db")) > 0:
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )
        print(f"Loaded {vectorstore._collection.count()} documents.")
        return

    print("Embedding knowledge base into ChromaDB...")
    df = pd.read_csv("data/knowledge_base.csv")

    docs = [
        Document(
            page_content=row["response_text"],
            metadata={
                "query_text":  row["query_text"],
                "category":    row["category"],
                "intent":      row["intent"],
                "document_id": str(row["document_id"])
            }
        )
        for _, row in df.iterrows()
    ]

    # Embed in batches
    batch_size = 50
    all_docs   = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        all_docs.extend(batch)
        print(f"Prepared {min(i+batch_size, len(docs))}/{len(docs)} documents...")

    vectorstore = Chroma.from_documents(
        documents         = all_docs,
        embedding         = embeddings,
        persist_directory = "data/chroma_db",
        collection_name   = "knowledge_base"
    )
    print(f"Stored {vectorstore._collection.count()} documents in ChromaDB.")


# ── GraphRAG: Build knowledge graph ──────────────────────────────────
def build_knowledge_graph():
    """Build a graph where categories and intents are nodes with relationships."""
    G = nx.DiGraph()

    category_intent_map = {
        "ACCOUNT":      ["create_account","delete_account","edit_account","switch_account","recover_password","registration_problems"],
        "ORDER":        ["cancel_order","change_order","place_order","track_order"],
        "REFUND":       ["check_refund_policy","get_refund","track_refund"],
        "CONTACT":      ["contact_customer_service","contact_human_agent"],
        "INVOICE":      ["check_invoice","get_invoice"],
        "PAYMENT":      ["check_payment_methods","payment_issue"],
        "FEEDBACK":     ["complaint","review"],
        "DELIVERY":     ["delivery_options","delivery_period"],
        "SHIPPING":     ["change_shipping_address","set_up_shipping_address"],
        "SUBSCRIPTION": ["newsletter_subscription"],
        "CANCEL":       ["check_cancellation_fee"],
    }

    for category, intents in category_intent_map.items():
        G.add_node(category, type="category")
        for intent in intents:
            G.add_node(intent, type="intent")
            G.add_edge(category, intent, relation="contains")

    cross_relations = [
        ("ORDER",   "REFUND",   "may_lead_to"),
        ("ORDER",   "CANCEL",   "may_lead_to"),
        ("ORDER",   "DELIVERY", "related_to"),
        ("ORDER",   "SHIPPING", "related_to"),
        ("PAYMENT", "REFUND",   "related_to"),
        ("PAYMENT", "INVOICE",  "related_to"),
        ("ACCOUNT", "CONTACT",  "related_to"),
        ("REFUND",  "CONTACT",  "related_to"),
    ]
    for src, dst, rel in cross_relations:
        G.add_edge(src, dst, relation=rel)

    return G

knowledge_graph = build_knowledge_graph()

def get_related_categories(category):
    """Use graph to find related categories for broader search."""
    related = set()
    if category in knowledge_graph:
        for neighbor in knowledge_graph.neighbors(category):
            if knowledge_graph.nodes[neighbor].get("type") == "category":
                related.add(neighbor)
        for predecessor in knowledge_graph.predecessors(category):
            if knowledge_graph.nodes[predecessor].get("type") == "category":
                related.add(predecessor)
    return list(related)


# ── Pipeline Steps ────────────────────────────────────────────────────

def rewrite_query(inputs):
    """Step 1: Rewrite query for better search."""
    query   = inputs["query"]
    history = inputs.get("history", "")

    prompt = f"""Rewrite this customer support query to be clear and specific.
Remove pronouns, make it self-contained.

Conversation so far: {history if history else 'None'}
Query: {query}

Rewritten query (return ONLY the rewritten query, no explanation):"""

    rewritten = llm.invoke(prompt).strip()
    # Safety: if model returns something too long, fall back to original
    if len(rewritten) > 300:
        rewritten = query
    print(f"[Rewriter] '{query}' → '{rewritten}'")
    return {**inputs, "rewritten_query": rewritten}


def detect_intent(inputs):
    """Step 2: Classify intent using LLM."""
    query = inputs["rewritten_query"]

    prompt = f"""Classify this customer support query into ONE category:
ACCOUNT, ORDER, REFUND, CONTACT, INVOICE, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL, UNKNOWN

Query: {query}
Return ONLY the category name, nothing else:"""

    category = llm.invoke(prompt).strip().upper()
    valid    = {"ACCOUNT","ORDER","REFUND","CONTACT","INVOICE","PAYMENT",
                "FEEDBACK","DELIVERY","SHIPPING","SUBSCRIPTION","CANCEL","UNKNOWN"}

    # Extract valid category if model returned extra text
    for word in category.split():
        if word in valid:
            category = word
            break
    else:
        category = "UNKNOWN"

    related = get_related_categories(category)
    print(f"[Intent] Category: {category} | Related: {related}")
    return {**inputs, "category": category, "related_categories": related}


def search_and_rerank(inputs):
    """Step 3: RAG search + GraphRAG expansion + reranking."""
    global vectorstore
    query    = inputs["rewritten_query"]
    category = inputs["category"]
    related  = inputs["related_categories"]

    # Primary RAG search
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    # GraphRAG: boost results from related categories
    boosted = []
    for doc, score in results:
        doc_category = doc.metadata.get("category", "")
        similarity   = 1 - (score / 2)

        if doc_category == category:
            similarity = min(1.0, similarity * 1.15)  # 15% boost
        elif doc_category in related:
            similarity = min(1.0, similarity * 1.05)  # 5% boost

        boosted.append((doc, similarity))

    boosted.sort(key=lambda x: x[1], reverse=True)
    top_candidates = boosted[:5]
    best_score     = top_candidates[0][1]

    if best_score < SIMILARITY_THRESHOLD:
        print(f"[Search] No good match (best: {best_score:.4f}), using LLM fallback")
        return {
            **inputs,
            "retrieved_response": None,
            "confidence_score":   round(best_score, 4),
            "source":             "llm",
            "intent":             "unknown"
        }

    # LLM Reranking
    candidates_text = "\n".join(
        [f"{i+1}. {doc.page_content}" for i, (doc, _) in enumerate(top_candidates)]
    )
    rerank_prompt = f"""You are a reranker for a customer support system.
Pick the BEST response for this customer query.

Query: {query}

Candidates:
{candidates_text}

Return ONLY the number (1-5) of the best response, nothing else:"""

    try:
        rerank_result = llm.invoke(rerank_prompt).strip()
        # Extract first digit found
        best_idx = int(next(c for c in rerank_result if c.isdigit())) - 1
        best_idx = max(0, min(best_idx, 4))
    except (ValueError, StopIteration):
        best_idx = 0

    best_doc, best_score = top_candidates[best_idx]
    print(f"[Reranker] Selected #{best_idx+1} | Score: {best_score:.4f} | Intent: {best_doc.metadata.get('intent')}")

    return {
        **inputs,
        "retrieved_response": best_doc.page_content,
        "confidence_score":   round(best_score, 4),
        "source":             "knowledge_base",
        "intent":             best_doc.metadata.get("intent", "unknown")
    }


def generate_response(inputs):
    """Step 4: Generate final natural response."""
    query    = inputs["query"]
    source   = inputs["source"]
    history  = inputs.get("history", "")
    category = inputs["category"]

    if source == "knowledge_base":
        reference = inputs["retrieved_response"]
        prompt = f"""You are Alex, a friendly customer support agent for ShopEase (e-commerce).

Conversation history: {history if history else 'None'}

Reference answer: {reference}

Customer asked: {query}

Respond naturally and concisely in max 3 sentences.
Address the customer directly. Do not repeat the reference word for word:"""
    else:
        prompt = f"""You are Alex, a friendly customer support agent for ShopEase (e-commerce).

ShopEase info:
- Website: www.shopease.com
- Email: support@shopease.com
- Phone: +1-800-555-0199
- Hours: 9AM-6PM EST Mon-Fri
- Returns: 30 days with receipt
- Cancellations: only before shipment

Conversation history: {history if history else 'None'}

Customer asked: {query}

Respond warmly and concisely. If unrelated to shopping/support, politely say so.
Max 3 sentences:"""

    response = llm.invoke(prompt).strip()
    print(f"[Generator] Source: {source} | Category: {category}")

    return {**inputs, "final_response": response}


# ── LangChain Pipeline ────────────────────────────────────────────────
pipeline = RunnableSequence(
    RunnableLambda(rewrite_query),
    RunnableLambda(detect_intent),
    RunnableLambda(search_and_rerank),
    RunnableLambda(generate_response)
)


# ── Main entry point ──────────────────────────────────────────────────
def get_response(query_text, conversation_history):
    """Run the full pipeline and return structured result."""
    global conversation_memory

    # Build history string from last N messages
    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in conversation_memory[-MAX_MEMORY_MESSAGES:]
    )

    # Run pipeline
    result = pipeline.invoke({
        "query":   query_text,
        "history": history_str
    })

    # Save to memory
    conversation_memory.append({"role": "user",      "content": query_text})
    conversation_memory.append({"role": "assistant",  "content": result["final_response"]})

    # Update conversation history for app.py compatibility
    conversation_history.append({"role": "user",      "content": query_text})
    conversation_history.append({"role": "assistant",  "content": result["final_response"]})

    return {
        "response":         result["final_response"],
        "confidence_score": result["confidence_score"],
        "source":           result["source"],
        "category":         result["category"],
        "intent":           result["intent"]
    }


if __name__ == "__main__":
    load_knowledge_base()

    history = []
    test_queries = [
        "How do I cancel my order?",
        "When can I talk to someone from support?",
        "What is your return policy for damaged items?"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Q: {query}")
        result = get_response(query, history)
        print(f"A: {result['response']}")
        print(f"Source: {result['source']} | Confidence: {result['confidence_score']} | Category: {result['category']} | Intent: {result['intent']}")