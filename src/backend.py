import os
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "qwen3-embedding:8b"
CHAT_MODEL           = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
SIMILARITY_THRESHOLD = 0.70
TOP_K                = 10
MAX_MEMORY_MESSAGES  = 6

# ── Groq AI Gateway client ──────────────────────────────────────────
client = OpenAI(
    base_url=os.getenv("VERCEL_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Embeddings still local via Ollama ────────────────────────────────
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# ── Unified LLM call ─────────────────────────────────────────────────
def call_llm(prompt, temperature=0.5):
    """Unified LLM call via Vercel AI Gateway."""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""

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

def get_related_categories(category, hops=2):
    """Multi-hop graph traversal — follow edges up to N levels deep."""
    related  = set()
    visited  = {category}
    frontier = {category}

    for hop in range(hops):
        next_frontier = set()
        for node in frontier:
            if node not in knowledge_graph:
                continue
            for neighbor in knowledge_graph.neighbors(node):
                if knowledge_graph.nodes[neighbor].get("type") == "category":
                    if neighbor not in visited:
                        related.add(neighbor)
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            for predecessor in knowledge_graph.predecessors(node):
                if knowledge_graph.nodes[predecessor].get("type") == "category":
                    if predecessor not in visited:
                        related.add(predecessor)
                        next_frontier.add(predecessor)
                        visited.add(predecessor)

        frontier = next_frontier
        if not frontier:
            break

    print(f"[GraphRAG] {hops}-hop traversal from '{category}' → {related}")
    return list(related)


# ── Fast path detector ────────────────────────────────────────────────
SIMPLE_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "bye", "goodbye", "ok", "okay", "sure",
    "great", "nice", "cool", "alright", "got it", "perfect", "awesome"
}

def is_simple_message(text):
    """Detect greetings and simple messages that don't need full pipeline."""
    cleaned = text.lower().strip().rstrip("!.,?")
    return cleaned in SIMPLE_PATTERNS or len(cleaned.split()) <= 2

def handle_simple_message(query_text, history_str):
    """Handle simple messages with a single LLM call."""
    prompt = f"""You are Alex, a friendly customer support agent for ShopEase.
Respond briefly and warmly to this greeting or simple message.
Max 1-2 sentences.

Conversation history: {history_str if history_str else 'None'}
Message: {query_text}"""

    return call_llm(prompt)


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

    rewritten = call_llm(prompt, temperature=0.1)
    if len(rewritten) > 300:
        rewritten = query
    print(f"[Rewriter] '{query}' → '{rewritten}'")
    return {**inputs, "rewritten_query": rewritten}


def detect_intent(inputs):
    """Step 2: Keyword-based intent detection with LLM fallback."""
    query = inputs["rewritten_query"]

    keyword_map = {
        "ACCOUNT":      ["account","login","password","register","sign up","signup","profile","delete account","log in"],
        "ORDER":        ["order","purchase","bought","buy","placed","my order"],
        "REFUND":       ["refund","money back","reimbursement","return money","damaged","broken","defective","return"],
        "CONTACT":      ["contact","support","agent","human","talk","speak","call","chat","reach"],
        "INVOICE":      ["invoice","receipt","bill","billing"],
        "PAYMENT":      ["payment","pay","charge","credit card","transaction","method"],
        "FEEDBACK":     ["complaint","review","feedback","unhappy","disappointed","complain"],
        "DELIVERY":     ["delivery","deliver","shipping time","arrive","arrival","when will"],
        "SHIPPING":     ["shipping","address","ship to","send to","change address"],
        "SUBSCRIPTION": ["subscription","newsletter","subscribe","unsubscribe"],
        "CANCEL":       ["cancel","cancellation","cancelling"],
    }

    query_lower = query.lower()
    for category, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            related = get_related_categories(category)
            print(f"[Intent] Category: {category} (keyword) | Related: {related}")
            return {**inputs, "category": category, "related_categories": related}

    # LLM fallback only if no keyword matched
    try:
        prompt = f"""Classify into ONE: ACCOUNT, ORDER, REFUND, CONTACT, INVOICE, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL, UNKNOWN
Query: {query}
Return ONLY the category name:"""
        category = call_llm(prompt, temperature=0.0).upper()
        valid = {"ACCOUNT","ORDER","REFUND","CONTACT","INVOICE","PAYMENT",
                 "FEEDBACK","DELIVERY","SHIPPING","SUBSCRIPTION","CANCEL","UNKNOWN"}
        for word in category.split():
            if word in valid:
                category = word
                break
        else:
            category = "UNKNOWN"
    except Exception:
        category = "UNKNOWN"

    related = get_related_categories(category)
    print(f"[Intent] Category: {category} (LLM) | Related: {related}")
    return {**inputs, "category": category, "related_categories": related}


def search_and_rerank(inputs):
    """Step 3: RAG search + GraphRAG boosting + score-based reranking."""
    global vectorstore
    query    = inputs["rewritten_query"]
    category = inputs["category"]
    related  = inputs["related_categories"]

    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    boosted = []
    for doc, score in results:
        doc_category = doc.metadata.get("category", "")
        similarity   = 1 - (score / 2)

        if doc_category == category:
            similarity = min(1.0, similarity * 1.15)
        elif doc_category in related:
            similarity = min(1.0, similarity * 1.05)

        boosted.append((doc, similarity))

    boosted.sort(key=lambda x: x[1], reverse=True)
    best_doc, best_score = boosted[0]

    if best_score < SIMILARITY_THRESHOLD:
        print(f"[Search] No good match (best: {best_score:.4f}), using LLM fallback")
        return {
            **inputs,
            "retrieved_response": None,
            "confidence_score":   round(best_score, 4),
            "source":             "llm",
            "intent":             "unknown"
        }

    print(f"[Search] Best match | Score: {best_score:.4f} | Intent: {best_doc.metadata.get('intent')}")

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

    response = call_llm(prompt)
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

    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in conversation_memory[-MAX_MEMORY_MESSAGES:]
    )

    # ── Fast path for greetings and simple messages ───────────────────
    if is_simple_message(query_text):
        print(f"[FastPath] Simple message: '{query_text}'")
        response = handle_simple_message(query_text, history_str)

        conversation_memory.append({"role": "user",      "content": query_text})
        conversation_memory.append({"role": "assistant",  "content": response})
        conversation_history.append({"role": "user",      "content": query_text})
        conversation_history.append({"role": "assistant",  "content": response})

        return {
            "response":         response,
            "confidence_score": 1.0,
            "source":           "llm",
            "category":         "GREETING",
            "intent":           "greeting"
        }

    # ── Full pipeline for real queries ────────────────────────────────
    result = pipeline.invoke({
        "query":   query_text,
        "history": history_str
    })

    conversation_memory.append({"role": "user",      "content": query_text})
    conversation_memory.append({"role": "assistant",  "content": result["final_response"]})
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
        "Hi",
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

