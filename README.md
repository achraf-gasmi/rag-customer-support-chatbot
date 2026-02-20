# 🛍️ RAG Customer Support Chatbot

A locally-run, privacy-first customer support chatbot built with a hybrid **RAG + GraphRAG** pipeline using LangChain, ChromaDB, and Ollama. No API costs, no data sent to the cloud — everything runs on your machine.

---

## 🎯 Project Overview

This project simulates a production-grade AI customer support system for a fictional e-commerce store called **ShopEase**. It classifies customer queries, retrieves relevant responses from a knowledge base using semantic search, and generates natural conversational replies — all locally.
---

## 🎬 Demo

[![Watch the demo](https://img.shields.io/badge/Watch-Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/E479DO3jqTA)

---

## 🏗️ Architecture
```
User Query
    ↓
Fast Path Check (greetings/simple messages)
    ↓
Query Rewriter (qwen3:8b)         — clarifies vague queries
    ↓
Intent Detector (keyword + LLM)   — classifies into 11 categories
    ↓
GraphRAG Traversal (NetworkX)     — finds related categories (2-hop)
    ↓
ChromaDB Similarity Search        — retrieves top 10 candidates
    ↓
GraphRAG Score Boosting           — boosts scores by category relevance
    ↓
Response Generator (qwen3:8b)     — generates natural reply
    ↓
SQLite Logger                     — logs every interaction
```

---

## ✨ Features

- **Hybrid RAG + GraphRAG pipeline** — combines vector similarity search with knowledge graph traversal for smarter retrieval
- **Multi-hop graph traversal** — discovers related categories up to 2 levels deep using NetworkX
- **Dynamic knowledge graph** — learns new relationships from user interactions over time
- **Query rewriting** — rewrites vague queries for better semantic search
- **Keyword + LLM intent detection** — fast keyword matching with LLM fallback
- **Score-based reranking** — GraphRAG-boosted scoring selects the best response
- **Conversation memory** — maintains context across the session
- **Fast path** — instant responses for greetings and simple messages
- **Interactive graph visualization** — explore the knowledge graph with pyvis
- **Session analytics** — confidence scores, KB vs LLM usage, intent tracking
- **SQLite interaction logging** — logs every query for monitoring and analysis
- **100% local** — no API keys, no internet required after setup

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama + qwen3:8b |
| Embeddings | Ollama + qwen3-embedding:8b |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Knowledge Graph | NetworkX + pyvis |
| Frontend | Streamlit |
| Interaction Logging | SQLite |
| Dataset | Bitext Customer Support (Hugging Face) |
| Language | Python 3.10+ |

---

## 📁 Project Structure
```
rag-customer-support-chatbot/
│
├── data/
│   ├── knowledge_base.csv       # cleaned & sampled dataset (1350 rows)
│   └── chroma_db/               # persisted vector embeddings
│
├── logs/
│   ├── interactions.db          # SQLite interaction log
│   └── graph.html               # generated graph visualization
│
├── src/
│   ├── backend.py               # RAG + GraphRAG pipeline
│   ├── database.py              # SQLite logging functions
│   └── prepare_data.py          # data cleaning & preparation
│
├── app.py                       # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-customer-support-chatbot.git
cd rag-customer-support-chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull required Ollama models
```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b
```

### 5. Prepare the knowledge base
```bash
python src/prepare_data.py
```

### 6. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

> **Note:** The first run will embed 1,350 documents into ChromaDB using `qwen3-embedding:8b`. This takes a few minutes but only happens once.

---

## 🧠 How It Works

### RAG Pipeline
On every customer query, the system:
1. Rewrites the query to be clear and self-contained
2. Detects the intent category using keyword matching
3. Searches ChromaDB for the top 10 most semantically similar responses
4. Returns the best match or falls back to LLM generation

### GraphRAG Enhancement
A knowledge graph connects categories and intents with typed relationships:
- `ORDER → REFUND` (may_lead_to)
- `ORDER → DELIVERY` (related_to)
- `PAYMENT → INVOICE` (related_to)

When a query is classified as ORDER, the graph traversal finds related categories (REFUND, CANCEL, DELIVERY, SHIPPING) up to 2 hops away. Results from related categories receive a score boost, improving retrieval accuracy for complex queries.

### Dynamic Graph Learning
Every interaction is logged to SQLite. When consecutive queries belong to different categories, a weighted edge is created between them. These **learned edges** appear as yellow connections in the graph visualization, showing real usage patterns.

### Local & Private
Unlike cloud-based solutions, this chatbot runs entirely on your machine using Ollama. No data is sent to external servers, making it suitable for privacy-sensitive environments.

---

## 📊 Dataset

Uses the [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) from Hugging Face:
- 26,872 question/answer pairs across 27 intents and 11 categories
- Sampled to 1,350 balanced rows (50 per intent)
- Placeholders replaced with realistic ShopEase values

---

## 📝 ChromaDB Note

ChromaDB runs entirely as a local library — no account or external service required. It stores vector embeddings as local files in `data/chroma_db/`, providing efficient semantic search without any cloud dependency.

---

## 🔮 Future Improvements

- [ ] Add support for document upload (PDF knowledge base)
- [ ] Implement streaming responses for faster UI feedback
- [ ] Add user authentication for multi-tenant support
- [ ] Expand knowledge base to full 26,872 rows
- [ ] Add evaluation metrics (RAGAS framework)

---

## 👤 Author

**Achraf**
- GitHub: [@achraf-gasmi](https://github.com/achraf-gasmi)
- LinkedIn: [My-linkedin](https://www.linkedin.com/in/achraf-gasmi-592766134/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.