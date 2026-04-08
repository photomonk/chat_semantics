# 🔍 Semantic Chat Finder

> Find relevant past chat messages using natural language queries — powered by **BAAI/bge-m3** embeddings and **ChromaDB** vector search.

---

## 📌 Overview

This project builds a **semantic search system** over synthetic chat data. Instead of keyword matching, it understands the *meaning* of your query and retrieves the most relevant messages — even if the exact words don't match.

For example, querying `"money and stock market"` will surface messages about inflation, investments, and economic forecasts — without those exact words needing to appear in the query.

---

## 🗂️ Project Structure

```
├── semantic_chat_model.py   # Main script — data generation, indexing, search
├── Shlok.csv                # Auto-generated per-user chat CSV
├── Ram.csv
├── Bob.csv
├── John.csv
└── chat_DB/                 # ChromaDB persistent vector store (auto-created)
```

---

## ⚙️ How It Works

```
Synthetic Chat Data (1000 messages, 4 users, 7 topics)
        │
        ▼
  Deduplicate messages (258 → 28 unique)
        │
        ▼
  BAAI/bge-m3  →  1024-dim dense embeddings
        │
        ▼
  ChromaDB  →  HNSW index (cosine similarity)
        │
        ▼
  Query  →  Encode with same model  →  Top-N similar messages
```

### Topics Covered
`Climate Change` · `Technology` · `Health` · `Politics` · `Economy` · `Sports` · `Entertainment`

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install FlagEmbedding chromadb pandas numpy
```

### 2. Run the model

```bash
python semantic_chat_model.py
```

This will:
- Generate 1000 synthetic chat messages
- Save per-user CSVs (`Shlok.csv`, `Ram.csv`, etc.)
- Encode unique messages using `BAAI/bge-m3`
- Index them in ChromaDB under `./chat_DB`
- Run 4 example semantic search queries and print ranked results

---

## 🔎 Example Queries & Output

```
Query: "I want health related chats"
────────────────────────────────────────────────────────────
[1] (0.731)  [Health]   Have you tried the new health app? It's quite useful.
             ↳ 2025-12-01 03:05:00
[2] (0.756)  [Health]   Eating more plant-based food has improved my energy levels.
             ↳ 2025-12-01 01:45:00
[3] (0.785)  [Politics] There's a lot of discussions around healthcare reform.
             ↳ 2025-12-01 09:35:00
```

---

## 🛠️ Key Design Decisions

| Decision | Reason |
|---|---|
| `query_embeddings=` instead of `query_texts=` | `query_texts=` triggers ChromaDB's built-in 384-dim model, causing a dimension mismatch with bge-m3's 1024-dim vectors |
| Deduplicate before indexing | Storing duplicate messages skews results — same message returned multiple times with identical scores |
| Store `topic` + `timestamp` in metadata | Gives context to search results beyond just the message text |
| Cosine similarity (HNSW) | Best suited for normalized embedding vectors from transformer models |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `FlagEmbedding` | BAAI/bge-m3 model for 1024-dim dense embeddings |
| `chromadb` | Local persistent vector database |
| `pandas` | CSV handling and data manipulation |
| `numpy` | Embedding matrix operations |

---

## 📝 Notes

- The model is currently built for user **Shlok**. To index all 4 users, loop `TARGET_USER` over `["Shlok", "Ram", "Bob", "John"]` in the script — each gets its own ChromaDB collection.
- First run downloads the `BAAI/bge-m3` model (~2 GB) from HuggingFace.
- Runs on CPU; set `use_fp16=False` if you encounter issues on CPU-only machines.
