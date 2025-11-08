# Persona-Based LinkedIn Writing Agent using RAG (Retrieval-Augmented Generation)

## 🧠 Project Overview

This project aims to build a **persona-based LinkedIn writing agent** that generates new posts mimicking a specific person’s tone and writing style using **Retrieval-Augmented Generation (RAG)**. It combines prior LinkedIn posts, personal details, and company context to produce authentic, grounded LinkedIn posts.

The goal is to show strong implementation intuition, modular design, and clear reasoning for each step — demonstrating a practical, high-quality RAG pipeline.

---

## 🧩 System Architecture

**Pipeline:**

```
Ingest → Index → Retrieve → Generate → Memory → Evaluate
```

**Core Modules:**

1. **Ingestion:** Clean and chunk prior LinkedIn posts while preserving writing style.
2. **Indexing:** Create vector embeddings and store them in FAISS or Chroma.
3. **Retrieval:** Fetch the most stylistically relevant chunks for each new prompt.
4. **Generation:** Assemble a structured prompt and generate a new post.
5. **Memory:** Maintain persona preferences (hashtags, banned words, themes).
6. **Evaluation:** Compare RAG vs Non-RAG generations.

---

## ⚙️ Tech Stack

| Component       | Tool                                                                     |
| --------------- | ------------------------------------------------------------------------ |
| Language        | Python 3.10+                                                             |
| LLM             | **gpt-4o-mini** (fast, cost-efficient) or **gpt-4-turbo** (high quality) |
| Embedding Model | **text-embedding-3-small** / `all-MiniLM-L6-v2`                          |
| Vector Store    | FAISS / Chroma                                                           |
| Framework       | LangChain (for modular RAG design)                                       |
| Interface       | Streamlit (optional)                                                     |
| Memory          | JSON-based persistent memory                                             |

---

## 🧭 Phase-wise Plan & GitHub Commit Guide

### **Commit 1 – Project Initialization**

* Create repo `linkedin-rag-agent`
* Add folders: `/data`, `/src`, `/interface`, `/memory`, `/eval`
* Add `.gitignore`, `requirements.txt`, and basic `README.md`
* Commit Message: `Initial setup: repo structure + dependencies`

### **Commit 2 – Persona Data Ingestion**

* Collect 3–6 LinkedIn posts (plain text)
* Clean and chunk by paragraph/sentence
* Store metadata (date, link, etc.)
* Save as `data/cleaned_chunks.json`
* Commit Message: `Added ingestion module: cleaned and chunked posts`

### **Commit 3 – Embedding and Index Creation**

* Use OpenAI or SentenceTransformers to embed chunks
* Store in FAISS or Chroma vector DB
* Log embedding shape and stats
* Commit Message: `Implemented vector embedding and indexing`

### **Commit 4 – Retrieval Pipeline**

* Build `retrieve.py` to fetch top-k (k=5) relevant chunks
* Implement MMR (optional) for diversity
* Input: profile + bullet ideas → query → retrieved snippets
* Commit Message: `Built retrieval system with similarity and MMR`

### **Commit 5 – Prompt Engineering**

* Define structured prompt template:

  ```
  You are a LinkedIn assistant trained to mimic {Person_Name}.
  Tone: professional, optimistic, insightful.
  Structure: Hook → Insight → Example → Reflection.
  No emojis, ≤4 hashtags.
  Context: {Retrieved_Snippets}
  ```
* Merge user input + memory context
* Commit Message: `Added structured prompt assembly for persona writing`

### **Commit 6 – Generation Pipeline**

* Use LLM (`gpt-4o-mini` or `gpt-4-turbo`) to generate posts
* Output: 120–220 words per post
* Save generations in `/outputs/generated_posts.json`
* Commit Message: `Integrated LLM generation with RAG pipeline`

### **Commit 7 – Memory System**

* Create `/memory/memory.json` to store:

  ```json
  {
    "preferred_hashtags": ["#leadership", "#innovation"],
    "banned_phrases": ["click the link"],
    "recurring_themes": ["AI ethics", "sustainability"],
    "previous_posts": []
  }
  ```
* Merge memory context during generation
* Commit Message: `Added memory persistence for persona preferences`

### **Commit 8 – Streamlit Interface**

* Inputs: Name, Title, Company, Industry, Bullets
* Outputs: Generated Post + Used Snippets
* Optional: Re-generate button for plagiarism fallback
* Commit Message: `Developed Streamlit interface for LinkedIn RAG agent`

### **Commit 9 – Anti-Plagiarism Check**

* Detect >25 consecutive word overlap with source
* If detected → re-generate with stronger paraphrase instruction
* Commit Message: `Added plagiarism detection and paraphrasing fallback`

### **Commit 10 – Evaluation (RAG vs Non-RAG)**

* Generate multiple posts with & without retrieval
* Compare tone and style consistency
* Save results in `/eval/comparison.json`
* Commit Message: `Evaluation: RAG vs Non-RAG performance analysis`

### **Commit 11 – Final Documentation**

* Add architecture diagram
* Update README with inputs, outputs, examples, and setup
* Commit Message: `Final documentation + architecture diagram`

---

## 💾 Directory Structure

```
linkedin-rag-agent/
│
├── data/
│   ├── posts.txt
│   └── cleaned_chunks.json
│
├── src/
│   ├── ingest.py
│   ├── indexer.py
│   ├── retrieve.py
│   ├── prompter.py
│   ├── generate.py
│
├── interface/
│   └── app.py
│
├── memory/
│   └── memory.json
│
├── eval/
│   └── comparison.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Model Recommendations

### 🔹 **Embeddings**

* **Option 1 (Best):** `text-embedding-3-small` — cost-effective, robust for writing style retrieval.
* **Option 2 (Free):** `all-MiniLM-L6-v2` from SentenceTransformers.

### 🔹 **Generation Agent**

* **Primary:** `gpt-4o-mini` — affordable, style-accurate, fast for experimentation.
* **Secondary:** `gpt-4-turbo` — deeper persona consistency and creativity.

### 🔹 **Agent Integration**

Wrap the retrieval + generation functions using LangChain’s agent pattern:

```python
from langchain.agents import initialize_agent, Tool
```

This makes the LLM act as an **autonomous LinkedIn writer** that understands persona context.

---

## 📊 Evaluation Metrics

| Metric            | Description                               |
| ----------------- | ----------------------------------------- |
| Style Consistency | Matches tone and phrasing of prior posts  |
| Plagiarism Rate   | Measures similarity with retrieved chunks |
| Readability       | Flesch Reading Ease test                  |
| Hashtag Balance   | ≤4, relevant and consistent               |
| Persona Coherence | Maintains subject voice and perspective   |

---

## 🧠 Example

**Input:**

```
Person: Sundar Pichai
Title: CEO, Google
Company: Google (Tech Industry, 150K employees)
Bullets: "AI for productivity", "Sustainable innovation"
```

**Output:**

> *AI continues to redefine how we work, learn, and create.*
> At Google, we see this transformation as an opportunity to empower people — not replace them.
> From developing tools that help businesses scale to reducing our environmental footprint, our goal remains clear: use technology to make life better for everyone.
> #AI #Sustainability #Innovation

---

## 🪶 Extras (If Time Allows)

* Add BM25 + Embedding re-ranker comparison
* Integrate FastAPI backend
* Include auto-checks for hashtag count, emoji usage, and first-person presence

---

## ✅ Summary of Commit Plan

| Commit | Focus             |
| ------ | ----------------- |
| 1      | Setup & Structure |
| 2      | Data Ingestion    |
| 3      | Embedding & Index |
| 4      | Retrieval         |
| 5      | Prompting         |
| 6      | Generation        |
| 7      | Memory            |
| 8      | Interface         |
| 9      | Anti-Plagiarism   |
| 10     | Evaluation        |
| 11     | Documentation     |

---

## 🚀 Conclusion

By following this commit plan, you’ll build a fully functional **RAG-based LinkedIn writing agent** capable of mimicking real personas while grounding outputs in authentic past data. This structure demonstrates professional intuition, organized project management, and clean engineering practices — exactly what a recruiter wants to see in an AI intern candidate.
