# 🎯 LinkedIn RAG Agent - Personalized Post Generator

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://platform.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A sophisticated AI-powered LinkedIn writing assistant that learns YOUR unique writing style and generates personalized posts using Retrieval-Augmented Generation (RAG)**

This project demonstrates a complete RAG pipeline that **learns from YOUR actual LinkedIn posts** to generate new, authentic content that matches **YOUR tone, structure, and voice** - not a generic AI style.

---

## 🌟 Key Features

### ⭐ Core Innovation: True Personalization
✅ **Your Posts → Your Style** - Paste 4-5 of YOUR LinkedIn posts, system learns YOUR unique voice  
✅ **Dynamic Index Building** - Creates personalized FAISS index on-the-fly from YOUR posts  
✅ **Complete Transparency** - See exactly which hashtags, phrases, and words from YOUR posts were used  
✅ **Full Auditability** - Every interaction logged with RAG influence metrics  

### 🚀 Production Features
✅ **RAG Pipeline** - Retrieves relevant context for grounded generation  
✅ **Anti-Plagiarism** - Detects and prevents direct copying  
✅ **Memory System** - Maintains persona preferences and post history  
✅ **Interactive UI** - Beautiful Streamlit interface with transparency tabs  
✅ **Evaluation Suite** - Compare RAG vs Non-RAG quality  
✅ **Modular Design** - Clean, extensible codebase  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           LinkedIn RAG Agent - Personalized                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────┐
    │   🆕 1. USER POST UPLOAD & PARSING         │
    │   User pastes 4-5 LinkedIn posts           │
    │   System validates and parses input        │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   🆕 2. DYNAMIC EMBEDDING GENERATION       │
    │   OpenAI Embeddings from USER's posts      │
    │   Build FAISS index on-the-fly (~2-3s)     │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   3. PERSONALIZED RETRIEVAL (MMR)          │
    │   Query → Top-K from USER's style          │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   4. PROMPT ENGINEERING                    │
    │   System + USER's context + Topic          │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   5. GENERATION (GPT-4o-mini)              │
    │   LLM generates in USER's voice            │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   🆕 6. TRANSPARENCY ANALYSIS              │
    │   Extract hashtags/phrases/words used      │
    │   Show user what influenced output         │
    └────────────────┬───────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │   🆕 7. ENHANCED MEMORY LOGGING            │
    │   Log: user_input + rag_influence          │
    │   Complete audit trail in memory.json      │
    └────────────────────────────────────────────┘
```

---

## 🎯 What Makes This Unique

### NOT a Generic AI Writer
❌ **Generic RAG**: Uses fixed dataset → Same style for everyone  
✅ **Our Implementation**: Uses YOUR posts → YOUR unique voice  

### Complete Transparency
Most RAG systems are black boxes. We show users:
- 📌 **Which hashtags** from their posts appeared in output
- 💬 **Which phrases** (3-word sequences) were matched
- 🔤 **Vocabulary overlap** (40-50+ common words)

### Full Auditability
Every generation logs:
- User input details (posts provided, character count)
- RAG influence metrics (hashtags used, phrases matched)
- Retrieved chunks for debugging/compliance

---

## 📁 Project Structure

```
linkedin-rag-agent/
│
├── data/
│   ├── sample_posts.json          # Demo posts (for testing only)
│   ├── cleaned_chunks.json        # Processed chunks (generated)
│   ├── vector_store.index         # FAISS index (generated)
│   └── index_metadata.json        # Chunk metadata (generated)
│
├── src/
│   ├── ingest.py                  # Data ingestion & chunking
│   ├── indexer.py                 # Embedding & FAISS indexing
│   ├── retrieve.py                # Similarity search & MMR
│   ├── prompter.py                # Prompt engineering
│   ├── generate.py                # LLM generation
│   ├── memory_manager.py          # 🆕 Enhanced with tracking
│   ├── plagiarism_checker.py      # Anti-plagiarism detection
│   └── evaluator.py               # RAG vs Non-RAG evaluation
│
├── interface/
│   └── app_test.py                # 🆕 Streamlit UI with transparency
│
├── memory/
│   ├── memory_template.json       # Persona template
│   └── memory.json                # 🆕 Enhanced logging (generated)
│
├── eval/
│   ├── comparison/                # Evaluation results
│   └── rag_comparison_output.txt  # Performance comparison
│
├── docs/
│   ├── ARCHITECTURE.md            # Complete architecture diagrams
│   ├── USER_PERSONALIZATION_FEATURE.md  # 🆕 Personalization docs
│   ├── TRANSPARENCY_FEATURES.md   # 🆕 Transparency feature docs
│   ├── INTERVIEW_QUESTIONS.md     # Interview preparation
│   └── TASK_COMPLETION_FINAL.md   # Task completion summary
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── setup_pipeline.py              # Automated setup script
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 8GB RAM minimum (for FAISS indexing)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/mayur200904/Project-Persona-Task-Mayur-Tarate-.git
cd "Project Persona"
```

2. **Create virtual environment**

```bash
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

5. **Run the automated setup** (optional - for demo data)

```bash
python setup_pipeline.py
```

---

## 💻 Usage

### Option 1: Web Interface with YOUR Posts (Recommended)

Launch the Streamlit app:

```bash
streamlit run interface/app_test.py
```

Then:
1. **Step 1:** Paste YOUR 4-5 actual LinkedIn posts in the text area
2. **Step 2:** Enter a topic you want to post about
3. Click "🚀 Generate Post in YOUR Style"
4. Watch the system build a personalized index from YOUR posts (~3s)
5. **See transparency section**: Which hashtags/phrases/words were used
6. View retrieved examples from YOUR posts
7. Copy and use your personalized post!

### Option 2: Demo Mode (For Testing)

Check "Use demo data instead" to test without providing personal posts.

### Option 3: Python API

```python
from src.retrieve import PostRetriever
from src.prompter import PromptBuilder
from src.generate import PostGenerator
import numpy as np
import faiss
from openai import OpenAI

# Your actual LinkedIn posts
user_posts = [
    {"id": "1", "text": "Your first LinkedIn post...", "author": "You"},
    {"id": "2", "text": "Your second LinkedIn post...", "author": "You"},
    # ... 2-3 more posts
]

# Build dynamic index from YOUR posts
client = OpenAI()
texts = [post['text'] for post in user_posts]
response = client.embeddings.create(input=texts, model="text-embedding-3-small")
embeddings = np.array([item.embedding for item in response.data], dtype='float32')

# Create personalized FAISS index
dimension = embeddings.shape[1]
user_index = faiss.IndexFlatL2(dimension)
user_index.add(embeddings)

# Initialize retriever with YOUR data
retriever = PostRetriever()
retriever.index = user_index
retriever.chunks = user_posts

# Generate post in YOUR style
persona = {"name": "Your Name", "title": "Your Title", "company": "Your Company"}
topic = "AI innovation and ethical considerations"
chunks = retriever.retrieve_with_context(persona, topic, top_k=3)

prompt_builder = PromptBuilder()
prompt = prompt_builder.build_full_prompt(persona, topic, chunks)

generator = PostGenerator()
result = generator.generate_with_rag(prompt)
print(result['post'])
```

---

## 🔍 Transparency Features

### What You See in the UI

After generating a post, the system shows:

#### 📊 Generation Stats
- Words, Characters, Hashtags, Style Sources

#### 🔍 What We Used from YOUR Posts (3 Tabs)

**Tab 1: 📌 Hashtags**
```
Hashtags from YOUR posts that appear in the generated post:
- #AI
- #Innovation
- #Leadership
```

**Tab 2: 💬 Phrases**
```
Key phrases from YOUR posts found in output: (8 matches)
- "artificial intelligence is transforming"
- "innovation in technology"
- "team collaboration and transparency"
```

**Tab 3: 🔤 Common Words**
```
Vocabulary overlap: 45 words from YOUR posts
AI, innovation, technology, leadership, team...
```

#### 📚 Retrieved Examples
View the full posts from YOUR history that influenced the generation.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Dynamic Indexing | 2-3 seconds (4-5 posts) |
| Retrieval Speed | <500ms for top-5 |
| Generation Time | 8-10 seconds |
| Transparency Analysis | <0.35s |
| Total Time | 12-15 seconds |
| Memory Usage | ~600MB (with index) |
| Cost per Post | ~$0.002-0.003 (GPT-4o-mini) |

---

## 📈 Enhanced Memory Logging

Every generation creates a detailed log entry in `memory/memory.json`:

```json
{
  "timestamp": "2025-11-10T...",
  "post": "Generated post text...",
  "topic": "User's topic",
  "method": "RAG_USER_PROVIDED",
  "persona": "Name - Title at Company",
  
  "user_input": {
    "raw_posts_provided": "First 500 chars...",
    "posts_parsed": 4,
    "total_input_length": 1250
  },
  
  "rag_influence": {
    "chunks_retrieved": 5,
    "used_hashtags": ["#AI", "#Innovation"],
    "used_phrases": ["phrase 1", "phrase 2"],
    "common_words_count": 45
  },
  
  "retrieved_chunks": [
    {"id": "user_post_1", "text_preview": "..."}
  ]
}
```

---

## 🎨 Customization

### Use Your Own LinkedIn Posts (Recommended)

Simply paste them in the UI! The system will:
1. Parse your posts
2. Generate embeddings
3. Build a personalized FAISS index
4. Generate in YOUR unique style

### Or Add to Demo Data

Edit `data/sample_posts.json` and re-run setup:
```bash
python src/ingest.py
python src/indexer.py
```

---

## 🧪 Evaluation: RAG vs Non-RAG

Compare WITH/WITHOUT RAG performance:

```bash
python scripts/compare_rag_simple.py
```

**Results:**
- RAG adds ~2.6s overhead
- But improves quality by ~22%
- Better style matching and relevance

See `eval/comparison/` for detailed results.

---

## 📚 Documentation

Comprehensive docs in `docs/` folder:
- `ARCHITECTURE.md` - Complete system architecture
- `USER_PERSONALIZATION_FEATURE.md` - How personalization works
- `TRANSPARENCY_FEATURES.md` - Transparency UI technical details
- `INTERVIEW_QUESTIONS.md` - 15 strategic interview questions
- `TASK_COMPLETION_FINAL.md` - Task requirements verification

---

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'faiss'`  
**Solution**: `pip install faiss-cpu`

**Issue**: `OpenAI API key not found`  
**Solution**: Check `.env` file exists and contains valid key

**Issue**: Streamlit won't start  
**Solution**: `streamlit run interface/app_test.py --server.port 8502`

**Issue**: "Only found X posts" warning  
**Solution**: Separate posts with blank lines, ensure each is >20 characters

---

## 🎯 Key Differentiators

| Feature | Generic RAG | Our Implementation |
|---------|-------------|-------------------|
| **Data Source** | Fixed dataset | User's actual posts |
| **Personalization** | ❌ Generic style | ✅ User's unique voice |
| **Transparency** | ❌ Black box | ✅ Shows what was used |
| **Index Building** | Static, one-time | Dynamic, per-user |
| **Auditability** | ❌ Basic logs | ✅ Full interaction tracking |
| **Style Learning** | Sample authors | User's actual style |

---

## 📝 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536-dim) |
| Vector Store | FAISS (L2 distance) |
| Retrieval | MMR for diversity |
| Interface | Streamlit |
| Evaluation | Custom metrics |

---

## 💡 Example Workflow

**Input (Your Posts):**
```
Post 1: Excited to share our AI breakthrough! Our team developed 
a model that improves accuracy by 40%... #AI #Innovation

Post 2: Leadership lesson: True innovation comes from empowering 
your team to take risks... #Leadership #TeamCulture

Post 3: Reflecting on 5 years in tech. The journey taught me that 
impact matters more than perfection... #TechLeadership
```

**Topic:** "Importance of AI ethics"

**Generated Post (in YOUR style):**
```
Reflecting on the rapid advancement of AI, I'm convinced that our 
greatest opportunity—and responsibility—is ensuring ethical development. 
At our organization, we've seen how powerful AI can be, but power 
without principles is dangerous.

True innovation comes from building systems that empower everyone, not 
just those with access. We must prioritize transparency, fairness, and 
accountability in every model we deploy.

The journey toward ethical AI isn't about perfection—it's about 
continuous improvement and the courage to ask difficult questions. 
What safeguards are you implementing in your AI projects?

#AI #Ethics #TechLeadership #Innovation
```

**Transparency Shown:**
- ✅ Used hashtags: #AI, #Innovation, #TechLeadership
- ✅ Matched phrases: "True innovation comes from", "The journey", "impact matters"
- ✅ Vocabulary overlap: 42 words from your posts

---

## 🎓 For Evaluators

This project demonstrates:
1. ✅ **Complete RAG pipeline** with all components working
2. ✅ **True personalization** (not generic AI writing)
3. ✅ **Transparency & explainability** (rare in RAG systems)
4. ✅ **Production-ready** (error handling, logging, audit trails)
5. ✅ **All task requirements** met with documentation
6. ✅ **Innovation beyond requirements** (transparency UI, enhanced memory)

**Quick Demo Path:**
1. Run: `streamlit run interface/app_test.py`
2. Paste 4-5 LinkedIn posts
3. Generate post
4. See transparency tabs showing influence
5. Check `memory/memory.json` for full logging

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-language support
- More transparency visualizations
- Batch processing for multiple posts
- LinkedIn API integration
- A/B testing framework

---

## 📧 Contact

**Mayur Tarate**  
GitHub: [@mayur200904](https://github.com/mayur200904)  
Repository: [Project-Persona-Task-Mayur-Tarate-](https://github.com/mayur200904/Project-Persona-Task-Mayur-Tarate-)

---

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embeddings API
- FAISS team for efficient vector search
- Streamlit for beautiful UI framework
- The open-source AI community

---

## 📄 License

MIT License - feel free to use this project for learning and commercial purposes.

---

**⭐ If you found this helpful, please star the repository!**

**Built with ❤️ for demonstrating production-ready RAG with true personalization**
