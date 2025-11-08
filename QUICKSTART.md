# 🚀 Quick Start Guide

## Get Started in 5 Minutes!

### Step 1: Setup Environment (2 min)

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### Step 2: Run Setup Pipeline (2 min)

```bash
python setup_pipeline.py
```

This will automatically:
- ✅ Ingest and chunk sample posts
- ✅ Create embeddings
- ✅ Build vector index
- ✅ Initialize memory

### Step 3: Launch Interface (1 min)

```bash
streamlit run interface/app.py
```

Your browser will open to `http://localhost:8501`

### Step 4: Generate Your First Post!

1. **Configure Persona** (sidebar):
   - Name: Your name
   - Title: Your role
   - Company: Your company
   - Industry: Your industry

2. **Enter Topic**:
   ```
   The future of AI in healthcare and ethical considerations
   ```

3. **Click "Generate Post"**

4. **View Results**:
   - Generated post
   - Word count and metrics
   - Retrieved context
   - Plagiarism check

---

## 🎯 Next Steps

### Add Your Own Posts

1. Edit `data/sample_posts.json`
2. Add your actual LinkedIn posts
3. Run:
   ```bash
   python src/ingest.py
   python src/indexer.py
   ```

### Run Evaluation

Compare RAG vs Non-RAG:
```bash
python scripts/run_evaluation.py
```

### Test Components

```bash
# Test retrieval
python src/retrieve.py

# Test generation
python src/generate.py

# Test plagiarism
python src/plagiarism_checker.py
```

---

## 💡 Tips for Best Results

1. **Use 5-10 sample posts** minimum for better style matching
2. **Adjust temperature** (0.7-0.9) for more creativity
3. **Enable MMR** for diverse context retrieval
4. **Check plagiarism** before posting
5. **Save persona settings** for consistency

---

## ❓ Troubleshooting

**Can't find vector store?**
```bash
python src/indexer.py
```

**API key error?**
- Check `.env` file exists
- Verify key starts with `sk-`

**Import errors?**
```bash
pip install -r requirements.txt
```

---

## 📚 Learn More

- [Full README](README.md)
- [Architecture Details](README.md#architecture)
- [Customization Guide](README.md#customization)

---

**Ready to impress with AI-powered LinkedIn posts! 🎉**
