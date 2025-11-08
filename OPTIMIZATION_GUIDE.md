
# Performance Optimization Guide

## Overview
This guide covers strategies to optimize your LinkedIn RAG Agent's performance across multiple dimensions: generation quality, speed, consistency, and user satisfaction.

## 🎯 Optimization Areas

### 1. **Temperature Tuning** (Generation Creativity vs Consistency)

**What it does**: Controls randomness in text generation
- **Lower (0.3-0.5)**: More focused, consistent, predictable
- **Higher (0.7-1.0)**: More creative, diverse, varied

**Recommended approach**:
```python
# Test range
temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]

# Optimal for LinkedIn (professional + engaging):
optimal_temperature = 0.7
```

**When to adjust**:
- Posts feel repetitive → Increase temperature
- Posts are off-brand or too random → Decrease temperature

---

### 2. **Retrieval K (Context Window Size)**

**What it does**: Number of similar past posts to retrieve as context
- **Lower K (3-5)**: Faster, focused context
- **Higher K (7-10)**: More comprehensive, but slower and potentially noisy

**Trade-offs**:
| K Value | Speed | Context Quality | Token Cost |
|---------|-------|----------------|------------|
| 3       | ⚡⚡⚡  | ⭐⭐          | 💰        |
| 5       | ⚡⚡   | ⭐⭐⭐        | 💰💰      |
| 7       | ⚡     | ⭐⭐⭐⭐      | 💰💰💰    |
| 10      | 🐌     | ⭐⭐⭐⭐⭐    | 💰💰💰💰  |

**Recommendation**: Start with K=5, optimize based on use case

---

### 3. **MMR Lambda (Diversity vs Relevance)**

**What it does**: Balances similarity vs diversity in retrieved chunks
- **λ = 1.0**: Pure relevance (most similar chunks)
- **λ = 0.5**: Balanced (default)
- **λ = 0.0**: Pure diversity (most different chunks)

**Formula**: `MMR Score = λ × Relevance - (1-λ) × Similarity to Selected`

**Use cases**:
```python
# Narrow, focused topic (technical deep-dive)
lambda_mult = 0.7  # Prioritize relevance

# Broad topic (innovation, leadership)
lambda_mult = 0.5  # Balance relevance and diversity

# Need varied perspectives
lambda_mult = 0.3  # Prioritize diversity
```

---

### 4. **Prompt Engineering Strategies**

#### **A. Baseline Prompt** (Simple, direct)
```
"You are a LinkedIn writing assistant. Write about [topic]."
```
- **Pros**: Fast, straightforward
- **Cons**: Generic, lacks personality

#### **B. Detailed Instructions** (Comprehensive guidelines)
```
"You are a LinkedIn assistant for [persona].
- Use first-person
- 150-200 words
- No emojis
- Professional tone
- Include insights from: [context]"
```
- **Pros**: Better control, consistent style
- **Cons**: Can feel constrained

#### **C. Example-Driven** (Style mimicry)
```
"Here are examples of [persona]'s writing:
[examples]

Write a similar post about [topic]."
```
- **Pros**: Authentic voice, good style matching
- **Cons**: Risk of copying phrases

#### **D. Constraint-Focused** (Rule-based)
```
"MUST: 150-200 words, first-person, no emojis
MUST NOT: Promotional language, CTAs
Topic: [topic]"
```
- **Pros**: Guaranteed compliance
- **Cons**: Can feel robotic

#### **E. Creative Freedom** (Inspirational)
```
"Draw inspiration from these examples but make it your own:
[examples]

Topic: [topic]"
```
- **Pros**: Fresh, original content
- **Cons**: Less predictable, may drift from brand

**Recommendation**: Use **Example-Driven** or **Detailed Instructions** for best results

---

### 5. **Chunk Size Optimization**

**Current setting**: Paragraph-based chunking

**Alternatives**:
```python
# Option 1: Fixed token chunks
chunk_size = 512  # tokens
overlap = 50      # token overlap

# Option 2: Sentence-based
min_sentences = 3
max_sentences = 6

# Option 3: Semantic chunking
# Split at topic boundaries
```

**Recommendation**: Paragraph-based works well for LinkedIn posts (natural boundaries)

---

### 6. **Embedding Model Selection**

**Current**: `text-embedding-3-small` (1536 dimensions)

**Options**:
| Model | Dimensions | Performance | Cost |
|-------|-----------|-------------|------|
| text-embedding-3-small | 1536 | Good | $ |
| text-embedding-3-large | 3072 | Better | $$ |
| text-embedding-ada-002 | 1536 | Good (legacy) | $ |

**When to upgrade**:
- Poor retrieval quality → Try `text-embedding-3-large`
- Cost concerns → Stick with `text-embedding-3-small`

---

### 7. **Generation Model Selection**

**Current**: `gpt-4o-mini` (fast, cost-effective)

**Options**:
| Model | Quality | Speed | Cost | Best For |
|-------|---------|-------|------|----------|
| gpt-4o-mini | ⭐⭐⭐ | ⚡⚡⚡ | $ | High-volume, good quality |
| gpt-4o | ⭐⭐⭐⭐ | ⚡⚡ | $$ | Better coherence |
| gpt-4-turbo | ⭐⭐⭐⭐⭐ | ⚡ | $$$ | Best quality |

**Recommendation**: 
- Development/testing: `gpt-4o-mini`
- Production (quality matters): `gpt-4o`

---

## 🔬 Running Optimization Experiments

### Step 1: Run Optimization Suite
```bash
cd "/Users/mayursantoshtarate/Desktop/Project Persona"
python scripts/run_optimization.py
```

This will:
1. Test temperature values (0.5 - 0.9)
2. Test retrieval K values (3, 5, 7, 10)
3. Test prompt strategies (5 variations)
4. Test MMR lambda values (0.3, 0.5, 0.7, 0.9)

### Step 2: Review Results
Check `eval/optimization_results.json` for:
- Best temperature: `optimizations.temperature.best_temperature`
- Best K: `optimizations.retrieval_k.best_k`
- Best strategy: `optimizations.prompt_strategy.best_strategy`
- Best lambda: `optimizations.mmr_lambda.best_lambda`

### Step 3: Apply Recommendations
Update configuration in `eval/optimized_config.json`:
```json
{
  "model_config": {
    "temperature": 0.7,
    "max_tokens": 500
  },
  "retrieval_config": {
    "top_k": 5,
    "mmr_lambda": 0.5
  }
}
```

---

## 📊 Performance Metrics to Track

### Quality Metrics
1. **Lexical Diversity**: Unique words / Total words
   - Target: > 0.7
   
2. **Style Compliance**: 
   - Word count (150-200): ✓/✗
   - Hashtag count (≤4): ✓/✗
   - No emojis: ✓/✗
   
3. **Readability** (Flesch Reading Ease):
   - Target: 60-80 (Standard to Fairly Easy)

4. **Authenticity**:
   - First-person usage: ✓/✗
   - Brand voice match: 1-5 scale

### Efficiency Metrics
1. **Generation Speed**: Time to generate one post
   - Target: < 3 seconds
   
2. **Token Usage**: Tokens consumed per generation
   - Monitor to control costs
   
3. **Retrieval Speed**: Time to fetch K chunks
   - Should be < 500ms

---

## 🎯 Optimization Workflow

```
1. Baseline Test
   ├─ Run with default settings
   ├─ Generate 10 posts
   └─ Measure quality & speed

2. Parameter Sweep
   ├─ Temperature: 0.5 → 0.9
   ├─ K: 3 → 10
   ├─ Lambda: 0.3 → 0.9
   └─ Record metrics for each

3. Prompt Engineering
   ├─ Test 5 strategies
   ├─ Compare outputs
   └─ Select best 2

4. Final Configuration
   ├─ Combine best settings
   ├─ Generate 20 test posts
   └─ Validate with evaluator

5. Production Deploy
   ├─ Update config files
   ├─ Monitor performance
   └─ Iterate as needed
```

---

## 🛠️ Quick Tuning Tips

### Issue: Posts too generic
**Fix**: 
- Increase K (more context)
- Use "Example-Driven" prompt
- Check retrieval quality

### Issue: Posts too similar to source
**Fix**:
- Add anti-plagiarism checks
- Use "Creative Freedom" prompt
- Increase temperature

### Issue: Posts off-brand
**Fix**:
- Decrease temperature
- Use "Constraint-Focused" prompt
- Improve memory/persona definition

### Issue: Slow generation
**Fix**:
- Decrease K (less context)
- Use `gpt-4o-mini` model
- Cache frequent retrievals

### Issue: High costs
**Fix**:
- Use `gpt-4o-mini` instead of `gpt-4o`
- Decrease max_tokens
- Reduce K value

---

## 📈 Advanced Techniques

### 1. **Ensemble Generation**
Generate multiple versions and select best:
```python
temperatures = [0.6, 0.7, 0.8]
posts = [generate(temp=t) for t in temperatures]
best = select_best(posts, criteria=['diversity', 'compliance'])
```

### 2. **Iterative Refinement**
```python
post_v1 = generate(topic)
feedback = check_plagiarism(post_v1)
if feedback['has_overlap']:
    post_v2 = regenerate_with_paraphrase(post_v1)
```

### 3. **Adaptive K Selection**
```python
if topic_complexity == "high":
    k = 7  # More context needed
else:
    k = 3  # Simple topic, less context
```

### 4. **Dynamic Prompt Selection**
```python
if brand_match_critical:
    use "Example-Driven"
elif creative_freedom_ok:
    use "Creative Freedom"
else:
    use "Detailed Instructions"
```

---

## 📚 References

- OpenAI Best Practices: https://platform.openai.com/docs/guides/prompt-engineering
- RAG Optimization: https://arxiv.org/abs/2312.10997
- MMR Algorithm: Carbonell & Goldstein (1998)

---

## 🔄 Continuous Improvement

1. **Weekly**: Review generation logs, spot patterns
2. **Monthly**: Re-run optimization suite with new data
3. **Quarterly**: Evaluate model upgrades (GPT-4o → GPT-5)
4. **Ad-hoc**: Test when quality degrades

---

*Last updated: November 8, 2025*
