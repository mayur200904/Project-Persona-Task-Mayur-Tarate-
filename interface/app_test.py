"""
Streamlit Test Version - Ultra Simple with Timeouts
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="LinkedIn RAG Agent - Personalized", page_icon="🚀", layout="wide")

st.title("🚀 LinkedIn RAG Agent - Your Personal Style")
st.caption("✨ Upload YOUR LinkedIn posts → Generate in YOUR unique voice")

# Sidebar
with st.sidebar:
    st.header("👤 Your Persona Details")
    name = st.text_input("Name", value="Sundar Pichai")
    title = st.text_input("Role/Title", value="CEO")
    company = st.text_input("Company", value="Google")
    industry = st.text_input("Industry", value="Technology")

# Main area - CRITICAL: User Upload Section
st.header("� Step 1: Upload YOUR LinkedIn Posts")
st.markdown("""
**🎯 This is the core feature:** Paste 4-5 of YOUR past LinkedIn posts below.  
The model will learn YOUR unique writing style, tone, and voice to generate new posts that sound like YOU.
""")

user_posts_input = st.text_area(
    "Paste Your 4-5 LinkedIn Posts (one per line, or separated by blank lines)",
    height=200,
    placeholder="""Example format:
    
Post 1: Just shipped our new feature! After 6 months of work, we finally...

Post 2: Reflecting on my journey in tech. When I started coding...

Post 3: Big announcement! Our team just reached 1M users...

(Paste your actual LinkedIn posts here)"""
)

use_sample_data = st.checkbox("🔧 Use demo data instead (for testing only)", value=False)

st.divider()
st.header("📝 Step 2: What Do You Want to Post About?")

topic = st.text_area(
    "Post Topic",
    value="The importance of developing AI responsibly",
    height=100
)

if st.button("🚀 Generate Post in YOUR Style", type="primary"):
    # Validation
    if not topic:
        st.error("❌ Please enter a topic!")
    elif not user_posts_input.strip() and not use_sample_data:
        st.error("❌ Please paste YOUR 4-5 LinkedIn posts, or check 'Use demo data' for testing!")
    else:
        log_area = st.empty()
        result_area = st.empty()
        
        def log(msg):
            st.write(f"**{msg}**")
        
        try:
            # Track overall generation time
            generation_start_time = time.time()
            
            # Step 1: Import
            log("⏱️ Step 1/6: Importing modules...")
            start = time.time()
            
            from retrieve import PostRetriever
            from prompter import PromptBuilder
            from generate import PostGenerator
            from plagiarism_checker import PlagiarismChecker
            from memory_manager import MemoryManager
            import json
            import numpy as np
            import faiss
            from openai import OpenAI
            
            log(f"✅ Imports done in {time.time()-start:.2f}s")
            
            # Step 2: Parse user posts OR use sample data
            log("⏱️ Step 2/6: Processing YOUR posts...")
            start = time.time()
            
            if use_sample_data:
                # Load sample data for demo
                retriever = PostRetriever(verbose=False)
                log(f"✅ Using demo data: {retriever.index.ntotal} sample posts")
                user_provided = False
            else:
                # Parse user's posts
                user_posts = []
                raw_posts = user_posts_input.strip().split('\n\n')  # Split by blank lines
                
                print("\n" + "="*70)
                print("📥 STEP 2: PARSING USER INPUT")
                print("="*70)
                print(f"📝 Raw input length: {len(user_posts_input)} characters")
                print(f"📄 Split into {len(raw_posts)} potential posts (by blank lines)")
                
                for i, post_text in enumerate(raw_posts):
                    post_text = post_text.strip()
                    if post_text and len(post_text) > 20:  # Ignore very short lines
                        user_posts.append({
                            "id": f"user_post_{i+1}",
                            "text": post_text,
                            "date": "2024",
                            "author": name
                        })
                        print(f"  ✅ Post #{i+1}: {len(post_text)} chars, {len(post_text.split())} words")
                        print(f"     Preview: {post_text[:80]}...")
                    else:
                        print(f"  ⚠️ Skipped chunk #{i+1}: Too short ({len(post_text)} chars)")
                
                print(f"\n✅ Successfully parsed {len(user_posts)} posts")
                print(f"📊 Total content: {sum(len(p['text']) for p in user_posts)} characters")
                print("="*70 + "\n")
                
                if len(user_posts) < 3:
                    st.warning(f"⚠️ Only found {len(user_posts)} posts. For best results, provide 4-5 posts.")
                
                log(f"✅ Parsed {len(user_posts)} of YOUR posts in {time.time()-start:.2f}s")
                
                # Step 3: Build dynamic index from user's posts
                log("⏱️ Step 3/6: Building personalized index from YOUR style...")
                start = time.time()
                
                print("\n" + "="*70)
                print("🔄 STEP 3: GENERATING EMBEDDINGS & BUILDING INDEX")
                print("="*70)
                
                client = OpenAI()
                
                # Generate embeddings for user's posts
                texts_to_embed = [post['text'] for post in user_posts]
                
                print(f"📤 Sending {len(texts_to_embed)} texts to OpenAI for embedding...")
                print(f"🤖 Model: text-embedding-3-small (1536 dimensions)")
                
                for i, text in enumerate(texts_to_embed):
                    word_count = len(text.split())
                    char_count = len(text)
                    print(f"  📝 Text #{i+1}: {word_count} words, {char_count} chars")
                
                response = client.embeddings.create(
                    input=texts_to_embed,
                    model="text-embedding-3-small"
                )
                
                print(f"\n✅ Received {len(response.data)} embeddings from OpenAI")
                
                embeddings = np.array([item.embedding for item in response.data], dtype='float32')
                
                print(f"📊 Embeddings shape: {embeddings.shape}")
                print(f"   - {embeddings.shape[0]} vectors")
                print(f"   - {embeddings.shape[1]} dimensions each")
                print(f"   - Data type: {embeddings.dtype}")
                print(f"   - Memory size: {embeddings.nbytes / 1024:.2f} KB")
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                user_index = faiss.IndexFlatL2(dimension)
                
                print(f"\n🔨 Building FAISS index...")
                print(f"   - Index type: IndexFlatL2 (L2 distance)")
                print(f"   - Dimension: {dimension}")
                
                user_index.add(embeddings)
                
                print(f"✅ FAISS index built successfully")
                print(f"   - Total vectors indexed: {user_index.ntotal}")
                print(f"   - Index trained: {user_index.is_trained}")
                
                # Create custom retriever with user's data
                retriever = PostRetriever(verbose=False)
                retriever.index = user_index  # Replace with user's index
                retriever.chunks = user_posts  # Replace with user's posts
                
                print(f"\n✅ Personalized retriever ready with YOUR {len(user_posts)} posts")
                print("="*70 + "\n")
                
                log(f"✅ Personalized index built in {time.time()-start:.2f}s ({len(user_posts)} YOUR posts indexed)")
                user_provided = True
            
            # Step 4: Retrieve from user's style
            log("⏱️ Step 4/6: Retrieving from YOUR writing style...")
            start = time.time()
            
            print("\n" + "="*70)
            print("🔍 STEP 4: RETRIEVAL FROM YOUR POSTS")
            print("="*70)
            
            persona_info = {
                "name": name,
                "title": title,
                "company": company,
                "industry": industry
            }
            
            print(f"🎯 Query topic: '{topic}'")
            print(f"👤 Persona: {name} - {title} at {company}")
            print(f"📊 Retrieval settings:")
            print(f"   - Top-K: {min(5, len(retriever.chunks))}")
            print(f"   - MMR (diversity): Enabled")
            print(f"   - Lambda: 0.9")
            
            chunks = retriever.retrieve_with_context(
                persona_info, 
                topic, 
                top_k=min(5, len(retriever.chunks)),  # Adapt to user's post count
                use_mmr=True
            )
            
            print(f"\n✅ Retrieved {len(chunks)} relevant chunks")
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', chunk.get('content', ''))
                print(f"\n  📄 Chunk #{i+1} (ID: {chunk.get('id', 'N/A')})")
                print(f"     Length: {len(chunk_text)} chars, {len(chunk_text.split())} words")
                print(f"     Preview: {chunk_text[:100]}...")
            
            print("="*70 + "\n")
            
            log(f"✅ Retrieved {len(chunks)} examples of YOUR style in {time.time()-start:.2f}s")
            
            # Step 5: Build prompt
            log("⏱️ Step 5/6: Building prompt with YOUR style...")
            start = time.time()
            
            print("\n" + "="*70)
            print("✍️  STEP 5: BUILDING PROMPT")
            print("="*70)
            
            prompt_builder = PromptBuilder()
            prompt_dict = prompt_builder.build_full_prompt(
                persona_info, 
                topic, 
                chunks
            )
            
            print(f"📝 Prompt components:")
            print(f"   - System message: {len(prompt_dict.get('system', ''))} chars")
            print(f"   - User message: {len(prompt_dict.get('user', ''))} chars")
            print(f"   - Context chunks: {len(chunks)}")
            print(f"   - Topic: '{topic}'")
            
            total_prompt_length = len(prompt_dict.get('system', '')) + len(prompt_dict.get('user', ''))
            estimated_tokens = total_prompt_length // 4  # Rough estimate: 1 token ≈ 4 chars
            print(f"\n📊 Estimated prompt tokens: ~{estimated_tokens}")
            print(f"   (Total chars: {total_prompt_length})")
            print("="*70 + "\n")
            
            log(f"✅ Prompt built in {time.time()-start:.2f}s")
            
            # Step 6: Generate
            log("⏱️ Step 6/6: Generating post in YOUR voice...")
            start = time.time()
            
            print("\n" + "="*70)
            print("🤖 STEP 6: LLM GENERATION")
            print("="*70)
            print(f"🔧 Model: GPT-4o-mini")
            print(f"🌡️  Temperature: 0.7")
            print(f"📏 Max tokens: 500")
            print(f"\n🔄 Sending request to OpenAI...")
            
            generator = PostGenerator(temperature=0.7, max_tokens=500)
            result = generator.generate_with_rag(prompt_dict)
            post = result['post']
            
            # Count tokens in generated post (rough estimate)
            generated_tokens = len(post) // 4
            generated_words = len(post.split())
            generated_chars = len(post)
            
            print(f"\n✅ Post generated successfully!")
            print(f"📊 Generated content stats:")
            print(f"   - Characters: {generated_chars}")
            print(f"   - Words: {generated_words}")
            print(f"   - Estimated tokens: ~{generated_tokens}")
            print(f"   - Hashtags: {post.count('#')}")
            print(f"   - Line breaks: {post.count(chr(10))}")
            print("="*70 + "\n")
            
            log(f"✅ Post generated in {time.time()-start:.2f}s")
            
            # ENHANCEMENT: Extract key phrases/tokens from user's posts used in generation
            log("⏱️ Analyzing which parts of YOUR posts influenced the output...")
            analysis_start = time.time()
            
            print("\n" + "="*70)
            print("🔍 STEP 7: TRANSPARENCY ANALYSIS")
            print("="*70)
            
            used_phrases = []
            used_hashtags = []
            used_words = set()
            
            if user_provided:
                print(f"📊 Analyzing influence from YOUR {len(user_posts)} posts...")
                
                # Extract hashtags from generated post
                generated_hashtags = [word for word in post.split() if word.startswith('#')]
                print(f"\n1️⃣ Hashtag Analysis:")
                print(f"   - Generated post has {len(generated_hashtags)} hashtags: {generated_hashtags}")
                
                # Find which hashtags came from user's posts
                for chunk in chunks:
                    chunk_text = chunk.get('text', chunk.get('content', ''))
                    chunk_hashtags = [word for word in chunk_text.split() if word.startswith('#')]
                    
                    # Track matching hashtags
                    for hashtag in generated_hashtags:
                        if hashtag in chunk_hashtags and hashtag not in used_hashtags:
                            used_hashtags.append(hashtag)
                
                print(f"   - Matched {len(used_hashtags)} hashtags from YOUR posts: {used_hashtags}")
                
                print(f"\n2️⃣ Phrase Analysis (3-word sequences):")
                # Extract key phrases (3-5 word sequences) from retrieved chunks
                for chunk in chunks:
                    chunk_text = chunk.get('text', chunk.get('content', ''))
                    words = chunk_text.split()
                    for j in range(len(words) - 2):
                        phrase = ' '.join(words[j:j+3])
                        # Check if this phrase appears in generated post
                        if phrase.lower() in post.lower() and len(phrase) > 10:
                            if phrase not in used_phrases:
                                used_phrases.append(phrase)
                
                print(f"   - Found {len(used_phrases)} matching phrases")
                if used_phrases:
                    for i, phrase in enumerate(used_phrases[:5]):  # Show first 5
                        print(f"     • '{phrase}'")
                    if len(used_phrases) > 5:
                        print(f"     ... and {len(used_phrases) - 5} more")
                
                print(f"\n3️⃣ Vocabulary Analysis:")
                # Track common words (excluding stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they', 'it'}
                
                for chunk in chunks:
                    chunk_words = set(chunk.get('text', '').lower().split())
                    generated_words = set(post.lower().split())
                    common = (chunk_words & generated_words) - stop_words
                    used_words.update(common)
                
                print(f"   - Common vocabulary: {len(used_words)} words")
                sample_words = sorted(list(used_words))[:15]
                print(f"   - Sample: {', '.join(sample_words)}{'...' if len(used_words) > 15 else ''}")
            
            print(f"\n✅ Transparency analysis complete!")
            print(f"   - Hashtag matches: {len(used_hashtags)}")
            print(f"   - Phrase matches: {len(used_phrases)}")
            print(f"   - Common words: {len(used_words)}")
            print("="*70 + "\n")
            
            log(f"✅ Analysis complete in {time.time()-analysis_start:.2f}s")
            
            # Log to memory with ENHANCED interaction tracking
            log("⏱️ Logging interaction to memory...")
            memory_start = time.time()
            
            print("\n" + "="*70)
            print("💾 STEP 8: MEMORY LOGGING")
            print("="*70)
            
            memory_manager = MemoryManager(memory_path="memory/memory.json", verbose=False)
            
            # Count hashtags and words
            hashtag_count = post.count('#')
            word_count = len(post.split())
            
            print(f"📋 Preparing memory entry...")
            print(f"   - Generated post: {len(post)} characters, {word_count} words, {hashtag_count} hashtags")
            print(f"   - Topic: '{topic}'")
            print(f"   - Method: {'RAG_USER_PROVIDED' if user_provided else 'RAG_DEMO'}")
            print(f"   - Persona: {name} - {title} at {company}")
            
            if user_provided:
                print(f"\n📊 User Input Section:")
                print(f"   - Raw posts provided: {len(user_posts_input)} characters")
                print(f"   - Posts parsed: {len(user_posts)}")
                print(f"   - First 100 chars: {user_posts_input[:100]}...")
            
            print(f"\n🔍 RAG Influence Section:")
            print(f"   - Chunks retrieved: {len(chunks)}")
            if user_provided:
                print(f"   - Hashtags reused: {len(used_hashtags)} → {used_hashtags}")
                print(f"   - Phrases matched: {len(used_phrases)}")
                if used_phrases:
                    print(f"     Sample: {used_phrases[:3]}")
                print(f"   - Common words: {len(used_words)}")
            
            print(f"\n📦 Retrieved Chunks Section:")
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk.get('text', chunk.get('content', ''))[:80]
                print(f"   - Chunk {i+1}: {preview}...")
            if len(chunks) > 3:
                print(f"   - ... and {len(chunks) - 3} more chunks")
            
            # ENHANCED: Log detailed interaction data
            interaction_data = {
                'post': post,
                'topic': topic,
                'method': 'RAG_USER_PROVIDED' if user_provided else 'RAG_DEMO',
                'word_count': word_count,
                'hashtag_count': hashtag_count,
                'persona': f"{name} - {title} at {company}",
                'user_posts_count': len(user_posts) if user_provided else 0,
                # NEW: User interaction tracking
                'user_input': {
                    'raw_posts_provided': user_posts_input[:500] if user_provided else None,  # First 500 chars
                    'posts_parsed': len(user_posts) if user_provided else 0,
                    'total_input_length': len(user_posts_input) if user_provided else 0
                },
                # NEW: RAG influence tracking
                'rag_influence': {
                    'chunks_retrieved': len(chunks),
                    'used_hashtags': used_hashtags if user_provided else [],
                    'used_phrases': used_phrases[:10] if user_provided else [],  # Top 10
                    'common_words_count': len(used_words) if user_provided else 0
                },
                # NEW: Retrieved content for reference
                'retrieved_chunks': [
                    {
                        'id': chunk.get('id', f'chunk_{i}'),
                        'text_preview': chunk.get('text', chunk.get('content', ''))[:200]
                    }
                    for i, chunk in enumerate(chunks)
                ]
            }
            
            print(f"\n💾 Writing to memory/memory.json...")
            memory_manager.log_generated_post(interaction_data)
            
            # Calculate memory file size
            import os
            memory_file = "memory/memory.json"
            file_size_kb = 0
            if os.path.exists(memory_file):
                file_size_kb = os.path.getsize(memory_file) / 1024
            
            print(f"✅ Successfully logged to memory!")
            print(f"   - Memory file size: {file_size_kb:.2f} KB")
            print(f"   - Total sections: 3 (user_input, rag_influence, retrieved_chunks)")
            estimated_entry_tokens = (len(str(interaction_data)) // 4)
            print(f"   - Estimated tokens logged: ~{estimated_entry_tokens}")
            print("="*70 + "\n")
            
            log(f"✅ Logged to memory in {time.time()-memory_start:.2f}s")
            
            print("\n" + "="*70)
            print("🎉 GENERATION COMPLETE!")
            print("="*70)
            total_time = time.time() - generation_start_time
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"📊 Steps: Parse → Embed → Retrieve → Generate → Analyze → Log")
            if user_provided:
                print(f"✨ This post reflects YOUR unique style from {len(user_posts)} posts!")
            else:
                print(f"ℹ️  Using demo data - upload YOUR posts for true personalization")
            print("="*70 + "\n")
            
            # Show result
            if user_provided:
                st.success("🎉 Generation Complete! This post matches YOUR unique style.")
            else:
                st.success("🎉 Generation Complete! (Using demo data - upload YOUR posts for personalization)")
                
            st.markdown("### 📄 Generated Post:")
            st.info(post)
            
            # Show stats
            st.markdown("### 📊 Generation Stats:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", word_count)
            with col2:
                st.metric("Characters", len(post))
            with col3:
                st.metric("Hashtags", hashtag_count)
            with col4:
                st.metric("Style Sources", len(chunks))
            
            if user_provided:
                st.success(f"✅ **Personalized using YOUR {len(user_posts)} posts**")
            
            # NEW: Show what was used from user's posts (TRANSPARENCY FEATURE)
            if user_provided and (used_hashtags or used_phrases or used_words):
                st.markdown("### 🔍 What We Used from YOUR Posts:")
                st.markdown("""
                <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                This shows exactly which elements from YOUR posts influenced the generated content.
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2, tab3 = st.tabs(["📌 Hashtags", "💬 Phrases", "🔤 Common Words"])
                
                with tab1:
                    if used_hashtags:
                        st.markdown("**Hashtags from YOUR posts that appear in the generated post:**")
                        for hashtag in used_hashtags:
                            st.markdown(f"- `{hashtag}`")
                    else:
                        st.info("No exact hashtag matches (model created new ones based on your style)")
                
                with tab2:
                    if used_phrases:
                        st.markdown(f"**Key phrases from YOUR posts found in output:** ({len(used_phrases)} matches)")
                        for phrase in used_phrases[:8]:  # Show top 8
                            st.markdown(f"- *\"{phrase}\"*")
                        if len(used_phrases) > 8:
                            st.caption(f"+ {len(used_phrases) - 8} more phrases...")
                    else:
                        st.info("No exact phrase matches (model learned your style patterns instead)")
                
                with tab3:
                    if used_words:
                        st.markdown(f"**Vocabulary overlap:** {len(used_words)} words from YOUR posts")
                        # Show sample of words
                        sample_words = sorted(list(used_words))[:20]
                        st.markdown(", ".join([f"`{word}`" for word in sample_words]))
                        if len(used_words) > 20:
                            st.caption(f"+ {len(used_words) - 20} more words...")
                    else:
                        st.info("Model used new vocabulary while maintaining your tone")
            
            # Show retrieved context
            with st.expander("� Retrieved Examples from YOUR Posts" if user_provided else "� Retrieved Context"):
                st.markdown("*These are the examples the AI used to learn your style:*")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"#### Example {i}:")
                    chunk_text = chunk.get('text', chunk.get('content', 'No text available'))
                    st.markdown(f"```\n{chunk_text[:400]}{'...' if len(chunk_text) > 400 else ''}\n```")
                    st.divider()
            
        except Exception as e:
            st.error(f"❌ ERROR: {str(e)}")
            with st.expander("🔍 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.markdown("""
### 🎯 How This Works:
1. **Upload YOUR Posts:** Paste 4-5 of your actual LinkedIn posts above
2. **Dynamic Indexing:** The system creates embeddings from YOUR posts in real-time
3. **Style Learning:** RAG retrieves examples from YOUR writing style
4. **Personalized Generation:** GPT-4 generates new posts that sound like YOU

**This is the core innovation:** Unlike generic AI writers, this learns YOUR unique voice! 🚀
""")
st.caption("✨ Personalized LinkedIn RAG Agent - Your Style, Your Voice")
