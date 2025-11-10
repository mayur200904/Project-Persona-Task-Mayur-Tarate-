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
                
                for i, post_text in enumerate(raw_posts):
                    post_text = post_text.strip()
                    if post_text and len(post_text) > 20:  # Ignore very short lines
                        user_posts.append({
                            "id": f"user_post_{i+1}",
                            "text": post_text,
                            "date": "2024",
                            "author": name
                        })
                
                if len(user_posts) < 3:
                    st.warning(f"⚠️ Only found {len(user_posts)} posts. For best results, provide 4-5 posts.")
                
                log(f"✅ Parsed {len(user_posts)} of YOUR posts in {time.time()-start:.2f}s")
                
                # Step 3: Build dynamic index from user's posts
                log("⏱️ Step 3/6: Building personalized index from YOUR style...")
                start = time.time()
                
                client = OpenAI()
                
                # Generate embeddings for user's posts
                texts_to_embed = [post['text'] for post in user_posts]
                response = client.embeddings.create(
                    input=texts_to_embed,
                    model="text-embedding-3-small"
                )
                
                embeddings = np.array([item.embedding for item in response.data], dtype='float32')
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                user_index = faiss.IndexFlatL2(dimension)
                user_index.add(embeddings)
                
                # Create custom retriever with user's data
                retriever = PostRetriever(verbose=False)
                retriever.index = user_index  # Replace with user's index
                retriever.chunks = user_posts  # Replace with user's posts
                
                log(f"✅ Personalized index built in {time.time()-start:.2f}s ({len(user_posts)} YOUR posts indexed)")
                user_provided = True
            
            # Step 4: Retrieve from user's style
            log("⏱️ Step 4/6: Retrieving from YOUR writing style...")
            start = time.time()
            
            persona_info = {
                "name": name,
                "title": title,
                "company": company,
                "industry": industry
            }
            
            chunks = retriever.retrieve_with_context(
                persona_info, 
                topic, 
                top_k=min(5, len(retriever.chunks)),  # Adapt to user's post count
                use_mmr=True
            )
            
            log(f"✅ Retrieved {len(chunks)} examples of YOUR style in {time.time()-start:.2f}s")
            
            # Step 5: Build prompt
            log("⏱️ Step 5/6: Building prompt with YOUR style...")
            start = time.time()
            
            prompt_builder = PromptBuilder()
            prompt_dict = prompt_builder.build_full_prompt(
                persona_info, 
                topic, 
                chunks
            )
            
            log(f"✅ Prompt built in {time.time()-start:.2f}s")
            
            # Step 6: Generate
            log("⏱️ Step 6/6: Generating post in YOUR voice...")
            start = time.time()
            
            generator = PostGenerator(temperature=0.7, max_tokens=500)
            result = generator.generate_with_rag(prompt_dict)
            post = result['post']
            
            log(f"✅ Post generated in {time.time()-start:.2f}s")
            
            # ENHANCEMENT: Extract key phrases/tokens from user's posts used in generation
            log("⏱️ Analyzing which parts of YOUR posts influenced the output...")
            analysis_start = time.time()
            
            used_phrases = []
            used_hashtags = []
            used_words = set()
            
            if user_provided:
                # Extract hashtags from generated post
                generated_hashtags = [word for word in post.split() if word.startswith('#')]
                
                # Find which hashtags came from user's posts
                for chunk in chunks:
                    chunk_text = chunk.get('text', chunk.get('content', ''))
                    chunk_hashtags = [word for word in chunk_text.split() if word.startswith('#')]
                    
                    # Track matching hashtags
                    for hashtag in generated_hashtags:
                        if hashtag in chunk_hashtags and hashtag not in used_hashtags:
                            used_hashtags.append(hashtag)
                    
                    # Extract key phrases (3-5 word sequences) from retrieved chunks
                    words = chunk_text.split()
                    for j in range(len(words) - 2):
                        phrase = ' '.join(words[j:j+3])
                        # Check if this phrase appears in generated post
                        if phrase.lower() in post.lower() and len(phrase) > 10:
                            if phrase not in used_phrases:
                                used_phrases.append(phrase)
                
                # Track common words (excluding stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they', 'it'}
                
                for chunk in chunks:
                    chunk_words = set(chunk.get('text', '').lower().split())
                    generated_words = set(post.lower().split())
                    common = (chunk_words & generated_words) - stop_words
                    used_words.update(common)
            
            log(f"✅ Analysis complete in {time.time()-analysis_start:.2f}s")
            
            # Log to memory with ENHANCED interaction tracking
            log("⏱️ Logging interaction to memory...")
            memory_start = time.time()
            
            memory_manager = MemoryManager(memory_path="memory/memory.json", verbose=False)
            
            # Count hashtags and words
            hashtag_count = post.count('#')
            word_count = len(post.split())
            
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
            
            memory_manager.log_generated_post(interaction_data)
            
            log(f"✅ Logged to memory in {time.time()-memory_start:.2f}s")
            
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
