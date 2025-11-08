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

st.set_page_config(page_title="LinkedIn RAG Agent - TEST", page_icon="🚀", layout="wide")

st.title("🚀 LinkedIn RAG Agent - TEST VERSION")
st.caption("With strict timeouts and detailed logging")

# Sidebar
with st.sidebar:
    st.header("👤 Persona Details")
    name = st.text_input("Name", value="Sundar Pichai")
    title = st.text_input("Role/Title", value="CEO")
    company = st.text_input("Company", value="Google")
    industry = st.text_input("Industry", value="Technology")

# Main area
st.header("📝 Input")

topic = st.text_area(
    "Post Topic",
    value="The importance of developing AI responsibly",
    height=100
)

if st.button("🚀 Generate Post", type="primary"):
    if not topic:
        st.error("Please enter a topic!")
    else:
        log_area = st.empty()
        result_area = st.empty()
        
        def log(msg):
            st.write(f"**{msg}**")
        
        try:
            # Step 1: Import
            log("⏱️ Step 1/5: Importing modules...")
            start = time.time()
            
            from retrieve import PostRetriever
            from prompter import PromptBuilder
            from generate import PostGenerator
            from plagiarism_checker import PlagiarismChecker
            from memory_manager import MemoryManager
            
            log(f"✅ Imports done in {time.time()-start:.2f}s")
            
            # Step 2: Initialize retriever
            log("⏱️ Step 2/5: Initializing retriever...")
            start = time.time()
            
            retriever = PostRetriever(verbose=False)
            
            log(f"✅ Retriever initialized in {time.time()-start:.2f}s (Index: {retriever.index.ntotal} vectors)")
            
            # Step 3: Retrieve
            log("⏱️ Step 3/5: Retrieving similar posts...")
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
                top_k=5,  # Reduced from 10 for speed
                use_mmr=True
            )
            
            log(f"✅ Retrieved {len(chunks)} chunks in {time.time()-start:.2f}s")
            
            # Step 4: Build prompt
            log("⏱️ Step 4/5: Building prompt...")
            start = time.time()
            
            prompt_builder = PromptBuilder()
            prompt_dict = prompt_builder.build_full_prompt(
                persona_info, 
                topic, 
                chunks
            )
            
            log(f"✅ Prompt built in {time.time()-start:.2f}s")
            
            # Step 5: Generate
            log("⏱️ Step 5/5: Generating post with GPT...")
            start = time.time()
            
            generator = PostGenerator(temperature=0.7, max_tokens=500)
            result = generator.generate_with_rag(prompt_dict)
            post = result['post']
            
            log(f"✅ Post generated in {time.time()-start:.2f}s")
            
            # Log to memory
            log("⏱️ Logging to memory...")
            memory_start = time.time()
            
            memory_manager = MemoryManager(memory_path="memory/memory.json", verbose=False)
            
            # Count hashtags and words
            hashtag_count = post.count('#')
            word_count = len(post.split())
            
            memory_manager.log_generated_post({
                'post': post,
                'topic': topic,
                'method': 'RAG',
                'word_count': word_count,
                'hashtag_count': hashtag_count,
                'persona': f"{name} - {title} at {company}"
            })
            
            log(f"✅ Logged to memory in {time.time()-memory_start:.2f}s")
            
            # Show result
            st.success("🎉 Generation Complete!")
            st.markdown("### 📄 Generated Post:")
            st.info(post)
            
            # Show stats
            st.markdown("### 📊 Stats:")
            st.write(f"- Post length: {len(post)} characters")
            st.write(f"- Chunks used: {len(chunks)}")
            
            # Show retrieved context
            with st.expander("🔍 Retrieved Context"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk.get('text', chunk.get('content', 'No text available'))[:200] + "...")
                    st.divider()
            
        except Exception as e:
            st.error(f"❌ ERROR: {str(e)}")
            with st.expander("🔍 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.caption("Test version with detailed timing for each step")
