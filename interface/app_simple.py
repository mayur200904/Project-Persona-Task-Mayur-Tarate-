"""
Streamlit Web Interface - Simplified Version
Interactive UI for LinkedIn RAG Agent with reliable performance
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from retrieve import PostRetriever
from prompter import PromptBuilder
from generate import PostGenerator
from plagiarism_checker import PlagiarismChecker
from memory_manager import MemoryManager


# Page configuration
st.set_page_config(
    page_title="LinkedIn RAG Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0077B5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .post-output {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0077B5;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #0077B5;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'generated_post' not in st.session_state:
        st.session_state.generated_post = None
    if 'used_chunks' not in st.session_state:
        st.session_state.used_chunks = []
    if 'plagiarism_check' not in st.session_state:
        st.session_state.plagiarism_check = None
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0


def main():
    """Main Streamlit application"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">✍️ LinkedIn RAG Agent</div>', unsafe_allow_html=True)
    st.markdown("**Generate persona-based LinkedIn posts with AI-powered style matching**")
    
    # Check if index exists
    if not os.path.exists("data/vector_store.index"):
        st.error("⚠️ Vector store not found! Please run the setup first.")
        st.info("Run: `python setup_pipeline.py`")
        st.stop()
    
    # Load optimized config if available
    optimized_config = {}
    if os.path.exists("eval/optimized_config.json"):
        with open("eval/optimized_config.json", 'r') as f:
            optimized_config = json.load(f)
    
    model_config = optimized_config.get('model_config', {})
    retrieval_config = optimized_config.get('retrieval_config', {})
    
    # Sidebar - Persona Configuration
    with st.sidebar:
        st.header("👤 Persona Configuration")
        
        # Load memory for defaults
        memory_manager = MemoryManager(verbose=False)
        persona = memory_manager.get_persona_info()
        
        name = st.text_input("Name", value=persona.get('name', 'Your Name'))
        title = st.text_input("Title", value=persona.get('title', 'Your Title'))
        company = st.text_input("Company", value=persona.get('company', 'Your Company'))
        industry = st.text_input("Industry", value=persona.get('industry', 'Technology'))
        
        # Update persona if changed
        if st.button("💾 Save Persona"):
            memory_manager.update_persona({
                "name": name,
                "title": title,
                "company": company,
                "industry": industry
            })
            st.success("✅ Persona saved!")
        
        st.divider()
        
        # Generation settings
        st.header("⚙️ Settings")
        
        use_mmr = st.checkbox("Use MMR (Diverse Retrieval)", 
                             value=retrieval_config.get('use_mmr', True))
        top_k = st.slider("Retrieved Chunks", 3, 10, 
                         retrieval_config.get('top_k', 5))
        temperature = st.slider("Creativity", 0.0, 1.0, 
                               model_config.get('temperature', 0.7), 0.1)
        
        # Show optimization status
        if optimized_config:
            st.success("✅ Using optimized settings")
            with st.expander("📊 Optimized Config"):
                st.json(optimized_config)
        
        st.divider()
        
        # Statistics
        st.header("📊 Stats")
        st.metric("Posts Generated", st.session_state.generation_count)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        topic = st.text_area(
            "Post Topic / Bullet Points",
            placeholder="Enter the main topic or key points for your LinkedIn post...",
            height=150
        )
        
        additional_context = st.text_area(
            "Additional Context (Optional)",
            placeholder="Any specific points, recent events, or details to include...",
            height=100
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            generate_btn = st.button("🚀 Generate Post", type="primary", use_container_width=True)
        
        with col_btn2:
            regenerate_btn = st.button("🔄 Regenerate", use_container_width=True)
    
    with col2:
        st.header("✨ Generated Post")
        
        if generate_btn or regenerate_btn:
            if not topic:
                st.warning("⚠️ Please enter a topic for the post.")
            else:
                # Create placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Build persona info
                    persona_info = {
                        "name": name,
                        "title": title,
                        "company": company,
                        "industry": industry
                    }
                    
                    # Step 1: Initialize components
                    status_text.text("🔧 Initializing AI components...")
                    progress_bar.progress(10)
                    
                    retriever = PostRetriever(verbose=False)
                    prompt_builder = PromptBuilder()
                    generator = PostGenerator(
                        temperature=temperature,
                        max_tokens=model_config.get('max_tokens', 500)
                    )
                    plagiarism_checker = PlagiarismChecker(threshold=15)
                    
                    progress_bar.progress(20)
                    
                    # Step 2: Retrieve
                    status_text.text("🔍 Retrieving relevant context...")
                    
                    start_time = time.time()
                    chunks = retriever.retrieve_with_context(
                        persona_info, 
                        topic, 
                        top_k=top_k, 
                        use_mmr=use_mmr
                    )
                    retrieval_time = time.time() - start_time
                    
                    progress_bar.progress(40)
                    status_text.text(f"✅ Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
                    time.sleep(0.5)
                    
                    # Step 3: Build prompt
                    status_text.text("✍️ Building prompt...")
                    prompt = prompt_builder.build_full_prompt(
                        persona_info, 
                        topic, 
                        chunks, 
                        additional_context
                    )
                    progress_bar.progress(50)
                    
                    # Step 4: Generate
                    status_text.text("🤖 Generating post (5-15 seconds)...")
                    
                    start_time = time.time()
                    result = generator.generate_with_rag(prompt)
                    gen_time = time.time() - start_time
                    
                    progress_bar.progress(80)
                    status_text.text(f"✅ Generated in {gen_time:.2f}s")
                    time.sleep(0.5)
                    
                    # Step 5: Check plagiarism
                    status_text.text("🔍 Checking for plagiarism...")
                    is_plagiarized, explanation = plagiarism_checker.check_with_explanation(
                        result['post'], 
                        chunks
                    )
                    
                    # If plagiarized and regenerating, try paraphrase
                    if is_plagiarized and regenerate_btn:
                        status_text.text("🔄 Paraphrasing to avoid plagiarism...")
                        paraphrase_prompt = prompt_builder.build_paraphrase_prompt(
                            result['post'], 
                            chunks
                        )
                        result['post'] = generator.regenerate_with_paraphrase(paraphrase_prompt)
                        
                        # Recheck
                        is_plagiarized, explanation = plagiarism_checker.check_with_explanation(
                            result['post'], 
                            chunks
                        )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    time.sleep(0.5)
                    
                    # Save to session state
                    st.session_state.generated_post = result['post']
                    st.session_state.used_chunks = chunks
                    st.session_state.plagiarism_check = explanation
                    st.session_state.generation_count += 1
                    
                    # Log to memory
                    memory_manager.log_generated_post({
                        **result,
                        'topic': topic
                    })
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show success
                    st.success("✅ Post generated successfully!")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Generation error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        
        # Display generated post
        if st.session_state.generated_post:
            st.markdown('<div class="post-output">', unsafe_allow_html=True)
            st.write(st.session_state.generated_post)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Copy button
            st.text_area("Copy Post", st.session_state.generated_post, height=200, key="copy_area")
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            
            word_count = len(st.session_state.generated_post.split())
            hashtag_count = st.session_state.generated_post.count('#')
            
            with col_m1:
                st.metric("Word Count", word_count)
            
            with col_m2:
                st.metric("Hashtags", hashtag_count)
            
            with col_m3:
                plagiarism_status = "⚠️ Issues" if st.session_state.plagiarism_check.get('has_overlap', False) else "✅ Clean"
                st.metric("Plagiarism", plagiarism_status)
            
            # Show tabs for additional info
            tab1, tab2, tab3 = st.tabs(["📚 Retrieved Context", "🔍 Plagiarism Check", "📊 Stats"])
            
            with tab1:
                st.subheader("Retrieved Context Chunks")
                for i, chunk in enumerate(st.session_state.used_chunks, 1):
                    with st.expander(f"Chunk {i} - Score: {chunk.get('similarity_score', 'N/A')}"):
                        st.write(chunk['text'])
                        st.caption(f"Post ID: {chunk.get('post_id', 'N/A')} | Words: {chunk.get('word_count', 'N/A')}")
            
            with tab2:
                st.subheader("Plagiarism Check Results")
                if st.session_state.plagiarism_check:
                    st.json(st.session_state.plagiarism_check)
                else:
                    st.info("No plagiarism check performed yet.")
            
            with tab3:
                st.subheader("Generation Statistics")
                stats = {
                    "Total Generations": st.session_state.generation_count,
                    "Current Word Count": word_count,
                    "Current Hashtag Count": hashtag_count,
                    "Retrieved Chunks Used": len(st.session_state.used_chunks)
                }
                st.json(stats)


if __name__ == "__main__":
    main()
