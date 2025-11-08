"""
Streamlit Web Interface
Interactive UI for LinkedIn RAG Agent
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json

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
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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


@st.cache_resource
def load_components():
    """Load all necessary components with optimized settings"""
    try:
        # Check if index exists
        if not os.path.exists("data/vector_store.index"):
            st.error("⚠️ Vector store not found! Please run the setup first.")
            st.info("Run: `python src/ingest.py` and `python src/indexer.py`")
            return None, None, None, None, None, {}
        
        # Load optimized config if available
        optimized_config = {}
        if os.path.exists("eval/optimized_config.json"):
            with open("eval/optimized_config.json", 'r') as f:
                optimized_config = json.load(f)
            st.sidebar.success("✅ Using optimized settings")
        
        # Initialize with optimized or default settings
        model_config = optimized_config.get('model_config', {})
        retriever = PostRetriever(verbose=False)  # Suppress console output
        prompt_builder = PromptBuilder()
        generator = PostGenerator(
            temperature=model_config.get('temperature', 0.7),
            max_tokens=model_config.get('max_tokens', 500)
        )
        plagiarism_checker = PlagiarismChecker(threshold=15)
        memory_manager = MemoryManager(verbose=False)  # Suppress console output
        
        return retriever, prompt_builder, generator, plagiarism_checker, memory_manager, optimized_config
    
    except Exception as e:
        st.error(f"❌ Error loading components: {str(e)}")
        return None, None, None, None, None, {}


def main():
    """Main Streamlit application"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">✍️ LinkedIn RAG Agent</div>', unsafe_allow_html=True)
    st.markdown("**Generate persona-based LinkedIn posts with AI-powered style matching**")
    
    # Load components
    with st.spinner("🔄 Loading AI components..."):
        components = load_components()
        if components[0] is None:
            st.stop()
        retriever, prompt_builder, generator, plagiarism_checker, memory_manager, optimized_config = components
    
    # Sidebar - Persona Configuration
    with st.sidebar:
        st.header("👤 Persona Configuration")
        
        persona = memory_manager.get_persona_info()
        
        name = st.text_input("Name", value=persona['name'])
        title = st.text_input("Title", value=persona['title'])
        company = st.text_input("Company", value=persona['company'])
        industry = st.text_input("Industry", value=persona['industry'])
        
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
        
        # Load optimized retrieval config
        model_config = optimized_config.get('model_config', {})
        retrieval_config = optimized_config.get('retrieval_config', {})
        
        use_mmr = st.checkbox("Use MMR (Diverse Retrieval)", value=retrieval_config.get('use_mmr', True))
        top_k = st.slider("Retrieved Chunks", 3, 10, retrieval_config.get('top_k', 5))
        temperature = st.slider("Creativity", 0.0, 1.0, model_config.get('temperature', 0.7), 0.1)
        
        # Show optimization status
        if optimized_config:
            st.info(f"🎯 Optimized settings loaded\n- Temperature: {model_config.get('temperature', 0.7)}\n- Top-K: {retrieval_config.get('top_k', 5)}\n- Strategy: {optimized_config.get('prompt_strategy', 'default')}")
        
        st.divider()
        
        # Memory context
        if st.checkbox("📚 Show Memory Context"):
            st.text(memory_manager.get_context_summary())
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        topic = st.text_area(
            "Post Topic / Bullet Points",
            placeholder="E.g., AI innovation in healthcare, ethical considerations, team collaboration...",
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
                # Create progress steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Retrieve
                    status_text.text("🔍 Retrieving relevant context...")
                    progress_bar.progress(20)
                    
                    # Build persona info
                    persona_info = {
                        "name": name,
                        "title": title,
                        "company": company,
                        "industry": industry
                    }
                    
                    # Retrieve relevant chunks
                    chunks = retriever.retrieve_with_context(
                        persona_info, 
                        topic, 
                        top_k=top_k, 
                        use_mmr=use_mmr
                    )
                    progress_bar.progress(40)
                    
                    # Step 2: Build prompt
                    status_text.text("✍️ Building prompt...")
                    prompt = prompt_builder.build_full_prompt(
                        persona_info, 
                        topic, 
                        chunks, 
                        additional_context
                    )
                    progress_bar.progress(50)
                    
                    # Step 3: Generate
                    status_text.text("🤖 Generating post (this may take 5-10 seconds)...")
                    generator.temperature = temperature
                    result = generator.generate_with_rag(prompt)
                    progress_bar.progress(80)
                    
                    # Step 4: Check plagiarism
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
                    
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")
        
        # Display generated post
        if st.session_state.generated_post:
            st.markdown('<div class="post-output">', unsafe_allow_html=True)
            st.write(st.session_state.generated_post)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Copy button
            st.code(st.session_state.generated_post, language=None)
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                word_count = len(st.session_state.generated_post.split())
                st.metric("Words", word_count)
            
            with col_m2:
                hashtag_count = st.session_state.generated_post.count('#')
                st.metric("Hashtags", hashtag_count)
            
            with col_m3:
                st.metric("Generated", st.session_state.generation_count)
    
    # Additional sections
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📚 Retrieved Context", "🔍 Plagiarism Check", "📊 Stats"])
    
    with tab1:
        if st.session_state.used_chunks:
            st.subheader("Retrieved Chunks Used for Generation")
            for i, chunk in enumerate(st.session_state.used_chunks, 1):
                with st.expander(f"Chunk {i} - Score: {chunk.get('similarity_score', chunk.get('mmr_score', 'N/A'))}"):
                    st.write(chunk['text'])
                    st.caption(f"Post ID: {chunk['post_id']} | Date: {chunk['date']}")
        else:
            st.info("Generate a post to see retrieved context")
    
    with tab2:
        if st.session_state.plagiarism_check:
            st.text(st.session_state.plagiarism_check)
        else:
            st.info("Generate a post to see plagiarism check")
    
    with tab3:
        st.subheader("Generation Statistics")
        stats = generator.get_statistics()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.json(stats)
        
        with col_s2:
            recent_posts = memory_manager.get_recent_posts(5)
            st.write(f"**Recent Posts:** {len(recent_posts)}")
            for post in recent_posts:
                st.caption(f"• {post['timestamp'][:10]}: {post['topic'][:50]}...")


if __name__ == "__main__":
    main()
