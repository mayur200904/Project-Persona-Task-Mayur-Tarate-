"""
Setup and Pipeline Script
Runs the complete RAG pipeline
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and display status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        print(f"✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup and run pipeline"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║         LinkedIn RAG Agent - Setup & Pipeline              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("\n⚠️  No .env file found!")
        print("📝 Please create a .env file with your OPENAI_API_KEY")
        print("   You can copy .env.example to .env and add your key")
        
        create_env = input("\n❓ Create .env file now? (y/n): ")
        if create_env.lower() == 'y':
            api_key = input("🔑 Enter your OpenAI API key: ")
            
            with open('.env', 'w') as f:
                f.write(f"# OpenAI API Configuration\n")
                f.write(f"OPENAI_API_KEY={api_key}\n\n")
                f.write(f"# Model Configuration\n")
                f.write(f"EMBEDDING_MODEL=text-embedding-3-small\n")
                f.write(f"GENERATION_MODEL=gpt-4o-mini\n\n")
                f.write(f"# Vector Store Configuration\n")
                f.write(f"VECTOR_STORE_PATH=./data/vector_store\n")
                f.write(f"TOP_K_RESULTS=5\n\n")
                f.write(f"# Generation Parameters\n")
                f.write(f"MAX_TOKENS=500\n")
                f.write(f"TEMPERATURE=0.7\n")
            
            print("✅ Created .env file")
        else:
            print("❌ Cannot proceed without API key. Exiting...")
            return
    
    print("\n" + "="*60)
    print("📦 STEP 1: Data Ingestion")
    print("="*60)
    
    if run_command("python src/ingest.py", "Ingesting and chunking posts"):
        print("✅ Posts successfully ingested and chunked")
    else:
        print("❌ Failed at ingestion step")
        return
    
    print("\n" + "="*60)
    print("🔮 STEP 2: Embedding & Indexing")
    print("="*60)
    
    if run_command("python src/indexer.py", "Creating embeddings and vector index"):
        print("✅ Vector index successfully created")
    else:
        print("❌ Failed at indexing step")
        return
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("""
✅ All components are ready!

You can now:

1. 🖥️  Run the Streamlit interface:
   streamlit run interface/app.py

2. 🧪 Test individual components:
   python src/retrieve.py        # Test retrieval
   python src/generate.py        # Test generation
   python src/evaluator.py       # Test evaluation

3. 📊 Run evaluation:
   python scripts/run_evaluation.py

4. 🔍 Check plagiarism:
   python src/plagiarism_checker.py

Enjoy building with LinkedIn RAG Agent! 🚀
    """)


if __name__ == "__main__":
    main()
