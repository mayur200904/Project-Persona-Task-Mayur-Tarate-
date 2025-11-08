"""
Project Utilities
Helper functions for the LinkedIn RAG Agent
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def check_setup() -> Dict[str, bool]:
    """
    Check if all required files and directories exist
    
    Returns:
        Dictionary with setup status
    """
    status = {
        "env_file": os.path.exists(".env"),
        "sample_data": os.path.exists("data/sample_posts.json"),
        "cleaned_chunks": os.path.exists("data/cleaned_chunks.json"),
        "vector_index": os.path.exists("data/vector_store.index"),
        "metadata": os.path.exists("data/index_metadata.json"),
        "memory": os.path.exists("memory/memory_template.json")
    }
    
    return status


def print_setup_status():
    """Print colored setup status"""
    status = check_setup()
    
    print("\n" + "="*60)
    print("📋 PROJECT SETUP STATUS")
    print("="*60 + "\n")
    
    for item, exists in status.items():
        icon = "✅" if exists else "❌"
        print(f"{icon} {item.replace('_', ' ').title()}")
    
    all_ready = all(status.values())
    
    print("\n" + "="*60)
    if all_ready:
        print("✅ ALL SYSTEMS READY!")
        print("\nYou can now run:")
        print("  streamlit run interface/app.py")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("\nPlease run:")
        print("  python setup_pipeline.py")
    print("="*60 + "\n")


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of tokens (1 token ≈ 4 characters)
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def calculate_cost_estimate(posts_count: int, 
                           embedding_model: str = "text-embedding-3-small",
                           generation_model: str = "gpt-4o-mini") -> Dict:
    """
    Calculate estimated API costs
    
    Args:
        posts_count: Number of posts to generate
        embedding_model: Embedding model name
        generation_model: Generation model name
        
    Returns:
        Cost breakdown dictionary
    """
    # Pricing (as of Nov 2024)
    pricing = {
        "text-embedding-3-small": 0.00002 / 1000,  # per token
        "gpt-4o-mini": {
            "input": 0.00015 / 1000,
            "output": 0.0006 / 1000
        }
    }
    
    # Estimates
    avg_chunk_tokens = 100
    chunks_per_post = 6  # sample posts
    retrieved_chunks = 5
    prompt_tokens = 500
    output_tokens = 300
    
    # Embedding cost (one-time for indexing)
    embedding_cost = (chunks_per_post * avg_chunk_tokens * 
                     pricing[embedding_model] * posts_count)
    
    # Generation cost per post
    input_cost = (prompt_tokens + retrieved_chunks * avg_chunk_tokens) * \
                 pricing[generation_model]["input"]
    output_cost = output_tokens * pricing[generation_model]["output"]
    generation_cost_per_post = input_cost + output_cost
    
    total_generation_cost = generation_cost_per_post * posts_count
    total_cost = embedding_cost + total_generation_cost
    
    return {
        "embedding_cost": round(embedding_cost, 4),
        "generation_cost_per_post": round(generation_cost_per_post, 4),
        "total_generation_cost": round(total_generation_cost, 4),
        "total_cost": round(total_cost, 4),
        "posts_count": posts_count
    }


def get_project_stats() -> Dict:
    """Get project statistics"""
    stats = {}
    
    # Check for generated chunks
    if os.path.exists("data/cleaned_chunks.json"):
        with open("data/cleaned_chunks.json", 'r') as f:
            chunks = json.load(f)
            stats['total_chunks'] = len(chunks)
            stats['total_words'] = sum(chunk['word_count'] for chunk in chunks)
    
    # Check for memory
    if os.path.exists("memory/memory.json"):
        with open("memory/memory.json", 'r') as f:
            memory = json.load(f)
            stats['generated_posts'] = len(memory.get('previous_posts', []))
    
    # Check for evaluations
    if os.path.exists("eval/comparison.json"):
        with open("eval/comparison.json", 'r') as f:
            evals = json.load(f)
            stats['evaluations_run'] = len(evals)
    
    return stats


def print_project_stats():
    """Print project statistics"""
    stats = get_project_stats()
    
    print("\n" + "="*60)
    print("📊 PROJECT STATISTICS")
    print("="*60 + "\n")
    
    for key, value in stats.items():
        label = key.replace('_', ' ').title()
        print(f"{label}: {value}")
    
    if not stats:
        print("No statistics available yet. Run the pipeline first!")
    
    print("\n" + "="*60 + "\n")


def clean_generated_files():
    """Remove generated files (for fresh start)"""
    files_to_remove = [
        "data/cleaned_chunks.json",
        "data/vector_store.index",
        "data/index_metadata.json",
        "memory/memory.json",
        "eval/comparison.json",
        "outputs/generated_posts.json"
    ]
    
    removed = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed += 1
            print(f"🗑️  Removed: {file_path}")
    
    if removed == 0:
        print("✨ No generated files to clean")
    else:
        print(f"\n✅ Cleaned {removed} files. Ready for fresh start!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            print_setup_status()
        elif command == "stats":
            print_project_stats()
        elif command == "cost":
            posts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            costs = calculate_cost_estimate(posts)
            print(f"\n💰 Cost Estimate for {posts} posts:")
            print(json.dumps(costs, indent=2))
        elif command == "clean":
            confirm = input("⚠️  This will remove all generated files. Continue? (yes/no): ")
            if confirm.lower() == 'yes':
                clean_generated_files()
        else:
            print("Unknown command. Use: status, stats, cost, or clean")
    else:
        print_setup_status()
