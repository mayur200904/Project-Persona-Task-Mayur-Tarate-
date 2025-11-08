"""
Performance Optimization Runner
Runs comprehensive optimization suite and saves recommendations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer import PerformanceOptimizer
from src.indexer import EmbeddingIndexer
from src.retrieve import PostRetriever
from src.prompter import PromptBuilder
from src.generate import PostGenerator
from src.evaluator import PostEvaluator
import json


def main():
    """Run full optimization suite"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       Performance Optimization Suite                       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check if index exists
    if not os.path.exists('data/vector_store.index'):
        print("❌ Vector store not found. Run setup_pipeline.py first.")
        return
    
    # Initialize components
    print("📦 Initializing components...")
    
    try:
        retriever = PostRetriever()
        prompter = PromptBuilder()
        generator = PostGenerator()
        evaluator = PostEvaluator()
        optimizer = PerformanceOptimizer()
        
        print("✅ All components loaded\n")
        
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return
    
    # Define test topics
    test_topics = [
        "The importance of continuous learning in technology",
        "Building sustainable AI systems",
        "Leadership lessons from scaling a startup",
        "The future of remote work and collaboration",
        "Innovation through diverse perspectives"
    ]
    
    print(f"📝 Test topics: {len(test_topics)}")
    for i, topic in enumerate(test_topics, 1):
        print(f"   {i}. {topic}")
    
    print("\n" + "="*60)
    input("Press Enter to start optimization (this may take 2-3 minutes)...")
    print("="*60 + "\n")
    
    # Run optimization
    try:
        results = optimizer.run_full_optimization(
            generator=generator,
            prompter=prompter,
            retriever=retriever,
            evaluator=evaluator,
            test_topics=test_topics
        )
        
        # Save results
        optimizer.save_optimization_results(results)
        
        # Generate config file with recommendations
        config = {
            "model_config": {
                "model_name": generator.model,
                "temperature": optimizer.best_config['temperature'],
                "max_tokens": generator.max_tokens
            },
            "retrieval_config": {
                "top_k": optimizer.best_config['retrieval_k'],
                "use_mmr": True,
                "mmr_lambda": optimizer.best_config['mmr_lambda']
            },
            "prompt_strategy": optimizer.best_config['prompt_strategy'],
            "optimization_date": results['timestamp']
        }
        
        # Save optimized config
        with open('eval/optimized_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "="*60)
        print("📄 Configuration files saved:")
        print("   - eval/optimization_results.json (full results)")
        print("   - eval/optimized_config.json (recommended settings)")
        print("="*60)
        
        # Display actionable recommendations
        print("\n" + "🎯 NEXT STEPS:")
        print("\n1. Review detailed results in eval/optimization_results.json")
        print("\n2. Update your .env or config with:")
        print(f"   MODEL_TEMPERATURE={config['model_config']['temperature']}")
        print(f"   RETRIEVAL_TOP_K={config['retrieval_config']['top_k']}")
        print(f"   MMR_LAMBDA={config['retrieval_config']['mmr_lambda']}")
        print("\n3. Use the optimized settings in your Streamlit app")
        print("\n4. Run comparative evaluation:")
        print("   python scripts/run_evaluation.py")
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
