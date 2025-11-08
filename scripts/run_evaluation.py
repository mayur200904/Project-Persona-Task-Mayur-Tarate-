"""
Evaluation Script - RAG vs Non-RAG Comparison
Demonstrates the value of retrieval-augmented generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
from retrieve import PostRetriever
from prompter import PromptBuilder
from generate import PostGenerator
from evaluator import PostEvaluator
from plagiarism_checker import PlagiarismChecker


def main():
    """Run comprehensive RAG vs Non-RAG evaluation"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║         RAG vs Non-RAG Evaluation Suite                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize components
    print("🔄 Loading components...")
    retriever = PostRetriever()
    prompt_builder = PromptBuilder()
    generator = PostGenerator()
    evaluator = PostEvaluator()
    plagiarism_checker = PlagiarismChecker()
    
    # Test scenarios
    test_cases = [
        {
            "persona": {
                "name": "Tech Leader",
                "title": "CEO",
                "company": "Innovation Labs",
                "industry": "Technology"
            },
            "topic": "AI innovation and sustainable technology for future generations"
        },
        {
            "persona": {
                "name": "Business Executive",
                "title": "VP of Product",
                "company": "Digital Solutions Inc",
                "industry": "Software"
            },
            "topic": "Building diverse teams and fostering inclusive workplace culture"
        },
        {
            "persona": {
                "name": "Industry Expert",
                "title": "CTO",
                "company": "CloudTech",
                "industry": "Cloud Computing"
            },
            "topic": "Leadership lessons from scaling technology teams"
        }
    ]
    
    print(f"\n📊 Running {len(test_cases)} test cases...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"Topic: {test_case['topic']}")
        print(f"{'='*60}\n")
        
        persona = test_case['persona']
        topic = test_case['topic']
        
        # Generate with RAG
        print("🔍 Generating with RAG...")
        chunks = retriever.retrieve_with_context(persona, topic, top_k=5, use_mmr=True)
        rag_prompt = prompt_builder.build_full_prompt(persona, topic, chunks)
        rag_result = generator.generate_with_rag(rag_prompt)
        
        print("✅ RAG Generation Complete")
        print(f"   Word count: {rag_result['word_count']}")
        print(f"   Hashtags: {rag_result['hashtag_count']}")
        
        # Check RAG plagiarism
        rag_plagiarism = plagiarism_checker.check_against_chunks(rag_result['post'], chunks)
        
        # Generate without RAG
        print("\n📝 Generating without RAG...")
        nonrag_prompt = prompt_builder.build_non_rag_prompt(persona, topic)
        nonrag_result = generator.generate_without_rag(nonrag_prompt)
        
        print("✅ Non-RAG Generation Complete")
        print(f"   Word count: {nonrag_result['word_count']}")
        print(f"   Hashtags: {nonrag_result['hashtag_count']}")
        
        # Evaluate comparison
        print("\n📊 Evaluating...")
        
        guidelines = {
            "word_count_range": [120, 220],
            "max_hashtags": 4,
            "use_emojis": False
        }
        
        comparison = evaluator.compare_rag_vs_nonrag(
            rag_result['post'],
            nonrag_result['post'],
            guidelines
        )
        
        # Add posts to results
        comparison['test_case'] = i
        comparison['topic'] = topic
        comparison['rag_post'] = rag_result['post']
        comparison['nonrag_post'] = nonrag_result['post']
        comparison['rag_plagiarism'] = rag_plagiarism
        comparison['retrieved_chunks_count'] = len(chunks)
        
        results.append(comparison)
        
        print(f"\n✅ Test Case {i} Complete\n")
    
    # Save results
    print("\n" + "="*60)
    print("💾 Saving Results...")
    print("="*60)
    
    output_path = "eval/comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved detailed results to {output_path}")
    
    # Generate report
    print("\n" + "="*60)
    print("📋 EVALUATION REPORT")
    print("="*60)
    
    report = evaluator.generate_report()
    print(report)
    
    # Summary statistics
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60 + "\n")
    
    rag_wins = {
        "compliance": 0,
        "readability": 0,
        "authenticity": 0
    }
    
    for result in results:
        for metric, winner in result['winner'].items():
            if winner == "RAG":
                rag_wins[metric] += 1
    
    total = len(results)
    
    print(f"RAG Performance:")
    print(f"  ✓ Compliance wins: {rag_wins['compliance']}/{total} ({rag_wins['compliance']/total*100:.1f}%)")
    print(f"  ✓ Readability wins: {rag_wins['readability']}/{total} ({rag_wins['readability']/total*100:.1f}%)")
    print(f"  ✓ Authenticity wins: {rag_wins['authenticity']}/{total} ({rag_wins['authenticity']/total*100:.1f}%)")
    
    # Plagiarism check
    plagiarism_detected = sum(1 for r in results if r['rag_plagiarism']['is_plagiarized'])
    print(f"\n  ⚠️  Plagiarism detected: {plagiarism_detected}/{total} cases")
    
    # Sample posts
    print("\n" + "="*60)
    print("📝 SAMPLE OUTPUTS (Test Case 1)")
    print("="*60 + "\n")
    
    print("RAG-Generated Post:")
    print("-" * 60)
    print(results[0]['rag_post'])
    print("-" * 60)
    
    print("\nNon-RAG-Generated Post:")
    print("-" * 60)
    print(results[0]['nonrag_post'])
    print("-" * 60)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nFull results saved to: {output_path}")
    print("\n💡 Key Takeaway:")
    print("   RAG provides more contextual, style-consistent posts")
    print("   by leveraging previous writing examples.\n")


if __name__ == "__main__":
    main()
