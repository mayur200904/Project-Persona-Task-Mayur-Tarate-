"""
Model Performance Optimization Module
Implements various techniques to improve generation quality and efficiency
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


class PerformanceOptimizer:
    """Handles model performance optimization strategies"""
    
    def __init__(self):
        """Initialize optimizer"""
        self.optimization_history = []
        self.best_config = None
        self.best_score = 0.0
    
    def optimize_temperature(self, generator, prompter, retriever,
                            test_topics: List[str], 
                            temperature_range: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Dict:
        """
        Test different temperature values to find optimal setting
        
        Args:
            generator: PostGenerator instance
            prompter: PromptBuilder instance
            retriever: PostRetriever instance
            test_topics: List of test topics
            temperature_range: Temperature values to test
            
        Returns:
            Optimization results with best temperature
        """
        print("🌡️  Testing temperature optimization...")
        
        results = []
        persona_info = {
            "name": "Test User",
            "title": "Technology Leader",
            "company": "Tech Corp"
        }
        
        for temp in temperature_range:
            print(f"\n📊 Testing temperature: {temp}")
            
            # Update generator temperature
            generator.temperature = temp
            
            # Generate test posts
            posts = []
            for topic in test_topics[:2]:  # Test with 2 topics for speed
                chunks = retriever.retrieve_similar(topic, top_k=3)
                prompt = prompter.build_full_prompt(persona_info, topic, chunks)
                post = generator.generate_post(prompt['system'], prompt['user'])
                posts.append(post)
            
            # Calculate metrics
            avg_length = np.mean([len(post.split()) for post in posts])
            avg_variety = self._calculate_lexical_diversity(posts)
            
            result = {
                "temperature": temp,
                "avg_word_count": round(avg_length, 2),
                "lexical_diversity": round(avg_variety, 3),
                "posts": posts
            }
            results.append(result)
            
            print(f"  Avg words: {result['avg_word_count']}")
            print(f"  Diversity: {result['lexical_diversity']}")
        
        # Find best temperature (balance between length and diversity)
        best = max(results, key=lambda x: x['lexical_diversity'])
        
        print(f"\n✅ Best temperature: {best['temperature']}")
        print(f"   Diversity: {best['lexical_diversity']}")
        
        return {
            "best_temperature": best['temperature'],
            "all_results": results,
            "recommendation": f"Set temperature to {best['temperature']} for optimal diversity"
        }
    
    def optimize_retrieval_k(self, retriever, prompter, generator,
                            test_topics: List[str],
                            k_values: List[int] = [3, 5, 7, 10]) -> Dict:
        """
        Test different retrieval K values
        
        Args:
            retriever: PostRetriever instance
            prompter: PromptBuilder instance
            generator: PostGenerator instance
            test_topics: Test topics
            k_values: K values to test
            
        Returns:
            Optimization results
        """
        print("🔍 Testing retrieval K optimization...")
        
        results = []
        persona_info = {
            "name": "Test User",
            "title": "Technology Leader",
            "company": "Tech Corp"
        }
        
        for k in k_values:
            print(f"\n📊 Testing K={k}")
            
            generation_times = []
            posts = []
            
            for topic in test_topics[:2]:
                start_time = time.time()
                
                chunks = retriever.retrieve_similar(topic, top_k=k)
                prompt = prompter.build_full_prompt(persona_info, topic, chunks)
                post = generator.generate_post(prompt['system'], prompt['user'])
                
                generation_times.append(time.time() - start_time)
                posts.append(post)
            
            avg_time = np.mean(generation_times)
            avg_quality = self._calculate_lexical_diversity(posts)
            
            result = {
                "k_value": k,
                "avg_generation_time": round(avg_time, 2),
                "quality_score": round(avg_quality, 3),
                "efficiency_score": round(avg_quality / avg_time, 3)
            }
            results.append(result)
            
            print(f"  Time: {result['avg_generation_time']}s")
            print(f"  Quality: {result['quality_score']}")
            print(f"  Efficiency: {result['efficiency_score']}")
        
        # Best = highest efficiency (quality/time ratio)
        best = max(results, key=lambda x: x['efficiency_score'])
        
        print(f"\n✅ Best K value: {best['k_value']}")
        print(f"   Efficiency: {best['efficiency_score']}")
        
        return {
            "best_k": best['k_value'],
            "all_results": results,
            "recommendation": f"Use top_k={best['k_value']} for best quality/speed balance"
        }
    
    def optimize_prompt_engineering(self, generator, retriever,
                                    test_topic: str) -> Dict:
        """
        Test different prompt engineering strategies
        
        Args:
            generator: PostGenerator instance
            retriever: PostRetriever instance
            test_topic: Topic to test with
            
        Returns:
            Results from different prompt strategies
        """
        print("✍️  Testing prompt engineering variations...")
        
        chunks = retriever.retrieve_similar(test_topic, top_k=5)
        context = "\n\n".join([c['text'] for c in chunks])
        
        strategies = {
            "baseline": self._create_baseline_prompt(test_topic, context),
            "detailed_instructions": self._create_detailed_prompt(test_topic, context),
            "example_driven": self._create_example_driven_prompt(test_topic, context),
            "constraint_focused": self._create_constraint_prompt(test_topic, context),
            "creative_freedom": self._create_creative_prompt(test_topic, context)
        }
        
        results = []
        
        for strategy_name, prompt_dict in strategies.items():
            print(f"\n📊 Testing: {strategy_name}")
            
            post = generator.generate_post(prompt_dict['system'], prompt_dict['user'])
            
            word_count = len(post.split())
            diversity = self._calculate_lexical_diversity([post])
            has_first_person = any(w in post.lower().split() for w in ['i', 'my', 'we', 'our'])
            
            result = {
                "strategy": strategy_name,
                "word_count": word_count,
                "lexical_diversity": round(diversity, 3),
                "uses_first_person": has_first_person,
                "post_preview": post[:150] + "..."
            }
            results.append(result)
            
            print(f"  Words: {word_count}")
            print(f"  Diversity: {result['lexical_diversity']}")
            print(f"  First-person: {has_first_person}")
        
        # Best = highest diversity + first person usage
        best = max(results, key=lambda x: (x['uses_first_person'], x['lexical_diversity']))
        
        print(f"\n✅ Best strategy: {best['strategy']}")
        
        return {
            "best_strategy": best['strategy'],
            "all_results": results,
            "recommendation": f"Use '{best['strategy']}' prompt strategy for best results"
        }
    
    def optimize_mmr_lambda(self, retriever, test_queries: List[str],
                           lambda_values: List[float] = [0.3, 0.5, 0.7, 0.9]) -> Dict:
        """
        Test different MMR lambda values for diversity vs relevance
        
        Args:
            retriever: PostRetriever instance
            test_queries: Test queries
            lambda_values: Lambda values to test
            
        Returns:
            Optimization results
        """
        print("🎯 Testing MMR lambda optimization...")
        
        results = []
        
        for lambda_mult in lambda_values:
            print(f"\n📊 Testing lambda={lambda_mult}")
            
            diversity_scores = []
            
            for query in test_queries[:3]:
                chunks = retriever.retrieve_with_mmr(query, top_k=5, lambda_mult=lambda_mult)
                
                # Calculate diversity (unique words across chunks)
                all_text = " ".join([c['text'] for c in chunks])
                diversity = len(set(all_text.split())) / len(all_text.split())
                diversity_scores.append(diversity)
            
            avg_diversity = np.mean(diversity_scores)
            
            result = {
                "lambda": lambda_mult,
                "diversity_score": round(avg_diversity, 3)
            }
            results.append(result)
            
            print(f"  Diversity: {result['diversity_score']}")
        
        best = max(results, key=lambda x: x['diversity_score'])
        
        print(f"\n✅ Best lambda: {best['lambda']}")
        
        return {
            "best_lambda": best['lambda'],
            "all_results": results,
            "recommendation": f"Use lambda={best['lambda']} in MMR for optimal diversity"
        }
    
    def run_full_optimization(self, generator, prompter, retriever,
                             evaluator, test_topics: List[str]) -> Dict:
        """
        Run comprehensive optimization suite
        
        Args:
            generator: PostGenerator instance
            prompter: PromptBuilder instance
            retriever: PostRetriever instance
            evaluator: PostEvaluator instance
            test_topics: Test topics
            
        Returns:
            Complete optimization report
        """
        print("\n" + "="*60)
        print("🚀 RUNNING FULL PERFORMANCE OPTIMIZATION")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations": {}
        }
        
        # 1. Temperature optimization
        print("\n[1/4] Temperature Optimization")
        print("-" * 60)
        temp_results = self.optimize_temperature(generator, prompter, retriever, test_topics)
        results['optimizations']['temperature'] = temp_results
        
        # Apply best temperature
        generator.temperature = temp_results['best_temperature']
        
        # 2. Retrieval K optimization
        print("\n[2/4] Retrieval K Optimization")
        print("-" * 60)
        k_results = self.optimize_retrieval_k(retriever, prompter, generator, test_topics)
        results['optimizations']['retrieval_k'] = k_results
        
        # 3. Prompt engineering
        print("\n[3/4] Prompt Engineering Optimization")
        print("-" * 60)
        prompt_results = self.optimize_prompt_engineering(generator, retriever, test_topics[0])
        results['optimizations']['prompt_strategy'] = prompt_results
        
        # 4. MMR lambda
        print("\n[4/4] MMR Lambda Optimization")
        print("-" * 60)
        mmr_results = self.optimize_mmr_lambda(retriever, test_topics)
        results['optimizations']['mmr_lambda'] = mmr_results
        
        # Summary
        duration = time.time() - start_time
        
        results['summary'] = {
            "total_duration": round(duration, 2),
            "recommendations": {
                "temperature": temp_results['best_temperature'],
                "retrieval_k": k_results['best_k'],
                "prompt_strategy": prompt_results['best_strategy'],
                "mmr_lambda": mmr_results['best_lambda']
            }
        }
        
        # Save best config
        self.best_config = results['summary']['recommendations']
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\n⏱️  Duration: {duration:.2f}s")
        print("\n🎯 RECOMMENDED SETTINGS:")
        print(f"   Temperature: {self.best_config['temperature']}")
        print(f"   Retrieval K: {self.best_config['retrieval_k']}")
        print(f"   Prompt Strategy: {self.best_config['prompt_strategy']}")
        print(f"   MMR Lambda: {self.best_config['mmr_lambda']}")
        
        return results
    
    def save_optimization_results(self, results: Dict, 
                                  output_path: str = "eval/optimization_results.json"):
        """Save optimization results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved optimization results to {output_path}")
    
    # Helper methods
    
    def _calculate_lexical_diversity(self, posts: List[str]) -> float:
        """Calculate lexical diversity (type-token ratio)"""
        all_words = []
        for post in posts:
            words = post.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        return len(set(all_words)) / len(all_words)
    
    def _create_baseline_prompt(self, topic: str, context: str) -> Dict[str, str]:
        """Create baseline prompt"""
        return {
            "system": "You are a LinkedIn writing assistant. Write professional posts.",
            "user": f"Context:\n{context}\n\nWrite a post about: {topic}"
        }
    
    def _create_detailed_prompt(self, topic: str, context: str) -> Dict[str, str]:
        """Create detailed instruction prompt"""
        return {
            "system": """You are a LinkedIn writing assistant specializing in authentic, engaging content.
            
Write posts that:
- Use first-person perspective naturally
- Mix short and long sentences for rhythm
- Focus on insights and experiences
- Avoid corporate jargon
- Include 2-4 hashtags at the end
- Stay between 150-200 words""",
            "user": f"""Based on these writing examples:
{context}

Create a LinkedIn post about: {topic}

Make it personal, insightful, and authentic."""
        }
    
    def _create_example_driven_prompt(self, topic: str, context: str) -> Dict[str, str]:
        """Create example-driven prompt"""
        return {
            "system": "You are mimicking the exact writing style shown in examples below.",
            "user": f"""EXAMPLES OF TARGET STYLE:
{context}

YOUR TASK:
Write a similar post about: {topic}

Match the tone, structure, and voice of the examples exactly."""
        }
    
    def _create_constraint_prompt(self, topic: str, context: str) -> Dict[str, str]:
        """Create constraint-focused prompt"""
        return {
            "system": """You are a LinkedIn writer following strict constraints:
- EXACTLY 150-200 words
- NO emojis
- 2-4 hashtags only
- Use "I" or "we" at least once
- No promotional language""",
            "user": f"Context:\n{context}\n\nTopic: {topic}"
        }
    
    def _create_creative_prompt(self, topic: str, context: str) -> Dict[str, str]:
        """Create creative freedom prompt"""
        return {
            "system": """You are a creative LinkedIn writer with your own authentic voice.
Draw inspiration from the examples but make the post uniquely yours.""",
            "user": f"""Inspiration:
{context}

Write an original post about: {topic}

Be creative, authentic, and engaging."""
        }


def main():
    """Test optimization module"""
    print("🚀 Testing Performance Optimization...")
    
    # This would normally use actual components
    print("\n✅ Performance optimizer initialized")
    print("💡 Use run_full_optimization() with your pipeline components")
    
    optimizer = PerformanceOptimizer()
    print(f"\n📊 Optimizer ready with {len(optimizer.optimization_history)} recorded optimizations")


if __name__ == "__main__":
    main()
