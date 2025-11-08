"""
Evaluation Module
Compares RAG vs Non-RAG generation quality
"""

import json
import textstat
from typing import List, Dict
from datetime import datetime
import re


class PostEvaluator:
    """Evaluates and compares generated LinkedIn posts"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_results = []
    
    def count_hashtags(self, post: str) -> int:
        """Count hashtags in post"""
        return len(re.findall(r'#\w+', post))
    
    def has_emojis(self, post: str) -> bool:
        """Check if post contains emojis"""
        emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
        return bool(emoji_pattern.search(post))
    
    def calculate_readability(self, post: str) -> Dict[str, float]:
        """
        Calculate readability metrics
        
        Args:
            post: Post text
            
        Returns:
            Dictionary with readability scores
        """
        # Remove hashtags for readability calculation
        clean_text = re.sub(r'#\w+', '', post)
        
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(clean_text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(clean_text),
            "automated_readability_index": textstat.automated_readability_index(clean_text)
        }
    
    def check_style_compliance(self, post: str, guidelines: Dict) -> Dict:
        """
        Check if post follows style guidelines
        
        Args:
            post: Post text
            guidelines: Style guidelines dictionary
            
        Returns:
            Compliance results
        """
        word_count = len(post.split())
        hashtag_count = self.count_hashtags(post)
        has_emoji = self.has_emojis(post)
        
        word_range = guidelines.get('word_count_range', [120, 220])
        max_hashtags = guidelines.get('max_hashtags', 4)
        use_emojis = guidelines.get('use_emojis', False)
        
        return {
            "word_count_compliant": word_range[0] <= word_count <= word_range[1],
            "word_count": word_count,
            "hashtag_compliant": hashtag_count <= max_hashtags,
            "hashtag_count": hashtag_count,
            "emoji_compliant": has_emoji == use_emojis,
            "has_emoji": has_emoji
        }
    
    def evaluate_single_post(self, post: str, method: str,
                            guidelines: Dict, source_chunks: List[Dict] = None) -> Dict:
        """
        Evaluate a single post
        
        Args:
            post: Generated post text
            method: 'RAG' or 'Non-RAG'
            guidelines: Style guidelines
            source_chunks: Optional source chunks for context analysis
            
        Returns:
            Evaluation results dictionary
        """
        # Basic metrics
        word_count = len(post.split())
        sentence_count = len(re.split(r'[.!?]+', post))
        
        # Style compliance
        compliance = self.check_style_compliance(post, guidelines)
        
        # Readability
        readability = self.calculate_readability(post)
        
        # First-person presence
        first_person_words = ['I', 'we', 'my', 'our', 'me', 'us']
        uses_first_person = any(word in post.split() for word in first_person_words)
        
        evaluation = {
            "method": method,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": round(word_count / sentence_count, 2) if sentence_count > 0 else 0,
            "compliance": compliance,
            "readability": readability,
            "uses_first_person": uses_first_person,
            "timestamp": datetime.now().isoformat()
        }
        
        return evaluation
    
    def compare_rag_vs_nonrag(self, rag_post: str, nonrag_post: str,
                             guidelines: Dict) -> Dict:
        """
        Compare RAG vs Non-RAG generation
        
        Args:
            rag_post: Post generated with RAG
            nonrag_post: Post generated without RAG
            guidelines: Style guidelines
            
        Returns:
            Comparison results
        """
        rag_eval = self.evaluate_single_post(rag_post, "RAG", guidelines)
        nonrag_eval = self.evaluate_single_post(nonrag_post, "Non-RAG", guidelines)
        
        # Calculate compliance scores
        rag_compliance_score = sum([
            rag_eval['compliance']['word_count_compliant'],
            rag_eval['compliance']['hashtag_compliant'],
            rag_eval['compliance']['emoji_compliant']
        ])
        
        nonrag_compliance_score = sum([
            nonrag_eval['compliance']['word_count_compliant'],
            nonrag_eval['compliance']['hashtag_compliant'],
            nonrag_eval['compliance']['emoji_compliant']
        ])
        
        comparison = {
            "rag_evaluation": rag_eval,
            "nonrag_evaluation": nonrag_eval,
            "winner": {
                "compliance": "RAG" if rag_compliance_score > nonrag_compliance_score else "Non-RAG",
                "readability": "RAG" if rag_eval['readability']['flesch_reading_ease'] > 
                                       nonrag_eval['readability']['flesch_reading_ease'] else "Non-RAG",
                "authenticity": "RAG" if rag_eval['uses_first_person'] else "Non-RAG"
            },
            "summary": {
                "rag_compliance_score": f"{rag_compliance_score}/3",
                "nonrag_compliance_score": f"{nonrag_compliance_score}/3",
                "rag_readability": round(rag_eval['readability']['flesch_reading_ease'], 2),
                "nonrag_readability": round(nonrag_eval['readability']['flesch_reading_ease'], 2)
            }
        }
        
        self.evaluation_results.append(comparison)
        
        return comparison
    
    def batch_evaluate(self, posts: List[Dict], guidelines: Dict) -> List[Dict]:
        """
        Evaluate multiple posts
        
        Args:
            posts: List of post dictionaries with 'post' and 'method' keys
            guidelines: Style guidelines
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, post_data in enumerate(posts):
            print(f"📊 Evaluating post {i+1}/{len(posts)}...")
            
            evaluation = self.evaluate_single_post(
                post_data['post'],
                post_data.get('method', 'Unknown'),
                guidelines
            )
            
            results.append(evaluation)
        
        return results
    
    def save_evaluation(self, output_path: str = "eval/comparison.json"):
        """
        Save evaluation results to file
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved evaluation results to {output_path}")
    
    def generate_report(self) -> str:
        """
        Generate human-readable evaluation report
        
        Returns:
            Formatted report string
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = "=" * 60 + "\n"
        report += "📊 RAG vs Non-RAG EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for i, result in enumerate(self.evaluation_results, 1):
            report += f"Comparison {i}:\n"
            report += "-" * 60 + "\n"
            
            # Compliance
            report += f"✓ Compliance Winner: {result['winner']['compliance']}\n"
            report += f"  RAG: {result['summary']['rag_compliance_score']}\n"
            report += f"  Non-RAG: {result['summary']['nonrag_compliance_score']}\n\n"
            
            # Readability
            report += f"✓ Readability Winner: {result['winner']['readability']}\n"
            report += f"  RAG: {result['summary']['rag_readability']}\n"
            report += f"  Non-RAG: {result['summary']['nonrag_readability']}\n\n"
            
            # Authenticity
            report += f"✓ Authenticity Winner: {result['winner']['authenticity']}\n\n"
        
        return report


def main():
    """Test evaluation"""
    print("🚀 Testing evaluation system...")
    
    evaluator = PostEvaluator()
    
    # Test posts
    rag_post = """AI continues to redefine how we work and create. 
In my experience, the best innovations come from collaboration. 
We're building tools that empower people, not replace them. 
The future is about human-AI partnership. 
#AI #Innovation #Leadership #Technology"""
    
    nonrag_post = """Technology is changing rapidly.
Companies need to adapt.
Innovation is important.
We should focus on the future.
#Tech #Business #Innovation #Future"""
    
    # Style guidelines
    guidelines = {
        "word_count_range": [120, 220],
        "max_hashtags": 4,
        "use_emojis": False
    }
    
    # Compare
    print("\n📊 Comparing RAG vs Non-RAG...")
    comparison = evaluator.compare_rag_vs_nonrag(rag_post, nonrag_post, guidelines)
    
    # Generate report
    report = evaluator.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
