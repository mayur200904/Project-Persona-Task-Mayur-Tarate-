"""
Plagiarism Detection Module
Checks for excessive similarity with source material
"""

import json
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class PlagiarismChecker:
    """Detects plagiarism in generated posts"""
    
    def __init__(self, threshold: int = 25):
        """
        Initialize plagiarism checker
        
        Args:
            threshold: Minimum consecutive words to flag as plagiarism
        """
        self.threshold = threshold
    
    def find_longest_match(self, generated: str, source: str) -> Tuple[int, str]:
        """
        Find longest consecutive word match
        
        Args:
            generated: Generated post text
            source: Source text to check against
            
        Returns:
            Tuple of (match_length, matched_text)
        """
        gen_words = generated.lower().split()
        src_words = source.lower().split()
        
        max_length = 0
        max_match = ""
        
        for i in range(len(gen_words)):
            for j in range(len(src_words)):
                length = 0
                while (i + length < len(gen_words) and 
                       j + length < len(src_words) and 
                       gen_words[i + length] == src_words[j + length]):
                    length += 1
                
                if length > max_length:
                    max_length = length
                    max_match = ' '.join(gen_words[i:i+length])
        
        return max_length, max_match
    
    def check_against_chunks(self, generated_post: str, 
                            source_chunks: List[Dict]) -> Dict:
        """
        Check generated post against source chunks
        
        Args:
            generated_post: The generated LinkedIn post
            source_chunks: List of source chunks used for RAG
            
        Returns:
            Dictionary with plagiarism check results
        """
        issues = []
        max_overlap = 0
        most_similar_chunk = None
        
        for chunk in source_chunks:
            match_length, matched_text = self.find_longest_match(
                generated_post, 
                chunk['text']
            )
            
            if match_length >= self.threshold:
                issues.append({
                    "chunk_id": chunk.get('chunk_id', 'unknown'),
                    "match_length": match_length,
                    "matched_text": matched_text,
                    "source_text": chunk['text']
                })
            
            if match_length > max_overlap:
                max_overlap = match_length
                most_similar_chunk = chunk
        
        # Calculate overall similarity ratio
        if most_similar_chunk:
            similarity_ratio = SequenceMatcher(
                None, 
                generated_post.lower(), 
                most_similar_chunk['text'].lower()
            ).ratio()
        else:
            similarity_ratio = 0
        
        return {
            "is_plagiarized": len(issues) > 0,
            "max_consecutive_words": max_overlap,
            "similarity_ratio": round(similarity_ratio, 3),
            "issues": issues,
            "threshold": self.threshold
        }
    
    def check_with_explanation(self, generated_post: str,
                              source_chunks: List[Dict]) -> Tuple[bool, str]:
        """
        Check plagiarism and return explanation
        
        Args:
            generated_post: The generated post
            source_chunks: Source chunks
            
        Returns:
            Tuple of (is_plagiarized, explanation)
        """
        result = self.check_against_chunks(generated_post, source_chunks)
        
        if result['is_plagiarized']:
            issues = result['issues']
            explanation = f"""⚠️ PLAGIARISM DETECTED

Found {len(issues)} instance(s) of excessive copying:

"""
            for i, issue in enumerate(issues, 1):
                explanation += f"""{i}. {issue['match_length']} consecutive words matched:
   "{issue['matched_text'][:100]}..."
   
"""
            explanation += f"\nSimilarity ratio: {result['similarity_ratio']:.1%}"
            explanation += f"\nThreshold: {self.threshold} consecutive words"
            
            return True, explanation
        else:
            explanation = f"""✅ NO PLAGIARISM DETECTED

Max consecutive words: {result['max_consecutive_words']}
Similarity ratio: {result['similarity_ratio']:.1%}
Threshold: {self.threshold} consecutive words

The post appears to be original content."""
            
            return False, explanation
    
    def batch_check(self, posts_with_sources: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        """
        Check multiple posts for plagiarism
        
        Args:
            posts_with_sources: List of (generated_post, source_chunks) tuples
            
        Returns:
            List of check results
        """
        results = []
        
        for i, (post, chunks) in enumerate(posts_with_sources):
            print(f"🔍 Checking post {i+1}/{len(posts_with_sources)}...")
            result = self.check_against_chunks(post, chunks)
            result['post_index'] = i
            results.append(result)
        
        return results
    
    def get_statistics(self, check_results: List[Dict]) -> Dict:
        """
        Get statistics from batch plagiarism checks
        
        Args:
            check_results: List of check results
            
        Returns:
            Statistics dictionary
        """
        total = len(check_results)
        plagiarized = sum(1 for r in check_results if r['is_plagiarized'])
        
        avg_similarity = sum(r['similarity_ratio'] for r in check_results) / total if total > 0 else 0
        max_overlap = max((r['max_consecutive_words'] for r in check_results), default=0)
        
        return {
            "total_checks": total,
            "plagiarized_count": plagiarized,
            "plagiarism_rate": round(plagiarized / total * 100, 2) if total > 0 else 0,
            "average_similarity": round(avg_similarity, 3),
            "max_word_overlap": max_overlap
        }


def main():
    """Test plagiarism detection"""
    print("🚀 Testing plagiarism checker...")
    
    checker = PlagiarismChecker(threshold=10)
    
    # Test case 1: High similarity
    generated = "AI continues to transform industries. We must use it responsibly and ethically in all our work."
    source_chunks = [
        {"chunk_id": "1", "text": "AI continues to transform industries. We must use it responsibly."}
    ]
    
    print("\n📝 Test 1: High Similarity")
    is_plagiarized, explanation = checker.check_with_explanation(generated, source_chunks)
    print(explanation)
    
    # Test case 2: Low similarity
    generated2 = "Artificial intelligence is changing how businesses operate today."
    
    print("\n📝 Test 2: Low Similarity")
    is_plagiarized2, explanation2 = checker.check_with_explanation(generated2, source_chunks)
    print(explanation2)


if __name__ == "__main__":
    main()
