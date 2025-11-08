"""
Generation Module
Generates LinkedIn posts using OpenAI GPT models
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class PostGenerator:
    """Handles LinkedIn post generation using LLM"""
    
    def __init__(self, model: str = "gpt-4o-mini", 
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """
        Initialize post generator
        
        Args:
            model: OpenAI model name
            temperature: Generation temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        # Initialize OpenAI client (reads OPENAI_API_KEY from environment)
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generation_history = []
    
    def generate_post(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a LinkedIn post
        
        Args:
            system_prompt: System instructions
            user_prompt: User request with context
            
        Returns:
            Generated post text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=60.0  # 60 second timeout for generation
            )

            generated_text = response.choices[0].message.content.strip()
            
            # Store in history
            self.generation_history.append({
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "temperature": self.temperature,
                "prompt_length": len(system_prompt) + len(user_prompt),
                "generated_length": len(generated_text),
                "tokens_used": response.usage.total_tokens
            })
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            return ""
    
    def generate_with_rag(self, prompt_dict: Dict[str, str]) -> Dict[str, any]:
        """
        Generate post with RAG context
        
        Args:
            prompt_dict: Dictionary with 'system' and 'user' keys
            
        Returns:
            Dictionary with generated post and metadata
        """
        generated_text = self.generate_post(
            prompt_dict['system'], 
            prompt_dict['user']
        )
        
        # Extract metadata
        word_count = len(generated_text.split())
        hashtag_count = generated_text.count('#')
        
        return {
            "post": generated_text,
            "word_count": word_count,
            "hashtag_count": hashtag_count,
            "method": "RAG",
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_without_rag(self, prompt_dict: Dict[str, str]) -> Dict[str, any]:
        """
        Generate post WITHOUT RAG context (for comparison)
        
        Args:
            prompt_dict: Dictionary with 'system' and 'user' keys
            
        Returns:
            Dictionary with generated post and metadata
        """
        generated_text = self.generate_post(
            prompt_dict['system'], 
            prompt_dict['user']
        )
        
        word_count = len(generated_text.split())
        hashtag_count = generated_text.count('#')
        
        return {
            "post": generated_text,
            "word_count": word_count,
            "hashtag_count": hashtag_count,
            "method": "Non-RAG",
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
    
    def regenerate_with_paraphrase(self, paraphrase_prompt: Dict[str, str]) -> str:
        """
        Regenerate post with stronger paraphrasing
        
        Args:
            paraphrase_prompt: Dictionary with paraphrase instructions
            
        Returns:
            Paraphrased post text
        """
        return self.generate_post(
            paraphrase_prompt['system'],
            paraphrase_prompt['user']
        )
    
    def batch_generate(self, prompts: List[Dict[str, str]], 
                      use_rag: bool = True) -> List[Dict]:
        """
        Generate multiple posts
        
        Args:
            prompts: List of prompt dictionaries
            use_rag: Whether these are RAG-based prompts
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"🔄 Generating post {i+1}/{len(prompts)}...")
            
            if use_rag:
                result = self.generate_with_rag(prompt)
            else:
                result = self.generate_without_rag(prompt)
            
            results.append(result)
        
        return results
    
    def save_generations(self, generations: List[Dict], 
                        output_path: str = "outputs/generated_posts.json"):
        """
        Save generated posts to file
        
        Args:
            generations: List of generation dictionaries
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(generations)} generations to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        if not self.generation_history:
            return {"message": "No generations yet"}
        
        total_tokens = sum(g['tokens_used'] for g in self.generation_history)
        avg_length = sum(g['generated_length'] for g in self.generation_history) / len(self.generation_history)
        
        return {
            "total_generations": len(self.generation_history),
            "total_tokens_used": total_tokens,
            "average_post_length": round(avg_length, 2),
            "model": self.model
        }


def main():
    """Test generation pipeline"""
    print("🚀 Testing post generation...")
    
    # Initialize generator
    generator = PostGenerator()
    
    # Simple test prompt
    system = """You are a LinkedIn writing assistant. Write professional, engaging posts.
    
Guidelines:
- 120-220 words
- No emojis
- ≤4 hashtags
- Professional tone"""
    
    user = """Write a LinkedIn post about:
The importance of continuous learning in technology careers.

Make it insightful and authentic."""
    
    # Generate
    print("\n📝 Generating post...")
    result = generator.generate_with_rag({
        'system': system,
        'user': user
    })
    
    print("\n✅ Generated Post:")
    print("-" * 50)
    print(result['post'])
    print("-" * 50)
    print(f"Word count: {result['word_count']}")
    print(f"Hashtags: {result['hashtag_count']}")
    
    # Show stats
    stats = generator.get_statistics()
    print("\n📊 Generation Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
