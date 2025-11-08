"""
Prompt Engineering Module
Creates structured prompts for persona-based generation
"""

import json
from typing import Dict, List, Optional


class PromptBuilder:
    """Handles prompt construction for LinkedIn post generation"""
    
    def __init__(self, memory_path: str = "memory/memory_template.json"):
        """
        Initialize prompt builder
        
        Args:
            memory_path: Path to memory/persona configuration
        """
        with open(memory_path, 'r', encoding='utf-8') as f:
            self.memory = json.load(f)
    
    def build_system_prompt(self, persona_info: Optional[Dict] = None) -> str:
        """
        Build system prompt defining the persona
        
        Args:
            persona_info: Optional persona details to override memory
            
        Returns:
            System prompt string
        """
        # Use provided info or fall back to memory
        if persona_info:
            name = persona_info.get('name', self.memory['persona']['name'])
            title = persona_info.get('title', self.memory['persona']['title'])
            company = persona_info.get('company', self.memory['persona']['company'])
        else:
            name = self.memory['persona']['name']
            title = self.memory['persona']['title']
            company = self.memory['persona']['company']
        
        tone = self.memory['preferences']['tone']
        structure = self.memory['preferences']['structure']
        max_hashtags = self.memory['style_guidelines']['max_hashtags']
        word_range = self.memory['style_guidelines']['word_count_range']
        
        system_prompt = f"""You are a professional LinkedIn writing assistant trained to mimic the writing style of {name}, a {title} at {company}.

TONE AND STYLE:
- Tone: {tone}
- Structure: {structure}
- Use first-person perspective authentically
- Write with varied sentence lengths (mix short punchy sentences with longer reflective ones)
- Be genuine and avoid corporate jargon

CONTENT GUIDELINES:
- Word count: {word_range[0]}-{word_range[1]} words
- Use NO emojis
- Include ≤{max_hashtags} relevant hashtags at the end
- Focus on insights, experiences, and authentic reflections
- Avoid promotional language or calls-to-action like "click the link" or "follow for more"

PROHIBITED PHRASES:
{', '.join(self.memory['preferences']['banned_phrases'])}

Your goal is to create a LinkedIn post that sounds naturally written by {name}, incorporating their unique voice and perspective."""
        
        return system_prompt
    
    def build_user_prompt(self, topic: str, retrieved_chunks: List[Dict],
                         additional_context: Optional[str] = None) -> str:
        """
        Build user prompt with context and topic
        
        Args:
            topic: Main topic or bullet points for the post
            retrieved_chunks: Retrieved relevant chunks from past posts
            additional_context: Optional additional context
            
        Returns:
            User prompt string
        """
        # Format retrieved context
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_sections.append(f"Reference {i}:\n{chunk['text']}")
        
        context_text = "\n\n".join(context_sections)
        
        # Build themes reminder
        themes = ", ".join(self.memory['preferences']['recurring_themes'])
        
        user_prompt = f"""WRITING CONTEXT:
Based on the following examples of previous writing style:

{context_text}

RECURRING THEMES TO CONSIDER:
{themes}

TASK:
Write a LinkedIn post about: {topic}
"""
        
        if additional_context:
            user_prompt += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
        
        user_prompt += """

IMPORTANT:
- Match the tone and style of the reference examples
- Create original content - do NOT copy phrases directly
- Stay authentic to the persona
- Follow the word count and hashtag guidelines
- Make it engaging and insightful"""
        
        return user_prompt
    
    def build_full_prompt(self, persona_info: Dict, topic: str,
                         retrieved_chunks: List[Dict],
                         additional_context: Optional[str] = None) -> Dict[str, str]:
        """
        Build complete prompt with system and user messages
        
        Args:
            persona_info: Persona details
            topic: Topic for the post
            retrieved_chunks: Retrieved context chunks
            additional_context: Optional additional context
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        return {
            'system': self.build_system_prompt(persona_info),
            'user': self.build_user_prompt(topic, retrieved_chunks, additional_context)
        }
    
    def build_non_rag_prompt(self, persona_info: Dict, topic: str) -> Dict[str, str]:
        """
        Build prompt WITHOUT retrieval (for comparison)
        
        Args:
            persona_info: Persona details
            topic: Topic for the post
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = self.build_system_prompt(persona_info)
        
        user_prompt = f"""Write a LinkedIn post about: {topic}

Follow the style guidelines and create an engaging, professional post."""
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }
    
    def build_paraphrase_prompt(self, original_post: str, similar_chunks: List[Dict]) -> Dict[str, str]:
        """
        Build prompt for anti-plagiarism paraphrasing
        
        Args:
            original_post: The generated post with potential plagiarism
            similar_chunks: Chunks that were too similar
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = "You are an expert at paraphrasing while maintaining tone and meaning."
        
        user_prompt = f"""The following LinkedIn post is too similar to source material:

POST:
{original_post}

SOURCE MATERIAL TO AVOID:
{similar_chunks[0]['text']}

TASK:
Rewrite this post to convey the same ideas and maintain the same tone, but use completely different phrasing and sentence structures. Be creative and original while keeping the core message."""
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }


def main():
    """Test prompt building"""
    print("🚀 Testing prompt builder...")
    
    builder = PromptBuilder()
    
    # Test data
    persona = {
        "name": "Jane Smith",
        "title": "CTO",
        "company": "Tech Innovations Inc",
        "industry": "Technology"
    }
    
    topic = "The future of AI in healthcare and ethical considerations"
    
    chunks = [
        {"text": "AI continues to transform industries. We must use it responsibly."},
        {"text": "Innovation means asking hard questions about impact and ethics."}
    ]
    
    # Build full prompt
    prompt = builder.build_full_prompt(persona, topic, chunks)
    
    print("\n📝 SYSTEM PROMPT:")
    print("-" * 50)
    print(prompt['system'])
    
    print("\n📝 USER PROMPT:")
    print("-" * 50)
    print(prompt['user'])


if __name__ == "__main__":
    main()
