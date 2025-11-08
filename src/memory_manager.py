"""
Memory Management Module
Handles persona preferences and post history
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class MemoryManager:
    """Manages persistent memory for persona preferences"""
    
    def __init__(self, memory_path: str = "memory/memory.json", verbose: bool = False):
        """
        Initialize memory manager
        
        Args:
            memory_path: Path to memory JSON file
            verbose: Whether to print status messages
        """
        self.memory_path = memory_path
        self.verbose = verbose
        self.memory = self._load_or_create_memory()
    
    def _load_or_create_memory(self) -> Dict:
        """Load existing memory or create from template"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            if self.verbose:
                print(f"📂 Loaded memory from {self.memory_path}")
            return memory
        else:
            # Load template
            template_path = "memory/memory_template.json"
            with open(template_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            if self.verbose:
                print(f"🆕 Created new memory from template")
            return memory
    
    def save_memory(self):
        """Save current memory state to disk"""
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        if self.verbose:
            print(f"💾 Saved memory to {self.memory_path}")
    
    def update_persona(self, persona_info: Dict):
        """
        Update persona information
        
        Args:
            persona_info: Dictionary with name, title, company, industry
        """
        self.memory['persona'].update(persona_info)
        self.save_memory()
        print("✅ Updated persona information")
    
    def add_preferred_hashtag(self, hashtag: str):
        """Add a new preferred hashtag"""
        if hashtag not in self.memory['preferences']['preferred_hashtags']:
            self.memory['preferences']['preferred_hashtags'].append(hashtag)
            self.save_memory()
            print(f"✅ Added hashtag: {hashtag}")
    
    def add_banned_phrase(self, phrase: str):
        """Add a phrase to ban from generation"""
        if phrase not in self.memory['preferences']['banned_phrases']:
            self.memory['preferences']['banned_phrases'].append(phrase)
            self.save_memory()
            print(f"✅ Added banned phrase: {phrase}")
    
    def add_theme(self, theme: str):
        """Add a recurring theme"""
        if theme not in self.memory['preferences']['recurring_themes']:
            self.memory['preferences']['recurring_themes'].append(theme)
            self.save_memory()
            print(f"✅ Added theme: {theme}")
    
    def log_generated_post(self, post_data: Dict):
        """
        Log a generated post to memory
        
        Args:
            post_data: Dictionary with post content and metadata
        """
        post_entry = {
            "timestamp": datetime.now().isoformat(),
            "post": post_data.get('post', ''),
            "topic": post_data.get('topic', ''),
            "method": post_data.get('method', 'RAG'),
            "word_count": post_data.get('word_count', 0),
            "hashtag_count": post_data.get('hashtag_count', 0)
        }
        
        self.memory['previous_posts'].append(post_entry)
        
        # Keep only last 50 posts to avoid memory bloat
        if len(self.memory['previous_posts']) > 50:
            self.memory['previous_posts'] = self.memory['previous_posts'][-50:]
        
        self.save_memory()
        print("✅ Logged generated post to memory")
    
    def get_persona_info(self) -> Dict:
        """Get current persona information"""
        return self.memory['persona']
    
    def get_preferences(self) -> Dict:
        """Get current preferences"""
        return self.memory['preferences']
    
    def get_style_guidelines(self) -> Dict:
        """Get style guidelines"""
        return self.memory['style_guidelines']
    
    def get_recent_posts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent generated posts
        
        Args:
            limit: Number of recent posts to return
            
        Returns:
            List of recent posts
        """
        return self.memory['previous_posts'][-limit:]
    
    def export_memory(self, export_path: str):
        """
        Export memory to a different location
        
        Args:
            export_path: Path to export memory
        """
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        print(f"📤 Exported memory to {export_path}")
    
    def get_context_summary(self) -> str:
        """
        Get a summary of memory for prompt context
        
        Returns:
            Formatted string with key preferences
        """
        prefs = self.memory['preferences']
        style = self.memory['style_guidelines']
        
        summary = f"""Preferred hashtags: {', '.join(prefs['preferred_hashtags'][:4])}
Recurring themes: {', '.join(prefs['recurring_themes'][:3])}
Tone: {prefs['tone']}
Word count range: {style['word_count_range'][0]}-{style['word_count_range'][1]}
Max hashtags: {style['max_hashtags']}"""
        
        return summary


def main():
    """Test memory management"""
    print("🚀 Testing memory manager...")
    
    # Initialize manager
    manager = MemoryManager()
    
    # Display current persona
    persona = manager.get_persona_info()
    print("\n👤 Current Persona:")
    print(json.dumps(persona, indent=2))
    
    # Update persona
    manager.update_persona({
        "name": "Alex Johnson",
        "title": "VP of Engineering",
        "company": "CloudTech Solutions"
    })
    
    # Add preferences
    manager.add_preferred_hashtag("#CloudComputing")
    manager.add_theme("digital transformation")
    
    # Log a test post
    test_post = {
        "post": "Test post about innovation...",
        "topic": "Innovation in tech",
        "method": "RAG",
        "word_count": 150,
        "hashtag_count": 3
    }
    manager.log_generated_post(test_post)
    
    # Get context summary
    print("\n📝 Context Summary:")
    print(manager.get_context_summary())
    
    # Show recent posts
    recent = manager.get_recent_posts(limit=3)
    print(f"\n📚 Recent posts: {len(recent)}")


if __name__ == "__main__":
    main()
