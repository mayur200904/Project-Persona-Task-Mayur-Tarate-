"""
Data Ingestion Module
Cleans and chunks LinkedIn posts while preserving writing style
"""

import json
import re
from typing import List, Dict
from pathlib import Path


class PostIngester:
    """Handles ingestion and chunking of LinkedIn posts"""
    
    def __init__(self):
        self.chunks = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving style markers
        
        Args:
            text: Raw post text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace but preserve single newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove URLs but keep the text structure
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        return re.findall(r'#\w+', text)
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Split text into meaningful chunks (paragraphs)
        
        Args:
            text: Post text
            
        Returns:
            List of text chunks
        """
        # Split by double newlines or sentence groups
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback: split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Group sentences into chunks of 2-3
            paragraphs = []
            for i in range(0, len(sentences), 2):
                chunk = ' '.join(sentences[i:i+2])
                if chunk.strip():
                    paragraphs.append(chunk)
        
        return paragraphs
    
    def ingest_posts(self, posts_data: List[Dict]) -> List[Dict]:
        """
        Process and chunk multiple posts
        
        Args:
            posts_data: List of post dictionaries
            
        Returns:
            List of chunked posts with metadata
        """
        all_chunks = []
        
        for post in posts_data:
            cleaned_text = self.clean_text(post['content'])
            hashtags = self.extract_hashtags(post['content'])
            
            # Chunk the post
            chunks = self.chunk_by_paragraph(cleaned_text)
            
            # Create metadata for each chunk
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': f"{post['post_id']}_chunk_{idx}",
                    'post_id': post['post_id'],
                    'chunk_index': idx,
                    'text': chunk,
                    'hashtags': hashtags,
                    'date': post.get('date', ''),
                    'link': post.get('link', ''),
                    'word_count': len(chunk.split())
                }
                all_chunks.append(chunk_data)
        
        self.chunks = all_chunks
        return all_chunks
    
    def save_chunks(self, output_path: str):
        """Save processed chunks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.chunks)} chunks to {output_path}")
    
    @staticmethod
    def load_posts(input_path: str) -> List[Dict]:
        """Load posts from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Main ingestion pipeline"""
    print("🚀 Starting data ingestion...")
    
    # Initialize ingester
    ingester = PostIngester()
    
    # Load sample posts
    posts = ingester.load_posts('data/sample_posts.json')
    print(f"📥 Loaded {len(posts)} posts")
    
    # Process and chunk
    chunks = ingester.ingest_posts(posts)
    print(f"✂️  Created {len(chunks)} chunks")
    
    # Save processed chunks
    ingester.save_chunks('data/cleaned_chunks.json')
    
    # Display sample
    print("\n📝 Sample chunk:")
    print(json.dumps(chunks[0], indent=2))


if __name__ == "__main__":
    main()
