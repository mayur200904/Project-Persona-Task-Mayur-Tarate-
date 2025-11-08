"""
Embedding and Indexing Module
Creates vector embeddings and stores them in FAISS
"""

import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingIndexer:
    """Handles embedding generation and vector storage"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize indexer with OpenAI client
        
        Args:
            model_name: OpenAI embedding model name
        """
        # Initialize OpenAI client (reads OPENAI_API_KEY from environment)
        self.client = OpenAI()
        self.model_name = model_name
        self.index = None
        self.chunks_metadata = []
        self.dimension = 1536  # text-embedding-3-small dimension
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Use the openai module directly
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings, dtype='float32')
        print(f"✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def create_index(self, chunks: List[Dict]) -> faiss.IndexFlatL2:
        """
        Create FAISS index from chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            FAISS index
        """
        print("🏗️  Building FAISS index...")
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        self.chunks_metadata = chunks
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"✅ Index created with {self.index.ntotal} vectors")
        
        return self.index
    
    def save_index(self, index_path: str = "data/vector_store.index", 
                   metadata_path: str = "data/index_metadata.json"):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Create index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"💾 Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str = "data/vector_store.index",
                   metadata_path: str = "data/index_metadata.json"):
        """
        Load FAISS index and metadata from disk
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"📂 Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)
        print(f"📂 Loaded {len(self.chunks_metadata)} metadata entries")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        if self.index is None:
            return {"error": "No index created yet"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks_metadata),
            "model": self.model_name
        }


def main():
    """Main indexing pipeline"""
    print("🚀 Starting embedding and indexing...")
    
    # Initialize indexer
    indexer = EmbeddingIndexer()
    
    # Load cleaned chunks
    with open('data/cleaned_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📥 Loaded {len(chunks)} chunks")
    
    # Create index
    indexer.create_index(chunks)
    
    # Save index
    indexer.save_index()
    
    # Display stats
    stats = indexer.get_stats()
    print("\n📊 Index Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
