"""
Retrieval Module
Fetches relevant chunks using similarity search and MMR
"""

import json
import numpy as np
from typing import List, Dict, Optional
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class PostRetriever:
    """Handles retrieval of relevant post chunks"""
    
    def __init__(self, index_path: str = "data/vector_store.index",
                 metadata_path: str = "data/index_metadata.json",
                 model_name: str = "text-embedding-3-small",
                 verbose: bool = False):
        """
        Initialize retriever
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            model_name: OpenAI embedding model name
            verbose: Whether to print status messages
        """
        # Initialize OpenAI client (reads OPENAI_API_KEY from environment)
        self.client = OpenAI()
        self.model_name = model_name
        self.verbose = verbose
        
        # Load index and metadata
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)
        
        if self.verbose:
            print(f"✅ Loaded index with {self.index.ntotal} vectors")
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for query text
        
        Args:
            query: Query string
            
        Returns:
            Query embedding as numpy array
        """
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name,
                timeout=30.0  # 30 second timeout
            )
            embedding = np.array([response.data[0].embedding], dtype='float32')
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
        return embedding
        return embedding
    
    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar chunks
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                results.append(chunk)
        
        return results
    
    def retrieve_with_mmr(self, query: str, top_k: int = 5, 
                          lambda_mult: float = 0.5, fetch_k: int = 20) -> List[Dict]:
        """
        Retrieve chunks using Maximal Marginal Relevance for diversity
        
        Args:
            query: Query string
            top_k: Number of results to return
            lambda_mult: Balance between relevance and diversity (0-1)
            fetch_k: Number of candidates to fetch before MMR
            
        Returns:
            List of diverse relevant chunks
        """
        # Get more candidates than needed
        query_embedding = self.generate_query_embedding(query)
        distances, indices = self.index.search(query_embedding, fetch_k)
        
        # Get embeddings for candidates
        candidate_indices = indices[0]
        candidate_chunks = [self.chunks_metadata[idx] for idx in candidate_indices 
                           if idx < len(self.chunks_metadata)]
        
        # MMR selection
        selected_indices = []
        selected_chunks = []
        
        # Start with most similar
        selected_indices.append(candidate_indices[0])
        selected_chunks.append(candidate_chunks[0])
        
        # Get embeddings for all candidates
        candidate_texts = [chunk['text'] for chunk in candidate_chunks]
        candidate_embeddings = self._get_batch_embeddings(candidate_texts)
        
        while len(selected_indices) < top_k and len(selected_indices) < len(candidate_indices):
            best_score = -float('inf')
            best_idx = None
            
            for i, idx in enumerate(candidate_indices):
                if idx in selected_indices:
                    continue
                
                # Relevance to query
                relevance = 1 / (1 + distances[0][i])
                
                # Max similarity to already selected
                max_sim = 0
                for selected_i in range(len(selected_indices)):
                    # Cosine similarity
                    sim = np.dot(candidate_embeddings[i], 
                               candidate_embeddings[candidate_indices.tolist().index(selected_indices[selected_i])])
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.append(candidate_indices[best_idx])
                chunk = candidate_chunks[best_idx].copy()
                chunk['mmr_score'] = float(best_score)
                selected_chunks.append(chunk)
        
        return selected_chunks
    
    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        embeddings = np.array([item.embedding for item in response.data], dtype='float32')
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
    
    def retrieve_with_context(self, persona_info: Dict, topic: str, 
                             top_k: int = 5, use_mmr: bool = True) -> List[Dict]:
        """
        Retrieve chunks with persona and topic context
        
        Args:
            persona_info: Dictionary with name, title, company, industry
            topic: Main topic or bullets for the post
            top_k: Number of results
            use_mmr: Whether to use MMR for diversity
            
        Returns:
            List of relevant chunks
        """
        # Build contextual query
        query = f"""
        Person: {persona_info.get('name', 'Professional')}
        Role: {persona_info.get('title', '')} at {persona_info.get('company', '')}
        Topic: {topic}
        """
        
        if use_mmr:
            return self.retrieve_with_mmr(query, top_k=top_k)
        else:
            return self.retrieve_similar(query, top_k=top_k)


def main():
    """Test retrieval pipeline"""
    print("🚀 Testing retrieval system...")
    
    # Initialize retriever
    retriever = PostRetriever()
    
    # Test query
    persona = {
        "name": "Tech Leader",
        "title": "CEO",
        "company": "Innovation Corp",
        "industry": "Technology"
    }
    topic = "AI innovation and sustainable technology"
    
    print(f"\n🔍 Retrieving for topic: {topic}")
    
    # Test similarity search
    print("\n📌 Similarity Search Results:")
    results = retriever.retrieve_similar(topic, top_k=3)
    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. Score: {chunk['similarity_score']:.3f}")
        print(f"   Text: {chunk['text'][:100]}...")
    
    # Test MMR
    print("\n📌 MMR Search Results:")
    mmr_results = retriever.retrieve_with_mmr(topic, top_k=3)
    for i, chunk in enumerate(mmr_results, 1):
        print(f"\n{i}. MMR Score: {chunk.get('mmr_score', 'N/A')}")
        print(f"   Text: {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
