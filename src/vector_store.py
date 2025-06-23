"""
Vector Store Module for RAG System
Handles embedding generation, vector storage, and similarity search
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from dataclasses import asdict

# Vector database imports
import chromadb
from chromadb.config import Settings
import faiss

# Embedding model imports
from sentence_transformers import SentenceTransformer
import torch

# LangChain imports
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Local imports
from document_processing import DocumentChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles embedding generation using sentence-transformers
    Provides caching and batch processing capabilities
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

class ChromaVectorStore:
    """
    Vector store implementation using ChromaDB
    Provides persistent storage and efficient similarity search
    """
    
    def __init__(self, collection_name: str = "ml_papers", 
                 persist_directory: str = "vector_db/chroma",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Model name for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            # First try to get the collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                # If it doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection: {e}")
            raise
    
    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """Embedding function for ChromaDB"""
        embeddings = self.embedding_generator.generate_embeddings(texts, show_progress=False)
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for adding documents
        """
        logger.info(f"Adding {len(chunks)} documents to vector store")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk.content)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {
                'paper_title': chunk.paper_title,
                'source': chunk.source,
                'section': chunk.section or 'Unknown',
                'page_number': str(chunk.page_number),
                'chunk_index': str(chunk.chunk_index),
                'chunk_length': str(len(chunk.content))
            }
            
            # Add additional metadata from chunk.metadata
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"meta_{key}"] = str(value)
            
            metadatas.append(metadata)
            ids.append(f"{chunk.source}_{chunk.chunk_index}")
        
        # Add documents in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        logger.info(f"Successfully added {len(chunks)} documents to ChromaDB")
    
    def similarity_search(self, query: str, top_k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Perform similarity search
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with metadata and scores
        """
        logger.info(f"Performing similarity search for: '{query[:50]}...'")
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} similar documents")
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.embedding_dim
        }

class FAISSVectorStore:
    """
    Vector store implementation using FAISS
    Provides high-performance similarity search
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 index_type: str = "flat"):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_model: Model name for embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.embedding_dim = self.embedding_generator.embedding_dim
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine sim with normalized vectors)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)  # 100 clusters
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Default to flat
        
        # Storage for documents and metadata
        self.documents = []
        self.metadatas = []
        self.id_to_index = {}
        
        logger.info(f"Initialized FAISS index: {index_type}, dimension: {self.embedding_dim}")
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to FAISS index"""
        logger.info(f"Adding {len(chunks)} documents to FAISS index")
        
        # Extract texts and generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add embeddings to index
        start_idx = len(self.documents)
        self.index.add(embeddings)
        
        # Store documents and metadata
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk.content)
            
            metadata = {
                'paper_title': chunk.paper_title,
                'source': chunk.source,
                'section': chunk.section,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'original_metadata': chunk.metadata
            }
            self.metadatas.append(metadata)
            self.id_to_index[f"{chunk.source}_{chunk.chunk_index}"] = start_idx + i
        
        logger.info(f"Successfully added {len(chunks)} documents to FAISS index")
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform similarity search using FAISS"""
        logger.info(f"Performing FAISS similarity search for: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                result = {
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'similarity_score': float(similarity),
                    'rank': i + 1
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'id_to_index': self.id_to_index,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }
        
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved FAISS index to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        filepath = Path(filepath)
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
        
        # Load metadata
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        self.metadatas = metadata['metadatas']
        self.id_to_index = metadata['id_to_index']
        
        logger.info(f"Loaded FAISS index from {filepath}")

class HybridVectorStore:
    """
    Hybrid vector store that combines ChromaDB and FAISS
    Provides both persistence and high-performance search
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "vector_db/hybrid"):
        """Initialize hybrid vector store"""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize both stores
        self.chroma_store = ChromaVectorStore(
            collection_name="ml_papers_chroma",
            persist_directory=str(self.persist_directory / "chroma"),
            embedding_model=embedding_model
        )
        
        self.faiss_store = FAISSVectorStore(embedding_model=embedding_model)
        
        logger.info("Initialized hybrid vector store")
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add documents to both stores"""
        logger.info("Adding documents to hybrid store")
        
        # Add to both stores
        self.chroma_store.add_documents(chunks)
        self.faiss_store.add_documents(chunks)
        
        # Save FAISS index
        self.faiss_store.save_index(str(self.persist_directory / "faiss_index"))
    
    def similarity_search(self, query: str, top_k: int = 5, 
                         use_faiss: bool = True) -> List[Dict]:
        """
        Perform similarity search using specified backend
        
        Args:
            query: Query string
            top_k: Number of results
            use_faiss: Whether to use FAISS (True) or ChromaDB (False)
        """
        if use_faiss:
            return self.faiss_store.similarity_search(query, top_k)
        else:
            return self.chroma_store.similarity_search(query, top_k)
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining both stores
        Returns reranked results from both systems
        """
        logger.info(f"Performing hybrid search for: '{query[:50]}...'")
        
        # Search both stores
        chroma_results = self.chroma_store.similarity_search(query, top_k)
        faiss_results = self.faiss_store.similarity_search(query, top_k)
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add ChromaDB results
        for result in chroma_results:
            key = result['content'][:100]  # Use content snippet as key
            if key not in combined_results:
                result['source_store'] = 'chroma'
                combined_results[key] = result
        
        # Add FAISS results (with score averaging if duplicate)
        for result in faiss_results:
            key = result['content'][:100]
            if key in combined_results:
                # Average the scores
                existing_score = combined_results[key]['similarity_score']
                new_score = result['similarity_score']
                combined_results[key]['similarity_score'] = (existing_score + new_score) / 2
                combined_results[key]['source_store'] = 'both'
            else:
                result['source_store'] = 'faiss'
                combined_results[key] = result
        
        # Sort by similarity score and return top_k
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:top_k]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results

class VectorStoreManager:
    """
    Manager class for vector store operations
    Provides high-level interface for RAG system
    """
    
    def __init__(self, store_type: str = "chroma", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "vector_db"):
        """
        Initialize vector store manager
        
        Args:
            store_type: Type of store ('chroma', 'faiss', 'hybrid')
            embedding_model: Model for embeddings
            persist_directory: Directory for persistence
        """
        self.store_type = store_type
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate store
        if store_type == "chroma":
            self.store = ChromaVectorStore(
                persist_directory=str(self.persist_directory),
                embedding_model=embedding_model
            )
        elif store_type == "faiss":
            self.store = FAISSVectorStore(embedding_model=embedding_model)
        elif store_type == "hybrid":
            self.store = HybridVectorStore(
                embedding_model=embedding_model,
                persist_directory=str(self.persist_directory)
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
        
        logger.info(f"Initialized vector store manager: {store_type}")
    
    def build_index(self, chunks: List[DocumentChunk]) -> Dict:
        """
        Build vector index from document chunks
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dictionary with build statistics
        """
        logger.info(f"Building vector index with {len(chunks)} chunks")
        
        # Add documents to store
        self.store.add_documents(chunks)
        
        # Save if FAISS
        if self.store_type == "faiss":
            index_path = self.persist_directory / "faiss_index"
            self.store.save_index(str(index_path))
        
        # Get statistics
        if hasattr(self.store, 'get_collection_stats'):
            stats = self.store.get_collection_stats()
        else:
            stats = {
                'total_documents': len(chunks),
                'store_type': self.store_type
            }
        
        logger.info(f"Vector index built successfully: {stats}")
        return stats
    
    def search(self, query: str, top_k: int = 5, 
               similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Search the vector store
        
        Args:
            query: Search query
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        # Perform search
        if hasattr(self.store, 'hybrid_search') and self.store_type == "hybrid":
            results = self.store.hybrid_search(query, top_k)
        else:
            results = self.store.similarity_search(query, top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= similarity_threshold
        ]
        
        # Log search results
        self._log_search_results(query, filtered_results)
        
        return filtered_results
    
    def _log_search_results(self, query: str, results: List[Dict]):
        """Log search results for debugging"""
        logger.info(f"Search query: '{query[:50]}...'")
        logger.info(f"Found {len(results)} results above threshold")
        
        for i, result in enumerate(results[:3]):  # Log top 3
            logger.info(f"Result {i+1}: Score={result['similarity_score']:.3f}, "
                       f"Paper={result['metadata'].get('paper_title', 'Unknown')}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        if hasattr(self.store, 'get_collection_stats'):
            return self.store.get_collection_stats()
        else:
            return {'store_type': self.store_type}

# Example usage and testing
if __name__ == "__main__":
    # Test vector store functionality
    from document_processing import DocumentProcessor
    
    print("Testing Vector Store functionality...")
    
    # Initialize vector store manager
    manager = VectorStoreManager(store_type="chroma", embedding_model="all-MiniLM-L6-v2")
    
    # Create sample chunks for testing
    sample_chunks = []
    sample_texts = [
        "The Transformer architecture uses self-attention mechanisms to process sequences.",
        "BERT employs bidirectional training to understand context from both directions.",
        "GPT-3 demonstrates few-shot learning capabilities without task-specific training.",
        "CLIP combines vision and language understanding through contrastive learning.",
        "LLaMA focuses on computational efficiency while maintaining performance."
    ]
    
    for i, text in enumerate(sample_texts):
        from document_processing import DocumentChunk
        chunk = DocumentChunk(
            content=text,
            metadata={'test': True},
            page_number=1,
            chunk_index=i,
            source=f"test_paper_{i}.pdf",
            paper_title=f"Test Paper {i}",
            section="Test Section"
        )
        sample_chunks.append(chunk)
    
    # Build index
    stats = manager.build_index(sample_chunks)
    print(f"✓ Built index: {stats}")
    
    # Test search
    query = "What is the main innovation in attention mechanisms?"
    results = manager.search(query, top_k=3)
    
    print(f"✓ Search results for '{query}':")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['similarity_score']:.3f}")
        print(f"     Content: {result['content'][:100]}...")
    
    print("\nVector store testing completed! ✓")