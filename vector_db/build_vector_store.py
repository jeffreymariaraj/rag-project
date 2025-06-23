#!/usr/bin/env python3
"""
Integration script to build the vector store from processed documents.
This script connects document processing with vector storage.
"""

import os
import sys
import logging
from typing import List, Dict
import argparse

# Add src directory to path
sys.path.append('src')

from document_processing import DocumentProcessor
from vector_store import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """
    Builder class to create and populate vector store from research papers.
    """
    
    def __init__(self, 
                 pdf_directory: str = "data/pdfs",
                 vector_store_dir: str = "vector_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the vector store builder.
        
        Args:
            pdf_directory: Directory containing PDF files
            vector_store_dir: Directory for vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_directory = pdf_directory
        self.vector_store_dir = vector_store_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(persist_directory=vector_store_dir)
        
        # Paper mapping for better metadata
        self.paper_mapping = {
            'attention_is_all_you_need': 'Attention Is All You Need',
            'bert': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'gpt3': 'GPT-3: Language Models are Few-Shot Learners',
            'clip': 'Contrastive Language-Image Pretraining with Knowledge Graphs',
            'llama': 'LLaMA: Open and Efficient Foundation Language Models'
        }
    
    def process_single_paper(self, pdf_path: str, paper_title: str) -> tuple[List[str], List[Dict]]:
        """
        Process a single research paper.
        
        Args:
            pdf_path: Path to the PDF file
            paper_title: Title of the paper
            
        Returns:
            Tuple of (text_chunks, metadata_list)
        """
        logger.info(f"Processing paper: {paper_title}")
        
        # Extract text from PDF
        extracted_text = self.doc_processor.extract_text_from_pdf(pdf_path)
        
        if not extracted_text:
            logger.error(f"Failed to extract text from {pdf_path}")
            return [], []
        
        # Clean and preprocess text
        cleaned_text = self.doc_processor.clean_text(extracted_text)
        
        # Create chunks
        chunks = self.doc_processor.create_chunks(cleaned_text)
        
        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'paper_title': paper_title,
                'chunk_id': f"{paper_title.lower().replace(' ', '_')}_chunk_{i+1}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'pdf_source': os.path.basename(pdf_path),
                'chunk_size': len(chunk),
                # Estimate page number (rough approximation)
                'page_number': max(1, (i * self.chunk_size) // 2000 + 1)
            }
            metadatas.append(metadata)
        
        logger.info(f"Created {len(chunks)} chunks for {paper_title}")
        return chunks, metadatas
    
    def build_vector_store(self, force_rebuild: bool = False) -> None:
        """
        Build the complete vector store from all research papers.
        
        Args:
            force_rebuild: Whether to rebuild even if vector store exists
        """
        logger.info("Starting vector store construction...")
        
        # Check if vector store already exists and has data
        if not force_rebuild:
            try:
                stats = self.vector_store.get_collection_stats()
                if stats['total_documents'] > 0:
                    logger.info(f"Vector store already exists with {stats['total_documents']} documents")
                    user_input = input("Vector store exists. Rebuild? (y/N): ").lower()
                    if user_input != 'y':
                        logger.info("Skipping vector store rebuild")
                        return
            except:
                pass  # Collection doesn't exist yet
        
        # Get all PDF files
        if not os.path.exists(self.pdf_directory):
            logger.error(f"PDF directory not found: {self.pdf_directory}")
            return
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_directory}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process all papers
        all_chunks = []
        all_metadatas = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            
            # Try to map filename to paper title
            filename_base = os.path.splitext(pdf_file)[0].lower()
            paper_title = self._get_paper_title(filename_base, pdf_file)
            
            try:
                chunks, metadatas = self.process_single_paper(pdf_path, paper_title)
                
                if chunks:
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    logger.info(f"Successfully processed {pdf_file}")
                else:
                    logger.warning(f"No chunks created for {pdf_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        if not all_chunks:
            logger.error("No chunks were created from any papers")
            return
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Clear existing collection if rebuilding
        if force_rebuild:
            try:
                self.vector_store.delete_collection()
                self.vector_store = VectorStore(persist_directory=self.vector_store_dir)
            except:
                pass
        
        # Add all documents to vector store
        try:
            self.vector_store.add_documents(all_chunks, all_metadatas)
            logger.info("Successfully built vector store!")
            
            # Create backup
            backup_path = f"{self.vector_store_dir}_backup.pkl"
            self.vector_store.save_backup(backup_path)
            logger.info(f"Backup saved to: {backup_path}")
            
            # Print final statistics
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Final stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
    
    def _get_paper_title(self, filename_base: str, original_filename: str) -> str:
        """
        Map filename to paper title.
        
        Args:
            filename_base: Base filename without extension
            original_filename: Original filename
            
        Returns:
            Paper title
        """
        # Try exact matches first
        for key, title in self.paper_mapping.items():
            if key in filename_base:
                return title
        
        # Try partial matches
        if 'attention' in filename_base or 'transformer' in filename_base:
            return self.paper_mapping['attention_is_all_you_need']
        elif 'bert' in filename_base:
            return self.paper_mapping['bert']
        elif 'gpt' in filename_base or 'gpt3' in filename_base:
            return self.paper_mapping['gpt3']
        elif 'clip' in filename_base:
            return self.paper_mapping['clip']
        elif 'llama' in filename_base:
            return self.paper_mapping['llama']
        
        # Fallback to filename
        return original_filename.replace('.pdf', '').replace('_', ' ').title()
    
    def test_vector_store(self) -> None:
        """Test the built vector store with sample queries."""
        logger.info("Testing vector store...")
        
        test_queries = [
            "attention mechanism",
            "BERT bidirectional",
            "GPT-3 few-shot learning",
            "CLIP contrastive learning",
            "LLaMA efficiency"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = self.vector_store.similarity_search(query, n_results=3)
            
            print(f"\nQuery: {query}")
            print("-" * 40)
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['similarity_score']:.3f}")
                print(f"   Paper: {result['metadata']['paper_title']}")
                print(f"   Preview: {result['document'][:100]}...")
                print()
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about the vector store."""
        stats = self.vector_store.get_collection_stats()
        
        # Get additional statistics
        try:
            # Sample some documents to analyze
            sample_results = self.vector_store.similarity_search("machine learning", n_results=50)
            
            # Count papers
            papers = set()
            chunk_counts = {}
            
            for result in sample_results:
                paper_title = result['metadata']['paper_title']
                papers.add(paper_title)
                chunk_counts[paper_title] = chunk_counts.get(paper_title, 0) + 1
            
            detailed_stats = {
                **stats,
                'unique_papers': len(papers),
                'paper_names': list(papers),
                'chunks_per_paper': chunk_counts,
                'avg_chunks_per_paper': sum(chunk_counts.values()) / len(chunk_counts) if chunk_counts else 0
            }
            
            return detailed_stats
            
        except Exception as e:
            logger.error(f"Error getting detailed statistics: {e}")
            return stats


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Build vector store from research papers")
    parser.add_argument("--pdf-dir", default="data/pdfs", 
                       help="Directory containing PDF files")
    parser.add_argument("--vector-dir", default="vector_db", 
                       help="Directory for vector database")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                       help="Overlap between chunks")
    parser.add_argument("--force-rebuild", action="store_true", 
                       help="Force rebuild even if vector store exists")
    parser.add_argument("--test", action="store_true", 
                       help="Test the vector store after building")
    parser.add_argument("--stats", action="store_true", 
                       help="Show detailed statistics")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = VectorStoreBuilder(
        pdf_directory=args.pdf_dir,
        vector_store_dir=args.vector_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        # Build vector store
        builder.build_vector_store(force_rebuild=args.force_rebuild)
        
        # Test if requested
        if args.test:
            builder.test_vector_store()
        
        # Show statistics if requested
        if args.stats:
            stats = builder.get_statistics()
            print("\nDETAILED STATISTICS:")
            print("=" * 50)
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        logger.info("Vector store building completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    main()