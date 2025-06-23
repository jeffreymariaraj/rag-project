"""
Build Vector Database Script
Processes PDFs and creates vector database for RAG system
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_processing import DocumentProcessor, DocumentChunk
from vector_store import VectorStoreManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vector_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    Path("logs").mkdir(exist_ok=True)

def validate_pdfs(pdf_directory: str) -> List[str]:
    """
    Validate that all required PDFs are present
    
    Args:
        pdf_directory: Directory containing PDFs
        
    Returns:
        List of valid PDF paths
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
    
    logger.info(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file.name}")
    
    return [str(pdf) for pdf in pdf_files]

def process_documents(pdf_paths: List[str], chunk_size: int = 800, 
                     chunk_overlap: int = 200) -> List[DocumentChunk]:
    """
    Process PDF documents into chunks
    
    Args:
        pdf_paths: List of PDF file paths
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of DocumentChunk objects
    """
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    
    logger.info(f"Processing {len(pdf_paths)} PDF files...")
    
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            chunks = processor.process_pdf(pdf_path)
            all_chunks.extend(chunks)
            logger.info(f"✓ {Path(pdf_path).name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to process {Path(pdf_path).name}: {e}")
            continue
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def build_vector_database(chunks: List[DocumentChunk], store_type: str = "chroma",
                         embedding_model: str = "all-MiniLM-L6-v2",
                         persist_directory: str = "vector_db") -> dict:
    """
    Build vector database from chunks
    
    Args:
        chunks: List of DocumentChunk objects
        store_type: Type of vector store ('chroma', 'faiss', 'hybrid')
        embedding_model: Embedding model to use
        persist_directory: Directory to persist the database
        
    Returns:
        Dictionary with build statistics
    """
    logger.info(f"Building {store_type} vector database...")
    
    # Initialize vector store manager
    manager = VectorStoreManager(
        store_type=store_type,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )
    
    # Build index
    start_time = time.time()
    stats = manager.build_index(chunks)
    build_time = time.time() - start_time
    
    stats['build_time_seconds'] = build_time
    stats['build_time_formatted'] = f"{build_time:.2f}s"
    
    logger.info(f"✓ Vector database built in {build_time:.2f}s")
    return stats

def test_vector_database(persist_directory: str, store_type: str = "chroma",
                        embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Test the built vector database with sample queries
    
    Args:
        persist_directory: Directory containing the vector database
        store_type: Type of vector store
        embedding_model: Embedding model used
    """
    logger.info("Testing vector database...")
    
    # Initialize manager
    manager = VectorStoreManager(
        store_type=store_type,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )
    
    # Test queries (benchmark questions)
    test_queries = [
        "What is the main innovation introduced in the Attention is All You Need paper?",
        "How does BERT differ from traditional left-to-right language models?",
        "Describe the few-shot learning capability of GPT-3",
        "What is the loss function used in CLIP?",
        "What approach does LLaMA take to reduce computational cost?"
    ]
    
    logger.info("Running test queries...")
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Test Query {i} ---")
        logger.info(f"Query: {query}")
        
        try:
            results = manager.search(query, top_k=3, similarity_threshold=0.5)
            
            if results:
                logger.info(f"✓ Found {len(results)} results")
                for j, result in enumerate(results, 1):
                    logger.info(f"  Result {j}: Score={result['similarity_score']:.3f}, "
                               f"Paper={result['metadata'].get('paper_title', 'Unknown')}")
            else:
                logger.warning("✗ No results found")
                
        except Exception as e:
            logger.error(f"✗ Query failed: {e}")
    
    # Get final stats
    stats = manager.get_stats()
    logger.info(f"\nFinal database stats: {stats}")

def main():
    """Main function to build vector database"""
    parser = argparse.ArgumentParser(description="Build vector database for RAG system")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Directory containing PDF files")
    parser.add_argument("--store-type", choices=["chroma", "faiss", "hybrid"], 
                        default="chroma", help="Vector store type")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                        help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=800, 
                        help="Size of each chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                        help="Overlap between chunks")
    parser.add_argument("--persist-dir", default="vector_db", 
                        help="Directory to persist the database")
    parser.add_argument("--test", action="store_true", 
                        help="Test the database after building")
    parser.add_argument("--only-test", action="store_true", 
                        help="Only test an existing database")
    
    args = parser.parse_args()
    
    # Create logs directory
    create_logs_directory()
    
    # Log arguments
    logger.info(f"Started build_vector_database with arguments: {args}")
    
    # Determine store path
    store_path = Path(args.persist_dir)
    if args.store_type != "hybrid":
        store_path = store_path / args.store_type
    
    # Test only mode
    if args.only_test:
        logger.info("Running in test-only mode")
        test_vector_database(
            persist_directory=str(store_path),
            store_type=args.store_type,
            embedding_model=args.embedding_model
        )
        return
    
    # Validate PDFs
    try:
        pdf_paths = validate_pdfs(args.pdf_dir)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return
    
    # Process documents
    chunks = process_documents(
        pdf_paths=pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if not chunks:
        logger.error("No chunks created. Aborting.")
        return
    
    # Build vector database
    stats = build_vector_database(
        chunks=chunks,
        store_type=args.store_type,
        embedding_model=args.embedding_model,
        persist_directory=str(store_path)
    )
    
    # Output stats
    logger.info(f"\nBuild completed successfully!")
    logger.info(f"Documents: {stats.get('document_count', 'N/A')}")
    logger.info(f"Embeddings: {stats.get('embedding_count', 'N/A')}")
    logger.info(f"Build time: {stats.get('build_time_formatted', 'N/A')}")
    
    # Test if requested
    if args.test:
        logger.info("\nTesting vector database...")
        test_vector_database(
            persist_directory=str(store_path),
            store_type=args.store_type,
            embedding_model=args.embedding_model
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)