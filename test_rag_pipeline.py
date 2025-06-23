#!/usr/bin/env python3
"""
Test script for the RAG Pipeline
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """Test the RAG pipeline with a simple question"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.error("OpenAI API key not set. Please set OPENAI_API_KEY in .env file.")
        print("\nPlease set your OpenAI API key in the .env file before running this test.")
        print("Example: OPENAI_API_KEY=sk-your-key-here")
        return False
    
    # Initialize RAG pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(
            vector_store_path="vector_db",
            openai_api_key=api_key
        )
        logger.info("RAG pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return False
    
    # Test question
    test_question = "What is the main innovation in the Attention is All You Need paper?"
    
    # Process question
    try:
        logger.info(f"Processing question: '{test_question}'")
        response = pipeline.ask_question(test_question)
        
        # Print results
        print("\n" + "="*80)
        print("RAG PIPELINE TEST RESULTS")
        print("="*80)
        print(f"Question: {test_question}")
        print(f"\nAnswer: {response['answer']}")
        print("\nMetadata:")
        print(f"  - Processing time: {response['processing_time_seconds']:.2f} seconds")
        print(f"  - Retrieved chunks: {response['num_retrieved_chunks']}")
        if 'token_usage' in response:
            print(f"  - Tokens used: {response['token_usage']['total_tokens']}")
            print(f"  - Estimated cost: ${response['token_usage']['total_cost']:.6f}")
        
        print("\nTop sources:")
        for i, source in enumerate(response['sources'][:3], 1):
            print(f"  {i}. {source['paper_title']} (Section: {source['section']}, "
                 f"Page: {source['page']}, Score: {source['similarity_score']:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return False

def main():
    """Main function"""
    print("Testing RAG Pipeline...")
    success = test_rag_pipeline()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. See error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 