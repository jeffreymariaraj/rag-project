"""
Test script for RAG system

This script tests the basic functionality of the RAG pipeline.
It can be used to validate that the system is working correctly.
"""

import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_rag_pipeline():
    """Test the RAG pipeline functionality"""
    try:
        from src.rag_pipeline import RAGPipeline
        from src.prompt_templates import PromptTemplates
        
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return False
        
        # Check if vector store path exists
        vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db")
        if not Path(vector_db_path).exists():
            logger.error(f"Vector database path not found: {vector_db_path}")
            return False
        
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(
            vector_store_path=vector_db_path,
            openai_api_key=api_key
        )
        
        # Test query
        test_query = "What is the main innovation in the Attention is All You Need paper?"
        logger.info(f"Testing query: '{test_query}'")
        
        start_time = time.time()
        response = pipeline.ask_question(test_query)
        processing_time = time.time() - start_time
        
        # Check response
        if not response or not response.get('answer'):
            logger.error("No answer received from RAG pipeline")
            return False
        
        logger.info(f"Response received in {processing_time:.2f} seconds")
        logger.info(f"Answer: {response['answer'][:100]}...")
        logger.info(f"Retrieved {response['num_retrieved_chunks']} chunks")
        
        if 'token_usage' in response:
            logger.info(f"Token usage: {response['token_usage']['total_tokens']} tokens")
        
        # Test prompt templates
        logger.info("Testing prompt templates...")
        templates = PromptTemplates()
        template_ids = templates.get_template_ids()
        logger.info(f"Available templates: {template_ids}")
        
        # Test a different template
        if 'technical' in template_ids:
            logger.info("Testing technical template...")
            response = pipeline.ask_question(
                test_query,
                prompt_template_id='technical'
            )
            logger.info(f"Technical template answer: {response['answer'][:100]}...")
        
        logger.info("All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing RAG pipeline: {e}")
        return False

def test_web_interface():
    """Test the web interface functionality"""
    try:
        # Import web interface module
        from src.web_interface import initialize_rag_pipeline
        
        logger.info("Testing web interface initialization...")
        
        # Try to initialize RAG pipeline
        pipeline = initialize_rag_pipeline()
        if pipeline:
            logger.info("Web interface successfully initialized RAG pipeline")
            return True
        else:
            logger.error("Web interface failed to initialize RAG pipeline")
            return False
            
    except Exception as e:
        logger.error(f"Error testing web interface: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("RAG System Test")
    print("=" * 50)
    
    # Test RAG pipeline
    print("\nTesting RAG pipeline...")
    rag_success = test_rag_pipeline()
    print(f"RAG pipeline test {'PASSED' if rag_success else 'FAILED'}")
    
    # Test web interface
    print("\nTesting web interface...")
    web_success = test_web_interface()
    print(f"Web interface test {'PASSED' if web_success else 'FAILED'}")
    
    # Overall result
    print("\n" + "=" * 50)
    if rag_success and web_success:
        print("All tests PASSED! The system is ready to use.")
        print("\nTo run the web interface:")
        print("  python run_web_interface.py")
    else:
        print("Some tests FAILED. Please check the logs for details.")
    print("=" * 50) 