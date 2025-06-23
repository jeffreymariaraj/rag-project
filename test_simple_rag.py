#!/usr/bin/env python3
"""
Simple test for the RAG pipeline without vector store dependencies
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAG Pipeline"""
    
    @patch('openai.OpenAI')
    def test_generate_answer(self, mock_openai):
        """Test answer generation with mocked OpenAI client"""
        # Set up the mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock completion response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="This is a test answer."))]
        mock_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Import RAGPipeline here to avoid import errors
        from src.rag_pipeline import RAGPipeline
        
        # Create RAGPipeline with mocked vector store
        mock_vector_store = MagicMock()
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            openai_api_key="test_key"
        )
        
        # Test generate_answer
        response = pipeline.generate_answer(
            "What is attention?",
            "Context about attention mechanisms."
        )
        
        # Check that OpenAI API was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Check response format
        self.assertEqual(response['answer'], "This is a test answer.")
        self.assertTrue(response['success'])
        self.assertIn('token_usage', response)
        self.assertIn('total_tokens', response['token_usage'])
        self.assertIn('total_cost', response['token_usage'])
        
        print("Test passed: generate_answer works correctly!")

if __name__ == '__main__':
    unittest.main() 