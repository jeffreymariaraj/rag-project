"""
Test suite for the RAG Pipeline
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAG Pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock the vector store
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.search.return_value = [
            {
                'content': 'This is a test document about attention mechanisms.',
                'metadata': {
                    'paper_title': 'Attention Is All You Need',
                    'section': 'Introduction',
                    'page_number': '1'
                },
                'similarity_score': 0.95
            },
            {
                'content': 'The transformer architecture uses self-attention.',
                'metadata': {
                    'paper_title': 'Attention Is All You Need',
                    'section': 'Model Architecture',
                    'page_number': '2'
                },
                'similarity_score': 0.85
            }
        ]
        
        # Mock OpenAI client
        self.patcher = patch('openai.OpenAI')
        self.mock_openai = self.patcher.start()
        
        # Mock completion response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="This is a test answer."))]
        mock_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        
        # Set up the mock OpenAI client
        self.mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        # Create RAG pipeline with mocks
        self.pipeline = RAGPipeline(
            vector_store=self.mock_vector_store,
            openai_api_key="test_key"
        )
        
        # Replace the OpenAI client with our mock
        self.pipeline.client = self.mock_openai.return_value
    
    def tearDown(self):
        """Clean up after tests"""
        self.patcher.stop()
    
    def test_preprocess_query(self):
        """Test query preprocessing"""
        # Test whitespace normalization
        self.assertEqual(
            self.pipeline.preprocess_query("  What   is  attention?  "),
            "What is attention?"
        )
        
        # Test filler word removal for longer queries
        self.assertEqual(
            self.pipeline.preprocess_query("What is the main innovation in the attention paper?"),
            "What main innovation attention paper?"
        )
    
    def test_retrieve_context(self):
        """Test context retrieval"""
        results = self.pipeline.retrieve_context("What is attention?")
        
        # Check that the vector store search was called
        self.mock_vector_store.search.assert_called_once()
        
        # Check that we got the expected results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['metadata']['paper_title'], 'Attention Is All You Need')
    
    def test_format_context(self):
        """Test context formatting"""
        mock_results = [
            {
                'content': 'This is document 1.',
                'metadata': {
                    'paper_title': 'Paper 1',
                    'section': 'Section 1',
                    'page_number': '1'
                },
                'similarity_score': 0.9
            },
            {
                'content': 'This is document 2.',
                'metadata': {
                    'paper_title': 'Paper 2',
                    'section': 'Section 2',
                    'page_number': '2'
                },
                'similarity_score': 0.8
            }
        ]
        
        context = self.pipeline.format_context(mock_results)
        
        # Check that both documents are included
        self.assertIn('Paper 1', context)
        self.assertIn('Paper 2', context)
        self.assertIn('This is document 1.', context)
        self.assertIn('This is document 2.', context)
    
    def test_generate_answer(self):
        """Test answer generation"""
        response = self.pipeline.generate_answer(
            "What is attention?",
            "Context about attention mechanisms."
        )
        
        # Check that OpenAI API was called
        self.pipeline.client.chat.completions.create.assert_called_once()
        
        # Check response format
        self.assertEqual(response['answer'], "This is a test answer.")
        self.assertTrue(response['success'])
        self.assertIn('token_usage', response)
        self.assertIn('total_tokens', response['token_usage'])
        self.assertIn('total_cost', response['token_usage'])
    
    def test_ask_question(self):
        """Test the complete pipeline"""
        response = self.pipeline.ask_question("What is attention?")
        
        # Check that the response contains all expected fields
        self.assertIn('answer', response)
        self.assertIn('question', response)
        self.assertIn('num_retrieved_chunks', response)
        self.assertIn('processing_time_seconds', response)
        self.assertIn('sources', response)
        
        # Check that sources were included
        self.assertEqual(len(response['sources']), 2)
        self.assertEqual(response['sources'][0]['paper_title'], 'Attention Is All You Need')
    
    def test_batch_questions(self):
        """Test batch question processing"""
        questions = [
            "What is attention?",
            "How does BERT work?"
        ]
        
        responses = self.pipeline.batch_questions(questions)
        
        # Check that we got responses for both questions
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]['question'], questions[0])
        self.assertEqual(responses[1]['question'], questions[1])
    
    def test_evaluate_benchmark_questions(self):
        """Test benchmark evaluation"""
        # Use a small set of test questions
        test_questions = ["What is attention?", "How does BERT work?"]
        
        results = self.pipeline.evaluate_benchmark_questions(test_questions)
        
        # Check that the results contain the expected metrics
        self.assertEqual(results['total_questions'], 2)
        self.assertEqual(results['successful_responses'], 2)
        self.assertEqual(results['success_rate'], 1.0)
        self.assertIn('total_tokens_used', results)
        self.assertIn('total_estimated_cost', results)
        self.assertIn('average_processing_time', results)
        self.assertIn('average_retrieved_chunks', results)
        self.assertEqual(len(results['responses']), 2)

if __name__ == '__main__':
    unittest.main() 