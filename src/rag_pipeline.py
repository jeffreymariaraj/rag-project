"""
RAG Pipeline Module
Handles query processing, context retrieval, and answer generation
"""

import os
import time
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# OpenAI imports
import openai
from openai import OpenAI

# Local imports - handle missing dependencies gracefully
try:
    from src.vector_store import VectorStoreManager
    from src.prompt_templates import PromptTemplates
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    try:
        from vector_store import VectorStoreManager
        from prompt_templates import PromptTemplates
        VECTOR_STORE_AVAILABLE = True
    except ImportError:
        VECTOR_STORE_AVAILABLE = False
        VectorStoreManager = None
        # Define a simple PromptTemplates class if it's not available
        class PromptTemplates:
            def __init__(self):
                self.templates = {
                    'default': {
                        'system_prompt': "You are a helpful AI assistant.",
                        'user_prompt': "{query}"
                    }
                }
            
            def format_prompt(self, template_id, context, query, topic=None):
                return {
                    'system_prompt': self.templates['default']['system_prompt'],
                    'user_prompt': query
                }
        
        print("Warning: Vector store dependencies not available. Some features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

class RAGPipeline:
    """
    RAG Pipeline for answering questions using vector store and LLM
    """
    
    def __init__(self, 
                 vector_store=None,
                 vector_store_path: str = None,
                 openai_api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 top_k: int = 5,
                 similarity_threshold: float = 0.7,
                 prompt_template_id: str = "default"):
        """
        Initialize the RAG Pipeline
        
        Args:
            vector_store: Initialized vector store (optional)
            vector_store_path: Path to vector store (if vector_store not provided)
            openai_api_key: OpenAI API key (defaults to env var)
            model_name: LLM model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM response
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieved documents
            prompt_template_id: ID of the prompt template to use
        """
        # Initialize vector store
        if vector_store is None and vector_store_path is None:
            vector_store_path = os.getenv("VECTOR_DB_PATH", "vector_db")
            
        if vector_store is None:
            logger.info(f"Initializing vector store from {vector_store_path}")
            store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.vector_store = VectorStoreManager(
                store_type=store_type,
                embedding_model=embedding_model,
                persist_directory=vector_store_path
            )
        else:
            self.vector_store = vector_store
        
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # LLM configuration
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "1000"))
        
        # Retrieval configuration
        self.top_k = top_k or int(os.getenv("TOP_K_RESULTS", "5"))
        self.similarity_threshold = similarity_threshold or float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates()
        self.prompt_template_id = prompt_template_id
        
        logger.info(f"RAG Pipeline initialized with model {self.model_name}")
        logger.info(f"Retrieval settings: top_k={self.top_k}, threshold={self.similarity_threshold}")
        logger.info(f"Using prompt template: {self.prompt_template_id}")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better retrieval
        
        Args:
            query: User query
            
        Returns:
            Preprocessed query
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove common filler words for better semantic matching
        filler_words = ['the', 'a', 'an', 'in', 'on', 'at', 'is', 'are', 'was', 'were']
        query_words = query.split()
        if len(query_words) > 5:  # Only remove if query is long enough
            query_words = [w for w in query_words if w.lower() not in filler_words]
            query = ' '.join(query_words)
        
        return query
    
    def retrieve_context(self, query: str, top_k: int = None, 
                        similarity_threshold: float = None) -> List[Dict]:
        """
        Retrieve relevant context from vector store
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents with metadata
        """
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Retrieve documents
        logger.info(f"Retrieving context for query: '{processed_query}'")
        results = self.vector_store.search(
            query=processed_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def format_context(self, results: List[Dict], max_tokens: int = 3500) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            results: Retrieved documents
            max_tokens: Maximum context tokens
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        # Sort by similarity score
        sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough approximation of tokens to chars
        
        for i, doc in enumerate(sorted_results):
            # Format document with metadata
            paper_title = doc['metadata'].get('paper_title', 'Unknown Paper')
            section = doc['metadata'].get('section', 'Unknown Section')
            page = doc['metadata'].get('page_number', 'Unknown Page')
            
            doc_text = f"[Document {i+1}] From '{paper_title}', Section: {section}, Page: {page}\n{doc['content']}\n"
            
            # Check if adding this document would exceed the limit
            if total_chars + len(doc_text) > char_limit:
                # If we have at least one document, stop
                if context_parts:
                    break
                # Otherwise, truncate the first document
                doc_text = doc_text[:char_limit]
                
            context_parts.append(doc_text)
            total_chars += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, prompt_template_id: str = None, topic: str = None) -> Dict:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            context: Retrieved context
            prompt_template_id: ID of the prompt template to use (optional)
            topic: Optional topic for templates that require it
            
        Returns:
            Dictionary with answer and metadata
        """
        template_id = prompt_template_id or self.prompt_template_id
        logger.info(f"Generating answer with {self.model_name} using template: {template_id}")
        
        # Format prompt using template
        formatted_prompt = self.prompt_templates.format_prompt(
            template_id=template_id,
            context=context,
            query=query,
            topic=topic
        )
        
        try:
            # Call OpenAI API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": formatted_prompt['system_prompt']},
                    {"role": "user", "content": formatted_prompt['user_prompt']}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            processing_time = time.time() - start_time
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Calculate token usage and cost
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            
            # Estimate cost (approximate rates)
            if self.model_name.startswith('gpt-4'):
                prompt_cost_per_1k = 0.03
                completion_cost_per_1k = 0.06
            else:  # gpt-3.5-turbo
                prompt_cost_per_1k = 0.0015
                completion_cost_per_1k = 0.002
                
            token_usage['prompt_cost'] = (token_usage['prompt_tokens'] / 1000) * prompt_cost_per_1k
            token_usage['completion_cost'] = (token_usage['completion_tokens'] / 1000) * completion_cost_per_1k
            token_usage['total_cost'] = token_usage['prompt_cost'] + token_usage['completion_cost']
            
            logger.info(f"Answer generated in {processing_time:.2f}s, {token_usage['total_tokens']} tokens")
            
            return {
                'answer': answer,
                'success': True,
                'processing_time_seconds': processing_time,
                'token_usage': token_usage,
                'model': self.model_name,
                'prompt_template': template_id
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def ask_question(self, query: str, n_results: int = None, 
                   prompt_template_id: str = None, topic: str = None) -> Dict:
        """
        Complete RAG pipeline: retrieve context and generate answer
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            prompt_template_id: ID of the prompt template to use (optional)
            topic: Optional topic for templates that require it
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Retrieve context
        results = self.retrieve_context(query, top_k=n_results or self.top_k)
        
        # Format context
        context = self.format_context(results)
        
        # Generate answer
        response = self.generate_answer(
            query=query, 
            context=context,
            prompt_template_id=prompt_template_id,
            topic=topic
        )
        
        # Add metadata
        response['question'] = query
        response['num_retrieved_chunks'] = len(results)
        response['processing_time_seconds'] = time.time() - start_time
        
        # Add sources
        sources = []
        for result in results:
            source = {
                'paper_title': result['metadata'].get('paper_title', 'Unknown'),
                'section': result['metadata'].get('section', 'Unknown'),
                'page': result['metadata'].get('page_number', 'Unknown'),
                'similarity_score': result['similarity_score']
            }
            sources.append(source)
        
        response['sources'] = sources
        
        # Log the response
        self._log_response(query, response)
        
        return response
    
    def batch_questions(self, questions: List[str], prompt_template_id: str = None) -> List[Dict]:
        """
        Process a batch of questions
        
        Args:
            questions: List of questions
            prompt_template_id: ID of the prompt template to use (optional)
            
        Returns:
            List of responses
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        responses = []
        for question in questions:
            try:
                response = self.ask_question(question, prompt_template_id=prompt_template_id)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                responses.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'success': False,
                    'error': str(e)
                })
        
        return responses
    
    def evaluate_benchmark_questions(self, questions: List[str] = None, 
                                   prompt_template_id: str = None) -> Dict:
        """
        Evaluate performance on benchmark questions
        
        Args:
            questions: List of benchmark questions (optional)
            prompt_template_id: ID of the prompt template to use (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if questions is None:
            # Default benchmark questions
            questions = [
                "What is the main innovation in 'Attention is All You Need'?",
                "How does BERT differ from traditional language models?",
                "Describe GPT-3's few-shot learning capabilities with an example.",
                "What is CLIP's loss function and how does it work?",
                "What approach does LLaMA use to reduce training costs?"
            ]
        
        logger.info(f"Evaluating {len(questions)} benchmark questions")
        
        # Process all questions
        responses = self.batch_questions(questions, prompt_template_id=prompt_template_id)
        
        # Calculate metrics
        successful_responses = sum(1 for r in responses if r['success'])
        total_tokens = sum(r['token_usage']['total_tokens'] if r.get('token_usage') else 0 for r in responses)
        total_cost = sum(r['token_usage']['total_cost'] if r.get('token_usage') else 0 for r in responses)
        avg_processing_time = sum(r['processing_time_seconds'] for r in responses) / len(responses)
        avg_retrieved_chunks = sum(r['num_retrieved_chunks'] for r in responses) / len(responses)
        
        # Compile results
        evaluation_results = {
            'total_questions': len(questions),
            'successful_responses': successful_responses,
            'success_rate': successful_responses / len(questions),
            'total_tokens_used': total_tokens,
            'total_estimated_cost': total_cost,
            'average_processing_time': avg_processing_time,
            'average_retrieved_chunks': avg_retrieved_chunks,
            'responses': responses,
            'timestamp': datetime.now().isoformat(),
            'prompt_template': prompt_template_id or self.prompt_template_id
        }
        
        # Save results to file
        self._save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _log_response(self, query: str, response: Dict) -> None:
        """
        Log response details
        
        Args:
            query: User query
            response: Response dictionary
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'success': response['success'],
            'num_chunks': response['num_retrieved_chunks'],
            'processing_time': response['processing_time_seconds'],
            'model': response.get('model', self.model_name),
            'prompt_template': response.get('prompt_template', self.prompt_template_id)
        }
        
        if 'token_usage' in response:
            log_data['tokens'] = response['token_usage']['total_tokens']
            log_data['estimated_cost'] = response['token_usage']['total_cost']
        
        logger.info(f"Query: '{query[:50]}...' | "
                   f"Chunks: {log_data['num_chunks']} | "
                   f"Time: {log_data['processing_time']:.2f}s | "
                   f"Success: {log_data['success']}")
    
    def _save_evaluation_results(self, results: Dict) -> None:
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
        """
        # Create directory if it doesn't exist
        Path("logs/evaluations").mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/evaluations/benchmark_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filename}")
