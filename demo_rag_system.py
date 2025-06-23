#!/usr/bin/env python3
"""
Complete demo script for the RAG system.
Tests both vector store and RAG pipeline functionality.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List

# Add src directory to path
sys.path.append('src')

from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystemDemo:
    """
    Comprehensive demo and testing class for the RAG system.
    """
    
    def __init__(self, 
                 vector_store_dir: str = "vector_db",
                 openai_api_key: str = None):
        """
        Initialize the demo system.
        
        Args:
            vector_store_dir: Directory containing the vector database
            openai_api_key: OpenAI API key
        """
        self.vector_store_dir = vector_store_dir
        
        # Get API key from environment if not provided
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Initialize components
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore(persist_directory=vector_store_dir)
        
        logger.info("Initializing RAG pipeline...")
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Benchmark questions from the project requirements
        self.benchmark_questions = [
            "What is the main innovation in 'Attention is All You Need'?",
            "How does BERT differ from traditional language models?",
            "Describe GPT-3's few-shot learning capabilities with an example.",
            "What is CLIP's loss function and how does it work?",
            "What approach does LLaMA use to reduce training costs?"
        ]
        
        # Additional test questions for comprehensive evaluation
        self.additional_questions = [
            "Compare the attention mechanisms used in BERT and GPT-3.",
            "What are the key architectural differences between transformers and RNNs?",
            "How does contrastive learning work in CLIP?",
            "What are the computational advantages of LLaMA over GPT-3?",
            "Explain the concept of self-attention in transformers.",
            "What is the difference between encoder-only and decoder-only models?",
            "How does BERT handle bidirectional context?",
            "What training techniques does GPT-3 use for few-shot learning?",
            "Describe the vision transformer architecture used in CLIP.",
            "What efficiency improvements does LLaMA implement?"
        ]
    
    def check_system_status(self) -> Dict:
        """
        Check the status of all system components.
        
        Returns:
            Dictionary with system status information
        """
        logger.info("Checking system status...")
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'vector_store_status': {},
            'rag_pipeline_status': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Check vector store
            vs_stats = self.vector_store.get_collection_stats()
            status['vector_store_status'] = {
                'initialized': True,
                'document_count': vs_stats['total_documents'],
                'collection_name': vs_stats['collection_name'],
                'embedding_model': vs_stats['embedding_model'],
                'persist_directory': vs_stats['persist_directory']
            }
            
            # Quick test of vector store
            test_results = self.vector_store.similarity_search("test query", n_results=1)
            status['vector_store_status']['search_functional'] = len(test_results) > 0
            
        except Exception as e:
            status['vector_store_status'] = {
                'initialized': False,
                'error': str(e)
            }
        
        try:
            # Check RAG pipeline with a simple question
            test_response = self.rag_pipeline.ask_question(
                "What is machine learning?", 
                n_results=2
            )
            status['rag_pipeline_status'] = {
                'initialized': True,
                'llm_functional': test_response['success'],
                'test_processing_time': test_response['processing_time_seconds'],
                'test_retrieved_chunks': test_response['num_retrieved_chunks']
            }
            
        except Exception as e:
            status['rag_pipeline_status'] = {
                'initialized': False,
                'error': str(e)
            }
        
        # Determine overall status
        vs_ok = status['vector_store_status'].get('initialized', False)
        rag_ok = status['rag_pipeline_status'].get('initialized', False)
        
        if vs_ok and rag_ok:
            status['overall_status'] = 'ready'
        elif vs_ok or rag_ok:
            status['overall_status'] = 'partial'
        else:
            status['overall_status'] = 'error'
        
        return status
    
    def run_benchmark_evaluation(self) -> Dict:
        """
        Run the official benchmark questions.
        
        Returns:
            Benchmark evaluation results
        """
        logger.info("Running benchmark evaluation...")
        
        print("\n" + "="*80)
        print("BENCHMARK EVALUATION - OFFICIAL QUESTIONS")
        print("="*80)
        
        results = self.rag_pipeline.evaluate_benchmark_questions()
        
        # Display results
        self._display_evaluation_results(results, "BENCHMARK")
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation with additional questions.
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Running comprehensive evaluation...")
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION - ADDITIONAL QUESTIONS")
        print("="*80)
        
        responses = self.rag_pipeline.batch_questions(self.additional_questions)
        
        # Calculate metrics similar to benchmark evaluation
        successful_responses = sum(1 for r in responses if r['success'])
        total_tokens = sum(r['token_usage']['total_tokens'] if r['token_usage'] else 0 for r in responses)
        total_cost = sum(r['token_usage']['total_cost'] if r['token_usage'] else 0 for r in responses)
        avg_processing_time = sum(r['processing_time_seconds'] for r in responses) / len(responses)
        avg_retrieved_chunks = sum(r['num_retrieved_chunks'] for r in responses) / len(responses)
        
        results = {
            'total_questions': len(self.additional_questions),
            'successful_responses': successful_responses,
            'success_rate': successful_responses / len(self.additional_questions),
            'total_tokens_used': total_tokens,
            'total_estimated_cost': total_cost,
            'average_processing_time': avg_processing_time,
            'average_retrieved_chunks': avg_retrieved_chunks,
            'responses': responses,
            'timestamp': datetime.now().isoformat()
        }
        
        # Display results
        self._display_evaluation_results(results, "COMPREHENSIVE")
        
        return results
    
    def _display_evaluation_results(self, results: Dict, evaluation_type: str) -> None:
        """
        Display evaluation results in a formatted way.
        
        Args:
            results: Evaluation results dictionary
            evaluation_type: Type of evaluation (BENCHMARK/COMPREHENSIVE)
        """
        print(f"\n{evaluation_type} EVALUATION SUMMARY:")
        print("-" * 50)
        print(f"Total Questions: {results['total_questions']}")
        print(f"Successful Responses: {results['successful_responses']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Total Tokens Used: {results['total_tokens_used']:,}")
        print(f"Estimated Cost: ${results['total_estimated_cost']:.4f}")
        print(f"Avg Processing Time: {results['average_processing_time']:.2f} seconds")
        print(f"Avg Retrieved Chunks: {results['average_retrieved_chunks']:.1f}")
        
        print(f"\nSAMPLE RESPONSES:")
        print("-" * 50)
        
        # Show first 3 responses as examples
        for i, response in enumerate(results['responses'][:3], 1):
            print(f"\n{i}. Question: {response['question']}")
            print(f"   Answer: {response['answer'][:200]}{'...' if len(response['answer']) > 200 else ''}")
            print(f"   Retrieved: {response['num_retrieved_chunks']} chunks, "
                  f"Time: {response['processing_time_seconds']:.2f}s")
    
    def interactive_demo(self) -> None:
        """
        Run an interactive demo where users can ask questions.
        """
        print("\n" + "="*80)
        print("INTERACTIVE RAG SYSTEM DEMO")
        print("="*80)
        print("Ask questions about the research papers!")
        print("Type 'quit' or 'exit' to stop, 'help' for sample questions.")
        print("-" * 80)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_sample_questions()
                    continue
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                print(f"\nProcessing: {question}")
                print("Please wait...")
                
                # Get response from RAG pipeline
                response = self.rag_pipeline.ask_question(question)
                
                # Display response
                print(f"\nAnswer:")
                print("-" * 40)
                print(response['answer'])
                
                print(f"\nMetadata:")
                print(f"  Retrieved chunks: {response['num_retrieved_chunks']}")
                print(f"  Processing time: {response['processing_time_seconds']:.2f} seconds")
                if response['token_usage']:
                    print(f"  Tokens used: {response['token_usage']['total_tokens']}")
                
                # Show retrieved sources
                if response['retrieval_results']:
                    print(f"\nSources:")
                    for i, result in enumerate(response['retrieval_results'][:3], 1):
                        paper = result['metadata'].get('paper_title', 'Unknown')
                        score = result['similarity_score']
                        print(f"  {i}. {paper} (similarity: {score:.3f})")
                
            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Error in interactive demo: {e}")
    
    def _show_sample_questions(self) -> None:
        """Show sample questions users can ask."""
        print("\nSample questions you can ask:")
        print("-" * 40)
        
        sample_questions = [
            "What is the transformer architecture?",
            "How does BERT work?",
            "What is few-shot learning in GPT-3?",
            "Explain attention mechanisms",
            "What is CLIP used for?",
            "How is LLaMA different from GPT-3?",
            "What are the advantages of transformers over RNNs?",
            "How does self-attention work?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. {question}")
    
    def save_results(self, 
                    benchmark_results: Dict, 
                    comprehensive_results: Dict, 
                    system_status: Dict) -> str:
        """
        Save all evaluation results to a JSON file.
        
        Args:
            benchmark_results: Benchmark evaluation results
            comprehensive_results: Comprehensive evaluation results
            system_status: System status information
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/demo_results_{timestamp}.json"
        
        os.makedirs("logs", exist_ok=True)
        
        combined_results = {
            'demo_timestamp': datetime.now().isoformat(),
            'system_status': system_status,
            'benchmark_evaluation': benchmark_results,
            'comprehensive_evaluation': comprehensive_results,
            'summary': {
                'total_questions_tested': (
                    benchmark_results['total_questions'] + 
                    comprehensive_results['total_questions']
                ),
                'overall_success_rate': (
                    (benchmark_results['successful_responses'] + 
                     comprehensive_results['successful_responses']) /
                    (benchmark_results['total_questions'] + 
                     comprehensive_results['total_questions'])
                ),
                'total_cost': (
                    benchmark_results['total_estimated_cost'] + 
                    comprehensive_results['total_estimated_cost']
                )
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
        return results_file


def main():
    """Main demo function."""
    print("RAG SYSTEM COMPREHENSIVE DEMO")
    print("="*50)
    
    try:
        # Initialize demo system
        demo = RAGSystemDemo()
        
        # Check system status
        print("\n1. SYSTEM STATUS CHECK")
        print("-" * 30)
        status = demo.check_system_status()
        
        print(f"Overall Status: {status['overall_status'].upper()}")
        print(f"Vector Store: {status['vector_store_status'].get('document_count', 0)} documents")
        print(f"RAG Pipeline: {'✓' if status['rag_pipeline_status'].get('initialized') else '✗'}")
        
        if status['overall_status'] != 'ready':
            print("⚠️  System not fully ready. Some components may not work properly.")
            user_input = input("Continue anyway? (y/N): ").lower()
            if user_input != 'y':
                return
        
        # Run evaluations
        print("\n2. RUNNING EVALUATIONS")
        print("-" * 30)
        
        # Benchmark evaluation
        benchmark_results = demo.run_benchmark_evaluation()
        
        # Comprehensive evaluation
        comprehensive_results = demo.run_comprehensive_evaluation()
        
        # Save results
        results_file = demo.save_results(benchmark_results, comprehensive_results, status)
        print(f"\n📊 All results saved to: {results_file}")
        
        # Interactive demo
        print("\n3. INTERACTIVE DEMO")
        print("-" * 30)
        user_input = input("Start interactive demo? (Y/n): ").lower()
        if user_input != 'n':
            demo.interactive_demo()
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()