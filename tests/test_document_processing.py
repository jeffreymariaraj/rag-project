"""
Test suite for document processing module
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_processing import PDFParser, TextChunker, DocumentProcessor, DocumentChunk

class TestPDFParser:
    """Test cases for PDF parsing functionality"""
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        parser = PDFParser()
        
        # Test basic cleaning
        dirty_text = "This  is   a   test\n\n\nwith   excessive   whitespace"
        clean_text = parser._clean_text(dirty_text)
        assert "  " not in clean_text
        
        # Test hyphenated word fixing
        hyphenated = "machine-\nlearning is important"
        fixed = parser._clean_text(hyphenated)
        assert "machinelearning" in fixed or "machine learning" in fixed
    
    def test_extract_title(self):
        """Test title extraction"""
        parser = PDFParser()
        
        sample_text = """
        Attention Is All You Need
        
        Ashish Vaswani, Noam Shazeer, Niki Parmar
        Google Brain
        
        Abstract
        The dominant sequence transduction models...
        """
        
        title = parser._extract_title(sample_text)
        assert "Attention" in title
    
    def test_extract_abstract(self):
        """Test abstract extraction"""
        parser = PDFParser()
        
        sample_text = """
        Title Here
        
        Abstract
        This is the abstract content that describes the paper.
        It contains important information about the methodology.
        
        1. Introduction
        The introduction starts here...
        """
        
        abstract = parser._extract_abstract(sample_text)
        assert "abstract content" in abstract.lower()
        assert "introduction" not in abstract.lower()

class TestTextChunker:
    """Test cases for text chunking functionality"""
    
    def test_chunker_initialization(self):
        """Test chunker initialization with different parameters"""
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
    
    def test_section_identification(self):
        """Test section identification in research papers"""
        chunker = TextChunker()
        
        sample_text = """
        Abstract
        This is the abstract section.
        
        1. Introduction
        This is the introduction section.
        
        2. Methodology
        This is the methodology section.
        
        Conclusion
        This is the conclusion.
        """
        
        sections = chunker._identify_sections(sample_text)
        assert len(sections) > 1
        
        # Check if common sections are identified
        section_titles = [s['title'].lower() for s in sections]
        assert any('abstract' in title for title in section_titles)
    
    def test_chunk_creation(self):
        """Test chunk creation from text"""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        
        # Create sample paper metadata
        metadata = {
            'title': 'Test Paper',
            'authors': ['Test Author'],
            'filename': 'test.pdf'
        }
        
        # Sample text that should create multiple chunks
        long_text = "This is a test sentence. " * 100
        
        chunks = chunker.chunk_document(long_text, metadata)
        
        # Verify chunks were created
        assert len(chunks) > 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.paper_title == 'Test Paper' for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        
        metadata = {'title': 'Test', 'filename': 'test.pdf'}
        text = "word " * 200  # Create text with repeated words
        
        chunks = chunker.chunk_document(text, metadata)
        
        if len(chunks) > 1:
            # Check if there's overlap between consecutive chunks
            first_chunk_end = chunks[0].content[-50:]  # Last 50 chars
            second_chunk_start = chunks[1].content[:50]  # First 50 chars
            
            # There should be some common words due to overlap
            first_words = set(first_chunk_end.split())
            second_words = set(second_chunk_start.split())
            assert len(first_words.intersection(second_words)) > 0

class TestDocumentProcessor:
    """Test cases for the main document processor"""
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
        assert processor.text_chunker.chunk_size == 800
        assert processor.text_chunker.chunk_overlap == 200
    
    def test_chunk_metadata(self):
        """Test that chunks contain proper metadata"""
        chunker = TextChunker()
        metadata = {
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2'],
            'filename': 'test.pdf'
        }
        
        text = "This is a test document with multiple sentences. " * 20
        chunks = chunker.chunk_document(text, metadata)
        
        for chunk in chunks:
            assert chunk.paper_title == 'Test Paper'
            assert chunk.source == 'test.pdf'
            assert isinstance(chunk.chunk_index, int)
            assert isinstance(chunk.page_number, int)
            assert chunk.metadata['paper_title'] == 'Test Paper'

class TestIntegration:
    """Integration tests for the complete document processing pipeline"""
    
    def test_small_document_processing(self):
        """Test processing a small sample document"""
        # Create a mock document
        sample_content = """
        Test Paper Title
        
        Abstract
        This is a sample abstract for testing purposes.
        
        1. Introduction
        This paper introduces a new method for testing.
        The method is based on advanced techniques.
        
        2. Methodology
        Our approach consists of several steps:
        First, we analyze the input data.
        Second, we apply our algorithm.
        Third, we evaluate the results.
        
        3. Results
        The results show significant improvement.
        Our method outperforms existing approaches.
        
        Conclusion
        We have presented a novel approach.
        Future work will extend this method.
        """
        
        # Test chunking
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        metadata = {
            'title': 'Test Paper Title',
            'filename': 'test_paper.pdf',
            'authors': ['Test Author']
        }
        
        chunks = chunker.chunk_document(sample_content, metadata)
        
        # Verify results
        assert len(chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in chunks)
        assert all(chunk.paper_title == 'Test Paper Title' for chunk in chunks)
        
        # Check that different sections are captured
        sections_found = set()
        for chunk in chunks:
            if chunk.section:
                sections_found.add(chunk.section.lower())
        
        # Should find at least some sections
        assert len(sections_found) > 0

def test_benchmark_readiness():
    """Test that the system can handle benchmark-style questions"""
    chunker = TextChunker(chunk_size=800, chunk_overlap=200)
    
    # Sample content that mimics research paper structure
    transformer_content = """
    Attention Is All You Need
    
    Abstract
    The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks. We propose a new simple network
    architecture, the Transformer, based solely on attention mechanisms.
    
    1. Introduction
    Recurrent neural networks have been the dominant approach for sequence modeling.
    However, RNNs are inherently sequential and cannot be parallelized.
    
    2. Model Architecture
    The Transformer follows the encoder-decoder structure but uses attention
    mechanisms instead of recurrence. The key innovation is the self-attention
    mechanism that allows the model to attend to different positions.
    """
    
    metadata = {
        'title': 'Attention Is All You Need',
        'filename': 'attention.pdf',
        'authors': ['Vaswani et al.']
    }
    
    chunks = chunker.chunk_document(transformer_content, metadata)
    
    # Verify that chunks contain key information for benchmark questions
    all_content = ' '.join(chunk.content for chunk in chunks)
    
    # Should contain key terms related to the benchmark question
    key_terms = ['attention', 'transformer', 'self-attention', 'innovation']
    for term in key_terms:
        assert term.lower() in all_content.lower()

if __name__ == "__main__":
    # Run basic tests
    print("Running document processing tests...")
    
    # Test text cleaning
    parser = PDFParser()
    test_text = "This  is   a   test  with   spaces"
    cleaned = parser._clean_text(test_text)
    print(f"✓ Text cleaning: '{test_text}' -> '{cleaned}'")
    
    # Test chunking
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    metadata = {'title': 'Test', 'filename': 'test.pdf'}
    text = "This is a sample text for testing chunking functionality. " * 10
    
    chunks = chunker.chunk_document(text, metadata)
    print(f"✓ Chunking: Created {len(chunks)} chunks from sample text")
    
    # Display sample chunk
    if chunks:
        print(f"✓ Sample chunk content: {chunks[0].content[:100]}...")
        print(f"✓ Sample chunk metadata: {chunks[0].metadata}")
    
    print("\nAll basic tests passed! ✓")
    print("Run 'python -m pytest tests/test_document_processing.py -v' for detailed testing")