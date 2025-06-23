"""
Document Processing Module for RAG System
Handles PDF parsing, text extraction, and chunking strategies
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import nltk
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    content: str
    metadata: Dict
    page_number: int
    chunk_index: int
    source: str
    paper_title: str
    section: Optional[str] = None

class PDFParser:
    """
    PDF Parser class that handles extraction from research papers
    Supports both pdfplumber and PyPDF2 for robust extraction
    """
    
    def __init__(self, fallback_parser: bool = True):
        self.fallback_parser = fallback_parser
        
    def extract_text_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text using pdfplumber (primary method)"""
        try:
            full_text = ""
            page_texts = []
            metadata = {
                'total_pages': 0,
                'extraction_method': 'pdfplumber',
                'file_name': Path(pdf_path).name
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata['total_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean and normalize text
                            page_text = self._clean_text(page_text)
                            page_texts.append({
                                'page': page_num,
                                'text': page_text
                            })
                            full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        
            metadata['page_texts'] = page_texts
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            if self.fallback_parser:
                return self.extract_text_pypdf2(pdf_path)
            raise
    
    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            full_text = ""
            page_texts = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                metadata = {
                    'total_pages': total_pages,
                    'extraction_method': 'PyPDF2',
                    'file_name': Path(pdf_path).name
                }
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_text = self._clean_text(page_text)
                            page_texts.append({
                                'page': page_num,
                                'text': page_text
                            })
                            full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        
            metadata['page_texts'] = page_texts
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        # Normalize quotes and dashes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        text = re.sub(r'[–—]', '-', text)
        
        return text.strip()
    
    def extract_paper_metadata(self, text: str, filename: str) -> Dict:
        """Extract paper-specific metadata like title, authors, abstract"""
        metadata = {
            'filename': filename,
            'title': self._extract_title(text),
            'abstract': self._extract_abstract(text),
            'authors': self._extract_authors(text)
        }
        return metadata
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title from text"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            # Skip if line is too short or too long
            if 10 < len(line) < 200:
                # Check if it looks like a title (capitalized, no periods at end)
                if line[0].isupper() and not line.endswith('.'):
                    return line
        
        # Fallback: use filename
        return Path(filename).stem.replace('_', ' ').title()
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from paper"""
        abstract_pattern = r'(?i)abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:1\.|introduction|keywords|\d+\s+introduction))'
        match = re.search(abstract_pattern, text, re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:1000]  # Limit length
        
        return ""
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors from paper (simple heuristic)"""
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for patterns that might indicate authors
            if any(indicator in line.lower() for indicator in ['university', 'institute', '@', 'department']):
                # Previous line might be authors
                if i > 0:
                    potential_authors = lines[i-1].strip()
                    if potential_authors and len(potential_authors) < 200:
                        # Split by common delimiters
                        authors = re.split(r'[,;&]', potential_authors)
                        return [author.strip() for author in authors if author.strip()]
        
        return []

class TextChunker:
    """
    Advanced text chunking strategy optimized for research papers
    Preserves semantic coherence and paper structure
    """
    
    def __init__(self, 
                 chunk_size: int = 800,
                 chunk_overlap: int = 200,
                 separator_priority: List[str] = None):
        """
        Initialize chunker with configurable parameters
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between consecutive chunks
            separator_priority: Custom separator priority for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separator priority optimized for research papers
        if separator_priority is None:
            self.separator_priority = [
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                ", ",        # Clause breaks
                " ",         # Word breaks
                ""           # Character breaks (last resort)
            ]
        else:
            self.separator_priority = separator_priority
            
        # Initialize LangChain's recursive splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separator_priority,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_document(self, text: str, paper_metadata: Dict) -> List[DocumentChunk]:
        """
        Chunk a document into semantically coherent pieces
        
        Args:
            text: Full document text
            paper_metadata: Metadata about the paper
            
        Returns:
            List of DocumentChunk objects
        """
        # Preprocess text to identify sections
        sections = self._identify_sections(text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._chunk_section(
                section['content'], 
                section['title'], 
                paper_metadata,
                chunk_index
            )
            
            for chunk in section_chunks:
                chunk.section = section['title']
                chunks.append(chunk)
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks from {paper_metadata.get('title', 'document')}")
        return chunks
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """Identify sections in research paper"""
        sections = []
        
        # Common section patterns in research papers
        section_patterns = [
            r'^\s*(?:abstract|introduction|related work|methodology|method|approach|experiments?|results?|discussion|conclusion|references?|acknowledgments?)\s*$',
            r'^\s*\d+\.?\s+([A-Z][^.]*?)(?=\n)',
            r'^\s*[A-Z][A-Z\s]{2,50}(?=\n)'
        ]
        
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': '', 'start_line': 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line matches section pattern
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_section_header = True
                    break
            
            if is_section_header and len(current_section['content']) > 100:
                # Save current section
                sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': line_stripped[:50],  # Limit title length
                    'content': '',
                    'start_line': i
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content']:
            sections.append(current_section)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{'title': 'Full Document', 'content': text, 'start_line': 0}]
        
        return sections
    
    def _chunk_section(self, section_text: str, section_title: str, 
                      paper_metadata: Dict, start_chunk_index: int) -> List[DocumentChunk]:
        """Chunk a specific section of the document"""
        
        # Use LangChain's splitter for initial chunking
        raw_chunks = self.splitter.split_text(section_text)
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
                
            # Create enhanced metadata
            metadata = {
                'paper_title': paper_metadata.get('title', ''),
                'authors': paper_metadata.get('authors', []),
                'filename': paper_metadata.get('filename', ''),
                'section': section_title,
                'chunk_index': start_chunk_index + i,
                'chunk_length': len(chunk_text),
                'chunk_overlap': self.chunk_overlap if i > 0 else 0
            }
            
            # Estimate page number (rough approximation)
            estimated_page = (start_chunk_index + i) // 5 + 1  # Assume ~5 chunks per page
            
            chunk = DocumentChunk(
                content=chunk_text.strip(),
                metadata=metadata,
                page_number=estimated_page,
                chunk_index=start_chunk_index + i,
                source=paper_metadata.get('filename', ''),
                paper_title=paper_metadata.get('title', ''),
                section=section_title
            )
            
            chunks.append(chunk)
        
        return chunks

class DocumentProcessor:
    """Main document processor that orchestrates PDF parsing and chunking"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text, extraction_metadata = self.pdf_parser.extract_text_pdfplumber(pdf_path)
        
        # Extract paper metadata
        paper_metadata = self.pdf_parser.extract_paper_metadata(text, pdf_path)
        paper_metadata.update(extraction_metadata)
        
        # Chunk the document
        chunks = self.text_chunker.chunk_document(text, paper_metadata)
        
        logger.info(f"Successfully processed {pdf_path}: {len(chunks)} chunks created")
        return chunks
    
    def process_directory(self, directory_path: str) -> List[DocumentChunk]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all DocumentChunk objects from all PDFs
        """
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory_path}")
        
        all_chunks = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} PDFs, created {len(all_chunks)} total chunks")
        return all_chunks

# Example usage and testing
if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
    
    # Process a single PDF (replace with actual path)
    # chunks = processor.process_pdf("path/to/your/paper.pdf")
    
    # Process all PDFs in a directory
    # chunks = processor.process_directory("data/pdfs/")
    
    # Print sample chunk information
    # for i, chunk in enumerate(chunks[:3]):
    #     print(f"\n--- Chunk {i+1} ---")
    #     print(f"Paper: {chunk.paper_title}")
    #     print(f"Section: {chunk.section}")
    #     print(f"Content: {chunk.content[:200]}...")
    #     print(f"Metadata: {chunk.metadata}")
    
    print("Document processing module loaded successfully!")