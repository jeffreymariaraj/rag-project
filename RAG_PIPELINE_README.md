# RAG Pipeline Implementation

This document provides detailed information about the Retrieval-Augmented Generation (RAG) pipeline implementation for querying machine learning research papers.

## Overview

The RAG pipeline combines vector search with large language models to provide accurate answers to questions about machine learning research papers. The system processes and indexes PDF documents, retrieves relevant context based on semantic similarity, and generates answers using OpenAI's language models.

## Components

### 1. Document Processing (`document_processing.py`)

- **PDF Parsing**: Extracts text from research papers using dual-parser system
- **Chunking**: Creates semantically coherent chunks with metadata
- **Metadata Enrichment**: Adds paper title, section, page number information

### 2. Vector Store (`vector_store.py`)

- **Embedding Generation**: Converts text chunks to vector embeddings
- **Vector Database**: Stores and indexes embeddings for fast retrieval
- **Similarity Search**: Retrieves relevant chunks based on query similarity

### 3. RAG Pipeline (`rag_pipeline.py`)

- **Query Processing**: Preprocesses and optimizes user queries
- **Context Retrieval**: Fetches relevant context from vector store
- **Answer Generation**: Uses OpenAI models to generate answers
- **Evaluation**: Provides metrics for system performance

### 4. Prompt Templates (`prompt_templates.py`)

- **Modular Templates**: Different prompt styles for various query types
- **Template Management**: Selection and customization of prompts
- **Context Formatting**: Optimized context presentation for LLMs

### 5. Web Interface (`web_interface.py`)

- **User Interface**: Clean, responsive Streamlit interface
- **Query History**: Session-based tracking of past queries
- **Context Navigation**: Tools to explore retrieved context
- **Settings Management**: Customization of model parameters
- **Error Handling**: Robust validation and error recovery

## Usage

### Environment Setup

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
VECTOR_DB_PATH=vector_db
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Running the Web Interface

```bash
python run_web_interface.py
```

This will start the Streamlit web interface, accessible at http://localhost:8501.

### Using the RAG Pipeline Programmatically

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    vector_store_path="vector_db",
    model_name="gpt-3.5-turbo"
)

# Ask a question
response = pipeline.ask_question(
    "What is the main innovation in the Attention is All You Need paper?",
    prompt_template_id="detailed"
)

# Print the answer
print(response['answer'])

# Print sources
for source in response['sources']:
    print(f"- {source['paper_title']}, {source['section']}, Page {source['page']}")
```

### Testing

Run the test script to validate the RAG system:

```bash
python test_rag_system.py
```

## Web Interface Features

### Phase A: Basic Interface
- Clean UI with text input and submit button
- Example questions users can click
- Answer display in a styled box
- Context display with expandable sections
- Metadata showing processing time and statistics

### Phase B: Session Tracking & Navigation
- Session state management for query history
- Sidebar for navigation through past queries
- Controls for navigating through context documents
- Settings panel for model configuration

### Phase C: Modular Prompt & Integration
- Template selection interface
- Multiple prompt templates for different query types
- Topic specification for specialized templates
- Template preview functionality

### Phase D: Validation & Error Handling
- Input validation for queries
- Robust error handling for API failures
- Environment validation on startup
- Detailed logging for troubleshooting
- Debug information panel

## Prompt Templates

The system includes several prompt templates:

1. **Default**: Standard RAG prompt for general questions
2. **Detailed with Citations**: Explicit citation instructions
3. **Multi-Document Synthesis**: For comparing information across papers
4. **Technical Explanation**: For detailed technical questions with equations

## Performance Considerations

- **API Usage**: The system uses OpenAI's API, which incurs costs
- **Vector Database Size**: Performance may degrade with very large document collections
- **Query Complexity**: Complex queries may require more tokens and processing time

## Troubleshooting

Common issues and solutions:

1. **OpenAI API Errors**: Check your API key and rate limits
2. **Vector Database Not Found**: Ensure the vector database has been built
3. **Slow Performance**: Adjust the number of retrieved chunks (top_k)
4. **Poor Answer Quality**: Try different prompt templates or adjust temperature

## Future Improvements

Potential enhancements for the system:

1. **Local LLM Support**: Add support for local language models
2. **Advanced Retrieval**: Implement hybrid search or re-ranking
3. **User Feedback Loop**: Incorporate user feedback for answer quality
4. **Custom Templates**: Allow users to create and save custom prompt templates
5. **Document Upload**: Enable users to upload their own research papers
