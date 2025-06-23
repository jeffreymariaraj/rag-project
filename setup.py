"""
Setup script for RAG ML Papers project
Creates directory structure and downloads papers from arXiv
"""

import os
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paper URLs from arXiv
PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert_pretraining.pdf": "https://arxiv.org/pdf/1810.04805.pdf", 
    "gpt3_language_models.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "clip_knowledge_graphs.pdf": "https://arxiv.org/pdf/2104.13478.pdf",
    "llama_foundation_models.pdf": "https://arxiv.org/pdf/2302.13971.pdf"
}

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        "data/pdfs",
        "src",
        "vector_db",
        "logs",
        "screenshots",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_papers():
    """Download research papers from arXiv"""
    pdf_dir = Path("data/pdfs")
    
    for filename, url in PAPERS.items():
        filepath = pdf_dir / filename
        
        if filepath.exists():
            logger.info(f"Paper already exists: {filename}")
            continue
            
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Successfully downloaded: {filename}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            continue

def create_env_template():
    """Create .env template file"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Parameters
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# Vector Database
VECTOR_DB_PATH=vector_db/

# Logging
LOG_LEVEL=INFO

# Web Interface
STREAMLIT_PORT=8501
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info("Created .env template file")
    else:
        logger.info(".env file already exists")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
rag_env/
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.production

# Vector Database
vector_db/
*.pkl
*.pickle

# Logs
logs/
*.log

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Data (uncomment if you don't want to track PDFs)
# data/pdfs/*.pdf
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        logger.info("Created .gitignore file")
    else:
        logger.info(".gitignore file already exists")

def verify_setup():
    """Verify that setup completed successfully"""
    logger.info("\n" + "="*50)
    logger.info("SETUP VERIFICATION")
    logger.info("="*50)
    
    # Check directories
    required_dirs = ["data/pdfs", "src", "vector_db", "logs"]
    for directory in required_dirs:
        if Path(directory).exists():
            logger.info(f"✓ Directory exists: {directory}")
        else:
            logger.error(f"✗ Missing directory: {directory}")
    
    # Check papers
    pdf_dir = Path("data/pdfs")
    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    logger.info(f"✓ Downloaded {pdf_count}/5 research papers")
    
    # Check files
    required_files = [".env", ".gitignore", "requirements.txt", "README.md"]
    for filename in required_files:
        if Path(filename).exists():
            logger.info(f"✓ File exists: {filename}")
        else:
            logger.error(f"✗ Missing file: {filename}")
    
    logger.info("\n" + "="*50)
    logger.info("NEXT STEPS")
    logger.info("="*50)
    logger.info("1. Edit .env file and add your OpenAI API key")
    logger.info("2. Create virtual environment: python -m venv rag_env")
    logger.info("3. Activate environment: source rag_env/bin/activate")
    logger.info("4. Install dependencies: pip install -r requirements.txt")
    logger.info("5. Run document processing: python src/document_processing.py")
    logger.info("6. Build vector database: python src/vector_store.py")
    logger.info("7. Test RAG pipeline: python src/rag_pipeline.py")
    logger.info("8. Launch web interface: streamlit run src/web_interface.py")

def main():
    """Main setup function"""
    logger.info("Starting RAG ML Papers project setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Download papers
    download_papers()
    
    # Create configuration files
    create_env_template()
    create_gitignore()
    
    # Verify setup
    verify_setup()
    
    logger.info("\nSetup completed! Check the verification results above.")

if __name__ == "__main__":
    main()