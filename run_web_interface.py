#!/usr/bin/env python3
"""
Script to run the RAG system web interface
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit web interface"""
    # Ensure we're in the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Add src to path
    sys.path.append(str(project_root))
    
    print("Starting RAG System Web Interface...")
    print("=" * 50)
    print("Access the interface at http://localhost:8501")
    print("=" * 50)
    
    # Run Streamlit
    subprocess.run(["streamlit", "run", "src/web_interface.py"])

if __name__ == "__main__":
    main() 