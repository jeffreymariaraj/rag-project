"""
Web Interface for RAG System
Streamlit-based interface for querying ML research papers
"""

import os
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

# Import RAG Pipeline
try:
    from src.rag_pipeline import RAGPipeline
    from src.prompt_templates import PromptTemplates
    RAG_AVAILABLE = True
except ImportError:
    # Try relative import for when running directly
    try:
        from rag_pipeline import RAGPipeline
        from prompt_templates import PromptTemplates
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        st.error("RAG Pipeline module not available. Some features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_interface.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ML Research Papers RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .context-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .metadata-text {
        font-size: 0.8rem;
        color: #616161;
    }
    .answer-box {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .score-high {
        color: #388e3c;
        font-weight: bold;
    }
    .score-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .score-low {
        color: #d32f2f;
        font-weight: bold;
    }
    .history-item {
        padding: 8px;
        border-radius: 4px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .history-item:hover {
        background-color: #f0f0f0;
    }
    .history-item-selected {
        background-color: #e3f2fd;
        border-left: 3px solid #1E88E5;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .navigation-btn {
        margin-right: 5px;
    }
    .template-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #1E88E5;
    }
    .template-card-selected {
        background-color: #e3f2fd;
        border-left: 3px solid #1E88E5;
    }
    .error-box {
        background-color: #ffebee;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #d32f2f;
    }
    .warning-box {
        background-color: #fff8e1;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ffa000;
    }
    .info-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)

def format_score(score: float) -> str:
    """Format similarity score with color coding"""
    if score >= 0.8:
        return f'<span class="score-high">{score:.3f}</span>'
    elif score >= 0.6:
        return f'<span class="score-medium">{score:.3f}</span>'
    else:
        return f'<span class="score-low">{score:.3f}</span>'

def display_context(context_items: List[Dict], current_paper_index: int = 0):
    """
    Display retrieved context with expandable sections
    
    Args:
        context_items: List of context items
        current_paper_index: Index of the current paper to focus on
    """
    if not context_items:
        st.info("No relevant context found.")
        return
    
    st.markdown("### 📄 Retrieved Context")
    
    # Group by paper title
    papers = {}
    for item in context_items:
        paper_title = item['metadata'].get('paper_title', 'Unknown Paper')
        if paper_title not in papers:
            papers[paper_title] = []
        papers[paper_title].append(item)
    
    # Get list of paper titles
    paper_titles = list(papers.keys())
    
    # Ensure current_paper_index is valid
    if paper_titles:
        current_paper_index = max(0, min(current_paper_index, len(paper_titles) - 1))
        
        # Navigation controls for papers
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if current_paper_index > 0:
                if st.button("⬅️ Previous Paper", key="prev_paper", 
                           help=f"Go to {paper_titles[current_paper_index-1]}"):
                    st.session_state.current_paper_index = current_paper_index - 1
                    st.rerun()
        with col2:
            st.write(f"Paper {current_paper_index + 1} of {len(paper_titles)}")
        with col3:
            if current_paper_index < len(paper_titles) - 1:
                if st.button("Next Paper ➡️", key="next_paper", 
                           help=f"Go to {paper_titles[current_paper_index+1]}"):
                    st.session_state.current_paper_index = current_paper_index + 1
                    st.rerun()
        
        # Display current paper
        current_paper_title = paper_titles[current_paper_index]
        items = papers[current_paper_title]
        
        st.markdown(f"## 📑 {current_paper_title}")
        
        # Display chunks for current paper
        for i, item in enumerate(items):
            with st.expander(f"Chunk {i+1} - Similarity: {item['similarity_score']:.3f}", expanded=(i==0)):
                section = item['metadata'].get('section', 'Unknown Section')
                page = item['metadata'].get('page_number', 'Unknown Page')
                
                st.markdown(f"**Section:** {section} | **Page:** {page}")
                st.markdown(f"<div class='context-box'>{item['content']}</div>", 
                           unsafe_allow_html=True)

def validate_environment() -> Tuple[bool, str]:
    """
    Validate environment variables and dependencies
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    
    # Check if vector store path exists
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db")
    if not Path(vector_db_path).exists():
        return False, f"Vector database path not found: {vector_db_path}. Please build the vector database first."
    
    # Check if RAG pipeline is available
    if not RAG_AVAILABLE:
        return False, "RAG pipeline dependencies not available. Please install the required packages."
    
    return True, ""

def initialize_rag_pipeline() -> Optional[RAGPipeline]:
    """Initialize the RAG pipeline"""
    # Validate environment
    is_valid, error_message = validate_environment()
    if not is_valid:
        logger.error(f"Environment validation failed: {error_message}")
        st.sidebar.error(error_message)
        return None
    
    try:
        # Get API key from environment or session state
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Get settings from session state if available
        model = st.session_state.get('model', os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
        temperature = st.session_state.get('temperature', float(os.getenv("LLM_TEMPERATURE", "0.7")))
        top_k = st.session_state.get('top_k', int(os.getenv("TOP_K_RESULTS", "5")))
        prompt_template_id = st.session_state.get('prompt_template_id', "default")
        
        # Initialize pipeline
        pipeline = RAGPipeline(
            vector_store_path=os.getenv("VECTOR_DB_PATH", "vector_db"),
            openai_api_key=api_key,
            model_name=model,
            temperature=temperature,
            top_k=top_k,
            prompt_template_id=prompt_template_id
        )
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        error_details = traceback.format_exc()
        logger.debug(f"Error details: {error_details}")
        
        # Display user-friendly error message
        error_message = f"Failed to initialize RAG pipeline: {str(e)}"
        st.sidebar.error(error_message)
        
        return None

def initialize_session_state():
    """Initialize session state variables"""
    # History of queries and responses
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Current history item index
    if 'current_history_index' not in st.session_state:
        st.session_state.current_history_index = -1
    
    # Current paper index for context navigation
    if 'current_paper_index' not in st.session_state:
        st.session_state.current_paper_index = 0
    
    # Prompt template settings
    if 'prompt_template_id' not in st.session_state:
        st.session_state.prompt_template_id = "default"
    
    # Model settings
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    
    # Temperature setting
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    # Top-k setting
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    
    # Topic for specialized prompts
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    
    # Error state
    if 'error' not in st.session_state:
        st.session_state.error = None

def add_to_history(query: str, response: Dict):
    """Add a query and its response to history"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    history_item = {
        'timestamp': timestamp,
        'query': query,
        'response': response,
        'context_items': [],
        'prompt_template_id': st.session_state.prompt_template_id,
        'topic': st.session_state.topic
    }
    
    # Add context items
    if response and 'sources' in response:
        pipeline = initialize_rag_pipeline()
        if pipeline:
            try:
                for source in response['sources']:
                    # Find the original context item
                    results = pipeline.vector_store.search(query, top_k=10)
                    for result in results:
                        if (result['metadata'].get('paper_title') == source['paper_title'] and
                            result['metadata'].get('section') == source['section'] and
                            result['metadata'].get('page_number') == source['page']):
                            history_item['context_items'].append(result)
                            break
            except Exception as e:
                logger.error(f"Error adding context to history: {e}")
                # Continue without context if there's an error
    
    st.session_state.history.append(history_item)
    st.session_state.current_history_index = len(st.session_state.history) - 1

def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query
    
    Args:
        query: User query
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Please enter a question."
    
    if len(query) < 5:
        return False, "Query is too short. Please provide a more detailed question."
    
    if len(query) > 1000:
        return False, "Query is too long. Please limit your question to 1000 characters."
    
    return True, ""

def display_prompt_templates():
    """Display and select prompt templates"""
    if RAG_AVAILABLE:
        try:
            prompt_templates = PromptTemplates()
            template_names = prompt_templates.get_template_names()
            
            st.markdown("### 📝 Prompt Templates")
            st.write("Select a prompt template for your query:")
            
            # Display template options
            for template_id, template_name in template_names.items():
                template = prompt_templates.get_template(template_id)
                
                # Determine if this is the selected template
                is_selected = template_id == st.session_state.prompt_template_id
                card_class = "template-card template-card-selected" if is_selected else "template-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}">
                        <strong>{template_name}</strong><br>
                        <small>{template['description']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("Select", key=f"select_{template_id}"):
                            st.session_state.prompt_template_id = template_id
                            st.rerun()
                    with col2:
                        if st.button("View", key=f"view_{template_id}"):
                            with st.expander("Template Details", expanded=True):
                                st.markdown("**System Prompt:**")
                                st.text(template['system_prompt'])
                                st.markdown("**User Prompt:**")
                                st.text(template['user_prompt'])
            
            # Topic input for specialized templates
            st.markdown("### 🏷️ Topic (Optional)")
            st.write("Specify a topic for templates that support it:")
            topic = st.text_input("Topic", value=st.session_state.get('topic', ''), 
                                help="Used in specialized templates like summarization or technical explanation")
            if topic != st.session_state.get('topic', ''):
                st.session_state.topic = topic
        except Exception as e:
            logger.error(f"Error displaying prompt templates: {e}")
            st.warning("Error loading prompt templates. Using default template.")

def render_sidebar():
    """Render the sidebar with history and controls"""
    st.sidebar.markdown('<p class="sidebar-header">📚 ML Research Papers RAG</p>', 
                      unsafe_allow_html=True)
    
    # System status
    pipeline = initialize_rag_pipeline()
    if pipeline:
        st.sidebar.success("RAG Pipeline: Ready")
    else:
        st.sidebar.error("RAG Pipeline: Not Available")
    
    st.sidebar.divider()
    
    # Query history
    st.sidebar.markdown('<p class="sidebar-header">📝 Query History</p>', 
                      unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.sidebar.info("No queries yet. Ask a question to get started!")
    else:
        # Display history items
        for i, item in enumerate(st.session_state.history):
            # Truncate query if too long
            query_display = item['query']
            if len(query_display) > 60:
                query_display = query_display[:57] + "..."
            
            # Determine if this is the selected item
            is_selected = i == st.session_state.current_history_index
            item_class = "history-item history-item-selected" if is_selected else "history-item"
            
            # Create a clickable history item
            if st.sidebar.markdown(f"""
            <div class="{item_class}" onclick="this.onclick=null; window.location.href='?history_index={i}';">
                <small>{item['timestamp']}</small><br>
                {query_display}
            </div>
            """, unsafe_allow_html=True):
                st.session_state.current_history_index = i
                st.rerun()
    
    st.sidebar.divider()
    
    # Controls
    st.sidebar.markdown('<p class="sidebar-header">⚙️ Controls</p>', 
                      unsafe_allow_html=True)
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.current_history_index = -1
        st.rerun()
    
    # Settings
    with st.sidebar.expander("🔧 Settings"):
        # Model selection
        model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        # Number of results
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        
        # Apply settings
        if st.button("Apply Settings"):
            # Store settings in session state
            st.session_state.model = model
            st.session_state.temperature = temperature
            st.session_state.top_k = top_k
            st.success("Settings applied!")

def handle_error(error: Exception, error_type: str = "general"):
    """
    Handle errors gracefully
    
    Args:
        error: Exception object
        error_type: Type of error (api, validation, general)
    """
    error_message = str(error)
    error_details = traceback.format_exc()
    
    # Log the error
    logger.error(f"{error_type.capitalize()} error: {error_message}")
    logger.debug(f"Error details: {error_details}")
    
    # Store error in session state
    st.session_state.error = {
        'type': error_type,
        'message': error_message,
        'details': error_details
    }
    
    # Display user-friendly error message
    if error_type == "api":
        st.error(f"API Error: {error_message}. Please check your API key and try again.")
    elif error_type == "validation":
        st.warning(f"Validation Error: {error_message}")
    else:
        st.error(f"Error: {error_message}. Please try again or contact support.")

def main():
    """Main function for the Streamlit app"""
    try:
        # Load custom CSS
        load_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        # Main content
        st.markdown('<h1 class="main-header">ML Research Papers RAG System</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Ask questions about machine learning research papers</p>', 
                   unsafe_allow_html=True)
        
        # Initialize RAG pipeline
        pipeline = initialize_rag_pipeline()
        
        # Display any stored errors
        if st.session_state.error:
            error_type = st.session_state.error['type']
            error_message = st.session_state.error['message']
            
            if error_type == "api":
                st.markdown(f"""
                <div class="error-box">
                    <strong>API Error:</strong> {error_message}<br>
                    Please check your API key and try again.
                </div>
                """, unsafe_allow_html=True)
            elif error_type == "validation":
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Validation Error:</strong> {error_message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Error:</strong> {error_message}<br>
                    Please try again or contact support.
                </div>
                """, unsafe_allow_html=True)
            
            # Clear error after displaying
            if st.button("Clear Error"):
                st.session_state.error = None
                st.rerun()
        
        # Check if we're viewing a history item
        viewing_history = st.session_state.current_history_index >= 0 and st.session_state.history
        
        if viewing_history:
            # Display the selected history item
            history_item = st.session_state.history[st.session_state.current_history_index]
            
            # Show the original query
            st.info(f"Viewing previous query: {history_item['query']}")
            
            # Navigation controls for history
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.session_state.current_history_index > 0:
                    if st.button("⬅️ Previous Query"):
                        st.session_state.current_history_index -= 1
                        st.rerun()
            with col2:
                if st.button("🔄 New Query"):
                    st.session_state.current_history_index = -1
                    st.rerun()
            with col3:
                if st.session_state.current_history_index < len(st.session_state.history) - 1:
                    if st.button("Next Query ➡️"):
                        st.session_state.current_history_index += 1
                        st.rerun()
            
            # Display the response
            response = history_item['response']
            
            # Display prompt template used
            if 'prompt_template_id' in history_item:
                st.markdown(f"**Prompt Template:** {history_item['prompt_template_id']}")
                if history_item.get('topic'):
                    st.markdown(f"**Topic:** {history_item['topic']}")
            
            # Display answer
            st.markdown("### 🤖 Answer")
            st.markdown(f"<div class='answer-box'>{response['answer']}</div>", 
                       unsafe_allow_html=True)
            
            # Display metadata
            st.markdown("#### ℹ️ Metadata")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{response['processing_time_seconds']:.2f}s")
            with col2:
                st.metric("Retrieved Chunks", str(response['num_retrieved_chunks']))
            with col3:
                if 'token_usage' in response:
                    st.metric("Tokens Used", str(response['token_usage']['total_tokens']))
            
            # Display context
            display_context(
                history_item['context_items'],
                st.session_state.current_paper_index
            )
        
        else:
            # Display prompt template selection
            display_prompt_templates()
            
            st.markdown("---")
            
            # Query input
            query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What is the main innovation in the Attention is All You Need paper?"
            )
            
            # Example questions
            with st.expander("Example questions"):
                example_questions = [
                    "What is the main innovation in 'Attention is All You Need'?",
                    "How does BERT differ from traditional language models?",
                    "Describe GPT-3's few-shot learning capabilities with an example.",
                    "What is CLIP's loss function and how does it work?",
                    "What approach does LLaMA use to reduce training costs?"
                ]
                for q in example_questions:
                    if st.button(q, key=f"example_{q[:20]}"):
                        query = q
                        st.rerun()
            
            # Submit button
            submit = st.button("Submit", type="primary")
            
            # Process query
            if submit:
                # Validate query
                is_valid, error_message = validate_query(query)
                
                if not is_valid:
                    # Display validation error
                    st.warning(error_message)
                elif not pipeline:
                    # Display pipeline error
                    st.error("RAG Pipeline is not available. Please check your installation.")
                else:
                    # Process valid query
                    with st.spinner("Processing your question..."):
                        try:
                            # Record start time
                            start_time = time.time()
                            
                            # Get settings from session state
                            model = st.session_state.get('model', "gpt-3.5-turbo")
                            temperature = st.session_state.get('temperature', 0.7)
                            top_k = st.session_state.get('top_k', 5)
                            prompt_template_id = st.session_state.get('prompt_template_id', "default")
                            topic = st.session_state.get('topic', None)
                            
                            # Process query with settings
                            response = pipeline.ask_question(
                                query=query,
                                n_results=top_k,
                                prompt_template_id=prompt_template_id,
                                topic=topic
                            )
                            
                            # Calculate processing time
                            processing_time = time.time() - start_time
                            response['processing_time_seconds'] = processing_time
                            
                            # Add to history
                            add_to_history(query, response)
                            
                            # Clear any previous errors
                            st.session_state.error = None
                            
                            # Rerun to display the history item
                            st.rerun()
                            
                        except Exception as e:
                            # Handle API errors
                            if "openai" in str(e).lower() or "api" in str(e).lower():
                                handle_error(e, "api")
                            else:
                                handle_error(e)
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in main: {e}")
        logger.debug(traceback.format_exc())
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # Simple test/demo when running directly
    try:
        main()
        
        # Add footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center">
            <p><small>ML Research Papers RAG System | Built with Streamlit</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display debug info in expander
        with st.expander("Debug Information", expanded=False):
            st.write("Environment:")
            st.json({
                "RAG_AVAILABLE": RAG_AVAILABLE,
                "OPENAI_API_KEY": "Present" if os.getenv("OPENAI_API_KEY") else "Missing",
                "VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_db"),
                "VECTOR_DB_EXISTS": Path(os.getenv("VECTOR_DB_PATH", "vector_db")).exists(),
                "SESSION_STATE": {
                    key: value for key, value in st.session_state.items() 
                    if key not in ['history', 'error'] and not isinstance(value, (dict, list))
                }
            })
            
            # Display system info
            st.write("System Information:")
            st.text(f"Python Version: {os.sys.version}")
            st.text(f"Working Directory: {os.getcwd()}")
            
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        logger.critical(f"Critical error in __main__: {e}")
        logger.critical(traceback.format_exc())
