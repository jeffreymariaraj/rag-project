"""
Prompt Templates Module
Contains modular prompt templates for the RAG system
"""

from typing import Dict, Any, Optional

# Default system prompt for research assistant
DEFAULT_SYSTEM_PROMPT = """You are an AI research assistant specialized in machine learning papers. 
Your task is to provide accurate, concise answers based on the provided context.
Only answer what can be inferred from the context. If the context doesn't contain 
relevant information, acknowledge the limitations of the provided information.
Always cite the specific paper and section when possible."""

# Default RAG prompt template
DEFAULT_RAG_PROMPT = """
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the following question:
{query}

Answer:
"""

# More detailed RAG prompt with citation instructions
DETAILED_RAG_PROMPT = """
Below is a set of documents retrieved based on your query.
---------------------
{context}
---------------------

Using ONLY the information provided in these documents, please answer the following question:
{query}

In your answer:
1. Be concise and specific
2. Cite the specific papers and sections that support your answer
3. If the information is not available in the context, acknowledge this limitation
4. Format any technical concepts, equations, or code appropriately

Answer:
"""

# Summarization prompt for when multiple papers are relevant
SUMMARIZATION_PROMPT = """
The following documents contain information about {topic}.
---------------------
{context}
---------------------

Please synthesize the key information from these documents to answer:
{query}

Focus on comparing and contrasting the different approaches or findings across the papers.
Include specific citations to the papers in your answer.

Answer:
"""

# Technical explanation prompt for complex topics
TECHNICAL_PROMPT = """
The following technical documents explain {topic}.
---------------------
{context}
---------------------

Please provide a detailed technical explanation in response to:
{query}

Include:
- Key mathematical concepts or equations if relevant
- Step-by-step explanation of algorithms or processes
- Specific implementation details mentioned in the papers
- Citations to the source papers and sections

Answer:
"""

class PromptTemplates:
    """Class to manage prompt templates"""
    
    def __init__(self):
        """Initialize with default templates"""
        self.templates = {
            'default': {
                'name': 'Default',
                'description': 'Standard RAG prompt for general questions',
                'system_prompt': DEFAULT_SYSTEM_PROMPT,
                'user_prompt': DEFAULT_RAG_PROMPT
            },
            'detailed': {
                'name': 'Detailed with Citations',
                'description': 'Detailed prompt with explicit citation instructions',
                'system_prompt': DEFAULT_SYSTEM_PROMPT,
                'user_prompt': DETAILED_RAG_PROMPT
            },
            'summarization': {
                'name': 'Multi-Document Synthesis',
                'description': 'Synthesize information across multiple papers',
                'system_prompt': DEFAULT_SYSTEM_PROMPT,
                'user_prompt': SUMMARIZATION_PROMPT
            },
            'technical': {
                'name': 'Technical Explanation',
                'description': 'Detailed technical explanation with equations and algorithms',
                'system_prompt': DEFAULT_SYSTEM_PROMPT,
                'user_prompt': TECHNICAL_PROMPT
            }
        }
    
    def get_template(self, template_id: str) -> Dict[str, str]:
        """Get a specific template by ID"""
        return self.templates.get(template_id, self.templates['default'])
    
    def get_template_ids(self) -> list:
        """Get list of available template IDs"""
        return list(self.templates.keys())
    
    def get_template_names(self) -> Dict[str, str]:
        """Get mapping of template IDs to names"""
        return {tid: template['name'] for tid, template in self.templates.items()}
    
    def format_prompt(self, template_id: str, context: str, query: str, topic: Optional[str] = None) -> Dict[str, str]:
        """
        Format a prompt using the specified template
        
        Args:
            template_id: ID of the template to use
            context: Context information
            query: User query
            topic: Optional topic for templates that require it
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        template = self.get_template(template_id)
        
        # Format user prompt
        user_prompt = template['user_prompt']
        
        # Replace placeholders
        format_dict = {'context': context, 'query': query}
        if topic:
            format_dict['topic'] = topic
        elif '{topic}' in user_prompt:
            # Extract topic from query if not provided
            format_dict['topic'] = query.split()[0] if query else 'this topic'
            
        user_prompt = user_prompt.format(**format_dict)
        
        return {
            'system_prompt': template['system_prompt'],
            'user_prompt': user_prompt
        }
    
    def add_custom_template(self, template_id: str, name: str, description: str, 
                          system_prompt: str, user_prompt: str) -> None:
        """Add a custom template"""
        self.templates[template_id] = {
            'name': name,
            'description': description,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
    
    def get_all_templates(self) -> Dict[str, Dict[str, str]]:
        """Get all templates"""
        return self.templates 