from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

class Qwen:
    """Qwen 2.5 client wrapper for SQL query generation"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen/qwen-2.5-72b-instruct:free"):
        """
        Initialize Qwen 2.5 client
        
        Args:
            api_key: OpenRouter API key (optional, uses hardcoded key if not provided)
            model_name: Model name to use
        """
        if api_key is None:
            api_key = 'openrouter_api_key_placeholder'  # Replace with your actual OpenRouter API key
        
        self.client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )
    
    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the model with given messages
        
        Args:
            messages: List of messages to send to the model
            
        Returns:
            Model response
        """
        return self.client.invoke(messages)


from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

class Qwen:
    """Qwen client wrapper for SQL query generation"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen/qwen-2.5-coder-32b-instruct"):
        """
        Initialize Qwen client
        
        Args:
            api_key: OpenRouter API key (optional, uses hardcoded key if not provided)
            model_name: Model name to use
        """
        if api_key is None:
            api_key = 'openrouter_api_key_placeholder'  # Replace with your actual OpenRouter API key
        
        self.client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )
    
    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the model with given messages
        
        Args:
            messages: List of messages to send to the model
            
        Returns:
            Model response
        """
        return self.client.invoke(messages)


class Meta:
    """Meta LLaMA client wrapper for SQL query evaluation"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/llama-3.2-3b-instruct"):
        """
        Initialize Meta LLaMA client
        
        Args:
            api_key: OpenRouter API key (optional, uses hardcoded key if not provided)
            model_name: Model name to use
        """
        if api_key is None:
            api_key = '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'
        
        self.client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )
    
    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the model with given messages
        
        Args:
            messages: List of messages to send to the model
            
        Returns:
            Model response
        """
        return self.client.invoke(messages)