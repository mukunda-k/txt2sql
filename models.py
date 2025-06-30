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


class Meta:
    """Meta LLaMA client wrapper for SQL query evaluation with fallback models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/llama-3.1-8b-instruct:free"):
        """
        Initialize Meta LLaMA client with fallback options
        
        Args:
            api_key: OpenRouter API key (optional, uses hardcoded key if not provided)
            model_name: Model name to use
        """
        
        api_key = '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'
       
        self.client = ChatOpenAI(
                    model='meta-llama/llama-3.1-8b-instruct:free',
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