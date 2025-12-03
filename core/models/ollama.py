"""
Ollama model implementation
"""
import requests
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class OllamaModel(BaseModel):
    """Ollama model wrapper"""
    
    def load(self, **kwargs):
        """Load Ollama model (no actual loading needed)"""
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        logger.info(f"Ollama model ready: {self.model_name}")
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with Ollama"""
        try:
            # Prepare request
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            # Make request
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            response_text = result.get('response', '')
            
            # Create messages list for reasoning trace
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, messages
            
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}] 