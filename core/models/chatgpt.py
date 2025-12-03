"""
ChatGPT model implementation
"""
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try to import openai, but don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not available. ChatGPT model will not work.")


class ChatGPTModel(BaseModel):
    """ChatGPT model wrapper"""
    
    def load(self, **kwargs):
        """Load ChatGPT model (no actual loading needed)"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is not installed. Please install it with: pip install openai")
        
        # Set API key if provided
        api_key = kwargs.get('api_key')
        if api_key:
            openai.api_key = api_key
        logger.info(f"ChatGPT model ready: {self.model_name}")
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with ChatGPT"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Create messages list for reasoning trace
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, messages
            
        except Exception as e:
            logger.error(f"ChatGPT inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}] 