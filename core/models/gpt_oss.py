"""
GPT-OSS model implementation
"""
from typing import Dict, List, Tuple
import logging
import requests
import json

from .base import BaseModel

logger = logging.getLogger(__name__)


class GptOssModel(BaseModel):
    """GPT-OSS model wrapper using HTTP API"""
    
    def __init__(self, model_name: str, api_url: str = "http://localhost:8001"):
        super().__init__(model_name)
        self.api_url = api_url
        self.api_endpoint = f"{api_url}/v1/completions"
    
    def load(self, **kwargs):
        """Load GPT-OSS model (no actual loading needed)"""
        # Skip connection test during load to avoid blocking
        logger.info(f"GPT-OSS model ready: {self.model_name}")
        logger.info(f"API endpoint: {self.api_endpoint}")
    
    def inference(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> Tuple[str, List[Dict]]:
        """Run inference with GPT-OSS via HTTP API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("choices", [{}])[0].get("text", "")
                
                # Create messages list for reasoning trace
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text}
                ]
                
                return response_text, messages
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}", [{"role": "error", "content": error_msg}]
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: Could not connect to GPT-OSS API at {self.api_url}. Please ensure the service is running."
            logger.error(error_msg)
            return f"Error: {error_msg}", [{"role": "error", "content": error_msg}]
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout error: Request to GPT-OSS API timed out after 30 seconds."
            logger.error(error_msg)
            return f"Error: {error_msg}", [{"role": "error", "content": error_msg}]
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}", [{"role": "error", "content": error_msg}]
        except Exception as e:
            error_msg = f"GPT-OSS inference error: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}", [{"role": "error", "content": error_msg}]
