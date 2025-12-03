"""
Local model implementation using vLLM
"""
import torch
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LocalModel(BaseModel):
    """Local model wrapper using vLLM"""
    
    def load(self, **kwargs):
        """Load local model using vLLM"""
        try:
            from vllm import LLM, SamplingParams
            
            # Set model parameters
            self.max_model_len = kwargs.get('max_model_len', 65536)
            self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.75)
            self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
            self.dtype = kwargs.get('dtype', 'float16')
            
            # Load model
            self.model = LLM(
                model=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype
            )
            
            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            logger.info(f"Local model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with local model"""
        try:
            # Update max_tokens if different
            if max_tokens != self.sampling_params.max_tokens:
                self.sampling_params.max_tokens = max_tokens
            
            # Run inference
            outputs = self.model.generate([prompt], self.sampling_params)
            
            # Extract response
            response_text = outputs[0].outputs[0].text
            
            # Create messages list for reasoning trace
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, messages
            
        except Exception as e:
            logger.error(f"Local model inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}] 