"""
Llama 3.1 8B model implementation
"""
import os
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LlamaModel(BaseModel):
    """Llama 3.1 8B model wrapper"""
    
    def load(self, **kwargs):
        """Load Llama 3.1 8B model"""
        try:
            from vllm import LLM, SamplingParams
            
            # 设置环境变量以避免vLLM V1 API问题
            os.environ["VLLM_USE_V1"] = "0"
            
            # 从kwargs或使用默认值获取模型配置
            model_name = kwargs.get("model_name", self.model_name)
            
            # 设置模型参数
            self.max_model_len = kwargs.get('max_model_len', 65536)
            self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.75)
            self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
            self.dtype = kwargs.get('dtype', 'float16')
            
            # 加载模型
            self.model = LLM(
                model=model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype
            )
            
            # 设置采样参数
            self.sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_new_tokens', 2048),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1)
            )
            
            logger.info(f"Llama 3.1 8B模型加载成功: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3.1 8B model: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with Llama 3.1 8B model"""
        try:
            # 更新max_tokens如果不同
            if max_tokens != self.sampling_params.max_tokens:
                self.sampling_params.max_tokens = max_tokens
            
            # 运行推理
            outputs = self.model.generate([prompt], self.sampling_params)
            
            # 提取响应
            response_text = outputs[0].outputs[0].text
            
            # 创建消息列表用于推理跟踪
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, messages
            
        except Exception as e:
            logger.error(f"Llama 3.1 8B inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}]