"""
Qwen3-8B model implementation
"""
import os
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class QwenModel(BaseModel):
    """Qwen3-8B model wrapper"""
    
    def load(self, **kwargs):
        """Load Qwen3-8B model"""
        try:
            # 检查是否为智能代理模式
            track = kwargs.get("track", "internal_reasoning")
            is_agentic = (track == "agentic_reasoning")
            
            if is_agentic:
                # 智能代理模式：直接使用TxAgent，避免双重加载
                logger.info("使用智能代理模式，直接加载TxAgent")
                self._setup_agentic_reasoning(**kwargs)
      
                return
            
            # 标准推理模式：使用vLLM
            from vllm import LLM, SamplingParams
            # 从kwargs或使用默认值获取模型配置
            model_name = kwargs.get("model_name", self.model_name)
            
            # 设置模型参数
            self.max_model_len = kwargs.get('max_model_len', 40960)
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
            
            # 设置推理参数
            self.temperature = kwargs.get('temperature', 0.3)
            self.max_new_tokens = kwargs.get('max_new_tokens', 8192)
            self.max_token = kwargs.get('max_token', 65536)
            self.max_round = kwargs.get('max_round', 20)
            self.multiagent = kwargs.get('multiagent', False)
            
            # 设置采样参数（用于标准推理）
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1)
            )
            
            logger.info(f"Qwen3-8B模型加载成功: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-8B model: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with Qwen3-8B model"""
        try:
            # 检查是否为智能代理模式
            if hasattr(self, 'is_agentic') and self.is_agentic and hasattr(self, 'txagent') and self.txagent is not None:
                logger.info(f"使用 {self.__class__.__name__} 智能代理推理模式")
                return self._agentic_inference(prompt, max_tokens)
            
            # 标准推理模式
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
            logger.error(f"Qwen3-8B inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}] 