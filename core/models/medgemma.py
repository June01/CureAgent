"""
MedGemma model implementation
"""
import os
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class MedGemmaModel(BaseModel):
    """MedGemma model wrapper"""
    
    def load(self, **kwargs):
        """Load MedGemma model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 从kwargs或使用默认值获取模型配置
            model_name = kwargs.get("model_name", self.model_name)
            
            # 设置模型参数
            self.torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
            self.device_map = kwargs.get('device_map', "auto")
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_new_tokens = kwargs.get('max_new_tokens', 2048)
            self.top_p = kwargs.get('top_p', 0.9)
            self.top_k = kwargs.get('top_k', 50)
            self.repetition_penalty = kwargs.get('repetition_penalty', 1.1)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            
            logger.info(f"MedGemma模型加载成功: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with MedGemma model"""
        try:
            import torch
            
            # 构建消息格式
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # 应用聊天模板
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # 生成参数
            generation_config = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # 运行推理
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    **generation_config
                )
                generation = generation[0][input_len:]
            
            # 解码响应
            response_text = self.tokenizer.decode(generation, skip_special_tokens=True)
            
            # 创建消息列表用于推理跟踪
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, messages
            
        except Exception as e:
            logger.error(f"MedGemma inference error: {e}")
            return f"Error: {e}", [{"role": "error", "content": str(e)}]