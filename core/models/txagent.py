"""
TxAgent model implementation
"""
import os
from typing import Dict, List, Tuple
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class TxAgentModel(BaseModel):
    """TxAgent model wrapper"""
    
    def load(self, **kwargs):
        """Load TxAgent model"""
        try:
            # 设置环境变量以避免vLLM V1 API问题
            os.environ["VLLM_USE_V1"] = "0"
            
            # 从kwargs或使用默认值获取模型配置
            model_name = kwargs.get("model_name", self.model_name)
            rag_model_name = kwargs.get(
                "rag_model_name", "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
            )

            # 导入TxAgent
            from txagent import TxAgent

            # Debug: Print summary_temperature value
            summary_temp = kwargs.get("summary_temperature", 0.3)
            print(f"\033[33m[DEBUG] TxAgentModel.py - summary_temperature from kwargs: {summary_temp}\033[0m")
            
            # 初始化TxAgent
            self.agent = TxAgent(
                model_name=model_name,
                rag_model_name=rag_model_name,
                enable_summary=kwargs.get("enable_summary", False),
                avoid_repeat=kwargs.get("avoid_repeat", True),
                enable_checker=kwargs.get("enable_checker", False),  # 从配置读取
                step_rag_num=kwargs.get("step_rag_num", 0),  # 从配置读取
                summary_temperature=summary_temp,  # 从配置读取
                seed=kwargs.get("seed", 100),
            )

            # 初始化模型
            self.agent.init_model()

            # 设置推理参数（从配置读取）
            self.temperature = kwargs.get("temperature", 0.3)
            self.max_new_tokens = kwargs.get("max_new_tokens", 2048)
            self.max_token = kwargs.get("max_token", 65536)
            self.max_round = kwargs.get("max_round", 20)
            self.multiagent = kwargs.get("multiagent", False)

            logger.info(f"TxAgent模型加载成功: {model_name}")
            
        except Exception as e:
            logger.error(f"TxAgent模型加载失败: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """TxAgent推理"""
        try:
            # 使用多步推理（支持工具调用）
            final_response, response_dict = self.agent.run_multistep_agent(
                message=prompt,
                temperature=self.temperature,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                max_token=self.max_token,
                call_agent=self.multiagent,
                max_round=self.max_round,
            )
            # import pdb; pdb.set_trace()
            # 从返回的字典中提取最终助手回复
            complete_messages = response_dict
                
            # 如果没有找到助手回复，使用默认错误信息
            if not final_response:
                final_response = "TxAgent未能生成有效回复"
                complete_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": final_response}
                ]
            
            # 清理响应文本
            if final_response:
                # 移除ANSI颜色代码和特殊标记
                import re
                final_response = re.sub(r"\033\[[0-9;]*m", "", final_response)  # 移除ANSI颜色
                final_response = final_response.replace("</s>", "").strip()  # 移除结束标记

            logger.info(f"TxAgent推理完成，响应长度: {len(final_response) if final_response else 0}")
            # import pdb; pdb.set_trace()
            
            return final_response, complete_messages

        except Exception as e:
            logger.error(f"TxAgent推理错误: {e}")
            # 返回错误信息
            error_response = f"TxAgent推理过程中发生错误: {str(e)}"
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": error_response},
            ]
            return error_response, error_messages 