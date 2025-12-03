"""
Base model abstract class for CURE-Bench evaluation framework
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.txagent = None
        self.is_agentic = False

    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass

    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass
    
    def _setup_agentic_reasoning(self, **kwargs):
        """Setup TxAgent for agentic reasoning"""
        try:
            # 检查是否为agentic_reasoning模式
            track = kwargs.get("track", "internal_reasoning")
            self.is_agentic = (track == "agentic_reasoning")
            
            if self.is_agentic:
                # 导入TxAgent - 尝试不同的导入路径
                try:
                    # 首先尝试从本地路径导入
                    import sys
                    sys.path.append('/root/code/TxAgent_AMD/src')
                    from txagent import TxAgent
                    logger.info("Successfully imported TxAgent from local path")
                except ImportError:
                    try:
                        # 然后尝试从安装的包导入
                        from txagent import TxAgent
                        logger.info("Successfully imported TxAgent from installed package")
                    except ImportError as e:
                        logger.error(f"TxAgent not available: {e}")
                        raise
                
                # 获取TxAgent配置参数
                model_name = kwargs.get("model_name", self.model_name)
                rag_model_name = kwargs.get("rag_model_name", "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")
                
                logger.info(f"使用模型: {model_name}")
                
                # 初始化TxAgent
                self.txagent = TxAgent(
                    model_name=model_name,
                    rag_model_name=rag_model_name,
                    tool_files_dict=kwargs.get("tool_files_dict", None),
                    enable_finish=kwargs.get("enable_finish", True),
                    enable_rag=kwargs.get("enable_rag", True),
                    enable_summary=kwargs.get("enable_summary", False),
                    init_rag_num=kwargs.get("init_rag_num", 0),
                    step_rag_num=kwargs.get("step_rag_num", 10),
                    summary_mode=kwargs.get("summary_mode", "step"),
                    summary_skip_last_k=kwargs.get("summary_skip_last_k", 0),
                    summary_context_length=kwargs.get("summary_context_length", None),
                    force_finish=kwargs.get("force_finish", True),
                    avoid_repeat=kwargs.get("avoid_repeat", True),
                    seed=kwargs.get("seed", 100),
                    enable_checker=kwargs.get("enable_checker", False),
                    enable_chat=kwargs.get("enable_chat", False),
                    additional_default_tools=kwargs.get("additional_default_tools", None),
                )
                
                # 初始化模型
                self.txagent.init_model()
                
                # 设置推理参数
                self.temperature = kwargs.get("temperature", 0.3)
                self.max_new_tokens = kwargs.get("max_new_tokens", 2048)
                self.max_token = kwargs.get("max_token", 65536)
                self.max_round = kwargs.get("max_round", 20)
                self.multiagent = kwargs.get("multiagent", False)
                
                logger.info(f"{self.__class__.__name__} agentic reasoning模型加载成功: {model_name}")
                
        except ImportError as e:
            logger.error(f"TxAgent not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load TxAgent for {self.__class__.__name__}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _agentic_inference(self, prompt: str, max_tokens: int) -> Tuple[str, List[Dict]]:
        """Agentic reasoning inference using TxAgent"""
        try:
            # 检查TxAgent是否正确初始化
            if not hasattr(self, 'txagent') or self.txagent is None:
                raise Exception("TxAgent not properly initialized")
            
            logger.info(f"Starting agentic inference with {self.__class__.__name__}")
            
            # 使用TxAgent的多步推理
            final_response, conversation = self.txagent.run_multistep_agent(
                message=prompt,
                temperature=self.temperature,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                max_token=self.max_token,
                call_agent=self.multiagent,
                max_round=self.max_round,
            )
            
            # 如果没有找到助手回复，使用默认错误信息
            if not final_response:
                final_response = f"{self.__class__.__name__} agentic reasoning未能生成有效回复"
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": final_response}
                ]
            
            # 清理响应文本
            if final_response:
                # 移除ANSI颜色代码和特殊标记
                import re
                final_response = re.sub(r"\033\[[0-9;]*m", "", final_response)
                final_response = final_response.replace("</s>", "").strip()
            
            logger.info(f"{self.__class__.__name__} agentic推理完成，响应长度: {len(final_response) if final_response else 0}")
            
            return final_response, conversation
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} agentic推理错误: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # 返回错误信息
            error_response = f"{self.__class__.__name__} agentic推理过程中发生错误: {str(e)}"
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": error_response}
            ]
            return error_response, error_messages 