"""
Gemini model implementation with Google Search support
"""
from typing import Dict, List, Tuple
import logging
import os

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try to import google.generativeai, but don't fail if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not available. Gemini model will not work.")

# Try to import google.genai for Google Search support
try:
    from google import genai as google_genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    logger.warning("google.genai not available. Google Search will not work.")


class GeminiModel(BaseModel):
    """Gemini model wrapper with Google Search support"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = None
        self.google_search_enabled = False
        self.client = None
        self.temperature = 0.7  # 默认temperature值
        # Debug信息展示

    
    def load(self, **kwargs):
        """Load Gemini model with optional Google Search support"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google.generativeai is not installed. Please install it with: pip install google-generativeai")
        
        # Get API key
        self.api_key = kwargs.get('api_key')
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google API key not provided. Set api_key parameter or GOOGLE_API_KEY environment variable.")
        
        # Check if Google Search is enabled
        self.google_search_enabled = kwargs.get('google_search_enabled', False)
        
        # Set temperature parameter
        self.temperature = kwargs.get('temperature', 0.7)
        logger.info(f"🌡️ Temperature set to: {self.temperature}")
        
        if self.google_search_enabled:
            if not GOOGLE_GENAI_AVAILABLE:
                logger.warning("google.genai not available. Falling back to standard Gemini without search.")
                self.google_search_enabled = False
            else:
                # Set API key in environment for google.genai
                os.environ["GOOGLE_API_KEY"] = self.api_key
                # Initialize the client for Google Search
                self.client = google_genai.Client()
                logger.info(f"Gemini model with Google Search ready: {self.model_name}")
        
        if not self.google_search_enabled:
            # Use standard google.generativeai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Gemini model ready: {self.model_name}")
        logger.info("🔧 DEBUG - Gemini模型参数:")
        logger.info(f"   📝 模型名称: {self.model_name}")
        logger.info(f"   🌡️ Temperature: {self.temperature}")
        logger.info(f"   🔍 Google搜索: {'启用' if self.google_search_enabled else '禁用'}")
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference with Gemini, optionally with Google Search"""
        try:
            
            messages = [{"role": "user", "content": prompt}]
            
            if self.google_search_enabled and self.client:
                enhanced_prompt = f"""Please use Google Search to find the most current and accurate medical information to answer this question. Search for relevant medical literature, guidelines, and evidence-based information.

                Question: {prompt}

                Please provide a comprehensive answer based on the search results and cite your sources."""
                # import pdb; pdb.set_trace()
                # Use Google Search enhanced inference
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=enhanced_prompt,
                    config={
                        "tools": [{"google_search": {}}],
                        "temperature": self.temperature,
                        "max_output_tokens": max_tokens
                    },
                )
                
                # Extract the text response
                if hasattr(response, "text") and response.text:
                    response_text = response.text
                else:
                    response_text = str(response) if response else "No response generated"
                
                # Log search information if available
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if (hasattr(candidate, "grounding_metadata") and 
                        candidate.grounding_metadata):
                        logger.info("🔍 Google Search was used for grounding")
                        
                        # Log search queries
                        if (hasattr(candidate.grounding_metadata, "web_search_queries") and 
                            candidate.grounding_metadata.web_search_queries):
                            search_queries = candidate.grounding_metadata.web_search_queries
                            logger.info(f"📝 Search queries: {search_queries}")
                        
                        # Log search pages
                        if (hasattr(candidate.grounding_metadata, "grounding_chunks") and 
                            candidate.grounding_metadata.grounding_chunks):
                            search_pages = []
                            for site in candidate.grounding_metadata.grounding_chunks:
                                if hasattr(site, "web") and hasattr(site.web, "title"):
                                    search_pages.append(site.web.title)
                            if search_pages:
                                logger.info(f"🌐 Search pages used: {', '.join(search_pages)}")
                    else:
                        logger.warning("⚠️ No grounding metadata found - Google Search may not have been used")
            else:
                # Use standard Gemini without search
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens
                )
                response = self.model.generate_content(
                    prompt, 
                    generation_config=generation_config
                )
                if hasattr(response, "text") and response.text:
                    response_text = response.text
                else:
                    response_text = str(response) if response else "No response generated"
                logger.info("🚫 Google Search disabled - using model knowledge only")
                logger.info(f"🔧 DEBUG - 使用标准Gemini，temperature: {self.temperature}")
            
            # Debug信息展示响应
            # logger.info("🔧 DEBUG - 响应信息:")
            # logger.info(f"   📝 响应长度: {len(response_text)} 字符")
            # logger.info(f"   📄 响应预览: {response_text[:100]}..." if len(response_text) > 100 else f"   📄 完整响应: {response_text}")
            
            # Create complete conversation history
            complete_messages = messages + [
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, complete_messages
            
        except Exception as e:
            logger.error(f"Gemini inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"Error occurred: {str(e)}"},
            ]
            return f"Error occurred: {str(e)}", error_messages 