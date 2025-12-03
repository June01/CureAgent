"""
Bio-Medical AI Competition Starter Kit - Modular Version

A modular framework for evaluating models on bio-medical datasets.
This version uses the new modular architecture for better maintainability.

Key Features:
- Modular design with separate components
- Easy model loading with factory pattern
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.save_submission(results, "my_submission.json")
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional
import torch

from tqdm import tqdm

# Setup environment variables
from env_setup import setup_environment
setup_environment()

# Import core modules
from core.models.factory import ModelFactory
from core.evaluation.evaluator import Evaluator
from core.evaluation.parallel_evaluator import ParallelEvaluator
from core.evaluation.metrics import EvaluationMetrics

# Import model implementations (these will be moved to separate files later)
from core.models import ChatGPTModel, LocalModel, OllamaModel, GeminiModel, TxAgentModel, LlamaModel, QwenModel, MedGemmaModel, GptOssModel, BaichuanModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CompetitionKit:
    """
    Modular competition framework using the new architecture
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit

        Args:
            config_path: Path to configuration file
        """
        self.model = None
        self.model_name = None
        self.config = {}
        self.evaluator = None
        
        # Load configuration
        if config_path:
            self.config = self._load_config_file(config_path)
        else:
            # Use default configuration
            self.config = {
                "model_name": "default-model",
                "dataset_name": "default-dataset",
                "dataset_path": "default-path",
                "output_dir": "results"
            }
        
        # Register model types with factory
        self._register_models()

    def _load_config_file(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract metadata if present
            metadata = config_data.get("metadata", {})
            
            # Extract dataset configuration
            dataset_config = config_data.get("dataset", {})
            
            # Create simplified config
            config = {
                "model_name": metadata.get("model_name", ""),
                "model_type": metadata.get("model_type", "auto"),
                "dataset_name": dataset_config.get("dataset_name", ""),
                "dataset_path": dataset_config.get("dataset_path", ""),
                "output_dir": config_data.get("output_dir", "results"),
                "track": metadata.get("track", "internal_reasoning"),
                "base_model_type": metadata.get("base_model_type", "API"),
                "base_model_name": metadata.get("base_model_name", ""),
                "additional_info": metadata.get("additional_info", ""),
                # TxAgent parameters from metadata
                "enable_checker": metadata.get("enable_checker", False),
                "step_rag_num": metadata.get("step_rag_num", 0),
                "temperature": metadata.get("temperature", 0.3),
                "summary_temperature": metadata.get("summary_temperature", 0.1),
                "max_new_tokens": metadata.get("max_new_tokens", 2048),
                "max_token": metadata.get("max_token", 65536),
                "max_round": metadata.get("max_round", 20),
                "multiagent": metadata.get("multiagent", False),
                "verbose": metadata.get("verbose", True),
                "save_mid_result": metadata.get("save_mid_result", False),
                # Gemini parameters from metadata
                "google_search_enabled": metadata.get("google_search_enabled", False),
                "api_key": metadata.get("api_key", None),
                "parallel_evaluation": metadata.get("parallel_evaluation", False),
                "max_workers": metadata.get("max_workers", 4)
            }
            
            # Debug: Print summary_temperature value
            print(f"\033[32m[DEBUG] eval_framework_v2.py - summary_temperature from config: {config['summary_temperature']}\033[0m")
            print(f"\033[32m[DEBUG] eval_framework_v2.py - google_search_enabled from config: {config['google_search_enabled']}\033[0m")
            print(f"\033[32m[DEBUG] eval_framework_v2.py - api_key from config: {config['api_key'] is not None}\033[0m")
            print(f"\033[32m[DEBUG] eval_framework_v2.py - parallel_evaluation from config: {config['parallel_evaluation']}\033[0m")
            print(f"\033[32m[DEBUG] eval_framework_v2.py - max_workers from config: {config['max_workers']}\033[0m")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")

    def _register_models(self):
        """Register all model types with the factory"""
        ModelFactory.register_model("chatgpt", ChatGPTModel)
        ModelFactory.register_model("local", LocalModel)
        ModelFactory.register_model("ollama", OllamaModel)
        ModelFactory.register_model("gemini", GeminiModel)
        ModelFactory.register_model("txagent", TxAgentModel)
        ModelFactory.register_model("llama", LlamaModel)
        ModelFactory.register_model("qwen", QwenModel)
        ModelFactory.register_model("medgemma", MedGemmaModel)
        ModelFactory.register_model("gpt_oss", GptOssModel)
        ModelFactory.register_model("baichuan", BaichuanModel)

    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation

        Args:
            model_name: Name/path of the model
            model_type: Type of model ("auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        
        # Auto-detect model type if needed
        if model_type == "auto":
            model_type = ModelFactory.auto_detect_type(model_name)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        # Create model using factory
        self.model = ModelFactory.create_model(model_name, model_type)
        
        # Prepare model parameters from config
        model_params = kwargs.copy()
        
        # Add TxAgent specific parameters from config if available
        if 'track' in self.config:
            model_params['track'] = self.config['track']
        if 'enable_checker' in self.config:
            model_params['enable_checker'] = self.config['enable_checker']
        if 'step_rag_num' in self.config:
            model_params['step_rag_num'] = self.config['step_rag_num']
        if 'init_rag_num' in self.config:
            model_params['init_rag_num'] = self.config['init_rag_num']
        if 'temperature' in self.config:
            model_params['temperature'] = self.config['temperature']
        if 'summary_temperature' in self.config:
            model_params['summary_temperature'] = self.config['summary_temperature']
        if 'max_new_tokens' in self.config:
            model_params['max_new_tokens'] = self.config['max_new_tokens']
        if 'max_token' in self.config:
            model_params['max_token'] = self.config['max_token']
        if 'max_round' in self.config:
            model_params['max_round'] = self.config['max_round']
        if 'multiagent' in self.config:
            model_params['multiagent'] = self.config['multiagent']
        if 'rag_model_name' in self.config:
            model_params['rag_model_name'] = self.config['rag_model_name']
        if 'enable_summary' in self.config:
            model_params['enable_summary'] = self.config['enable_summary']
        if 'avoid_repeat' in self.config:
            model_params['avoid_repeat'] = self.config['avoid_repeat']
        if 'seed' in self.config:
            model_params['seed'] = self.config['seed']
        if 'enable_finish' in self.config:
            model_params['enable_finish'] = self.config['enable_finish']
        if 'enable_rag' in self.config:
            model_params['enable_rag'] = self.config['enable_rag']
        if 'summary_mode' in self.config:
            model_params['summary_mode'] = self.config['summary_mode']
        if 'summary_skip_last_k' in self.config:
            model_params['summary_skip_last_k'] = self.config['summary_skip_last_k']
        if 'summary_context_length' in self.config:
            model_params['summary_context_length'] = self.config['summary_context_length']
        if 'force_finish' in self.config:
            model_params['force_finish'] = self.config['force_finish']
        if 'enable_chat' in self.config:
            model_params['enable_chat'] = self.config['enable_chat']
        if 'tool_files_dict' in self.config:
            model_params['tool_files_dict'] = self.config['tool_files_dict']
        if 'additional_default_tools' in self.config:
            model_params['additional_default_tools'] = self.config['additional_default_tools']
        
        # Add Gemini specific parameters from config if available
        if 'google_search_enabled' in self.config:
            model_params['google_search_enabled'] = self.config['google_search_enabled']
        if 'api_key' in self.config:
            model_params['api_key'] = self.config['api_key']
        
        # Debug: Print model_params before loading
        print(f"\033[34m[DEBUG] load_model - model_params['summary_temperature']: {model_params.get('summary_temperature', 'NOT_FOUND')}\033[0m")
        print(f"\033[34m[DEBUG] load_model - model_params['google_search_enabled']: {model_params.get('google_search_enabled', 'NOT_FOUND')}\033[0m")
        
        # Load the model with config parameters
        self.model.load(**model_params)
        
        # Create evaluator with verbose setting
        verbose = self.config.get('verbose', False)
        
        # Check if this is a Gemini model and if parallel evaluation is enabled
        print(f"\033[36m[DEBUG] Model type: {model_type}\033[0m")
        print(f"\033[36m[DEBUG] Google search enabled: {self.config.get('google_search_enabled', False)}\033[0m")
        print(f"\033[36m[DEBUG] Parallel evaluation enabled: {self.config.get('parallel_evaluation', False)}\033[0m")
        
        if (model_type == "gemini" and 
            self.config.get('google_search_enabled', False) and 
            self.config.get('parallel_evaluation', False)):
            # Use parallel evaluator for Gemini with Google Search
            max_workers = self.config.get('max_workers', 4)
            print(f"\033[32m[DEBUG] Creating ParallelEvaluator with {max_workers} workers\033[0m")
            self.evaluator = ParallelEvaluator(
                model_class=self.model.__class__,
                model_name=self.model_name,
                max_workers=max_workers,
                verbose=verbose
            )
            self.model_params = model_params  # Store for parallel evaluator
            print(f"\033[33m[INFO] ✅ Using parallel evaluation with {max_workers} workers for Gemini with Google Search\033[0m")
        else:
            # Use standard evaluator
            print(f"\033[33m[INFO] ⚠️ Using standard (single-threaded) evaluation\033[0m")
            self.evaluator = Evaluator(self.model, verbose=verbose)

    def evaluate(self, dataset_name: str) -> EvaluationMetrics:
        """
        Evaluate model on a dataset

        Args:
            dataset_name: Name of dataset to evaluate on

        Returns:
            EvaluationMetrics with results
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if not self.evaluator:
            raise ValueError("Evaluator not initialized. Call load_model() first.")

        logger.info(f"Evaluating on dataset: {dataset_name}")

        # Load dataset
        dataset = self._load_dataset(dataset_name)
        
        # Store dataset for later use in submission
        self._current_dataset = dataset

        # Run evaluation
        save_mid_result = self.config.get('save_mid_result', False)
        
        # Check if using parallel evaluator
        print(f"\033[36m[DEBUG] Evaluator type: {type(self.evaluator).__name__}\033[0m")
        if isinstance(self.evaluator, ParallelEvaluator):
            print(f"\033[32m[DEBUG] 🚀 Starting parallel evaluation with {self.evaluator.max_workers} workers\033[0m")
            # Use parallel evaluation with stored model parameters
            results = self.evaluator.evaluate_dataset(
                dataset, 
                dataset_name, 
                save_mid_result,
                **self.model_params
            )
        else:
            print(f"\033[33m[DEBUG] 🐌 Starting standard (single-threaded) evaluation\033[0m")
            # Use standard evaluation
            results = self.evaluator.evaluate_dataset(dataset, dataset_name, save_mid_result)

        return results

    def _load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load dataset based on configuration"""
        from torch.utils.data import DataLoader
        from dataset_utils import build_dataset
        
        # Get dataset path from config
        dataset_path = self.config.get("dataset_path")
        if not dataset_path:
            raise ValueError(f"No dataset path configured for dataset: {dataset_name}")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Build dataset
        dataset = build_dataset(dataset_path)
        
        # Convert to list of dictionaries for easier processing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []
        
        for batch in dataloader:
            question_type = batch[0][0]
            
            if question_type == "multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
            elif question_type == "open_ended_multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                    "meta_question": batch[4][0],
                })
            elif question_type == "open_ended":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
        
        logger.info(f"Loaded {len(dataset_list)} examples from dataset")
        return dataset_list

    def save_submission(self, results: EvaluationMetrics, output_dir, 
                       filename: str = "submission.csv") -> str:
        """
        Save results in competition submission format

        Args:
            results: Evaluation results
            output_dir: Output directory path
            filename: Output filename

        Returns:
            Path to saved submission file
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)
        
        # Use stored dataset if available
        dataset_examples = getattr(self, '_current_dataset', [])
        
        # Create CSV format
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'prediction', 'choice', 'reasoning'])
            
            for i, (prediction, reasoning) in enumerate(zip(results.predictions, results.reasoning_traces)):
                # Get example ID from dataset if available
                if i < len(dataset_examples):
                    example_id = dataset_examples[i].get("id", f"example_{i}")
                else:
                    example_id = f"example_{i}"
                
                # Clean up prediction text
                prediction_text = prediction.get("open_ended_answer", "") or ""
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"
                
                # Clean up choice
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ["NULL", "NONE", "NAN"]:
                    choice_clean = "NOTAVALUE"
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"
                else:
                    choice_clean = str(choice_raw).strip()
                
                
                writer.writerow([
                    str(example_id),
                    str(prediction_text),
                    str(choice_clean),
                    str(reasoning)
                ])
        
        logger.info(f"Submission saved to: {output_path}")
        
        # If save_mid_result is enabled, also save intermediate results
        if self.config.get('save_mid_result', False):
            dataset_name = self.config.get("dataset_name", "unknown")
            model_name = self.config.get("model_name", "unknown")
            mid_result_path = self.save_intermediate_results(results, dataset_name, model_name)
            logger.info(f"Intermediate results also saved to: {mid_result_path}")
        
        return output_path

    def _parse_tool_calls_from_message(self, content: str) -> tuple:
        """
        Parse [TOOL_CALLS] from message content
        
        Args:
            content: Message content that may contain [TOOL_CALLS]
            
        Returns:
            tuple: (message_before_tool_calls, tools_list)
        """
        import re
        import json
        
        # Pattern to match [TOOL_CALLS] and its content
        tool_calls_pattern = r'\[TOOL_CALLS\](.*?)(?=\[/TOOL_CALLS\]|$)'
        
        # Find all tool calls in the content
        tool_calls_matches = re.findall(tool_calls_pattern, content, re.DOTALL)
        
        if not tool_calls_matches:
            # No tool calls found, return original content and empty tools list
            return content, []
        
        # Extract the tools JSON from the first match
        tools_content = tool_calls_matches[0].strip()
        
        try:
            # Try to parse as JSON
            tools_list = json.loads(tools_content)
            # Ensure it's a list
            if not isinstance(tools_list, list):
                tools_list = [tools_list]
        except json.JSONDecodeError:
            # If not valid JSON, treat as string
            tools_list = [tools_content]
        
        # Remove [TOOL_CALLS] section from the original content
        message_before_tool_calls = re.sub(tool_calls_pattern, '', content, flags=re.DOTALL).strip()
        
        return message_before_tool_calls, tools_list

    def save_intermediate_results(self, results: EvaluationMetrics, 
                                dataset_name: str = "unknown",
                                model_name: str = "unknown") -> str:
        """
        Save intermediate results in JSONL format for dataset creation
        
        Args:
            results: Evaluation results
            dataset_name: Name of the dataset
            model_name: Name of the model
            
        Returns:
            Path to saved intermediate results file
        """
        output_dir = self.config.get("output_dir", "results")
        
        # Clean model name for filename (remove special characters)
        import re
        clean_model_name = re.sub(r'[^\w\-_.]', '_', model_name)
        
        # Create filename with format: reasoning_trace_{dataset_name}_{model_name}.jsonl
        filename = f"reasoning_trace_{dataset_name}_{clean_model_name}.jsonl"
        output_path = os.path.join(output_dir, filename)
        
        # Use stored dataset if available
        dataset_examples = getattr(self, '_current_dataset', [])
        
        # Create JSONL format with intermediate results
        with open(output_path, 'w', encoding='utf-8') as jsonlfile:
            for i, (prediction, reasoning_trace) in enumerate(zip(results.predictions, results.reasoning_traces)):
                # import pdb; pdb.set_trace()
                # Get example from dataset if available
                if i < len(dataset_examples):
                    example = dataset_examples[i]
                    question = example.get("question", "")
                    example_id = example.get("id", f"example_{i}")
                else:
                    question = "Unknown question"
                    example_id = f"example_{i}"
                
                # Format intermediate result
                intermediate_result = {
                    "tools": [],  # Default empty tools list
                    "messages": []
                }
                
                # Process reasoning trace to extract tools and store messages
                if isinstance(reasoning_trace, list):
                    for msg in reasoning_trace:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            role = msg['role']
                            content = msg['content']
                            
                            # Check if message has tool_calls field
                            if 'tool_calls' in msg:
                                try:
                                    tool_calls = json.loads(msg['tool_calls'])
                                    if isinstance(tool_calls, list):
                                        # Add tool calls to tools list
                                        for tool_call in tool_calls:
                                            intermediate_result["tools"].append(tool_call)
                                except:
                                    pass
                            
                            # Add the message to messages list
                            message_to_add = {
                                "role": role,
                                "content": content
                            }
                            
                            # If message has tool_calls, add it to the message
                            if 'tool_calls' in msg:
                                message_to_add["tool_calls"] = msg['tool_calls']
                            
                            intermediate_result["messages"].append(message_to_add)
                elif isinstance(reasoning_trace, str):
                    # If reasoning trace is a string, check for [TOOL_CALLS]
                    if '[TOOL_CALLS]' in reasoning_trace:
                        message_before_tool_calls, new_tools = self._parse_tool_calls_from_message(reasoning_trace)
                        
                        # Append new tools to existing tools list
                        if new_tools:
                            intermediate_result["tools"].extend(new_tools)
                        
                        # Add the message without tool calls
                        if message_before_tool_calls.strip():
                            intermediate_result["messages"].append({
                                "role": "assistant",
                                "content": message_before_tool_calls
                            })
                    else:
                        # If reasoning trace is a string, treat it as assistant content
                        intermediate_result["messages"].append({
                            "role": "assistant",
                            "content": reasoning_trace
                        })
                   
                # Keep tools as list, not JSON string
                output_result = {
                    "tools": intermediate_result["tools"],
                    "messages": intermediate_result["messages"]
                }
                
                # Write to JSONL file
                jsonlfile.write(json.dumps(output_result, ensure_ascii=False) + '\n')
        
        logger.info(f"Intermediate results saved to: {output_path}")
        return output_path

    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        dataset_name = self.config.get("dataset_name", "unknown")
        dataset_path = self.config.get("dataset_path", "unknown")
        print(f"  {dataset_name}: {dataset_path}")

