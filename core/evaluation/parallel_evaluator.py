"""
Parallel evaluator for Gemini model with Google Search support
"""
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from ..models.base import BaseModel
from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class ParallelEvaluator:
    """Parallel evaluator specifically for Gemini model with Google Search"""
    
    def __init__(self, model_class, model_name: str, max_workers: int = 4, verbose: bool = False):
        """
        Initialize parallel evaluator
        
        Args:
            model_class: The model class to use
            model_name: Name of the model
            max_workers: Maximum number of worker threads
            verbose: Whether to print detailed results
        """
        self.model_class = model_class
        self.model_name = model_name
        self.max_workers = max_workers
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.lock = Lock()
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], 
                        dataset_name: str = "unknown", 
                        save_mid_result: bool = False,
                        **model_kwargs) -> EvaluationMetrics:
        """
        Evaluate model on a dataset using parallel processing
        
        Args:
            dataset: List of dataset examples
            dataset_name: Name of the dataset
            model_kwargs: Additional model parameters
            
        Returns:
            EvaluationMetrics with results
        """
        self.logger.info(f"Starting parallel evaluation on {dataset_name} with {len(dataset)} examples using {self.max_workers} workers")
        print(f"\033[32m[DEBUG] 🔧 ParallelEvaluator initialized with {self.max_workers} workers\033[0m")
        print(f"\033[32m[DEBUG] 📊 Dataset size: {len(dataset)} examples\033[0m")
        print(f"\033[32m[DEBUG] 🧵 Thread pool will be created with max_workers={self.max_workers}\033[0m")
        
        # 使用字典来保持原始顺序
        results_by_index = {}
        accuracy_correct_count = 0
        accuracy_total_count = 0
        
        if self.verbose:
            print("\n" + "="*80)
            print("🔍 开始并行评估 - 详细结果输出")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("🔍 开始并行评估 - 仅显示工具调用信息")
            print("="*80)
        
        # 添加总计时
        total_start_time = time.time()
        
        # 准备任务参数，保持原始索引
        tasks = []
        for i, example in enumerate(dataset):
            tasks.append((i, example, model_kwargs))
        
        print(f"\033[32m[DEBUG] 📋 Prepared {len(tasks)} tasks with original indices\033[0m")
        
        # 使用线程池执行任务
        print(f"\033[32m[DEBUG] 🚀 Creating ThreadPoolExecutor with max_workers={self.max_workers}\033[0m")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            print(f"\033[32m[DEBUG] ✅ ThreadPoolExecutor created successfully\033[0m")
            # 提交所有任务，保持索引映射
            print(f"\033[32m[DEBUG] 📤 Submitting {len(tasks)} tasks to thread pool\033[0m")
            future_to_index = {}
            for task in tasks:
                future = executor.submit(self._evaluate_single_example_parallel, task)
                future_to_index[future] = task[0]  # 保存原始索引
            print(f"\033[32m[DEBUG] ✅ All {len(future_to_index)} tasks submitted successfully\033[0m")
            
            # 创建进度条
            with tqdm(total=len(dataset), desc="Evaluating", unit="example") as pbar:
                # 处理完成的任务
                completed_count = 0
                for future in as_completed(future_to_index):
                    try:
                        original_index = future_to_index[future]
                        print(f"\033[35m[DEBUG] 📥 Task completed: {completed_count + 1}/{len(future_to_index)} (original index: {original_index})\033[0m")
                        prediction, reasoning_trace, detailed_info = future.result()
                        
                        # 线程安全地存储结果，使用原始索引
                        with self.lock:
                            results_by_index[original_index] = {
                                'prediction': prediction,
                                'reasoning_trace': reasoning_trace,
                                'detailed_info': detailed_info
                            }
                            
                            # 更新准确率统计
                            is_correct = detailed_info["is_correct"]
                            question_type = detailed_info.get("question_type", "")
                            if question_type in ["multi_choice", "open_ended_multi_choice"]:
                                accuracy_total_count += 1
                                if is_correct:
                                    accuracy_correct_count += 1
                            
                            # 更新进度条
                            pbar.update(1)
                            completed_count += 1
                            
                            # 显示结果
                            if self.verbose:
                                self._print_example_result(detailed_info, completed_count, len(dataset))
                            else:
                                self._print_tool_calls(reasoning_trace, completed_count, len(dataset))
                            
                    except Exception as e:
                        original_index = future_to_index[future]
                        print(f"\033[31m[DEBUG] ❌ Error processing task {original_index}: {e}\033[0m")
                        self.logger.error(f"Error processing example {original_index}: {e}")
                        # 添加错误结果，使用原始索引
                        with self.lock:
                            error_prediction = {
                                "choice": "NOTAVALUE",
                                "open_ended_answer": "Error"
                            }
                            error_detailed_info = {
                                "id": f"error_{original_index}",
                                "question_type": "unknown",
                                "question": "Error",
                                "expected_answer": "",
                                "final_answer": "Error",
                                "is_correct": False
                            }
                            results_by_index[original_index] = {
                                'prediction': error_prediction,
                                'reasoning_trace': "Error occurred during inference",
                                'detailed_info': error_detailed_info
                            }
                            pbar.update(1)
        
        # 按原始顺序重新组织结果
        print(f"\033[32m[DEBUG] 🔄 Reordering results by original indices...\033[0m")
        predictions = []
        reasoning_traces = []
        detailed_results = []
        
        # 按原始索引顺序提取结果
        for i in range(len(dataset)):
            if i in results_by_index:
                result_data = results_by_index[i]
                predictions.append(result_data['prediction'])
                reasoning_traces.append(result_data['reasoning_trace'])
                detailed_results.append(result_data['detailed_info'])
                print(f"\033[32m[DEBUG] 📋 Reordered result {i+1}: {result_data['detailed_info']['id']}\033[0m")
            else:
                print(f"\033[31m[DEBUG] ❌ Missing result for index {i}\033[0m")
        
        # 重新计算准确率（按正确顺序）
        accuracy_correct_count = 0
        accuracy_total_count = 0
        for result in detailed_results:
            question_type = result.get("question_type", "")
            if question_type in ["multi_choice", "open_ended_multi_choice"]:
                accuracy_total_count += 1
                if result["is_correct"]:
                    accuracy_correct_count += 1
        
        # 计算最终准确率
        accuracy = (accuracy_correct_count / accuracy_total_count 
                   if accuracy_total_count > 0 else 0.0)
        
        # 计算总耗时
        total_time = time.time() - total_start_time
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces,
            dataset_name=dataset_name,
            model_name=self.model_name
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("📊 并行评估完成 - 总体统计")
        print("="*80)
        print(f"🎯 总测试样本数: {len(dataset)}（除open_ended以外样本数: {accuracy_total_count}）")
        print(f"✅ 正确答案数: {accuracy_correct_count}")
        print(f"❌ 错误答案数: {accuracy_total_count - accuracy_correct_count}")
        print(f"📈 准确率: {accuracy:.2%}")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        print(f"🚀 平均每样本耗时: {total_time/len(dataset):.2f} 秒")
        print(f"🔧 使用线程数: {self.max_workers}")
        
        # Print detailed results for each example
        print("\n" + "="*80)
        print("📋 详细结果分析")
        print("="*80)
        for i, result in enumerate(detailed_results):
            status = "✅ 正确" if result["is_correct"] else "❌ 错误"
            print(f"样本 {i+1:2d}: {status} | 期望: {result['expected_answer']:1s} | 预测: {result['final_answer']:1s} | 类型: {result['question_type']}")
        
        print(f"\033[32m[DEBUG] 🎯 Parallel evaluation completed successfully with {self.max_workers} workers\033[0m")
        print("="*80)
        
        # Debug: Print detailed analysis
        print(f"\033[36m[DEBUG] 📊 Detailed Analysis:\033[0m")
        for i, result in enumerate(detailed_results):
            print(f"\033[36m[DEBUG] Sample {i+1}: {result['id']} | Expected: {result['expected_answer']} | Predicted: {result['final_answer']} | Correct: {result['is_correct']}\033[0m")
        
        self.logger.info(f"Parallel evaluation completed: {accuracy:.2%} accuracy "
                        f"({accuracy_correct_count}/{accuracy_total_count})")
        
        return metrics
    
    def _evaluate_single_example_parallel(self, task: Tuple[int, Dict[str, Any], Dict]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Evaluate a single example in parallel
        
        Args:
            task: Tuple of (index, example, model_kwargs)
            
        Returns:
            Tuple of (prediction, reasoning_trace, detailed_info)
        """
        index, example, model_kwargs = task
        
        # 为每个线程创建独立的模型实例
        print(f"\033[36m[DEBUG] 🧵 Thread processing example {index + 1}\033[0m")
        model = self.model_class(self.model_name)
        print(f"\033[36m[DEBUG] 🔧 Creating independent model instance for thread\033[0m")
        model.load(**model_kwargs)
        print(f"\033[36m[DEBUG] Model instance loaded successfully in thread\033[0m")
        
        question = example["question"]
        question_type = example["question_type"]
        expected_answer = example.get("answer", "")
        
        # Format prompt based on question type
        prompt = self._format_prompt(question, question_type, example)
        
        # Get model response
        print(f"\033[36m[DEBUG] 🤖 Running inference in thread for example {index + 1}\033[0m")
        response, reasoning_trace = model.inference(prompt)
        print(f"\033[36m[DEBUG] Inference completed in thread for example {index + 1}\033[0m")
        
        # Extract prediction from response
        prediction, reasoning_trace = self._extract_prediction(response, question_type, example, reasoning_trace)
        
        # Debug: Print detailed prediction information
        print(f"\033[33m[DEBUG] 📊 Example {index + 1} Prediction Details:\033[0m")
        print(f"\033[33m[DEBUG]   - Question: {question[:100]}{'...' if len(question) > 100 else ''}\033[0m")
        print(f"\033[33m[DEBUG]   - Question Type: {question_type}\033[0m")
        print(f"\033[33m[DEBUG]   - Expected Answer: {expected_answer}\033[0m")
        print(f"\033[33m[DEBUG]   - Predicted Choice: {prediction.get('choice', 'NOTAVALUE')}\033[0m")
        print(f"\033[33m[DEBUG]   - Response Length: {len(response)} chars\033[0m")
        print(f"\033[33m[DEBUG]   - Response Preview: {response[:200]}{'...' if len(response) > 200 else ''}\033[0m")
        
        # Create detailed info for output
        is_correct = self._check_correctness(example, prediction)
        
        # Debug: Print correctness check details
        print(f"\033[33m[DEBUG] 🎯 Correctness Check for Example {index + 1}:\033[0m")
        print(f"\033[33m[DEBUG]   - Expected: '{expected_answer}'\033[0m")
        print(f"\033[33m[DEBUG]   - Predicted: '{prediction.get('choice', 'NOTAVALUE')}'\033[0m")
        print(f"\033[33m[DEBUG]   - Match: {expected_answer.upper() == prediction.get('choice', '').upper()}\033[0m")
        print(f"\033[33m[DEBUG]   - Is Correct: {is_correct}\033[0m")
        
        detailed_info = {
            "id": example.get("id", "unknown"),
            "question_type": question_type,
            "question": question,
            "expected_answer": expected_answer,
            "final_answer": prediction.get("choice", ""),
            "is_correct": is_correct,
            "reasoning_trace": reasoning_trace
        }
        
        return prediction, reasoning_trace, detailed_info
    
    def _format_prompt(self, question: str, question_type: str, example: Dict[str, Any] = None) -> str:
        """Format prompt based on question type"""
        if question_type == "multi_choice":
            # 检查是否有选项信息
            options = example.get("options", {}) if example else {}
            if options:
                # 使用原始代码的高性能prompt格式
                options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
                return f"""请回答以下医学问题。请仔细分析问题，并从给定选项中选择最佳答案。

问题: {question}

选项:
{options_text}

请提供详细的推理过程，然后在最后明确给出答案，格式为: The final answer is $\\boxed{{X}}$，其中X是选项字母。"""
            else:
                # 备用格式
                return (f"The following is a multiple choice question about medicine. "
                       f"Answer with only the letter (A, B, C, D, or E).\n\n"
                       f"Question: {question}\n\nAnswer:")
        elif question_type in ["open_ended_multi_choice", "open_ended"]:
            return (f'''请回答以下医学问题。请提供详细的分析和推理过程。

问题: {question}

请提供详细的推理过程，然后在最后明确给出答案，格式为: The final answer is $\\boxed{{答案}}$''')
        else:
            return f"Question: {question}\n\nAnswer:"
    
    def _extract_prediction(self, response: str, question_type: str, 
                           example: Dict[str, Any], reasoning_trace: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract prediction from model response"""
        import re
        
        print(f"\033[35m[DEBUG] 🔍 Extracting prediction from response:\033[0m")
        print(f"\033[35m[DEBUG]   - Question Type: {question_type}\033[0m")
        print(f"\033[35m[DEBUG]   - Response: {response[:300]}{'...' if len(response) > 300 else ''}\033[0m")
        
        if question_type == "multi_choice":
            # Extract choice from response - 匹配原始代码的高性能格式
            choice_patterns = [
                r'The final answer is \$\\boxed\{([A-E])\}\$',  # 原始代码的格式
                r'\\boxed\{([A-E])\}',  # 简化格式
                r'最终答案[:：]?\s*([A-E])',
                r'答案[:：]?\s*([A-E])',
                r'选择[:：]?\s*([A-E])',
                r'The final answer is.*?([A-E])',
                r'Answer:\s*([A-E])',
                r'\b([A-E])\b'  # 最后尝试简单匹配
            ]
            
            choice = "A"  # Default choice
            matched_pattern = None
            
            print(f"\033[35m[DEBUG]   - Trying {len(choice_patterns)} patterns to extract choice...\033[0m")
            
            for i, pattern in enumerate(choice_patterns):
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    choice = match.group(1).upper()
                    matched_pattern = pattern
                    print(f"\033[35m[DEBUG]   - ✅ Pattern {i+1} matched: '{pattern}' -> '{choice}'\033[0m")
                    break
                else:
                    print(f"\033[35m[DEBUG]   - ❌ Pattern {i+1} failed: '{pattern}'\033[0m")
            
            if matched_pattern is None:
                print(f"\033[35m[DEBUG]   - ⚠️ No pattern matched, using default choice 'A'\033[0m")
            
            print(f"\033[35m[DEBUG]   - Final extracted choice: '{choice}'\033[0m")
            
            return {
                "choice": choice,
                "open_ended_answer": response.strip()
            }, reasoning_trace
            
        elif question_type in ["open_ended_multi_choice", "open_ended"]:
            # For open-ended questions, extract the answer
            return {
                "choice": "NOTAVALUE",
                "open_ended_answer": response.strip()
            }, reasoning_trace
        
        else:
            return {
                "choice": "NOTAVALUE",
                "open_ended_answer": response.strip()
            }, reasoning_trace
    
    def _check_correctness(self, example: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
        """Check if prediction is correct"""
        expected_answer = example.get("answer", "")
        predicted_choice = prediction.get("choice", "")
        
        if not expected_answer or not predicted_choice:
            return False
        
        return predicted_choice.upper() == expected_answer.upper()
    
    def _print_example_result(self, detailed_info: Dict[str, Any], example_num: int, total_examples: int):
        """Print detailed result for a single example"""
        print(f"\n📝 测试样本 {example_num}/{total_examples}")
        print("-" * 60)
        print(f"🆔 ID: {detailed_info['id']}")
        print(f"📋 问题类型: {detailed_info['question_type']}")
        print(f"❓ 问题: {detailed_info['question'][:100]}{'...' if len(detailed_info['question']) > 100 else ''}")
        print(f"✅ 期望答案: {detailed_info['expected_answer']}")
        print(f"🤖 模型答案: {detailed_info['final_answer']}")
        
        # Show correctness with emoji
        if detailed_info['is_correct']:
            print(f"🎯 结果: ✅ 正确")
        else:
            print(f"🎯 结果: ❌ 错误")
        
        print("=" * 60)
    
    def _print_tool_calls(self, reasoning_trace, example_num: int, total_examples: int):
        """Print only tool call information for a single example"""
        print(f"\n🔧 测试样本 {example_num}/{total_examples}")
        print("-" * 40)
        
        # Extract tool calls from reasoning trace
        tool_calls = self._extract_tool_calls(reasoning_trace)
        
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                print(f"Tool Call {i+1}: {tool_call}")
        else:
            print("No tool calls detected")
        
        print("-" * 40)
    
    def _extract_tool_calls(self, reasoning_trace) -> List[Dict]:
        """Extract tool calls from reasoning trace"""
        tool_calls = []
        
        if isinstance(reasoning_trace, list):
            for msg in reasoning_trace:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    # Look for tool call patterns in content
                    tool_calls.extend(self._parse_tool_calls_from_text(content))
        
        return tool_calls
    
    def _parse_tool_calls_from_text(self, text: str) -> List[Dict]:
        """Parse tool calls from text content"""
        import re
        
        tool_calls = []
        
        # Look for Google Search patterns
        search_patterns = [
            r"🔍.*?Search.*?",
            r"📝.*?Search queries.*?",
            r"🌐.*?Search pages.*?",
            r"Google Search.*?",
            r"search.*?results.*?"
        ]
        
        for pattern in search_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                tool_calls.append({
                    "type": "google_search",
                    "content": match.strip()
                })
        
        return tool_calls
