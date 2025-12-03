"""
Evaluator for running model evaluations on datasets
"""
import logging
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

from ..models.base import BaseModel
from .metrics import EvaluationMetrics

import re

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator class for running model evaluations"""
    
    def __init__(self, model: BaseModel, verbose: bool = False):
        """
        Initialize evaluator with a model
        
        Args:
            model: The model to evaluate
            verbose: Whether to print detailed results for each example
        """
        self.model = model
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], 
                        dataset_name: str = "unknown", 
                        save_mid_result: bool = False) -> EvaluationMetrics:
        """
        Evaluate model on a dataset
        
        Args:
            dataset: List of dataset examples
            dataset_name: Name of the dataset
            
        Returns:
            EvaluationMetrics with results
        """
        self.logger.info(f"Starting evaluation on {dataset_name} with {len(dataset)} examples")
        
        predictions = []
        reasoning_traces = []
        detailed_results = []
        accuracy_correct_count = 0
        accuracy_total_count = 0
        
        if self.verbose:
            print("\n" + "="*80)
            print("🔍 Start evaluation - Detailed results output")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("🔍 Start evaluation - Only show tool call information")
            print("="*80)
        
        # Add total timing
        import time
        total_start_time = time.time()
        
        for i, example in enumerate(tqdm(dataset, desc="Evaluating", unit="example")):
            try:
                # Record single sample start time
                sample_start_time = time.time()
                
                prediction, reasoning_trace, detailed_info = self._evaluate_single_example(example)
                # import pdb; pdb.set_trace()
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)
                detailed_results.append(detailed_info)
                
                # Check if correct based on question type
                is_correct = detailed_info["is_correct"]
                
                question_type = example.get("question_type", "")
                if question_type in ["multi_choice", "open_ended_multi_choice"]:
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                
                # Calculate single sample elapsed time
                sample_time = time.time() - sample_start_time
                
                # Print result based on verbose mode - Use tqdm.write to avoid overriding progress bar
                if self.verbose:
                    self._print_example_result(detailed_info, i + 1, len(dataset), sample_time)
                else:
                    self._print_tool_calls(reasoning_trace, i + 1, len(dataset), sample_time)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    current_acc = (accuracy_correct_count / accuracy_total_count 
                                 if accuracy_total_count > 0 else 0.0)
                    self.logger.info(f"Progress: {i + 1}/{len(dataset)}, "
                                   f"Accuracy: {current_acc:.2%}")
                    
            except Exception as e:
                self.logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",
                    "open_ended_answer": "Error"
                }
                predictions.append(error_prediction)
                reasoning_traces.append("Error occurred during inference")
                
                # Add error info to detailed results
                detailed_results.append({
                    "id": example.get("id", f"example_{i}"),
                    "question_type": example.get("question_type", "unknown"),
                    "question": example.get("question", "Error"),
                    "expected_answer": example.get("answer", ""),
                    "final_answer": "Error",
                    "is_correct": False
                })
        
        # Calculate final accuracy
        accuracy = (accuracy_correct_count / accuracy_total_count 
                   if accuracy_total_count > 0 else 0.0)
        
        # Calculate total elapsed time
        total_time = time.time() - total_start_time
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces,
            dataset_name=dataset_name,
            model_name=self.model.model_name
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("📊 Evaluation Complete - Overall Statistics")
        print("="*80)
        print(f"🎯 Total test samples: {len(dataset)} (excluding open_ended samples: {accuracy_total_count})")
        print(f"✅ Number of correct answers: {accuracy_correct_count}")
        print(f"❌ Number of incorrect answers: {accuracy_total_count - accuracy_correct_count}")
        print(f"📈 Accuracy: {accuracy:.2%}")
        print(f"⏱️  Total elapsed time: {total_time:.2f} seconds")
        print(f"🚀 Average time per sample: {total_time/len(dataset):.2f} seconds")
        print("="*80)
        
        self.logger.info(f"Evaluation completed: {accuracy:.2%} accuracy "
                        f"({accuracy_correct_count}/{accuracy_total_count})")
        
        return metrics
    
    def _print_example_result(self, detailed_info: Dict[str, Any], example_num: int, total_examples: int, sample_time: float = 0.0):
        """Print detailed result for a single example"""
        from tqdm import tqdm
        tqdm.write(f"\n📝 Test sample {example_num}/{total_examples} (Time: {sample_time:.2f}s)")
        tqdm.write("-" * 60)
        tqdm.write(f"🆔 ID: {detailed_info['id']}")
        tqdm.write(f"📋 Question type: {detailed_info['question_type']}")
        tqdm.write(f"❓ Question: {detailed_info['question'][:100]}{'...' if len(detailed_info['question']) > 100 else ''}")
        tqdm.write(f"✅ Expected answer: {detailed_info['expected_answer']}")
        tqdm.write(f"🤖 Model answer: {detailed_info['final_answer']}")
        
        # Show correctness with emoji
        if detailed_info['is_correct']:
            tqdm.write(f"🎯 Result: ✅ Correct")
        else:
            tqdm.write(f"🎯 Result: ❌ Incorrect")
        
        tqdm.write("=" * 60)
    
    def _print_tool_calls(self, reasoning_trace, example_num: int, total_examples: int, sample_time: float = 0.0):
        """Print only tool call information for a single example"""
        from tqdm import tqdm
        tqdm.write(f"\n🔧 Test sample {example_num}/{total_examples} (Time: {sample_time:.2f}s)")
        tqdm.write("-" * 40)
        
        # Extract tool calls from reasoning trace
        tool_calls = self._extract_tool_calls(reasoning_trace)
        
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                tqdm.write(f"Tool Call {i+1}: {tool_call}")
        else:
            tqdm.write("No tool calls detected")
        
        tqdm.write("-" * 40)
    
    def _extract_tool_calls(self, reasoning_trace) -> List[Dict]:
        """Extract tool calls from reasoning trace"""
        tool_calls = []
        
        if isinstance(reasoning_trace, list):
            for msg in reasoning_trace:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    # Look for tool call patterns in content
                    tool_calls.extend(self._parse_tool_calls_from_text(content))
        elif isinstance(reasoning_trace, str):
            tool_calls.extend(self._parse_tool_calls_from_text(reasoning_trace))
        
        return tool_calls
    
    def _parse_tool_calls_from_text(self, text: str) -> List[Dict]:
        """Parse tool calls from text content"""
        tool_calls = []
        
        # Common patterns for tool calls
        import re
        
        # Pattern 1: Tool Call: {'name': 'xxx', 'arguments': {...}}
        pattern1 = r'Tool Call:\s*(\{[^}]+\})'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        
        # Pattern 2: Function call patterns
        pattern2 = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches2 = re.findall(pattern2, text)
        
        # Pattern 3: JSON-like tool calls
        pattern3 = r'\{[^}]*"name"[^}]*"arguments"[^}]*\}'
        matches3 = re.findall(pattern3, text)
        
        for match in matches1 + matches3:
            try:
                import json
                tool_call = json.loads(match)
                if 'name' in tool_call:
                    tool_calls.append(tool_call)
            except:
                # If not valid JSON, treat as raw text
                tool_calls.append({'raw_text': match})
        
        for name, args in matches2:
            if name.lower() in ['search', 'calculate', 'lookup', 'query', 'get', 'find']:
                tool_calls.append({'name': name, 'arguments': args})
        
        return tool_calls
    
    def _evaluate_single_example(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Evaluate a single example
        
        Args:
            example: Single dataset example
            
        Returns:
            Tuple of (prediction, reasoning_trace, detailed_info)
        """
        question = example["question"]
        question_type = example["question_type"]
        expected_answer = example.get("answer", "")
        
        # Format prompt based on question type
        prompt = self._format_prompt(question, question_type)
        
        # Get model response
        response, reasoning_trace = self.model.inference(prompt)
        
        # Extract prediction from response
        prediction, reasoning_trace = self._extract_prediction(response, question_type, example, reasoning_trace)
        
        # Create detailed info for output
        detailed_info = {
            "id": example.get("id", "unknown"),
            "question_type": question_type,
            "question": question,
            "expected_answer": expected_answer,
            "final_answer": prediction.get("choice", ""),
            "is_correct": self._check_correctness(example, prediction),
            "reasoning_trace": reasoning_trace  # Add reasoning trace to detailed info
        }
        
        return prediction, reasoning_trace, detailed_info
    
    def _format_prompt(self, question: str, question_type: str) -> str:
        """Format prompt based on question type"""
        if question_type in ["multi_choice", "open_ended_multi_choice"]:
            return (f"The following is a multiple choice question about medicine. "
                   f"Answer with only the letter (A, B, C, D, or E).\n\n"
                   f"Question: {question}\n\nAnswer:")
        elif question_type in ["open_ended"]:
            return (f"The following is an open-ended question about medicine. "
                   f"Provide a comprehensive answer.\n\n"
                   f"Question: {question}\n\nAnswer:")
        else:
            return f"Question: {question}\n\nAnswer:"
    
    def _extract_prediction(self, response: str, question_type: str, 
                           example: Dict[str, Any], reasoning_trace: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract prediction from model response"""
        prediction = {
            "choice": "",
            "open_ended_answer": ""
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Handle meta question if available
            if "meta_question" in example:
                meta_prompt = (f"{example['meta_question']}Agent's answer: {response.strip()}\n\n"
                             f"Multi-choice answer:")
                # import pdb; pdb.set_trace()
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                reasoning_trace += meta_reasoning
                choice = self._extract_multiple_choice_answer(meta_response)
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                choice = self._extract_multiple_choice_answer(response)
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
                
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        return prediction, reasoning_trace
    
    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response (English only)."""
        if not response:
            return ""
            
        response = response.strip().upper()
        
        # Look for letter at the beginning
        import re
        m = re.match(r"^\s*([ABCD])(?:[\)\.\:\-]\s|\s*$)", response)
        if m:
            return m.group(1)
        
        patterns = [
            # NEW: align/match/correspond patterns
            r"(?:ALIGNS?\s+WITH|MATCH(?:ES)?|CORRESPONDS?\s+TO)\s+(?:OPTION\s*)?([ABCD])\b",
            
            # Original rules (Step 1 refinement: drop bare '|is')
            r"(?:ANSWER IS|ANSWER:)\s*([ABCD])",
            r"([ABCD])\)",  
            r"\b([ABCD])\b"
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, response))
            if matches:
                return matches[0].group(1)  # take the last occurrence
        
        return ""


    def _extract_multiple_choice_answer(self, prediction: str) -> str:
        """
        Extract A/B/C/D after "Answer" from the prediction text.
        Combine the extraction logic in evaluator.py to handle various formats.
        
        Args:
            prediction (str): Prediction text containing Answer.
            
        Returns:
            str: Extracted choice (A/B/C/D) or the original choice if unable to extract.
        """
        if not prediction:
            return ""
                    
        # Convert to uppercase for processing
        response = prediction.strip().upper()
        
        # Method 1: Check if letter appears at the beginning (from evaluator.py)
        m = re.match(r"^\s*([ABCD])(?:[\)\.\:\-]\s|\s*$)", response)
        if m:
            return m.group(1)
        
        # Method 2: Find content after "Answer:" (priority for example format)
        answer_pattern = r'Answer:\s*\(?([A-D])\)?'
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1)
        
        # Method 3: Find content after "Answer" (no colon)
        answer_pattern2 = r'Answer\s+\(?([A-D])\)?'
        match = re.search(answer_pattern2, response)
        if match:
            return match.group(1)
        
        # Method 4: Find content after [FinalAnswer]
        final_answer_pattern = r'\[FinalAnswer\]\s+([A-D])'
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1)
        
        # Method 5: Find a line starting with the letter, e.g., "D: No treatment..." (handle example format 2)
        line_pattern = r'^([A-D]):\s*'
        match = re.search(line_pattern, response, re.MULTILINE)
        if match:
            return match.group(1)
        
        # Methods 6-9: Use patterns from evaluator.py (priority order)
        patterns = [
            # NEW: align/match/correspond patterns
            r"(?:ALIGNS?\s+WITH|MATCH(?:ES)?|CORRESPONDS?\s+TO)\s+(?:OPTION\s*)?([ABCD])\b",
            # Original rules
            r"(?:ANSWER IS|ANSWER:)\s*([ABCD])",
            r"([ABCD])\)",  
            r"\b([ABCD])\b"
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, response))
            if matches:
                return matches[-1].group(1)  # Take the last match (same as evaluator.py)
    
        return None



    def _check_correctness(self, example: Dict[str, Any], 
                          prediction: Dict[str, Any]) -> bool:
        """Check if prediction is correct"""
        expected_answer = example.get("answer", "")
        question_type = example.get("question_type", "")
        
        if question_type in ["multi_choice", "open_ended_multi_choice"]:
            if expected_answer:
                return prediction["choice"] == expected_answer
            return False
        elif question_type == "open_ended":
            if expected_answer:
                return prediction["open_ended_answer"] == expected_answer
            return False
        
        return False 