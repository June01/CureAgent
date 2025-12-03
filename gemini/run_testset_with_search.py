#!/usr/bin/env python3
"""
CUREBench Testset Multiprocessing Evaluation Script - Using Gemini Search
Specialized for testset evaluation and Kaggle submission file generation.
No ground truth for the testset; only prediction results CSV is generated.
"""

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Tuple

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_gemini_model_with_search(model_name: str, enable_search: bool, api_key: str):
    """Create a Gemini model instance for each worker process (using Google Search)."""
    try:
        from eval_framework import GeminiModel

        model = GeminiModel(
            model_name=model_name,
            api_key=api_key,
            google_search_enabled=enable_search,
        )
        model.load()
        return model
    except Exception as e:
        print(f"Failed to create model: {e}")
        return None


def extract_answer_from_response(response_text: str, options: Dict) -> str:
    """
    - If `options` are provided (multiple-choice), returns a valid option letter (A/B/C/D).
    - If `options` are not provided (open-ended), tries to return the final option letter, or returns truncated original text if extraction fails.
    """
    if not isinstance(response_text, str) or response_text.strip() == "":
        # Multiple-choice: return the first option; Open-ended: return empty string
        if options:
            option_keys = list(options.keys())
            return option_keys[0] if option_keys else "B"
        return ""

    response = response_text.strip()

    # Robust extraction patterns based on CURE-Bench `extract_choice_from_prediction`
    base_patterns = [
        # 1. LaTeX \boxed{A} style (most explicit)
        r"(?i)answer is.*?\\boxed\{([A-D])\}",
        r"(?i)\\boxed\{([A-D])\}",
        # 2. Phrases like "final answer is", "answer is", or "answer:"
        r"(?i)(?:the\s+)?final\s+answer\s+is\s*:?\s*\(?([A-D])\)?\b",
        r"(?i)answer\s+is\s*:?\s*\(?([A-D])\)?\b",
        r"(?i)answer\s*:\s*\(?([A-D])\)?\b",
        # 3. Statements like "corresponds to option A"
        r"(?i)(?:aligns\s+with|matches|corresponds\s+to)\s+(?:option\s+)?([A-D])\b",
        # 4. Formats like "A)" or "[B]"
        r"\[([A-D])\]",
        r"\b([A-D])\)",
        # 5. Fallback: single options as standalone letters
        r"\b([A-D])\b",
    ]

    patterns = base_patterns

    # Multiple-choice: the extracted letter must be in the provided options
    if options:
        option_keys = list(options.keys())

        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.MULTILINE))
            if matches:
                # Use the last match to avoid interference from earlier references
                choice = matches[-1].group(1).upper()
                if choice in option_keys:
                    return choice

        # If still not found, fall back to the first option
        return option_keys[0] if option_keys else "B"

    # Open-ended: try to extract a final option letter, else return truncated text
    for pattern in patterns:
        matches = list(re.finditer(pattern, response, re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    # If no clear answer found, return the first 50 characters of text as a fallback
    return response[:50].strip()

def process_question_with_search(args):
    """Process a single question (using Google Search)"""
    (
        question_data,
        worker_id,
        question_idx,
        total_questions,
        model_name,
        enable_search,
        api_key,
    ) = args

    try:
        # Create model instance for each worker process
        model = create_gemini_model_with_search(
            model_name=model_name,
            enable_search=enable_search,
            api_key=api_key,
        )
        if model is None:
            return {
                "question_id": question_data.get("id", "unknown"),
                "success": False,
                "error": "Failed to create model",
                "predicted_answer": None,
                "choice": "NOTAVALUE",
                "open_ended_answer": "Error: Failed to create model",
                "processing_time": 0,
                "response_text": "",
                "question_type": question_data.get("question_type", "unknown"),
            }

        start_time = time.time()

        # Build prompt
        question_text = question_data.get("question", "")
        options = question_data.get("options", {})

        if options:
            # Multiple-choice question
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            prompt = f"""Please answer the following medical question. Carefully analyze the question and select the best answer from the given options.

Question: {question_text}

Options:
{options_text}

Please provide detailed reasoning, then clearly state the answer at the end, in the format: The final answer is $\\boxed{{X}}$, where X is the option letter."""
        else:
            # Open-ended question
            prompt = f"""Please answer the following medical question. Please provide detailed analysis and reasoning.

Question: {question_text}

Please provide detailed reasoning, then clearly state the answer at the end, in the format: The final answer is $\\boxed{{answer}}$."""
        # if question_data.get("question_type", "unknown") == "open_ended":
        #     import pdb
        #     pdb.set_trace()
        # Call model
        response_result = model.inference(prompt)
        processing_time = time.time() - start_time

        # Handle response result (may be string or tuple)
        if isinstance(response_result, tuple):
            response = response_result[0]  # Use the first element as response text
        else:
            response = response_result

        # Extract answer
        predicted_answer = extract_answer_from_response(response, options)

        # Set choice and open_ended_answer according to question type
        question_type = question_data.get("question_type", "unknown")

        if question_type == "multi_choice":
            choice = predicted_answer
            open_ended_answer = response.strip()
        elif question_type == "open_ended_multi_choice":
            choice = predicted_answer
            open_ended_answer = response.strip()
        elif question_type == "open_ended":
            choice = "NOTAVALUE"  # No choices for open-ended questions
            open_ended_answer = response.strip()
        else:
            choice = predicted_answer if options else "NOTAVALUE"
            open_ended_answer = response.strip()

        # Record detailed info
        question_id = question_data.get("id", "unknown")

        # Calculate progress percent
        progress_percent = ((question_idx + 1) / total_questions) * 100

        # Print processing result and progress
        print(
            f"✅ Worker {worker_id} processing {question_id} [{question_idx + 1}/{total_questions}] ({progress_percent:.1f}%)"
        )
        print(
            f"   Type: {question_type} | Choice: {choice} | Open-ended answer length: {len(open_ended_answer)}"
        )
        print(f"   Time taken: {processing_time:.2f}s")
        print(f"   Response preview: {response[:100]}...")
        print("-" * 80)

        return {
            "question_id": question_id,
            "success": True,
            "error": None,
            "predicted_answer": predicted_answer,
            "choice": choice,
            "open_ended_answer": open_ended_answer,
            "processing_time": processing_time,
            "response_text": response,
            "question_type": question_type,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Worker {worker_id} failed to process question: {error_msg}")

        return {
            "question_id": question_data.get("id", "unknown"),
            "success": False,
            "error": error_msg,
            "predicted_answer": None,
            "choice": "NOTAVALUE",
            "open_ended_answer": f"Error: {error_msg}",
            "processing_time": 0,
            "response_text": "",
            "question_type": question_data.get("question_type", "unknown"),
        }


def setup_logging(log_file: str):
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_dataset(dataset_path: str, subset_size: int = None) -> List[Dict]:
    """Load dataset"""
    data = []

    if dataset_path.endswith(".jsonl"):
        # Handle JSONL format
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        # Handle JSON format
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if subset_size:
        data = data[:subset_size]

    return data


def calculate_processing_stats(results: List[Dict]) -> Dict:
    """Calculate processing statistics (excluding accuracy, as no ground truth in test set)"""
    total_questions = len(results)
    successful_questions = [r for r in results if r["success"]]

    # Group by question type
    type_stats = {}
    for result in successful_questions:
        q_type = result["question_type"]
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "processed": 0}
        type_stats[q_type]["total"] += 1
        if result["success"]:
            type_stats[q_type]["processed"] += 1

    # Calculate average processing time
    avg_time = (
        sum(r["processing_time"] for r in successful_questions)
        / len(successful_questions)
        if successful_questions
        else 0
    )

    return {
        "total_questions": total_questions,
        "successful_questions": len(successful_questions),
        "average_processing_time": avg_time,
        "type_statistics": type_stats,
        "error_count": total_questions - len(successful_questions),
    }


def save_results(results: List[Dict], stats: Dict, output_dir: str, timestamp: str):
    """Save results"""
    # Save detailed results
    results_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save statistical summary
    stats_file = os.path.join(output_dir, f"processing_stats_{timestamp}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return results_file, stats_file


def create_kaggle_submission_csv(results: List[Dict], output_dir: str, timestamp: str):
    """Directly create Kaggle submission CSV file"""
    try:
        import zipfile

        import pandas as pd

        # Prepare metadata
        metadata = {
            "model_name": "gemini-2.5-flash",
            "model_type": "GeminiModel",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": "gemini-2.5-flash",
            "dataset": "curebench_testset_phase1",
            "additional_info": "Test set submission with Google Search enabled",
            "timestamp": timestamp,
            "total_questions": len(results),
            "successful_questions": len([r for r in results if r["success"]]),
        }

        # Prepare CSV data
        csv_data = []

        for result in results:
            # Ensure choice and open_ended_answer are not empty
            choice = result.get("choice", "NOTAVALUE")
            if not choice or choice.strip() == "":
                choice = "NOTAVALUE"

            open_ended_answer = result.get("open_ended_answer", "")
            if not open_ended_answer or open_ended_answer.strip() == "":
                open_ended_answer = "No answer provided"

            reasoning = result.get("response_text", "")
            if not reasoning or reasoning.strip() == "":
                reasoning = "No reasoning available"

            # Create CSV row
            row = {
                "id": str(result["question_id"]),
                "prediction": str(open_ended_answer),
                "choice": str(choice),
                "reasoning": str(reasoning),
            }
            csv_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(csv_data)

        # Ensure no null values
        df = df.fillna("NOTAVALUE")

        # Save CSV file
        csv_filename = f"testset_submission_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False, quoting=1)

        # Create metadata file
        metadata_filename = f"metadata_{timestamp}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Create ZIP file
        zip_filename = f"testset_submission_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, csv_filename)
            zipf.write(metadata_path, "meta_data.json")

        print(f"✅ CSV file created: {csv_path}")
        print(f"✅ ZIP submission file created: {zip_path}")

        return zip_path

    except Exception as e:
        print(f"Failed to create Kaggle submission file: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main function"""
    # Try to load config from file
    config_file = "gemini_with_search_config.json"
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ Loaded settings from config file: {config_file}")

    # Read model-related settings from config
    model_name = config.get("model_name", "gemini-2.5-flash")
    enable_search = config.get("google_search_enabled", False)
    api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")

    # Show current config
    evaluation_mode = "Full Evaluation" if config.get("full_evaluation", True) else "Test Evaluation"
    dataset_size = (
        "All"
        if config.get("full_evaluation", True)
        else str(config.get("test_subset_size", 20))
    )
    print(f"📋 Evaluation Mode: {evaluation_mode}")
    print(f"📊 Dataset Size: {dataset_size} questions")
    print(f"⚡ Number of workers: {config.get('num_workers', 8)}")
    print(f"🔍 Google Search: {'Enabled' if enable_search else 'Disabled'}")
    print(f"📝 Output Format: CSV (Kaggle Submission)")
    print()

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Set timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    log_file = os.path.join(
        config["output_dir"], f"testset_with_search_{timestamp}.log"
    )
    setup_logging(log_file)

    logging.info("🚀 Starting test set multiprocessing evaluation (using Google Search)")
    logging.info(f"📁 Results will be saved to: {config['output_dir']}")
    logging.info(f"📋 Log file: {log_file}")

    # Load dataset
    logging.info("📂 Loading test set...")
    subset_size = None if config["full_evaluation"] else config["test_subset_size"]
    dataset = load_dataset(config["dataset_path"], subset_size)
    logging.info(f"📋 Loaded {len(dataset)} questions")

    # Prepare multiprocessing arguments
    total_questions = len(dataset)
    worker_args = [
        (
            question,
            i % config["num_workers"],
            i,
            total_questions,
            model_name,
            enable_search,
            api_key,
        )
        for i, question in enumerate(dataset)
    ]

    # Start multiprocessing evaluation
    logging.info(f"⚡ Starting evaluation with {config['num_workers']} worker processes...")
    logging.info(f"📊 Total number of questions: {total_questions}")
    start_time = time.time()

    # Start progress monitor thread
    def progress_monitor():
        while True:
            time.sleep(30)  # Update progress every 30 seconds
            current_time = time.time()
            elapsed = current_time - start_time

            # Estimate progress (based on time)
            if elapsed > 0:
                # Assume each question takes 15 seconds on average (based on previous observation)
                estimated_completed = min(int(elapsed / 15), total_questions)
                progress_percent = (estimated_completed / total_questions) * 100
                questions_per_minute = (
                    estimated_completed / (elapsed / 60) if elapsed > 0 else 0
                )

                if estimated_completed < total_questions:
                    remaining_questions = total_questions - estimated_completed
                    eta_seconds = remaining_questions * 15  # Assume 15s per question
                    eta_minutes = eta_seconds / 60

                    print(
                        f"\n🔄 Progress Estimate: ~{estimated_completed}/{total_questions} ({progress_percent:.1f}%) | "
                        f"Speed: ~{questions_per_minute:.1f} questions/min | "
                        f"ETA: ~{eta_minutes:.1f} min\n"
                    )
                else:
                    print(f"\n🔄 Evaluation almost finished... Total time: {elapsed/60:.1f} min\n")
                    break

    # Multiprocessing mode
    progress_thread = threading.Thread(target=progress_monitor, daemon=True)
    progress_thread.start()

    with Pool(config["num_workers"]) as pool:
        results = pool.map(process_question_with_search, worker_args)

    total_time = time.time() - start_time

    # Calculate statistics
    logging.info("📊 Calculating processing statistics...")
    stats = calculate_processing_stats(results)
    stats["total_evaluation_time"] = total_time
    stats["questions_per_second"] = len(dataset) / total_time if total_time > 0 else 0

    # Save results
    logging.info("💾 Saving evaluation results...")
    results_file, stats_file = save_results(
        results, stats, config["output_dir"], timestamp
    )

    # Create Kaggle submission CSV file
    logging.info("📝 Creating Kaggle submission file...")
    csv_zip_path = create_kaggle_submission_csv(
        results, config["output_dir"], timestamp
    )

    # Print summary
    logging.info("=" * 60)
    logging.info("📊 Test Set Evaluation Summary:")
    logging.info(f"Total questions: {stats['total_questions']}")
    logging.info(f"Successfully processed: {stats['successful_questions']}")
    logging.info(f"Average processing time: {stats['average_processing_time']:.2f}s")
    logging.info(f"Total evaluation time: {total_time:.2f}s")
    logging.info(f"Processing speed: {stats['questions_per_second']:.2f} questions/s")
    logging.info(f"Error count: {stats['error_count']}")
    logging.info("=" * 60)

    # Print stats per question type
    if stats["type_statistics"]:
        logging.info("📈 Statistics by question type:")
        for q_type, type_stats in stats["type_statistics"].items():
            logging.info(
                f"  {q_type}: {type_stats['processed']}/{type_stats['total']} successfully processed"
            )

    logging.info(f"📁 Detailed results: {results_file}")
    logging.info(f"📊 Statistics summary: {stats_file}")
    if csv_zip_path:
        logging.info(f"📝 Kaggle submission file: {csv_zip_path}")
    logging.info("🎉 Test set evaluation complete!")


if __name__ == "__main__":
    main()
