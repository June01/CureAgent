#!/usr/bin/env python3
"""
CUREBench Final Step Processing Script - Using Gemini Search
Specifically designed to process submission.csv, re-infer after removing the last message
"""

import json
import logging
import os
import re
import sys
import time
import threading
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, Manager
from typing import Dict, List, Tuple
import signal
import pickle

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Remove all messages with role 'tool', and those whose content deserializes to None
def is_valid_message(c):
    if c.get('role') != 'tool':
        return True
    content = c.get('content')
    # Try to deserialize content; if it is a string and contains "content": null, it's invalid
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get('content') is None:
                print(c)
                return False
            if "Invalid function call:" in content:
                return False
        except Exception:
            pass
    return content is not None

def create_gemini_model_with_search(model_name, enable_search, api_key):
    """Create Gemini model instance (with Google Search enabled)"""
    try:
        from eval_framework import GeminiModel

        model = GeminiModel(
            model_name=model_name,
            api_key=api_key,
            google_search_enabled=enable_search,  # Enable Google Search
        )
        model.load()
        return model
    except Exception as e:
        print(f"Model creation failed: {e}")
        return None


def extract_conversation_from_reasoning(reasoning):
    """
    Extract conversation from the 'reasoning' field

    Args:
        reasoning (str): Content of the reasoning field

    Returns:
        list: conversation list, or None if extraction fails
    """
    if reasoning is None or (isinstance(reasoning, str) and reasoning.strip() == ""):
        return None

    try:
        # Try parsing as JSON
        if isinstance(reasoning, str):
            # If string, attempt to parse
            # Try standard JSON parsing first
            try:
                conversation = json.loads(reasoning)
            except json.JSONDecodeError:
                # If standard JSON parsing fails, try ast.literal_eval
                import ast
                try:
                    conversation = ast.literal_eval(reasoning)
                except (ValueError, SyntaxError):
                    return None
        else:
            # If already list/dict, use directly
            conversation = reasoning

        # Ensure it is a list
        if isinstance(conversation, list):
            return conversation
        else:
            return None

    except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
        return None


def extract_choice_from_prediction(prediction):
    """
    Extract the A/B/C/D answer from prediction text.
    Replicates the extraction logic in evaluator.py to cover various formats.
    """
    if not prediction or prediction.strip() == "":
        return None

    # Convert to uppercase for handling
    response = prediction.strip().upper()

    # Method 1: Check for leading A/B/C/D
    m = re.match(r"^\s*([ABCD])(?:[\)\.\:\-]\s|\s*$)", response)
    if m:
        return m.group(1)

    # Method 2: Look for "Answer:" followed by option
    answer_pattern = r'Answer:\s*\(?([A-D])\)?'
    match = re.search(answer_pattern, response)
    if match:
        return match.group(1)

    # Method 3: Look for "Answer" (no colon)
    answer_pattern2 = r'Answer\s+\(?([A-D])\)?'
    match = re.search(answer_pattern2, response)
    if match:
        return match.group(1)

    # Method 4: [FinalAnswer] followed by option
    final_answer_pattern = r'\[FinalAnswer\]\s+([A-D])'
    match = re.search(final_answer_pattern, response)
    if match:
        return match.group(1)

    # Method 5: Line starting with option letter
    line_pattern = r'^([A-D]):\s*'
    match = re.search(line_pattern, response, re.MULTILINE)
    if match:
        return match.group(1)

    # Methods 6-9: Patterns from evaluator.py
    patterns = [
        r"(?:ALIGNS?\s+WITH|MATCH(?:ES)?|CORRESPONDS?\s+TO)\s+(?:OPTION\s*)?([ABCD])\b",
        r"(?:ANSWER IS|ANSWER:)\s*([ABCD])",
        r"([ABCD])\)",
        r"\b([ABCD])\b"
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, response))
        if matches:
            return matches[-1].group(1)

    return None


def extract_final_answer_from_response(response_text: str) -> str:
    """Extract content after [FinalAnswer] from model response"""
    if not response_text:
        return ""

    # Find content after [FinalAnswer]
    final_answer_pattern = r'\[FinalAnswer\]\s*(.+?)(?:\n|$)'
    match = re.search(final_answer_pattern, response_text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return response_text.strip()


def process_submission_row_worker(args):
    """Multiprocessing worker function - processes a single submission row"""
    row_data, worker_id, row_idx, total_rows, model_name, search_enabled, gemini_api_key = args

    try:
        # Each worker process creates its own model instance
        model = create_gemini_model_with_search(model_name, search_enabled, api_key=gemini_api_key)
        if model is None:
            return {
                'id': row_data['id'],
                'prediction': row_data['prediction'],
                'choice': row_data['choice'],
                'reasoning': row_data['reasoning'],
                'error': 'Failed to create model'
            }

        question_id = row_data['id']
        original_choice = row_data['choice']
        reasoning = row_data['reasoning']

        # Parse conversation from the reasoning field
        conversation = extract_conversation_from_reasoning(reasoning)
        if conversation is None:
            print(f"⚠️ Worker {worker_id} row {question_id}: Failed to parse reasoning field")
            return {
                'id': question_id,
                'prediction': row_data['prediction'],
                'choice': original_choice,
                'reasoning': reasoning,
                'error': 'Failed to parse reasoning'
            }

        # Check conversation length
        if not isinstance(conversation, list) or len(conversation) < 2:
            print(f"⚠️ Worker {worker_id} row {question_id}: Conversation length insufficient")
            return {
                'id': question_id,
                'prediction': row_data['prediction'],
                'choice': original_choice,
                'reasoning': reasoning,
                'error': 'Insufficient conversation length'
            }

        # Remove the last message
        conversation = conversation[1:]
        conversation_new = conversation[:-2]
        conversation_new = [c for c in conversation_new if is_valid_message(c)]

        # Inference with Gemini, add timeout
        start_time = time.time()
        try:
            # Set timeout to 60 seconds
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Inference timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 seconds timeout

            gemini_response = model.inference(prompt=json.dumps(conversation_new, ensure_ascii=False))
            signal.alarm(0)  # Cancel timeout

        except TimeoutError:
            signal.alarm(0)
            return {
                'id': question_id,
                'prediction': row_data['prediction'],
                'choice': original_choice,
                'reasoning': reasoning,
                'error': 'Inference timeout'
            }

        processing_time = time.time() - start_time

        # Handle response
        if isinstance(gemini_response, tuple):
            response_text = gemini_response[0]
        else:
            response_text = gemini_response

        # Extract new prediction from [FinalAnswer]
        new_prediction = extract_final_answer_from_response(response_text)

        # Extract choice from prediction
        new_choice = extract_choice_from_prediction(new_prediction)
        if new_choice is None:
            new_choice = original_choice

        # Build new reasoning (conversation_new + inference result)
        new_reasoning = conversation_new + [{
            'role': 'assistant',
            'content': response_text
        }]

        # Calculate progress percent
        progress_percent = ((row_idx + 1) / total_rows) * 100

        # Print progress and result
        print(f"✅ Worker {worker_id} processed {question_id} [{row_idx + 1}/{total_rows}] ({progress_percent:.1f}%)")
        print(f"   Original choice='{original_choice}' -> New choice='{new_choice}' (Elapsed: {processing_time:.2f}s)")
        print(f"   Response preview: {response_text[:100]}...")
        print("-" * 80)

        return {
            'id': question_id,
            'prediction': new_prediction,
            'choice': new_choice,
            'reasoning': json.dumps(new_reasoning, ensure_ascii=False),
            'error': None
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Worker {worker_id} processing {row_data['id']} failed: {error_msg}")
        return {
            'id': row_data['id'],
            'prediction': row_data['prediction'],
            'choice': row_data['choice'],
            'reasoning': row_data['reasoning'],
            'error': error_msg
        }


def process_submission_row(row_data, model, model_name, search_enabled):
    """Process a single submission row"""
    try:
        question_id = row_data['id']
        original_choice = row_data['choice']
        reasoning = row_data['reasoning']

        # Parse conversation from the reasoning field
        conversation = extract_conversation_from_reasoning(reasoning)
        if conversation is None:
            print(f"⚠️ Row {question_id}: Failed to parse reasoning field")
            return {
                'id': question_id,
                'prediction': row_data['prediction'],
                'choice': original_choice,
                'reasoning': reasoning
            }

        # Check conversation length
        if not isinstance(conversation, list) or len(conversation) < 2:
            print(f"⚠️ Row {question_id}: Conversation length insufficient")
            return {
                'id': question_id,
                'prediction': row_data['prediction'],
                'choice': original_choice,
                'reasoning': reasoning
            }

        # Remove the last message
        conversation = conversation[1:]
        conversation_new = conversation[:-2]

        conversation_new = [c for c in conversation_new if is_valid_message(c)]

        # Use Gemini for inference
        start_time = time.time()
        gemini_response = model.inference(prompt=json.dumps(conversation_new, ensure_ascii=False))
        processing_time = time.time() - start_time

        # Handle response
        if isinstance(gemini_response, tuple):
            response_text = gemini_response[0]
        else:
            response_text = gemini_response

        # Extract new prediction from [FinalAnswer]
        new_prediction = extract_final_answer_from_response(response_text)

        # Extract choice from prediction
        new_choice = extract_choice_from_prediction(new_prediction)
        if new_choice is None:
            new_choice = original_choice

        # Build new reasoning (conversation_new + inference result)
        new_reasoning = conversation_new + [{
            'role': 'assistant',
            'content': response_text
        }]

        print(f"✅ Processed {question_id}: original choice='{original_choice}' -> new choice='{new_choice}' (Elapsed: {processing_time:.2f}s)")

        return {
            'id': question_id,
            'prediction': new_prediction,
            'choice': new_choice,
            'reasoning': json.dumps(new_reasoning, ensure_ascii=False)
        }

    except Exception as e:
        print(f"❌ Processing {question_id} failed: {e}")
        return {
            'id': row_data['id'],
            'prediction': row_data['prediction'],
            'choice': row_data['choice'],
            'reasoning': row_data['reasoning']
        }


def load_submission_csv(input_file: str):
    """Load submission.csv file"""

    try:
        df = pd.read_csv(input_file)
        print(f"✅ Successfully loaded file: {input_file}")
        print(f"📊 Total rows: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return None


def save_results(df, output_file: str):
    """Save results to CSV file"""
    try:
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        return False


def save_progress(results: List[Dict], output_file: str, processed_count: int, total_count: int):
    """Save current progress to CSV file"""
    try:
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"💾 Progress saved: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%) -> {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save progress: {e}")
        return False


def load_progress(progress_file: str):
    """Load previous progress"""
    try:
        if os.path.exists(progress_file):
            df = pd.read_csv(progress_file)
            processed_ids = set(df['id'].tolist())
            print(f"📂 Found progress file: {progress_file}, processed {len(processed_ids)} records")
            return processed_ids
        return set()
    except Exception as e:
        print(f"⚠️ Failed to load progress file: {e}")
        return set()


def save_checkpoint(checkpoint_file: str, processed_count: int, total_count: int, start_idx: int):
    """Save checkpoint info"""
    try:
        checkpoint_data = {
            'processed_count': processed_count,
            'total_count': total_count,
            'start_idx': start_idx,
            'timestamp': time.time()
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_file: str):
    """Load checkpoint info"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"📂 Found checkpoint file: {checkpoint_file}")
            return checkpoint_data
        return None
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return None


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='CUREBench Final Step Processing Script')
    parser.add_argument('input_file', help='Path to input submission.csv')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--model', default='gemini-3-pro-preview', help='Model name')
    parser.add_argument('--search', action='store_true', help='Enable Google Search')
    parser.add_argument('--gemini-api-key', type=str, default="", help='Gemini API key')
    # parser.add_argument('--search-enabled', action='store_true', help='Enable Google Search')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker processes (default 4)')
    parser.add_argument('--single-thread', action='store_true', help='Run in single-thread mode')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size (default 50)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval (number of records between saves, default 10)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file does not exist: {args.input_file}")
        return

    # Set output file name
    if args.output:
        output_file = args.output
    else:
        # Auto-generate output file name
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        model_name = args.model.replace('-', '_')
        search_suffix = "withsearch_True" if args.search else "withsearch_False"
        # Save output file in the same folder as input file
        input_dir = os.path.dirname(os.path.abspath(args.input_file))
        output_file = os.path.join(input_dir, f"submission_finalstep_{model_name}_{search_suffix}.csv")
        # output_file = f"submission_finalstep_{model_name}_{search_suffix}.csv"

    print("🚀 Starting Final Step processing...")
    print(f"📁 Input file: {args.input_file}")
    print(f"📁 Output file: {output_file}")
    print(f"🤖 Model: {args.model}")
    print(f"🔍 Google Search: {'Enabled' if args.search else 'Disabled'}")
    print(f"⚡ Number of worker processes: {args.workers if not args.single_thread else 1}")
    print()

    # Load submission.csv
    df = load_submission_csv(args.input_file)
    if df is None:
        return
    # Only select first row for model test
    # df = df.head(1)
    # Check required columns
    required_columns = ['id', 'prediction', 'choice', 'reasoning']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return

    # Set progress file and checkpoint file
    progress_file = output_file.replace('.csv', '_progress.csv')
    checkpoint_file = output_file.replace('.csv', '_checkpoint.pkl')

    # Process rows
    print("⚡ Starting data processing...")
    start_time = time.time()

    results = []
    total_rows = len(df)
    processed_count = 0
    start_idx = 0

    # Check if resumpting is needed
    if args.resume:
        processed_ids = load_progress(progress_file)
        checkpoint_data = load_checkpoint(checkpoint_file)

        if processed_ids:
            # Filter out already-processed records
            df = df[~df['id'].isin(processed_ids)]
            processed_count = len(processed_ids)
            start_idx = processed_count
            print(f"🔄 Resuming from last checkpoint, skipping {processed_count} already-processed records")

            # Load previous results
            if os.path.exists(progress_file):
                try:
                    prev_df = pd.read_csv(progress_file)
                    results = prev_df.to_dict('records')
                    print(f"📂 Loaded {len(results)} previous results")
                except Exception as e:
                    print(f"⚠️ Failed to load previous results: {e}")
                    results = []

    remaining_rows = len(df)
    print(f"📊 Total rows: {total_rows}, Processed: {processed_count}, Remaining: {remaining_rows}")

    if remaining_rows == 0:
        print("✅ All data has been processed!")
        return

    # Batch processing config
    batch_size = args.batch_size
    save_interval = args.save_interval

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n⚠️ Received interrupt signal, saving current progress...")
        if results:
            save_progress(results, progress_file, processed_count, total_rows)
            save_checkpoint(checkpoint_file, processed_count, total_rows, start_idx)
        print("💾 Progress saved. You can resume later using --resume.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.single_thread:
            # Single-thread mode
            print("🔄 Running in single-thread mode...")
            model = create_gemini_model_with_search(args.model, args.search)
            if model is None:
                print("❌ Failed to initialize model")
                return

            for idx, (_, row) in enumerate(df.iterrows()):
                current_idx = start_idx + idx
                print(f"📊 Progress: {current_idx + 1}/{total_rows} ({(current_idx + 1)/total_rows*100:.1f}%)")

                result = process_submission_row(
                    row,
                    model,
                    args.model,
                    args.search
                )
                results.append(result)
                processed_count += 1

                # Periodically save progress
                if processed_count % save_interval == 0:
                    save_progress(results, progress_file, processed_count, total_rows)
                    save_checkpoint(checkpoint_file, processed_count, total_rows, start_idx)
        else:
            # Multiprocessing - batch processing
            print(f"⚡ Using {args.workers} worker processes for batch processing...")

            # Divide data into batches
            batches = []
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                batches.append(batch_df)

            print(f"📦 Total {len(batches)} batch(es), {batch_size} records each")

            for batch_idx, batch_df in enumerate(batches):
                print(f"\n🔄 Processing batch {batch_idx + 1}/{len(batches)} (contains {len(batch_df)} records)")

                # Prepare arguments for this batch
                worker_args = [
                    (row, i % args.workers, start_idx + i, total_rows, args.model, args.search, args.gemini_api_key)
                    for i, (_, row) in enumerate(batch_df.iterrows())
                ]

                # Process this batch
                with Pool(args.workers) as pool:
                    batch_results = pool.map(process_submission_row_worker, worker_args)

                # Add batch results to global results
                results.extend(batch_results)
                processed_count += len(batch_results)

                # Save progress
                save_progress(results, progress_file, processed_count, total_rows)
                save_checkpoint(checkpoint_file, processed_count, total_rows, start_idx)

                print(f"✅ Batch {batch_idx + 1} completed, total processed: {processed_count}/{total_rows}")

                # Short pause between batches to avoid API limits
                if batch_idx < len(batches) - 1:
                    time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n⚠️ User interrupted, saving current progress...")
        if results:
            save_progress(results, progress_file, processed_count, total_rows)
            save_checkpoint(checkpoint_file, processed_count, total_rows, start_idx)
        print("💾 Progress saved. You can resume later using --resume.")
        return
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if results:
            save_progress(results, progress_file, processed_count, total_rows)
            save_checkpoint(checkpoint_file, processed_count, total_rows, start_idx)
        print("💾 Progress saved. You can resume later using --resume.")
        return

    total_time = time.time() - start_time

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Save final results
    if save_results(result_df, output_file):
        print(f"\n🎉 Processing completed!")
        print(f"⏱️ Total elapsed time: {total_time:.2f}s")
        print(f"📊 Processing speed: {processed_count/total_time:.2f} rows/second")

        # Count the choice change stats
        if len(results) > 0:
            original_choices = [r.get('choice', '') for r in results]
            new_choices = [r.get('choice', '') for r in results]
            changes = sum(1 for orig, new in zip(original_choices, new_choices) if orig != new)

            print(f"🔄 Choice changes: {changes}/{len(results)} ({changes/len(results)*100:.1f}%)")

        # Count errors
        errors = [r for r in results if r.get('error')]
        if errors:
            print(f"⚠️ Number of errors: {len(errors)}")

        # Cleanup temporary files
        try:
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"🗑️ Progress file removed: {progress_file}")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"🗑️ Checkpoint file removed: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️ Failed to clean up temporary files: {e}")
    else:
        print("❌ Save failed")


if __name__ == "__main__":
    main()
