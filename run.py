#!/usr/bin/env python3
"""
Bio-Medical AI Competition - Evaluation Script

Simple evaluation script that supports metadata configuration
via command line arguments and configuration files.

Usage:
    # Basic usage
    python run.py                                      # Run with defaults

    # With metadata via config file
    python run.py --config metadata_config.json
"""

import os
import argparse
import sys
from datetime import datetime
from eval_framework_v2 import CompetitionKit


def load_config_file(config_path):
    """Load configuration from JSON file"""
    import json
    import os
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


def load_and_merge_config(args):
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args

    config = load_config_file(args.config)

    # First, handle the metadata section specially - merge its contents directly
    if "metadata" in config:
        metadata = config["metadata"]
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=""):
        for key, value in config_dict.items():
            if key in [
                "metadata",
                "dataset",
            ]:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)

    add_config_to_args(config)
    return args


def create_metadata_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="CURE-Bench Evaluation Framework")
    
    parser.add_argument("--model-name", type=str, help="Name of the model")
    parser.add_argument("--model-type", type=str, help="Type of model wrapper")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output-file", type=str, default="submission.csv", help="Output filename")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output with detailed results")
    parser.add_argument("--save-mid-result", action="store_true", help="Save intermediate results for dataset creation")
    parser.add_argument("--save-log", action="store_true", default=False, help="Save log to log.txt file")
    
    return parser


class Logger:
    """Logger class to capture both console and file output"""
    
    def __init__(self, output_dir, log_filename="log.txt"):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, log_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Store original stdout
        self.original_stdout = sys.stdout
        
        # Create a custom stdout that writes to both console and file
        class DualOutput:
            def __init__(self, console, file):
                self.console = console
                self.file = file
            
            def write(self, text):
                self.console.write(text)
                self.file.write(text)
                self.file.flush()  # Ensure immediate write to file
            
            def flush(self):
                self.console.flush()
                self.file.flush()
        
        # Replace stdout with dual output
        sys.stdout = DualOutput(self.original_stdout, self.log_file)
        
        # Write header to log file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"=== CURE-Bench Evaluation Log ===\n")
        self.log_file.write(f"Started at: {timestamp}\n")
        self.log_file.write(f"Log file: {self.log_path}\n")
        self.log_file.write("=" * 50 + "\n\n")
        self.log_file.flush()
    
    def close(self):
        """Close the logger and restore original stdout"""
        if hasattr(self, 'log_file') and self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\n" + "=" * 50 + "\n")
            self.log_file.write(f"Finished at: {timestamp}\n")
            self.log_file.close()
        
        # Restore original stdout
        sys.stdout = self.original_stdout


def main():
    # Create argument parser with metadata support
    parser = create_metadata_parser()

    args = parser.parse_args()

    # Load configuration from config file if provided and merge with args
    args = load_and_merge_config(args)

    # Extract values dynamically with fallback defaults
    output_file = getattr(args, "output_file", "submission.csv")
    dataset_name = getattr(args, "dataset")
    model_name = getattr(args, "model_path", None) or getattr(args, "model_name", None)
    model_type = getattr(args, "model_type", "auto")
    verbose = getattr(args, "verbose", False)
    save_mid_result = getattr(args, "save_mid_result", False)
    save_log = getattr(args, "save_log", False)

    # Get the correct output_dir - prioritize args.output_dir, fallback to config file
    output_dir = getattr(args, "output_dir", None)
    
    # Initialize the competition kit
    config_path = getattr(args, "config", None)
    # Use metadata_config.json as default if no config is specified
    if not config_path:
        default_config = "metadata_config.json"
        if os.path.exists(default_config):
            config_path = default_config

    kit = CompetitionKit(config_path=config_path)
    
    # If output_dir was not specified via command line, use the one from config
    if output_dir is None:
        output_dir = kit.config.get("output_dir", "results")
    
    # Initialize logger only if save_log is True
    logger = None
    if save_log:
        logger = Logger(output_dir, "log.txt")

    try:
        """Run evaluation with metadata support"""
        print("\n" + "=" * 60)
        print("🏥 CURE-Bench Competition - Modular Evaluation")
        print("=" * 60)
        print("Verbose mode: ", verbose)
        print("Save intermediate results: ", save_mid_result)
        print("Save log: ", save_log)
        if save_log:
            print(f"Log file: {logger.log_path}")
        print(f"Output directory: {output_dir}")
        
        # Set verbose mode if specified
        if verbose:
            kit.config['verbose'] = True
        
            # Set save_mid_result mode if specified
        if save_mid_result:
            kit.config['save_mid_result'] = True

        # Set random seed for reproducibility and variation
        import random
        import numpy as np
        import torch
        
        # 设置随机种子以获得不同的结果
        # 优先使用环境变量中的种子，否则随机生成
        random_seed = random.randint(1, 10000)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 同时更新配置文件中的种子
        kit.config['seed'] = random_seed
        
        print(f"Using random seed: {random_seed}")
        print(f"Loading model: {model_name}")
        kit.load_model(model_name, model_type)

        # Show available datasets
        print("Available datasets:")
        kit.list_datasets()

        # Run evaluation
        print(f"Running evaluation on dataset: {dataset_name}")

        results = kit.evaluate(dataset_name)

        # Generate submission
        print("Generating submission...")
        submission_path = kit.save_submission(results, output_dir, output_file)

        print("\n✅ Evaluation completed successfully!")
        print(
            f" Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})"
        )
        print(f"📄 Submission saved to: {submission_path}")

        # Show metadata summary
        print("\n Final metadata:")
        print(f"  Model: {model_name}")
        print(f"  Model Type: {model_type}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Output File: {output_file}")
        print(f"  Config Path: {config_path}")
        print(f"  Save Intermediate Results: {save_mid_result}")
        print(f"  Save Log: {save_log}")
        if save_log:
            print(f"  Log File: {logger.log_path}")
        print(f"  Output Directory: {output_dir}")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Close the logger if it was initialized
        if logger:
            logger.close()


if __name__ == "__main__":
    main()