"""
Submission generator for CURE-Bench evaluation framework
"""
import os
import json
import zipfile
import logging
from typing import Dict, List, Any
import pandas as pd

from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """Generator for competition submission files"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize submission generator
        
        Args:
            output_dir: Directory to save submission files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_csv(self, results: EvaluationMetrics, 
                    dataset_examples: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate CSV DataFrame from evaluation results
        
        Args:
            results: Evaluation results
            dataset_examples: Original dataset examples (optional)
            
        Returns:
            DataFrame ready for CSV export
        """
        submission_data = []
        
        # Use dataset examples if provided, otherwise create generic IDs
        examples = dataset_examples if dataset_examples else []
        
        for i, (prediction, reasoning_trace) in enumerate(zip(results.predictions, results.reasoning_traces)):
            # Get example ID
            if i < len(examples):
                example_id = examples[i].get("id", f"example_{i}")
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
            
            # Clean up reasoning trace
            if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                reasoning_trace = "No reasoning available"
            
            # Create row
            row = {
                "id": str(example_id),
                "prediction": str(prediction_text),
                "choice": str(choice_clean),
                "reasoning": str(reasoning_trace),
            }
            
            submission_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(submission_data)
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Clean up null values
        null_replacements = {
            "id": "unknown_id",
            "prediction": "No prediction available",
            "choice": "NOTAVALUE",
            "reasoning": "No reasoning available",
        }
        
        for col in df.columns:
            df[col] = df[col].fillna(null_replacements.get(col, "NOTAVALUE"))
            
            # Replace string representations of null
            null_like_values = ["nan", "NaN", "None", "null", "NULL", "<NA>", "nat", "NaT"]
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, "NOTAVALUE"))
        
        return df
    
    def save_csv(self, results: EvaluationMetrics, 
                filename: str = "submission.csv",
                dataset_examples: List[Dict[str, Any]] = None) -> str:
        """
        Save results as CSV file
        
        Args:
            results: Evaluation results
            filename: Output CSV filename
            dataset_examples: Original dataset examples (optional)
            
        Returns:
            Path to saved CSV file
        """
        df = self.generate_csv(results, dataset_examples)
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Save CSV with proper parameters
        df.to_csv(csv_path, index=False, na_rep="NOTAVALUE", quoting=1, encoding="utf-8")
        
        logger.info(f"CSV submission saved to: {csv_path}")
        return csv_path
    
    def create_metadata_json(self, metadata: Dict[str, Any], 
                           filename: str = "meta_data.json") -> str:
        """
        Create metadata JSON file
        
        Args:
            metadata: Metadata dictionary
            filename: Output JSON filename
            
        Returns:
            Path to saved JSON file
        """
        json_path = os.path.join(self.output_dir, filename)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to: {json_path}")
        return json_path
    
    def create_zip_package(self, csv_path: str, metadata_path: str,
                          zip_filename: str = "submission.zip") -> str:
        """
        Create ZIP package with CSV and metadata
        
        Args:
            csv_path: Path to CSV file
            metadata_path: Path to metadata JSON file
            zip_filename: Output ZIP filename
            
        Returns:
            Path to saved ZIP file
        """
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            csv_basename = os.path.basename(csv_path)
            zipf.write(csv_path, csv_basename)
            
            # Add metadata JSON to zip
            metadata_basename = os.path.basename(metadata_path)
            zipf.write(metadata_path, metadata_basename)
        
        logger.info(f"Submission package saved to: {zip_path}")
        return zip_path
    
    def generate_submission(self, results: EvaluationMetrics,
                          metadata: Dict[str, Any],
                          csv_filename: str = "submission.csv",
                          zip_filename: str = "submission.zip",
                          dataset_examples: List[Dict[str, Any]] = None) -> str:
        """
        Generate complete submission package
        
        Args:
            results: Evaluation results
            metadata: Metadata dictionary
            csv_filename: CSV filename
            zip_filename: ZIP filename
            dataset_examples: Original dataset examples (optional)
            
        Returns:
            Path to ZIP package
        """
        # Generate CSV
        csv_path = self.save_csv(results, csv_filename, dataset_examples)
        
        # Generate metadata JSON
        metadata_path = self.create_metadata_json(metadata)
        
        # Create ZIP package
        zip_path = self.create_zip_package(csv_path, metadata_path, zip_filename)
        
        return zip_path 