"""
Evaluation metrics for CURE-Bench evaluation framework
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics and results"""
    
    # Basic metrics
    accuracy: float
    correct_predictions: int
    total_examples: int
    
    # Detailed results
    predictions: List[Dict[str, Any]]
    reasoning_traces: List[str]
    
    # Metadata
    dataset_name: str
    model_name: str
    
    # Additional metrics
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate metrics after initialization"""
        if self.total_examples > 0:
            calculated_accuracy = self.correct_predictions / self.total_examples
            if abs(self.accuracy - calculated_accuracy) > 1e-6:
                logger.warning(f"Accuracy mismatch: provided {self.accuracy}, calculated {calculated_accuracy}")
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_examples == 0:
            return 0.0
        return 1.0 - self.accuracy
    
    @property
    def correct_rate(self) -> float:
        """Calculate correct rate (same as accuracy)"""
        return self.accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results"""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "correct_predictions": self.correct_predictions,
            "total_examples": self.total_examples,
            "error_rate": self.error_rate,
            "details": self.details or {}
        }
    
    def __str__(self) -> str:
        """String representation of metrics"""
        return (f"Evaluation Results for {self.model_name} on {self.dataset_name}: "
                f"{self.accuracy:.2%} accuracy ({self.correct_predictions}/{self.total_examples})")
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"EvaluationMetrics(accuracy={self.accuracy:.4f}, "
                f"correct={self.correct_predictions}/{self.total_examples}, "
                f"dataset='{self.dataset_name}', model='{self.model_name}')") 