"""
Metrics calculation for evaluation.

Calculates accuracy, confusion matrices, and per-subject/per-choice metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .chunk_schema import Prediction

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics."""

    def calculate(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """
        Calculate all metrics.

        Args:
            predictions: List of predictions

        Returns:
            Metrics dictionary
        """
        total = len(predictions)
        correct = sum(1 for p in predictions if p.is_correct)

        accuracy = correct / total if total > 0 else 0.0

        # With/without context
        with_context = [p for p in predictions if p.context and len(p.context.passages) > 0]
        without_context = [p for p in predictions if not p.context or len(p.context.passages) == 0]

        with_context_acc = (
            sum(1 for p in with_context if p.is_correct) / len(with_context) if with_context else 0.0
        )
        without_context_acc = (
            sum(1 for p in without_context if p.is_correct) / len(without_context)
            if without_context
            else 0.0
        )

        # Per-subject metrics (if available)
        by_subject = self._calculate_by_subject(predictions)

        # Per-choice-type metrics (if available)
        by_choice_type = self._calculate_by_choice_type(predictions)

        # Confusion matrix
        confusion = self._calculate_confusion(predictions)

        # Coverage statistics
        coverage = self._calculate_coverage(predictions)

        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "with_context_count": len(with_context),
            "with_context_accuracy": with_context_acc,
            "without_context_count": len(without_context),
            "without_context_accuracy": without_context_acc,
            "by_subject": by_subject,
            "by_choice_type": by_choice_type,
            "confusion_matrix": confusion,
            "coverage": coverage,
        }

        return metrics

    def _calculate_by_subject(self, predictions: List[Prediction]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-subject metrics.

        Args:
            predictions: List of predictions

        Returns:
            Dict mapping subject name to metrics
        """
        # Extract subject from metadata if available
        by_subject = defaultdict(lambda: {"total": 0, "correct": 0})

        for pred in predictions:
            # Try to get subject from metadata
            subject = pred.metadata.get("subject_name", "UNKNOWN")

            by_subject[subject]["total"] += 1
            if pred.is_correct:
                by_subject[subject]["correct"] += 1

        # Convert to final format
        result = {}
        for subject, counts in by_subject.items():
            result[subject] = {
                "count": counts["total"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
            }

        return result

    def _calculate_by_choice_type(self, predictions: List[Prediction]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-choice-type metrics (single vs multiple choice).

        Args:
            predictions: List of predictions

        Returns:
            Dict mapping choice type to metrics
        """
        by_type = defaultdict(lambda: {"total": 0, "correct": 0})

        for pred in predictions:
            choice_type = pred.metadata.get("choice_type", "single")
            by_type[choice_type]["total"] += 1
            if pred.is_correct:
                by_type[choice_type]["correct"] += 1

        # Convert to final format
        result = {}
        for choice_type, counts in by_type.items():
            result[choice_type] = {
                "count": counts["total"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
            }

        return result

    def _calculate_confusion(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """
        Calculate confusion matrix.

        Args:
            predictions: List of predictions

        Returns:
            Confusion matrix dictionary
        """
        # Track (gold, predicted) pairs
        pairs = defaultdict(int)

        for pred in predictions:
            if pred.gold_answer and pred.predicted_answer:
                pair = (pred.gold_answer, pred.predicted_answer)
                pairs[pair] += 1

        # Convert to matrix format
        all_answers = set()
        for gold, pred in pairs.keys():
            all_answers.add(gold)
            all_answers.add(pred)

        all_answers = sorted(all_answers)

        matrix = []
        for gold in all_answers:
            row = []
            for pred in all_answers:
                row.append(pairs.get((gold, pred), 0))
            matrix.append(row)

        return {
            "labels": all_answers,
            "matrix": matrix,
        }

    def _calculate_coverage(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """
        Calculate retrieval coverage statistics.

        Args:
            predictions: List of predictions

        Returns:
            Coverage statistics
        """
        context_counts = [len(p.context.passages) if p.context else 0 for p in predictions]

        if not context_counts:
            return {"mean": 0, "min": 0, "max": 0, "median": 0}

        sorted_counts = sorted(context_counts)
        n = len(sorted_counts)

        return {
            "mean": sum(context_counts) / n,
            "min": min(context_counts),
            "max": max(context_counts),
            "median": sorted_counts[n // 2] if n > 0 else 0,
            "p25": sorted_counts[n // 4] if n > 0 else 0,
            "p75": sorted_counts[3 * n // 4] if n > 0 else 0,
            "with_context": sum(1 for c in context_counts if c > 0),
            "without_context": sum(1 for c in context_counts if c == 0),
        }
