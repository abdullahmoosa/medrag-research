#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate RAG retrieval using pre-defined questions and known relevant passages.

This script:
1. Uses a test set of medical questions with known relevant document IDs
2. Evaluates precision, recall, and other IR metrics for different retrieval methods
3. Compares vector-only vs. hybrid search performance

Usage:
    python src/scripts/evaluate_retrieval.py --test-file data/medmcqa/retrieval_test.json
"""
import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

# Ensure project root on sys.path when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the RAG utilities
from src.rag_utils import (
    build_or_load_index,
    DEFAULT_KB_DIR,
    DEFAULT_INDEX_DIR,
    Passage,
    EmbeddingModel,
    HybridRetriever,
    ensure_dir,
)

DEFAULT_EMBED_MODEL = "oscardp96/medcpt-article"
DEFAULT_TEST_FILE = os.path.join(PROJECT_ROOT, "data", "medmcqa", "retrieval_test.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results", "retrieval_evaluation")


def create_sample_test_set(
    sample_count: int = 5,
    output_path: str = DEFAULT_TEST_FILE,
) -> None:
    """
    Create a sample test set of questions from MedMCQA with random 'relevant' passage IDs.
    This is a placeholder - in a real application, you would manually annotate relevant passages.
    """
    import glob
    from random import sample, randint
    
    # Get a sample of MedMCQA questions
    medmcqa_path = os.path.join(PROJECT_ROOT, "data", "medmcqa", "dev_stratified_sample.json")
    if not os.path.exists(medmcqa_path):
        print(f"MedMCQA file not found: {medmcqa_path}")
        return
        
    with open(medmcqa_path, "r", encoding="utf-8") as f:
        medmcqa_data = json.load(f)
    
    # Select a random sample
    questions = sample(medmcqa_data, min(sample_count, len(medmcqa_data)))
    
    # Create test set entries
    test_entries = []
    for q in questions:
        question_text = q.get("question", "")
        if not question_text:
            continue
            
        # For demonstration, create random "relevant" passage IDs
        # In a real application, these would be manually annotated
        relevant_ids = [str(randint(1000, 9999)) for _ in range(randint(3, 7))]
        
        test_entries.append({
            "question": question_text,
            "relevant_passages": relevant_ids,
            "subject": q.get("subject", ""),
        })
    
    # Save the test set
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_entries, f, indent=2)
    
    print(f"Created sample test set with {len(test_entries)} questions at {output_path}")


def calculate_metrics(
    retrieved_ids: Set[str],
    relevant_ids: Set[str],
) -> Dict[str, float]:
    """Calculate precision, recall, F1, etc."""
    true_positives = len(retrieved_ids & relevant_ids)
    false_positives = len(retrieved_ids - relevant_ids)
    false_negatives = len(relevant_ids - retrieved_ids)
    
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * (precision * recall) / max(1e-10, precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_retrieval(
    test_file: str = DEFAULT_TEST_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    kb_dir: str = DEFAULT_KB_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    embed_model_id: str = DEFAULT_EMBED_MODEL,
    device: Optional[str] = None,
    top_k: int = 10,
) -> None:
    """Evaluate retrieval performance on a test set."""
    ensure_dir(output_dir)
    
    # Check if test file exists, create a sample one if not
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}. Creating a sample test set...")
        create_sample_test_set(output_path=test_file)
    
    # Load the test set
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Load the index and embedder
    index, embedder = build_or_load_index(
        kb_dir=kb_dir,
        index_dir=index_dir,
        embed_model_id=embed_model_id,
        device=device,
    )
    
    # Initialize the hybrid retriever
    hybrid_retriever = HybridRetriever(index, embedder)
    
    # Results collection
    results = []
    metrics_vector = defaultdict(list)
    metrics_hybrid = defaultdict(list)
    
    # Process each test question
    for i, entry in enumerate(test_data, 1):
        question = entry["question"]
        relevant_ids = set(entry.get("relevant_passages", []))
        
        print(f"Processing question {i}/{len(test_data)}: {question[:50]}...")
        
        # Vector retrieval
        q_emb = embedder.embed([question])
        vector_hits = index.search(q_emb, top_k=top_k)
        vector_ids = {str(i) for i, (_, p) in enumerate(vector_hits)}
        
        # Hybrid retrieval
        hybrid_hits = hybrid_retriever.search(question, top_k=top_k)
        hybrid_ids = {str(i) for i, (_, p) in enumerate(hybrid_hits)}
        
        # Calculate metrics
        vector_metrics = calculate_metrics(vector_ids, relevant_ids)
        hybrid_metrics = calculate_metrics(hybrid_ids, relevant_ids)
        
        # Store metrics
        for k, v in vector_metrics.items():
            if isinstance(v, (int, float)):
                metrics_vector[k].append(v)
        
        for k, v in hybrid_metrics.items():
            if isinstance(v, (int, float)):
                metrics_hybrid[k].append(v)
        
        # Store individual result
        results.append({
            "question": question,
            "subject": entry.get("subject", ""),
            "relevant_count": len(relevant_ids),
            "vector_metrics": vector_metrics,
            "hybrid_metrics": hybrid_metrics,
        })
    
    # Calculate averages
    avg_vector = {k: np.mean(v) for k, v in metrics_vector.items()}
    avg_hybrid = {k: np.mean(v) for k, v in metrics_hybrid.items()}
    
    # Prepare the final report
    report = {
        "test_file": test_file,
        "kb_dir": kb_dir,
        "index_dir": index_dir,
        "embed_model": embed_model_id,
        "top_k": top_k,
        "question_count": len(test_data),
        "vector_avg_metrics": avg_vector,
        "hybrid_avg_metrics": avg_hybrid,
        "detailed_results": results,
    }
    
    # Save the report
    output_path = os.path.join(output_dir, "retrieval_evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nRetrieval Evaluation Summary:")
    print(f"Questions evaluated: {len(test_data)}")
    print("\nVector-only retrieval:")
    for k, v in avg_vector.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nHybrid retrieval:")
    for k, v in avg_hybrid.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval performance")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE,
                        help="Path to the test file with questions and relevant passage IDs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--kb-dir", type=str, default=DEFAULT_KB_DIR,
                        help="Path to the knowledge base directory")
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR,
                        help="Path to the index directory")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                        help="Embedding model to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages to retrieve")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a sample test set and exit")
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_test_set(output_path=args.test_file)
    else:
        evaluate_retrieval(
            test_file=args.test_file,
            output_dir=args.output_dir,
            kb_dir=args.kb_dir,
            index_dir=args.index_dir,
            embed_model_id=args.embed_model,
            device=args.device,
            top_k=args.top_k,
        )
