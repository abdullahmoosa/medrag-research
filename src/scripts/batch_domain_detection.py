#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Domain Detection Evaluation

This script runs domain detection evaluation across multiple models and configurations.
It supports both zero-shot and few-shot modes on both train and dev datasets.
"""

from domain_detection_llm import DomainDetectionEvaluator
import json
import time
from pathlib import Path

def run_evaluation(model_name, few_shot_mode, data_split, data_file, url):
    """
    Run a single evaluation configuration
    
    Args:
        model_name: Name of the model to evaluate
        few_shot_mode: Whether to use few-shot prompting
        data_split: Dataset split ('train' or 'dev')
        data_file: Path to the data file
        url: API endpoint URL
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"Mode: {'Few-shot' if few_shot_mode else 'Zero-shot'}")
    print(f"Dataset: {data_split}")
    print(f"{'='*80}")
    
    # Load data
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        print(f"Loaded {len(data)} questions")
    except Exception as e:
        print(f"Error loading data from {data_file}: {e}")
        return False
    
    # Initialize evaluator
    evaluator = DomainDetectionEvaluator(
        model_name=model_name,
        use_thinking_model=False,
        few_shot_mode=few_shot_mode,
        data_split=data_split
    )
    
    # Check if evaluation already exists
    progress = evaluator._load_progress()
    if progress.get("total_processed", 0) > 0:
        print(f"Found existing progress: {progress.get('total_processed', 0)} questions processed")
        user_input = input("Continue from where left off? (y/n): ").lower()
        if user_input != 'y':
            evaluator.reset_evaluation()
            print("Reset evaluation progress")
    
    try:
        # Run evaluation
        start_time = time.time()
        evaluator.evaluate_batch(data, url, batch_size=10)
        end_time = time.time()
        
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
        
        # Print final results
        final_progress = evaluator._load_progress()
        metrics = final_progress.get("cumulative_metrics", {})
        print(f"\nFinal Results:")
        print(f"  Total Questions: {metrics.get('total_questions', 0)}")
        print(f"  Correct Predictions: {metrics.get('correct_predictions', 0)}")
        print(f"  Accuracy: {metrics.get('overall_accuracy', 0):.2f}%")
        print(f"  Avg Inference Time: {metrics.get('average_inference_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False

def main():
    """Main function to run batch evaluations"""
    
    # Configuration
    base_dir = Path("C:/Users/User/Downloads/nusrat/medrag")
    data_dir = base_dir / "data" / "medmcqa"
    url = "http://localhost:11434/api/generate"
    
    # Data files
    data_files = {
        "dev": str(data_dir / "dev_stratified_sample.json"),
        "train": str(data_dir / "train_stratified_sample.json")
    }
    
    # Models to evaluate
    models = [
        "thewindmom/llama3-med42-8b:latest",
        "deepseek-r1:8b",
        "gemma3:12b-it-qat",
        "meditron:latest",
        "OussamaELALLAM/MedExpert:latest"
    ]
    
    # Modes to test
    modes = [
        {"few_shot": False, "name": "zero-shot"},
        {"few_shot": True, "name": "few-shot"}
    ]
    
    # Data splits
    splits = ["dev", "train"]
    
    print("DOMAIN DETECTION BATCH EVALUATION")
    print("="*50)
    print(f"Models: {len(models)}")
    print(f"Modes: {len(modes)}")
    print(f"Splits: {len(splits)}")
    print(f"Total combinations: {len(models) * len(modes) * len(splits)}")
    
    # Interactive selection
    print("\nSelect evaluation scope:")
    print("1. Run all combinations (full evaluation)")
    print("2. Select specific model")
    print("3. Select specific configuration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    evaluations_to_run = []
    
    if choice == "1":
        # Run all combinations
        for model in models:
            for mode in modes:
                for split in splits:
                    evaluations_to_run.append({
                        "model": model,
                        "few_shot": mode["few_shot"],
                        "split": split,
                        "data_file": data_files[split]
                    })
    
    elif choice == "2":
        # Select specific model
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        model_choice = input("Select model number: ").strip()
        try:
            selected_model = models[int(model_choice) - 1]
            for mode in modes:
                for split in splits:
                    evaluations_to_run.append({
                        "model": selected_model,
                        "few_shot": mode["few_shot"],
                        "split": split,
                        "data_file": data_files[split]
                    })
        except (ValueError, IndexError):
            print("Invalid model selection")
            return
    
    elif choice == "3":
        # Select specific configuration
        print("\nSelect model:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        model_choice = input("Model number: ").strip()
        
        print("\nSelect mode:")
        for i, mode in enumerate(modes, 1):
            print(f"{i}. {mode['name']}")
        
        mode_choice = input("Mode number: ").strip()
        
        print("\nSelect dataset:")
        for i, split in enumerate(splits, 1):
            print(f"{i}. {split}")
        
        split_choice = input("Dataset number: ").strip()
        
        try:
            selected_model = models[int(model_choice) - 1]
            selected_mode = modes[int(mode_choice) - 1]
            selected_split = splits[int(split_choice) - 1]
            
            evaluations_to_run.append({
                "model": selected_model,
                "few_shot": selected_mode["few_shot"],
                "split": selected_split,
                "data_file": data_files[selected_split]
            })
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    
    else:
        print("Invalid choice")
        return
    
    # Run evaluations
    print(f"\nStarting {len(evaluations_to_run)} evaluations...")
    
    successful = 0
    failed = 0
    
    for i, eval_config in enumerate(evaluations_to_run, 1):
        print(f"\n\nEvaluation {i}/{len(evaluations_to_run)}")
        
        success = run_evaluation(
            model_name=eval_config["model"],
            few_shot_mode=eval_config["few_shot"],
            data_split=eval_config["split"],
            data_file=eval_config["data_file"],
            url=url
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Ask whether to continue if there are more evaluations
        if i < len(evaluations_to_run):
            continue_eval = input(f"\nContinue to next evaluation? (y/n): ").lower()
            if continue_eval != 'y':
                break
    
    print(f"\n\nBATCH EVALUATION COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")

if __name__ == "__main__":
    main()
