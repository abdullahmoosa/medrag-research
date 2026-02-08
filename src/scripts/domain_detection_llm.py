#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical Domain Detection using LLM

This script uses a Large Language Model to detect the medical domain/subject of medical questions.
It processes questions from the MedMCQA dataset and evaluates the model's ability to 
correctly classify them into their respective medical domains using both zero-shot and few-shot approaches.
"""

import os
import json
import time
import datetime
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class DomainDetectionEvaluator:
    def __init__(
        self,
        base_output_dir: str = "evaluation_results",
        model_name: str = "thewindmom/llama3-med42-8b:latest",
        use_thinking_model: bool = False,
        few_shot_mode: bool = False,
        data_split: str = "dev"
    ) -> None:
        """
        Initialize the domain detection evaluator.
        
        Args:
            base_output_dir: Directory to store evaluation results
            model_name: Name of the model to use for prediction
            use_thinking_model: Whether to use thinking prompts
            few_shot_mode: Whether to use few-shot examples
            data_split: Which data split to use ('dev' or 'train')
        """
        self.base_output_dir = base_output_dir
        # Keep original model name for API calls
        self.original_model_name = model_name
        # Sanitize model name for filesystem only
        self.model_name = model_name.replace('/', '_').replace(':', '-').replace('\\', '_')
        
        # Define medical domains based on the prompt
        self.medical_domains = [
            "Anatomy", "Biochemistry", "Physiology", "Pharmacology", "Pathology", 
            "Medicine", "Surgery", "Ophthalmology", "ENT", "Orthopaedics", 
            "Pediatrics", "Gynaecology & Obstetrics", "Anaesthesia", "Psychiatry", 
            "Radiology", "Forensic Medicine", "Social & Preventive Medicine", 
            "Dental", "Microbiology", "Skin"
        ]
        
        # Set up directories for results - new structure: domain-detection/LLM/model_name/mode/split/
        mode_suffix = "_few_shot" if few_shot_mode else "_zero_shot"
        self.output_dir = os.path.join(
            base_output_dir, 
            "domain-detection",
            "LLM",
            f"{self.model_name}{mode_suffix}", 
            data_split
        )
        self.results_dir = os.path.join(self.output_dir, "batches")
        self.progress_file = os.path.join(self.output_dir, "evaluation_progress.json")
        
        self.use_thinking_model = use_thinking_model
        self.few_shot_mode = few_shot_mode
        self.data_split = data_split
        self._setup_directories()
        
        # Keep track of unique domains for metrics
        self.detected_domains = set()

    def _setup_directories(self) -> None:
        """Create necessary directories for storing results"""
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "domain-detection"), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "domain-detection", "LLM"), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from previous runs if any"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {"last_batch": -1, "total_processed": 0, "last_run": None}

    def _save_progress(
        self,
        batch_num: int,
        total_processed: int,
        batch_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Save progress and metrics from the current evaluation batch.
        
        Args:
            batch_num: Current batch number
            total_processed: Total number of questions processed
            batch_results: Results from the current batch
            
        Returns:
            Updated progress dictionary
        """
        progress: Dict[str, Any] = {
            "last_batch": batch_num,
            "total_processed": total_processed,
            "last_run": datetime.datetime.now().isoformat(),
            "evaluation_mode": "few_shot" if self.few_shot_mode else "zero_shot",
            "model_name": self.model_name,
            "data_split": self.data_split,
            "cumulative_metrics": {
                "total_questions": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0.0,
                "total_inference_time": 0.0,
                "average_inference_time": 0.0,
                "total_prompt_tokens": 0,
                "average_tokens_per_question": 0.0,
                "unique_domains_detected": 0,
                "domain_distribution": {}
            },
            "batches": [],
        }

        # Keep existing history
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                prior = json.load(f)
                progress["batches"] = prior.get("batches", [])
                progress["cumulative_metrics"] = prior.get(
                    "cumulative_metrics", progress["cumulative_metrics"]
                )

        # Add this batch
        if batch_results:
            correct = sum(1 for r in batch_results if r["is_correct"])
            tot_inf_time = sum(r["metrics"]["inference_time"] for r in batch_results)
            tot_tokens = sum(r["metrics"]["prompt_eval_count"] for r in batch_results)

            batch_metrics: Dict[str, Any] = {
                "batch_number": batch_num,
                "processed_count": len(batch_results),
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics": {
                    "correct_predictions": correct,
                    "batch_accuracy": (correct / len(batch_results)) * 100,
                    "total_inference_time": tot_inf_time,
                    "average_inference_time": tot_inf_time / len(batch_results),
                    "total_prompt_tokens": tot_tokens,
                },
            }
            progress["batches"].append(batch_metrics)

            # Update cumulative totals
            cum = progress["cumulative_metrics"]
            cum["total_questions"] += batch_metrics["processed_count"]
            cum["correct_predictions"] += correct
            cum["overall_accuracy"] = (cum["correct_predictions"] / cum["total_questions"]) * 100
            cum["total_inference_time"] += tot_inf_time
            cum["average_inference_time"] = cum["total_inference_time"] / cum["total_questions"]
            cum["total_prompt_tokens"] += tot_tokens
            cum["average_tokens_per_question"] = cum["total_prompt_tokens"] / cum["total_questions"]
            cum["unique_domains_detected"] = len(self.detected_domains)
            
            # Update domain distribution
            domain_counts = {}
            for domain in self.detected_domains:
                domain_counts[domain] = sum(1 for r in batch_results if r.get("predicted_domain") == domain)
            cum["domain_distribution"] = domain_counts

        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        return progress

    def reset_evaluation(self) -> None:
        """Reset the evaluation progress"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        
        # Clear any existing batch files
        for batch_file in Path(self.results_dir).glob("batch_*.json"):
            os.remove(batch_file)

    def detect_domain(self, question: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Send a question to the LLM and get its prediction for the medical domain.
        
        Args:
            question: Question data including the question text
            url: URL endpoint for the model API
            
        Returns:
            Dictionary containing the prediction results and metrics
        """
        start_time = datetime.datetime.now()
        
        # Store the actual domain/subject name
        true_domain = question.get("subject_name", "Unknown")
        
        # Prepare prompt for the model
        if self.few_shot_mode:
            prompt = self._create_few_shot_prompt(question)
        else:
            prompt = self._create_zero_shot_prompt(question)
        
        # Request parameters for the API
        request_data = {
            "model": self.original_model_name,  # Use original model name for API
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }
        
        try:
            # Debug: Print request details for first few requests
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 3:
                print(f"DEBUG - Request data: {request_data}")
            
            # Make the API request
            response = requests.post(url, json=request_data)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # Parse the response
            response_data = response.json()
            model_response = response_data.get("response", "")
            
            if self._debug_count <= 3:
                print(f"DEBUG - Response data: {response_data}")
                print(f"DEBUG - Model response: {model_response}")
            
            # Extract the predicted domain from the response
            predicted_domain = self._extract_domain_from_response(model_response)
            self.detected_domains.add(predicted_domain)
            
            # Calculate inference time
            end_time = datetime.datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            # Determine if the prediction is correct
            is_correct = self._check_prediction_correctness(predicted_domain, true_domain)
            
            # Compile results
            result = {
                "question_id": question.get("id", ""),
                "question_text": question.get("question", ""),
                "true_domain": true_domain,
                "predicted_domain": predicted_domain,
                "is_correct": is_correct,
                "raw_response": model_response,
                "prompt_type": "few_shot" if self.few_shot_mode else "zero_shot",
                "metrics": {
                    "inference_time": inference_time,
                    "prompt_eval_count": len(prompt.split()),
                    "eval_count": len(model_response.split()),
                }
            }
            
        except Exception as e:
            # Handle errors - print the actual error for debugging
            print(f"ERROR in detect_domain: {str(e)}")
            end_time = datetime.datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            result = {
                "question_id": question.get("id", ""),
                "question_text": question.get("question", ""),
                "true_domain": true_domain,
                "predicted_domain": "Error",
                "is_correct": False,
                "raw_response": f"Error: {str(e)}",
                "prompt_type": "few_shot" if self.few_shot_mode else "zero_shot",
                "metrics": {
                    "inference_time": inference_time,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                }
            }
            
        return result
        
    def _create_zero_shot_prompt(self, question: Dict[str, Any]) -> str:
        """
        Create a zero-shot prompt for domain detection.
        
        Args:
            question: Question data
            
        Returns:
            Formatted prompt string
        """
        domain_list = ", ".join(self.medical_domains)
        
        return f"""You are an expert medical doctor with extensive knowledge across all medical specialties. You have years of experience in medical education and are skilled at categorizing medical questions by their primary subject area.

You are given a medical multiple-choice question. Based on your expertise, choose the single most relevant subject area from the following list:
{{{domain_list}}}.

Question: "{question.get('question', '')}"

Answer with only the subject name from the list above."""

    def _create_few_shot_prompt(self, question: Dict[str, Any]) -> str:
        """
        Create a few-shot prompt for domain detection.
        
        Args:
            question: Question data
            
        Returns:
            Formatted prompt string
        """
        domain_list = ", ".join(self.medical_domains)
        
        return f"""You are an expert medical doctor with extensive knowledge across all medical specialties. You have years of experience in medical education and are highly skilled at categorizing medical questions by their primary subject area.

Classify each question into one of these medical subjects:
{{{domain_list}}}

Example 1:
Q: "Which vitamin is supplied from only animal source?"
Subject: Biochemistry

Example 2:
Q: "Following endarterectomy on the right common carotid, a patient is found to be blind in the right eye. Which artery would be blocked?"
Subject: Ophthalmology

Example 3:
Q: "A 6-month-old infant presents with failure to thrive and recurrent respiratory infections. What is the most likely diagnosis?"
Subject: Pediatrics

Now classify this question:
Q: "{question.get('question', '')}"
Subject:"""

    def _extract_domain_from_response(self, response_text: str) -> str:
        """
        Extract the predicted domain from the model's response.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Extracted domain name or "Unknown" if not found
        """
        # Clean the response
        response = response_text.strip()
        
        # Handle different response formats
        if response.lower().startswith("subject:"):
            # Extract after "Subject:"
            parts = response.split(":", 1)
            if len(parts) > 1:
                response = parts[1].strip()
        
        # Remove any extra text after the domain name
        response = response.split('\n')[0].strip()
        
        # Check if the response matches any of our known domains (case insensitive)
        response_lower = response.lower()
        for domain in self.medical_domains:
            if domain.lower() == response_lower:
                return domain
        
        # Check for common variations
        domain_variations = {
            "gynecology": "Gynaecology & Obstetrics",
            "obstetrics": "Gynaecology & Obstetrics",
            "ob/gyn": "Gynaecology & Obstetrics",
            "obgyn": "Gynaecology & Obstetrics",
            "gynecology & obstetrics": "Gynaecology & Obstetrics",
            "obstetrics & gynecology": "Gynaecology & Obstetrics",
            "dermatology": "Skin",
            "orthopedics": "Orthopaedics",
            "anesthesia": "Anaesthesia",
            "anesthesiology": "Anaesthesia",
            "preventive medicine": "Social & Preventive Medicine",
            "community medicine": "Social & Preventive Medicine",
            "public health": "Social & Preventive Medicine",
            "internal medicine": "Medicine",
            "general medicine": "Medicine",
            "general surgery": "Surgery",
        }
        
        if response_lower in domain_variations:
            return domain_variations[response_lower]
        
        # If no match found, return the original response (will be marked as incorrect)
        return response if response else "Unknown"

    def _check_prediction_correctness(self, predicted: str, actual: str) -> bool:
        """
        Check if the predicted domain matches the actual domain.
        Handles case insensitivity and some variations in naming.
        
        Args:
            predicted: Predicted domain name
            actual: Actual domain name
            
        Returns:
            True if prediction is correct, False otherwise
        """
        # Normalize both strings for comparison
        predicted_norm = predicted.lower().strip()
        actual_norm = actual.lower().strip()
        
        # Direct match
        if predicted_norm == actual_norm:
            return True
        
        # Handle common variations
        variations = {
            "gynaecology & obstetrics": ["gynecology", "obstetrics", "ob/gyn", "obgyn", 
                                        "gynecology & obstetrics", "obstetrics & gynecology"],
            "skin": ["dermatology"],
            "orthopaedics": ["orthopedics"],
            "anaesthesia": ["anesthesia", "anesthesiology"],
            "social & preventive medicine": ["preventive medicine", "community medicine", "public health"],
            "medicine": ["internal medicine", "general medicine"],
            "surgery": ["general surgery"],
        }
        
        # Check if actual matches any variation of predicted
        for standard, var_list in variations.items():
            if actual_norm == standard and predicted_norm in var_list:
                return True
            if predicted_norm == standard and actual_norm in var_list:
                return True
                
        return False

    def evaluate_batch(self, questions: List[Dict[str, Any]], url: str, batch_size: int = 10) -> None:
        """
        Evaluate a batch of questions for domain detection.
        
        Args:
            questions: List of question data dictionaries
            url: URL endpoint for the model API
            batch_size: Size of batches to process
        """
        # Load progress to determine where to restart from
        progress = self._load_progress()
        last_batch = progress["last_batch"]
        total_processed = progress["total_processed"]
        
        # Process in batches
        num_batches = (len(questions) + batch_size - 1) // batch_size
        
        print(f"Starting evaluation with {self.model_name} in {'few-shot' if self.few_shot_mode else 'zero-shot'} mode")
        print(f"Total batches to process: {num_batches}")
        
        for batch_idx in range(last_batch + 1, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}, questions {start_idx + 1} to {end_idx}")
            
            # Process each question in the batch
            batch_results = []
            for q_idx, question in enumerate(batch_questions):
                try:
                    print(f"  Processing question {start_idx + q_idx + 1}/{len(questions)}: {question.get('id', '')}")
                    result = self.detect_domain(question, url)
                    batch_results.append(result)
                    
                    # Print result summary
                    print(f"    True: {result['true_domain']}, Predicted: {result['predicted_domain']}, Correct: {result['is_correct']}")
                    
                except Exception as e:
                    print(f"Error processing question {question.get('id', '')}: {str(e)}")
            
            # Save batch results
            batch_file = os.path.join(self.results_dir, f"batch_{batch_idx}.json")
            with open(batch_file, "w") as f:
                json.dump(batch_results, f, indent=2)
            
            # Update progress
            total_processed += len(batch_results)
            progress = self._save_progress(batch_idx, total_processed, batch_results)
            
            # Print batch summary
            if batch_results:
                batch_accuracy = sum(1 for r in batch_results if r['is_correct']) / len(batch_results) * 100
                print(f"Batch {batch_idx + 1} accuracy: {batch_accuracy:.2f}%")
            
            print(f"Completed batch {batch_idx + 1}, total processed: {total_processed}/{len(questions)}")
            
            # Print overall progress
            if progress and "cumulative_metrics" in progress:
                overall_acc = progress["cumulative_metrics"].get("overall_accuracy", 0)
                print(f"Overall accuracy so far: {overall_acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on medical domain detection")
    parser.add_argument("--model", default="thewindmom/llama3-med42-8b:latest", 
                        help="Model name to use for prediction")
    parser.add_argument("--thinking", action="store_true", 
                        help="Use thinking mode to encourage step-by-step reasoning")
    parser.add_argument("--few-shot", action="store_true",
                        help="Use few-shot examples in prompts")
    parser.add_argument("--output-dir", default="evaluation_results",
                        help="Directory to store evaluation results")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of questions to process in each batch")
    parser.add_argument("--data-file", default="C:/Users/User/Downloads/nusrat/medrag/data/medmcqa/dev_stratified_sample.json",
                        help="Path to the evaluation data file")
    parser.add_argument("--data-split", default="dev", choices=["dev", "train"],
                        help="Data split to use for evaluation")
    parser.add_argument("--reset", action="store_true",
                        help="Reset evaluation progress and start fresh")
    parser.add_argument("--url", default="http://localhost:11434/api/generate",
                        help="URL for the model API")
    
    args = parser.parse_args()
    
    # Load evaluation data
    with open(args.data_file, 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(eval_data)} questions for evaluation")
    
    # Initialize evaluator
    evaluator = DomainDetectionEvaluator(
        base_output_dir=args.output_dir,
        model_name=args.model,
        use_thinking_model=args.thinking,
        few_shot_mode=args.few_shot,
        data_split=args.data_split
    )
    
    # Reset if requested
    if args.reset:
        evaluator.reset_evaluation()
        print("Evaluation progress reset")
    
    # Run evaluation
    evaluator.evaluate_batch(eval_data, args.url, batch_size=args.batch_size)
    
    print("\nEvaluation complete!")
    
    # Print final summary
    progress_file = evaluator.progress_file
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            final_progress = json.load(f)
            
        metrics = final_progress.get("cumulative_metrics", {})
        print(f"\n{'='*50}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Mode: {'Few-shot' if args.few_shot else 'Zero-shot'}")
        print(f"Data Split: {args.data_split}")
        print(f"Total Questions: {metrics.get('total_questions', 0)}")
        print(f"Correct Predictions: {metrics.get('correct_predictions', 0)}")
        print(f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.2f}%")
        print(f"Average Inference Time: {metrics.get('average_inference_time', 0):.2f}s")
        print(f"Unique Domains Detected: {metrics.get('unique_domains_detected', 0)}")
        
        if "domain_distribution" in metrics:
            print(f"\nDomain Distribution:")
            for domain, count in metrics["domain_distribution"].items():
                print(f"  {domain}: {count}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
