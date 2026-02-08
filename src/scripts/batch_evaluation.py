# batch_evaluation.py  –  Py 3.7+ compatible
import os
import json
import datetime
from typing import Optional, List, Dict, Any, Tuple

from evaluation import evaluate_thinking_model, evaluate_model


class BatchEvaluator:
    def __init__(
        self, 
        base_output_dir: str = "evaluation_results",
        model_name: str = "thewindmom/llama3-med42-8b:latest",
        use_thinking_model: bool = False,
        few_shot_mode: bool = False,
        num_examples: int = 3,  # Number of few-shot examples to use
        data_split: str = "train"  # Can be "train" or "dev"
    ) -> None:
        self.base_output_dir = base_output_dir
        # Sanitize model name for filesystem - replace invalid characters
        self.model_name = model_name.replace('/', '_').replace(':', '-').replace('\\', '_')
        
        # Adjust model directory based on mode and data split
        mode_suffix = "_few_shot" if few_shot_mode else "_zero_shot"
        self.model_dir = os.path.join(base_output_dir, f"{self.model_name}{mode_suffix}", data_split)
        self.results_dir = os.path.join(self.model_dir, "batches")
        self.progress_file = os.path.join(self.model_dir, "evaluation_progress.json")
        
        self.use_thinking_model = use_thinking_model
        self.few_shot_mode = few_shot_mode
        self.num_examples = num_examples
        self.data_split = data_split
        self._setup_directories()

    # ──────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────
    def _setup_directories(self) -> None:
        # Create model-specific directories
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_progress(self) -> Dict[str, Any]:
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
        Persist progress + aggregate metrics.
        Expects `batch_results` to be a list of per‑question dicts.
        """
        progress: Dict[str, Any] = {
            "last_batch": batch_num,
            "total_processed": total_processed,
            "last_run": datetime.datetime.now().isoformat(),
            "cumulative_metrics": {
                "total_questions": 0,
                "correct_answers": 0,
                "overall_accuracy": 0.0,
                "total_inference_time": 0.0,
                "average_inference_time": 0.0,
                "total_prompt_tokens": 0,
                "average_tokens_per_question": 0.0,
            },
            "batches": [],
        }

        # keep existing history
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                prior = json.load(f)
                progress["batches"] = prior.get("batches", [])
                progress["cumulative_metrics"] = prior.get(
                    "cumulative_metrics", progress["cumulative_metrics"]
                )

        # add this batch
        if batch_results:
            correct = sum(1 for r in batch_results if r["is_correct"])
            tot_inf_time = sum(r["metrics"]["inference_time"] for r in batch_results)
            tot_tokens = sum(r["metrics"]["prompt_eval_count"] for r in batch_results)

            batch_metrics: Dict[str, Any] = {
                "batch_number": batch_num,
                "processed_count": len(batch_results),
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics": {
                    "correct_answers": correct,
                    "batch_accuracy": (correct / len(batch_results)) * 100,
                    "total_inference_time": tot_inf_time,
                    "average_inference_time": tot_inf_time / len(batch_results),
                    "total_prompt_tokens": tot_tokens,
                },
            }
            progress["batches"].append(batch_metrics)

            # update cumulative totals
            cum = progress["cumulative_metrics"]
            cum["total_questions"] += batch_metrics["processed_count"]
            cum["correct_answers"] += correct
            cum["overall_accuracy"] = (cum["correct_answers"] / cum["total_questions"]) * 100
            cum["total_inference_time"] += tot_inf_time
            cum["average_inference_time"] = cum["total_inference_time"] / cum["total_questions"]
            cum["total_prompt_tokens"] += tot_tokens
            cum["average_tokens_per_question"] = cum["total_prompt_tokens"] / cum["total_questions"]

        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        return progress

    def _get_few_shot_examples(self, all_data: List[Dict], current_question: Dict) -> List[Dict]:
        """Get relevant few-shot examples from the training data"""
        # Filter examples from the same subject/topic
        relevant_examples = [
            q for q in all_data 
            if q['id'] != current_question['id'] and 
               q['subject_name'] == current_question['subject_name']
        ]
        
        # Sort by topic similarity if topic is available
        if current_question.get('topic_name'):
            relevant_examples.sort(
                key=lambda x: x.get('topic_name') == current_question['topic_name'], 
                reverse=True
            )
            
        # Return the top N most relevant examples
        return relevant_examples[:self.num_examples]

    def _format_few_shot_prompt(self, examples: List[Dict], question: Dict) -> str:
        """Format the few-shot examples and current question into a prompt"""
        prompt = "Answer the following medical questions. Here are some examples:\n\n"
        
        # Add examples
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"A) {ex['opa']}\n"
            prompt += f"B) {ex['opb']}\n"
            prompt += f"C) {ex['opc']}\n"
            prompt += f"D) {ex['opd']}\n"
            prompt += f"Answer: {chr(64 + ex['cop'])}\n"
            if ex.get('exp'):
                prompt += f"Explanation: {ex['exp']}\n"
            prompt += "\n"
            
        # Add current question
        prompt += "Now answer this question:\n"
        prompt += f"Question: {question['question']}\n"
        prompt += f"A) {question['opa']}\n"
        prompt += f"B) {question['opb']}\n"
        prompt += f"C) {question['opc']}\n"
        prompt += f"D) {question['opd']}\n"
        
        return prompt

    # ──────────────────────────────────────────────────────────────
    # main entry
    # ──────────────────────────────────────────────────────────────
    def evaluate_batch(
        self,
        train_data: List[Dict[str, Any]],
        url: str,
        batch_size: int = 10_000,
        start_batch: Optional[int] = None,
    ) -> None:
        """
        Evaluate one batch of questions using either evaluate_thinking_model or evaluate_model.
        """
        # decide where to start
        if start_batch is None:
            start_batch = self._load_progress()["last_batch"] + 1

        start_idx = start_batch * batch_size
        if start_idx >= len(train_data):
            print("All data has been processed!")
            return

        end_idx = min(start_idx + batch_size, len(train_data))
        print(f"\nProcessing Batch {start_batch + 1}")
        print(f"Questions {start_idx} to {end_idx - 1}")
        print(f"Using model: {self.model_name}")
        print(f"Evaluation mode: {'Thinking' if self.use_thinking_model else 'Standard'}")

        batch_data = train_data[start_idx:end_idx]

        # run evaluation based on selected mode
        if self.use_thinking_model:
            accuracy, evaluation_data = evaluate_thinking_model(
                batch_data, url, save_results=False
            )
        else:
            accuracy, evaluation_data = evaluate_model(
                batch_data, url, save_results=False
            )

        # new – extract list of per‑question dicts
        batch_results: List[Dict[str, Any]] = evaluation_data["results"]

        # save progress
        progress = self._save_progress(start_batch, end_idx, batch_results)

        # write batch file with model info
        batch_file_path = os.path.join(
            self.results_dir,
            f"{self.model_name}_batch_{start_batch}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json",
        )
        with open(batch_file_path, "w") as f:
            json.dump(
                {
                    "batch_metadata": {
                        "model_name": self.model_name,
                        "evaluation_mode": "thinking" if self.use_thinking_model else "standard",
                        "batch_number": start_batch,
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    "evaluation_results": evaluation_data,  # full dict
                    "accuracy": accuracy,
                },
                f,
                indent=2,
            )

        # console summary
        print(f"\nBatch {start_batch + 1} complete → saved to {batch_file_path}")
        cum = progress["cumulative_metrics"]
        print(
            f"Cumulative: {cum['total_questions']} Qs | "
            f"{cum['overall_accuracy']:.2f}% acc | "
            f"{cum['average_inference_time']:.2f}s avg inf."
        )

    # ──────────────────────────────────────────────────────────────
    # reset utility
    # ──────────────────────────────────────────────────────────────
    def reset_evaluation(self, confirm: bool = True) -> bool:
        if confirm:
            ans = input("This will delete ALL evaluation progress. Continue? (y/n): ")
            if ans.lower() != "y":
                print("Reset aborted.")
                return False
        try:
            if os.path.exists(self.results_dir):
                for f_name in os.listdir(self.results_dir):
                    f_path = os.path.join(self.results_dir, f_name)
                    if os.path.isfile(f_path):
                        os.remove(f_path)
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            self._setup_directories()
            print("Evaluation reset complete.")
            return True
        except Exception as exc:
            print(f"Error during reset: {exc}")
            return False
