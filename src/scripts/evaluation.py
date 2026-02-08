import requests
import json
import datetime
from pathlib import Path
from typing import Dict, Any


def parse_model_response(response_text):
    """Extract thinking process and final answer from model response"""
    
    # Initialize variables
    thinking = ""
    answer = ""
    answer_letter = None  # ensure defined even if pattern not found
    
    # Split the response into lines
    lines = response_text.strip().split('\n')
    
    # Track if we're in the thinking section
    in_thinking = False
    
    for line in lines:
        # Check for thinking section markers
        if line.strip() == "<think>":
            in_thinking = True
            continue
        elif line.strip() == "</think>":
            in_thinking = False
            continue
            
        # Collect thinking content
        if in_thinking:
            thinking += line + "\n"
        # Look for answer line
        elif line.startswith("The correct answer is"):
            answer = line.strip()
            # Extract just the letter from "The correct answer is X)"
            answer_letter = answer.split(")")[0].split()[-1]
    
    return {
        "thinking_process": thinking.strip(),
        "answer": answer_letter if answer else None
    }

def map_numeric_cop_to_letter(cop):
    """Map legacy numeric `cop` (1-4) to corresponding option letter."""
    option_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    return option_map.get(cop, None)


def _extract_options(question_data: Dict[str, Any]) -> Dict[str, str]:
    """Return a normalized options dict {letter: text}.

    Supports two schemas:
      1. Legacy MedMCQA style with opa/opb/opc/opd
      2. New MedQA style with an 'options' dict possibly containing A-E
    Ignores missing / None texts.
    """
    if isinstance(question_data.get("options"), dict):
        # Already in dict form (MedQA)
        # Keep natural letter ordering (A,B,C,...) by sorting keys
        return {k.upper(): v for k, v in sorted(question_data["options"].items()) if v is not None}
    # Legacy flat keys
    collected = {}
    for letter, key in zip(["A", "B", "C", "D", "E"], ["opa", "opb", "opc", "opd", "ope"]):
        if key in question_data and question_data[key] not in (None, ""):
            collected[letter] = question_data[key]
    return collected


def _determine_correct_option(question_data: Dict[str, Any]) -> str:
    """Determine the correct option letter from multiple schema variants.

    Priority order:
      1. Explicit 'correct_option_letter'
      2. 'answer_idx' (MedQA)
      3. Legacy numeric 'cop'
    Returns empty string if not resolvable.
    """
    if question_data.get("correct_option_letter"):
        return question_data["correct_option_letter"].upper()
    if question_data.get("answer_idx"):
        return str(question_data["answer_idx"]).upper()
    if question_data.get("cop") is not None:
        return map_numeric_cop_to_letter(question_data.get("cop")) or ""
    return ""


def process_single_question(question_data, url):
    """Process a single question through the model and verify the output.

    Dynamically adapts to question schemas (MedMCQA legacy or MedQA) and variable
    number of options (A-D or A-E). Ensures system prompt lists the allowable
    option letters so the model returns a single letter.
    """
    start_time = datetime.datetime.now()
    
    options_dict = _extract_options(question_data)
    allowed_letters = list(options_dict.keys())
    letters_joined = "/".join(allowed_letters)
    option_lines = "\n".join(f"{ltr}) {txt}" for ltr, txt in options_dict.items())

    system_suffix = f"({', '.join(allowed_letters[:-1])} or {allowed_letters[-1]})" if len(allowed_letters) > 1 else f"({allowed_letters[0]})"
    system_instruction = (
        "You are a medical doctor answering real-world exam questions. "
        f"Answer only with the single correct option letter {system_suffix} "
        "without any additional explanation."
    )
    payload = {
        "model": "thewindmom/llama3-med42-8b:latest",
        "temperature": 0,
        "system": system_instruction,
        "prompt": f"""
Question: "{question_data['question']}"

Options:
{option_lines}
""",
        "stream": False,
    }
    
    try:
        # Make API request
        response = requests.post(url, json=payload)
        data = response.json()
        end_time = datetime.datetime.now()
        inference_time = (end_time - start_time).total_seconds()

        question_domain = question_data.get('subject_name') or question_data.get('meta_info') or ""
        question_topic = question_data.get('topic_name') or ""
        question_id = question_data.get('id') or question_data.get('qid') or ""

        if "response" not in data:
            print(f"Error: Unexpected response format - {data}")
            return 0, None

        # Get model's prediction and correct answer
        model_response = data["response"].strip()
        correct_option = _determine_correct_option(question_data)
        # Extract first valid allowed letter from response
        predicted_option = None
        for ch in model_response:
            if ch.upper() in allowed_letters:
                predicted_option = ch.upper()
                break
        if predicted_option is None and model_response:
            # fallback naive parsing
            predicted_option = model_response.split(")")[0].strip().upper()[:1]
            if predicted_option not in allowed_letters:
                predicted_option = None

        # Extract additional metrics from the response
        metrics = {
            "inference_time": inference_time,
            "total_duration": data.get("total_duration", 0),
            "load_duration": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0)
        }

        # Create result dictionary
        result_dict = {
            "question_id": question_id,
            "question_domain": question_domain,
            "question_topic": question_topic,
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_option": predicted_option,
            "is_correct": (correct_option == predicted_option) if predicted_option else False,
            "model_response": model_response,
            "options": options_dict,
            "metrics": metrics
        }

        # Print metrics for current question
        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Total Duration: {metrics['total_duration']:.2f} seconds")
        print(f"Prompt Eval Count: {metrics['prompt_eval_count']}")

        return 1 if result_dict["is_correct"] else 0, result_dict

    except Exception as e:
        print(f"Error processing question: {e}")
        return 0, None

def evaluate_model(train_data, url, num_samples=None, save_results=True):
    """Evaluate model performance on multiple questions and save results"""
    
    start_time = datetime.datetime.now()
    samples = train_data if num_samples is None else train_data[:num_samples]
    
    total = len(samples)
    correct = 0
    all_results = []
    total_inference_time = 0
    total_prompt_tokens = 0
    
    for i, question in enumerate(samples):
        print(f"\nProcessing question {i+1}/{total}")
        result, result_dict = process_single_question(question, url)
        correct += result
        
        if result_dict:
            all_results.append(result_dict)
            total_inference_time += result_dict["metrics"]["inference_time"]
            total_prompt_tokens += result_dict["metrics"]["prompt_eval_count"]
        
        # Print current accuracy and average metrics
        current_accuracy = (correct / (i + 1)) * 100
        avg_inference_time = total_inference_time / (i + 1)
        print(f"Current Accuracy: {current_accuracy:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.2f} seconds")
    
    end_time = datetime.datetime.now()
    total_evaluation_time = (end_time - start_time).total_seconds()
    
    # Calculate final metrics
    final_accuracy = (correct / total) * 100
    avg_inference_time = total_inference_time / total
    avg_tokens_per_question = total_prompt_tokens / total
    
    print(f"\nFinal Results:")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time:.2f} seconds")
    print(f"Average Tokens per Question: {avg_tokens_per_question:.2f}")
    print(f"Total Evaluation Time: {total_evaluation_time:.2f} seconds")
    
    # Save results to a JSON file
    results_data = {
        "metadata": {
            "total_questions": total,
            "correct_answers": correct,
            "final_accuracy": final_accuracy,
            "model": "thewindmom/llama3-med42-8b:latest",
            "evaluation_type": "general_model",
            "timestamp": datetime.datetime.now().isoformat(),
            "performance_metrics": {
                "total_evaluation_time": total_evaluation_time,
                "total_inference_time": total_inference_time,
                "average_inference_time": avg_inference_time,
                "total_prompt_tokens": total_prompt_tokens,
                "average_tokens_per_question": avg_tokens_per_question
            }
        },
        "results": all_results
    }
        
        # output_file = "model_evaluation_results.json"
        # with open(output_file, "w") as f:
        #     json.dump(results_data, f, indent=2)
        # print(f"\nResults saved to {output_file}")
    
    return final_accuracy, results_data


def evaluate_thinking_model(train_data, url, num_samples=None, save_results=True, output_dir="results"):
    """
    Evaluate model performance with thinking process on multiple questions
    
    Args:
        train_data (list): List of question data dictionaries
        url (str): API endpoint URL
        num_samples (int, optional): Number of samples to evaluate. Defaults to None (all samples)
        save_results (bool, optional): Whether to save results to file. Defaults to True
        output_dir (str, optional): Directory to save results. Defaults to "results"
    
    Returns:
        tuple: (final_accuracy, results_data)
    """
    
    start_time = datetime.datetime.now()
    samples = train_data if num_samples is None else train_data[:num_samples]
    
    total = len(samples)
    correct = 0
    all_results = []
    total_inference_time = 0
    total_prompt_tokens = 0
    
    # Create output directory if it doesn't exist
    if save_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, question in enumerate(samples):
        print(f"\nProcessing question {i+1}/{total}")
        result, result_dict = process_single_question_with_thinking(question, url)
        correct += result
        
        if result_dict:
            all_results.append(result_dict)
            total_inference_time += result_dict["metrics"]["inference_time"]
            total_prompt_tokens += result_dict["metrics"]["prompt_eval_count"]
        
        # Print current metrics
        current_accuracy = (correct / (i + 1)) * 100
        avg_inference_time = total_inference_time / (i + 1)
        print(f"Current Accuracy: {current_accuracy:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.2f} seconds")
    
    end_time = datetime.datetime.now()
    total_evaluation_time = (end_time - start_time).total_seconds()
    
    # Calculate final metrics
    final_accuracy = (correct / total) * 100
    avg_inference_time = total_inference_time / total
    avg_tokens_per_question = total_prompt_tokens / total
    
    # Prepare results data
    results_data = {
        "metadata": {
            "total_questions": total,
            "correct_answers": correct,
            "final_accuracy": final_accuracy,
            "model": "deepseek-r1:8b",
            "evaluation_type": "thinking_model",
            "timestamp": datetime.datetime.now().isoformat(),
            "performance_metrics": {
                "total_evaluation_time": total_evaluation_time,
                "total_inference_time": total_inference_time,
                "average_inference_time": avg_inference_time,
                "total_prompt_tokens": total_prompt_tokens,
                "average_tokens_per_question": avg_tokens_per_question
            }
        },
        "results": all_results,
        "thinking_analysis": {
            "average_thinking_length": sum(len(r["thinking_process"].split()) 
                                        for r in all_results) / len(all_results),
            "correct_with_reasoning": sum(1 for r in all_results 
                                       if r["is_correct"] and r["thinking_process"]),
            "incorrect_with_reasoning": sum(1 for r in all_results 
                                         if not r["is_correct"] and r["thinking_process"])
        }
    }
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time:.2f} seconds")
    print(f"Average Tokens per Question: {avg_tokens_per_question:.2f}")
    print(f"Total Evaluation Time: {total_evaluation_time:.2f} seconds")
    
    # Save results
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"thinking_model_evaluation_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return final_accuracy, results_data

def process_single_question_with_thinking(question_data, url):
    """Process a single question through the model with thinking process.

    Adapted for variable option counts (A-D / A-E) and both schemas.
    """
    start_time = datetime.datetime.now()
    
    options_dict = _extract_options(question_data)
    allowed_letters = list(options_dict.keys())
    option_lines = "\n".join(f"{ltr}) {txt}" for ltr, txt in options_dict.items())
    allowed_str = ", ".join(allowed_letters[:-1]) + f" or {allowed_letters[-1]}" if len(allowed_letters) > 1 else allowed_letters[0]
    payload = {
    "model": "deepseek-r1:8b",
    "temperature": 0,
    "system": f"""You are a medical doctor answering real-world exam questions.
INSTRUCTIONS:
1. First, write your thinking process between <think> and </think> tags
2. Then, write ONLY ONE LINE with your answer in EXACTLY this format:
   'The correct answer is X)' where X is one of: {allowed_str}
3. Do not write anything else after your answer.
""",
    "prompt": f"""
Question: "{question_data['question']}"

Options:
{option_lines}

Remember: Your response must have thinking inside <think></think> tags and end with ONLY 'The correct answer is X)'
""",
    "stream": False,
    }
    
    try:
        # Make API request
        response = requests.post(url, json=payload)
        data = response.json()
        end_time = datetime.datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        
        if "response" not in data:
            print(f"Error: Unexpected response format - {data}")
            return 0, None
            
        # Parse and verify response
    parsed_response = parse_model_response(data["response"])
    correct_option = _determine_correct_option(question_data)
    predicted_option = parsed_response["answer"].upper() if parsed_response["answer"] else None
        
        # Create result dictionary
        result_dict = {
            "question_id": question_data.get('id', '') or question_data.get('qid', ''),
            "question_domain": question_data.get('subject_name', '') or question_data.get('meta_info', ''),
            "question_topic": question_data.get('topic_name', ''),
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_option": predicted_option,
            "is_correct": correct_option == predicted_option,
            "thinking_process": parsed_response["thinking_process"],
            "options": options_dict,
            "metrics": {
                "inference_time": inference_time,
                "total_duration": data.get("total_duration", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0)
            }
        }
        
        return 1 if result_dict["is_correct"] else 0, result_dict

    except Exception as e:
        print(f"Error processing question: {e}")
        return 0, None

# Import required modules
# import datetime

# # Run the evaluation pipeline
# url = "http://localhost:11434/api/generate"

# # Evaluate questions and save results
# accuracy, results = evaluate_model(train_data, url, num_samples=100, save_results=True)