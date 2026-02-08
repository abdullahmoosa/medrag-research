"""
Multi-Agent Medical Question Answering System using CrewAI
===========================================================

This script implements a two-agent system that mimics the real-world medical education
scenario of a senior resident presenting cases to an attending physician for validation.

Agents:
1. Senior Medical Resident - Provides initial medical analysis
2. Senior Attending Physician - Reviews and validates the analysis

The system includes a feedback loop where the attending can request revisions
if the initial analysis has issues.
"""

import os
import json
import datetime
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

try:
    from crewai import Agent, Task, Crew
    try:
        from langchain_community.llms import Ollama
        print("✅ Using LangChain Community Ollama LLM")
    except ImportError:
        try:
            from langchain.llms import Ollama  
            print("✅ Using LangChain Ollama LLM")
        except ImportError:
            print("❌ Could not import Ollama LLM")
            exit(1)
except ImportError as e:
    print(f"Error importing CrewAI: {e}")
    print("Please install CrewAI with: pip install crewai")
    print("Please install LangChain with: pip install langchain langchain-community")
    exit(1)

# Direct Ollama LLM wrapper that bypasses AgentExecutor issues
class DirectOllamaLLM:
    """Direct Ollama LLM wrapper that bypasses AgentExecutor for better performance"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """Direct generation without agent chains"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"

# Performance optimization utilities
def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Utility functions
def get_correct_option(cop):
    """Map numeric cop to corresponding option letter"""
    option_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    return option_map.get(cop, None)

def extract_answer_from_response(response: str) -> str:
    """Extract the final answer letter from agent response with improved accuracy"""
    import re
    
    if not response:
        return None
    
    # Print response for debugging
    print(f"🔍 Extracting answer from: {response[:200]}...")
    
    # More precise patterns in order of preference
    patterns = [
        # Exact patterns for our expected format
        r"(?:correct\s+answer\s+is\s+)([ABCD])\)",  # "The correct answer is X)"
        r"(?:answer\s+is\s+)([ABCD])\)",             # "answer is X)"
        r"(?:final\s+answer\s+is\s+)([ABCD])\)",     # "final answer is X)"
        
        # Common answer formats
        r"(?:answer[:\s]+)([ABCD])\b",               # "Answer: X"
        r"(?:select|choose)\s+([ABCD])\b",           # "Select X"
        r"(?:option|choice)\s+([ABCD])\b",           # "Option X"
        
        # Specific medical answer patterns
        r"\b([ABCD])\s*is\s*(?:the\s*)?correct",     # "X is correct"
        r"\b([ABCD])\s*is\s*(?:the\s*)?best",        # "X is the best"
        r"(?:should\s+be\s+)([ABCD])\b",             # "should be X"
        
        # Legacy patterns for compatibility
        r"(?:answer[:\s]+)([ABCD])\)",
        r"(?:answer[:\s]+)([ABCD])",
        r"^([ABCD])\)",
        r"answer:\s*([ABCD])",
        r"select\s*([ABCD])",
        
        # Last resort patterns
        r"\b([ABCD])\b(?=\s*(?:\.|$))",              # Single letter followed by period or end
        r"\b([ABCD])\b"                              # Any single letter as last resort
    ]
    
    # Clean the response for better matching
    cleaned_response = response.strip()
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].upper()  # Take the last match
            print(f"✅ Found answer '{answer}' using pattern {i+1}: '{pattern}'")
            return answer
    
    # If no pattern matches, look for any single A, B, C, D in the last line
    lines = cleaned_response.split('\n')
    for line in reversed(lines[-3:]):  # Check last 3 lines
        single_letter_match = re.search(r'\b([ABCD])\b', line.upper())
        if single_letter_match:
            answer = single_letter_match.group(1)
            print(f"✅ Found answer '{answer}' in line: '{line.strip()}'")
            return answer
    
    print("❌ No answer pattern found")
    return None

class CrewAIMultiAgentMedicalQA:
    """Multi-agent system for medical question answering using CrewAI"""
    
    def __init__(self, 
                 resident_model: str = "thewindmom/llama3-med42-8b:latest",  # Medical model by default
                 attending_model: str = "meditron:latest",  # Keep attending as medical expert
                 base_output_dir: str = "evaluation_results",
                 performance_mode: str = "fast"):  # "fast", "balanced", "accurate"
        
        # Adjust models based on performance mode
        if performance_mode == "fast":
            self.resident_model = "thewindmom/llama3-med42-8b:latest"  # Medical model for fast mode
            self.attending_model = "thewindmom/llama3-med42-8b:latest"  # Same model for consistency
            self.max_iterations = 1
            self.timeout = 30
        elif performance_mode == "balanced":
            self.resident_model = resident_model
            self.attending_model = "thewindmom/llama3-med42-8b:latest"  # Use consistent model
            self.max_iterations = 1
            self.timeout = 45
        else:  # accurate
            self.resident_model = "thewindmom/llama3-med42-8b:latest"
            self.attending_model = "thewindmom/llama3-med42-8b:latest"  # Use same model for both
            self.max_iterations = 2
            self.timeout = 60
        
        self.base_output_dir = base_output_dir
        
        # Initialize LLMs using native Ollama support
        self.resident_llm = Ollama(
            model=self.resident_model,
            base_url="http://localhost:11434",
            temperature=0.3
        )
        self.attending_llm = Ollama(
            model=self.attending_model,
            base_url="http://localhost:11434", 
            temperature=0.3
        )
        
        # Setup output directories
        self._setup_directories()
        
        # Initialize agents
        self._initialize_agents()
    
    def _setup_directories(self):
        """Setup output directory structure"""
        # For batch processing, use a consistent directory without timestamp
        self.output_dir = Path(self.base_output_dir) / "crewai_multiagent_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # For individual runs, create timestamped subdirectory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.output_dir / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking file
        self.progress_file = self.results_dir / "progress.json"
    
    def _initialize_agents(self):
        """Initialize the resident and attending agents"""
        
        # Senior Medical Resident Agent
        self.resident_agent = Agent(
            role="Senior Medical Resident",
            goal="Solve medical questions with high accuracy using evidence-based reasoning",
            backstory="""You are a chief resident in the final year of your medical residency at a prestigious academic medical center.
            You have extensive medical knowledge and strong command of clinical concepts across specialties.
            You scored in the top percentile on your in-training examinations and are known for your thorough, 
            methodical approach to clinical problems. You excel at applying foundational medical science to clinical scenarios.""",
            verbose=True,
            allow_delegation=False,
            llm=self.resident_llm,
            tools=[],  # Explicitly no tools to avoid AgentExecutor issues
            max_iter=1,
            max_execution_time=45,  # Reduced timeout
            step_callback=None  # Disable step callback to avoid tool loops
        )
        
        # Senior Attending Physician Agent
        self.attending_agent = Agent(
            role="Senior Attending Physician",
            goal="Ensure diagnostic accuracy through critical evaluation of medical reasoning",
            backstory="""You are a distinguished professor of medicine with 20+ years of clinical experience
            and a reputation for diagnostic excellence. You've authored textbook chapters in your specialty
            and have served as an examiner for medical licensing boards for over a decade.
            You have exceptional ability to identify subtle diagnostic errors and reasoning flaws.""",
            verbose=True,
            allow_delegation=False,
            llm=self.attending_llm,
            tools=[],  # Explicitly no tools to avoid AgentExecutor issues
            max_iter=1,
            max_execution_time=45,  # Reduced timeout
            step_callback=None  # Disable step callback to avoid tool loops
        )
    
    def create_resident_task(self, question_data: Dict) -> Task:
        """Create task for resident analysis with improved prompt"""
        description = f"""
        As a medical resident, analyze this question systematically:
        
        QUESTION: {question_data['question']}
        
        OPTIONS:
        A) {question_data['opa']}
        B) {question_data['opb']}
        C) {question_data['opc']}
        D) {question_data['opd']}
        
        APPROACH:
        1. Identify the key medical concept being tested
        2. Consider relevant pathophysiology, anatomy, or clinical guidelines
        3. Eliminate clearly incorrect options
        4. Choose the most appropriate answer based on medical evidence
        
        FORMAT YOUR RESPONSE:
        - Provide your clinical reasoning (2-3 sentences)
        - State your conclusion clearly
        - End with: "The correct answer is X)" where X is A, B, C, or D
        
        Be concise but thorough in your medical reasoning.
        """
        
        return Task(
            description=description,
            agent=self.resident_agent,
            expected_output="Medical analysis concluding with 'The correct answer is X)' where X is A, B, C, or D."
        )
    
    def create_attending_task(self, question_data: Dict, resident_analysis: str) -> Task:
        """Create task for attending review with improved oversight"""
        resident_answer = extract_answer_from_response(resident_analysis)
        
        description = f"""
        As an attending physician, review this resident's medical analysis:
        
        QUESTION: {question_data['question']}
        
        OPTIONS:
        A) {question_data['opa']}
        B) {question_data['opb']}
        C) {question_data['opc']}
        D) {question_data['opd']}
        
        RESIDENT'S CHOICE: {resident_answer}
        RESIDENT'S REASONING: {resident_analysis[:300]}...
        
        EVALUATION CRITERIA:
        - Is the medical reasoning sound?
        - Is the answer choice appropriate for the clinical scenario?
        - Are there any significant errors in understanding?
        
        RESPOND WITH ONE OF:
        1. "APPROVED - The analysis and answer are correct."
        2. "INCORRECT: The correct answer should be X" (where X is A, B, C, or D)
        
        Be decisive and provide clear feedback based on medical accuracy.
        """
        
        return Task(
            description=description,
            agent=self.attending_agent,
            expected_output="Either 'APPROVED - The analysis and answer are correct.' or 'INCORRECT: The correct answer should be X' where X is A, B, C, or D."
        )
    
    def process_single_question(self, question_data: Dict, max_iterations: int = 1) -> Dict:  # Reduced from 2 to 1
        """Process a single question through the multi-agent workflow"""
        
        start_time = datetime.datetime.now()
        iterations = []
        
        try:
            print(f"Processing question: {question_data['question'][:100]}...")
            
            # Create and execute resident task
            resident_task = self.create_resident_task(question_data)
            resident_crew = Crew(
                agents=[self.resident_agent],
                tasks=[resident_task],
                verbose=True,  # Re-enabled for debugging
                process="sequential"
            )
            
            print("Getting initial analysis from resident...")
            resident_result = resident_crew.kickoff()
            resident_analysis = str(resident_result)
            
            # Extract initial answer
            initial_answer = extract_answer_from_response(resident_analysis)
            print(f"Resident's initial answer: {initial_answer}")
            
            current_analysis = resident_analysis
            approved = False
            
            for iteration in range(max_iterations):
                print(f"Iteration {iteration + 1}: Getting feedback from attending...")
                
                # Create and execute attending task
                attending_task = self.create_attending_task(question_data, current_analysis)
                attending_crew = Crew(
                    agents=[self.attending_agent],
                    tasks=[attending_task],
                    verbose=True,  # Re-enabled for debugging
                    process="sequential"
                )
                
                attending_result = attending_crew.kickoff()
                attending_feedback = str(attending_result)
                
                # Check if approved
                if "APPROVED" in attending_feedback.upper():
                    approved = True
                    print("Attending approved the analysis!")
                    final_answer = extract_answer_from_response(current_analysis)
                    break
                
                print("Attending requested revision...")
                
                # If not approved and not the last iteration, get revision
                if iteration < max_iterations - 1:
                    revision_description = f"""
                    Please revise your analysis of this medical case based on the attending physician's feedback:
                    
                    Original Question: "{question_data['question']}"
                    
                    Options:
                    A) {question_data['opa']}
                    B) {question_data['opb']}
                    C) {question_data['opc']}
                    D) {question_data['opd']}
                    
                    Attending Physician's Feedback:
                    {attending_feedback}
                    
                    Please provide an improved analysis addressing the feedback and select your final answer.
                    End your response with: "The correct answer is X)" where X is A, B, C, or D.
                    """
                    
                    revision_task = Task(
                        description=revision_description,
                        agent=self.resident_agent,
                        expected_output="A revised medical analysis ending with 'The correct answer is X)' where X is A, B, C, or D."
                    )
                    
                    revision_crew = Crew(
                        agents=[self.resident_agent],
                        tasks=[revision_task],
                        verbose=True,  # Re-enabled for debugging
                        process="sequential"
                    )
                    
                    revision_result = revision_crew.kickoff()
                    current_analysis = str(revision_result)
                    print(f"Resident's revised answer: {extract_answer_from_response(current_analysis)}")
                
                # Store iteration details
                iterations.append({
                    "iteration": iteration + 1,
                    "resident_analysis": current_analysis,
                    "attending_feedback": attending_feedback,
                    "approved": "APPROVED" in attending_feedback.upper()
                })
            
            # Final answer extraction with attending corrections
            if approved:
                final_answer = extract_answer_from_response(current_analysis)
            else:
                # Check if attending provided a correction in their feedback
                attending_correction = extract_answer_from_response(attending_feedback)
                if attending_correction:
                    final_answer = attending_correction
                    print(f"Using attending's correction: {final_answer}")
                else:
                    final_answer = extract_answer_from_response(current_analysis)
                    print(f"Using resident's answer despite no approval: {final_answer}")
            
            correct_answer = get_correct_option(question_data['cop'])
            is_correct = correct_answer == final_answer
            
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Prepare result
            result = {
                "question_id": question_data.get('id', ''),
                "question": question_data['question'],
                "options": {
                    "A": question_data['opa'],
                    "B": question_data['opb'],
                    "C": question_data['opc'],
                    "D": question_data['opd']
                },
                "correct_answer": correct_answer,
                "predicted_answer": final_answer,
                "is_correct": is_correct,
                "initial_answer": initial_answer,
                "final_approved": approved,
                "iterations": iterations,
                "processing_time": processing_time,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            print(f"Final result: {correct_answer} (correct) vs {final_answer} (predicted) - {'✓' if result['is_correct'] else '✗'}")
            
            return result
            
        except Exception as e:
            print(f"Error processing question: {e}")
            return {
                "question_id": question_data.get('id', ''),
                "error": str(e),
                "processing_time": (datetime.datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def process_single_question_fast(self, question_data: Dict) -> Dict:
        """Improved fast processing method that bypasses CrewAI for speed while maintaining accuracy"""
        start_time = datetime.datetime.now()
        
        try:
            print(f"⚡ Fast processing question: {question_data['question'][:100]}...")
            
            # Direct LLM calls
            direct_resident = DirectOllamaLLM(self.resident_model)
            direct_attending = DirectOllamaLLM(self.attending_model)
            
            # Improved resident analysis prompt
            resident_prompt = f"""
As a medical expert, analyze this question systematically:

QUESTION: {question_data['question']}

OPTIONS:
A) {question_data['opa']}
B) {question_data['opb']}
C) {question_data['opc']}
D) {question_data['opd']}

ANALYSIS APPROACH:
1. Identify the key medical concept
2. Apply relevant clinical knowledge
3. Eliminate incorrect options
4. Select the best answer

Provide your reasoning briefly and conclude with: "The correct answer is X)" where X is A, B, C, or D.
"""
            
            print("🩺 Getting analysis from resident...")
            resident_response = direct_resident.generate(resident_prompt, max_tokens=200, temperature=0.1)
            resident_answer = extract_answer_from_response(resident_response)
            print(f"👨‍⚕️ Resident answer: {resident_answer}")
            
            # Improved attending validation with corrective feedback
            attending_prompt = f"""
As a senior attending physician, review this medical analysis:

QUESTION: {question_data['question']}

OPTIONS:
A) {question_data['opa']}
B) {question_data['opb']}
C) {question_data['opc']}
D) {question_data['opd']}

RESIDENT'S CHOICE: {resident_answer}

Evaluate the choice. Respond with:
- "APPROVED" if the answer is medically correct
- "INCORRECT: The correct answer should be X" if wrong (where X is A, B, C, or D)

Be decisive based on medical accuracy.
"""
            
            print("👩‍⚕️ Getting attending validation...")
            attending_response = direct_attending.generate(attending_prompt, max_tokens=80, temperature=0.1)
            
            # Check if attending provided a correction
            attending_correction = extract_answer_from_response(attending_response)
            if attending_correction and "INCORRECT" in attending_response.upper():
                final_answer = attending_correction
                approved = False
                print(f"👩‍⚕️ Attending corrected to: {final_answer}")
            else:
                final_answer = resident_answer
                approved = "APPROVED" in attending_response.upper()
                print(f"👩‍⚕️ Attending decision: {'APPROVED' if approved else 'NOT APPROVED'}")
            
            # Get correct answer for evaluation
            correct_answer = get_correct_option(question_data['cop'])
            is_correct = final_answer == correct_answer
            
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            
            print(f"✅ Result: {final_answer} ({'✓ CORRECT' if is_correct else '✗ INCORRECT, should be ' + correct_answer})")
            
            return {
                "question_id": question_data['id'],
                "resident_analysis": resident_response,
                "attending_feedback": attending_response,
                "iterations": 1,
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "approved": approved,
                "processing_time": processing_time,
                "timestamp": start_time.isoformat(),
                "method": "fast_improved"
            }
            
        except Exception as e:
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"❌ Error in fast processing: {e}")
            return {
                "question_id": question_data['id'],
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": start_time.isoformat(),
                "method": "fast_improved"
            }

    def evaluate_dataset(self, data: List[Dict], max_questions: Optional[int] = None, use_fast_mode: bool = False) -> Dict:
        """Evaluate a dataset using the multi-agent system"""
        
        start_time = datetime.datetime.now()
        
        # Limit questions if specified
        questions = data[:max_questions] if max_questions else data
        total_questions = len(questions)
        
        mode_str = "FAST" if use_fast_mode else "STANDARD"
        print(f"Starting CrewAI multi-agent evaluation of {total_questions} questions ({mode_str} MODE)")
        print(f"Resident Model: {self.resident_model}")
        print(f"Attending Model: {self.attending_model}")
        
        results = []
        correct_count = 0
        error_count = 0
        approved_count = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}/{total_questions}")
            print(f"{'='*60}")
            
            # Choose processing method based on mode
            if use_fast_mode:
                result = self.process_single_question_fast(question)
            else:
                result = self.process_single_question(question)
                
            results.append(result)
            
            if 'error' not in result:
                if result['is_correct']:
                    correct_count += 1
                if result.get('final_approved', False) or result.get('approved', False):
                    approved_count += 1
            else:
                error_count += 1
                print(f"Error in question {i}: {result['error']}")
            
            # Print progress
            current_accuracy = (correct_count / i) * 100
            print(f"Current accuracy: {current_accuracy:.2f}%")
            
            # Save progress every 5 questions
            if i % 5 == 0:
                self._save_progress(results, i, total_questions)
        
        end_time = datetime.datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate final metrics
        final_accuracy = (correct_count / total_questions) * 100
        approval_rate = (approved_count / (total_questions - error_count)) * 100 if total_questions > error_count else 0
        avg_processing_time = sum(r.get('processing_time', 0) for r in results if 'error' not in r) / (total_questions - error_count) if total_questions > error_count else 0
        
        # Prepare final results
        evaluation_results = {
            "metadata": {
                "total_questions": total_questions,
                "correct_answers": correct_count,
                "final_accuracy": final_accuracy,
                "approval_rate": approval_rate,
                "error_count": error_count,
                "resident_model": self.resident_model,
                "attending_model": self.attending_model,
                "evaluation_type": "crewai_multiagent",
                "timestamp": datetime.datetime.now().isoformat(),
                "total_evaluation_time": total_time,
                "average_processing_time": avg_processing_time
            },
            "results": results
        }
        
        # Save final results
        self._save_final_results(evaluation_results)
        
        print(f"\n{'='*60}")
        print(f"CREWAI MULTI-AGENT EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_count}")
        print(f"Final Accuracy: {final_accuracy:.2f}%")
        print(f"Approval Rate: {approval_rate:.2f}%")
        print(f"Error Count: {error_count}")
        print(f"Average Processing Time: {avg_processing_time:.2f} seconds")
        print(f"Total Evaluation Time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.results_dir}")
        
        return evaluation_results
    
    def _save_progress(self, results: List[Dict], current: int, total: int):
        """Save progress during evaluation"""
        progress_data = {
            "current_question": current,
            "total_questions": total,
            "last_updated": datetime.datetime.now().isoformat(),
            "results_so_far": len(results),
            "accuracy_so_far": sum(1 for r in results if r.get('is_correct', False)) / len(results) * 100 if results else 0
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _save_final_results(self, evaluation_results: Dict):
        """Save final evaluation results"""
        # Save detailed results
        results_file = self.results_dir / "crewai_multiagent_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save summary
        summary_file = self.results_dir / "evaluation_summary.json"
        summary = {
            "summary": evaluation_results["metadata"],
            "file_location": str(results_file)
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_next_batch_number(self) -> int:
        """Get the next batch number to process based on existing progress"""
        progress_file = os.path.join(str(self.output_dir), "evaluation_progress.json")
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                return progress.get("last_batch", -1) + 1
            except:
                return 0
        return 0
    
    def load_progress(self) -> Dict:
        """Load existing evaluation progress"""
        progress_file = os.path.join(str(self.output_dir), "evaluation_progress.json")
        
        default_progress = {
            "last_batch": -1,
            "total_processed": 0,
            "last_run": None,
            "cumulative_metrics": {
                "total_questions": 0,
                "correct_answers": 0,
                "overall_accuracy": 0.0,
                "total_time": 0.0,
                "average_time": 0.0,
                "approval_rate": 0.0
            },
            "batches": []
        }
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except:
                return default_progress
        return default_progress
    
    def save_batch_progress(self, batch_number: int, batch_results: List[Dict], progress: Dict):
        """Save batch results and update progress"""
        timestamp = datetime.datetime.now()
        
        # Save individual batch results
        batch_dir = os.path.join(str(self.output_dir), "batches")
        os.makedirs(batch_dir, exist_ok=True)
        
        batch_filename = f"crewai_multiagent_batch_{batch_number}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        batch_filepath = os.path.join(batch_dir, batch_filename)
        
        # Calculate batch metrics
        correct_count = sum(1 for r in batch_results if r.get('is_correct', False))
        total_time = sum(r.get('processing_time', 0) for r in batch_results)
        approval_count = sum(1 for r in batch_results if r.get('approved', False))
        
        batch_metrics = {
            "batch_number": batch_number,
            "total_questions": len(batch_results),
            "correct_answers": correct_count,
            "batch_accuracy": (correct_count / len(batch_results)) * 100 if batch_results else 0,
            "total_time": total_time,
            "average_time": total_time / len(batch_results) if batch_results else 0,
            "approval_rate": (approval_count / len(batch_results)) * 100 if batch_results else 0,
            "timestamp": timestamp.isoformat()
        }
        
        batch_data = {
            "metadata": batch_metrics,
            "results": batch_results
        }
        
        with open(batch_filepath, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        # Update cumulative progress
        progress["last_batch"] = batch_number
        progress["total_processed"] += len(batch_results)
        progress["last_run"] = timestamp.isoformat()
        
        # Update cumulative metrics
        cum_metrics = progress["cumulative_metrics"]
        cum_metrics["total_questions"] += len(batch_results)
        cum_metrics["correct_answers"] += correct_count
        cum_metrics["overall_accuracy"] = (cum_metrics["correct_answers"] / cum_metrics["total_questions"]) * 100 if cum_metrics["total_questions"] > 0 else 0
        cum_metrics["total_time"] += total_time
        cum_metrics["average_time"] = cum_metrics["total_time"] / cum_metrics["total_questions"] if cum_metrics["total_questions"] > 0 else 0
        
        # Calculate overall approval rate
        total_approved = sum(batch["metrics"]["approval_rate"] * batch["metrics"]["total_questions"] / 100 for batch in progress["batches"])
        total_approved += approval_count
        cum_metrics["approval_rate"] = (total_approved / cum_metrics["total_questions"]) * 100 if cum_metrics["total_questions"] > 0 else 0
        
        # Add batch info to progress
        progress["batches"].append({
            "batch_number": batch_number,
            "processed_count": len(batch_results),
            "timestamp": timestamp.isoformat(),
            "metrics": {
                "correct_answers": correct_count,
                "batch_accuracy": batch_metrics["batch_accuracy"],
                "total_time": total_time,
                "average_time": batch_metrics["average_time"],
                "approval_rate": batch_metrics["approval_rate"]
            }
        })
        
        # Save progress file
        progress_file = os.path.join(str(self.output_dir), "evaluation_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"📁 Batch {batch_number} saved: {batch_filepath}")
        print(f"📊 Batch accuracy: {batch_metrics['batch_accuracy']:.2f}%")
        return batch_filepath

    def evaluate_dataset_with_batches(self, data: List[Dict], batch_size: int = 100, 
                                    start_batch: Optional[int] = None, 
                                    use_fast_mode: bool = False) -> Dict:
        """Evaluate dataset in batches with resume capability"""
        
        # Load existing progress
        progress = self.load_progress()
        
        # Determine starting batch
        if start_batch is None:
            start_batch = self.get_next_batch_number()
        
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        print(f"🔄 Starting batch processing:")
        print(f"   Total questions: {len(data)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Total batches: {total_batches}")
        print(f"   Starting from batch: {start_batch}")
        print(f"   Previously processed: {progress['total_processed']} questions")
        print(f"   Mode: {'Fast' if use_fast_mode else 'Standard CrewAI'}")
        
        for batch_num in range(start_batch, total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"PROCESSING BATCH {batch_num}")
            print(f"Questions {start_idx + 1} to {end_idx} of {len(data)}")
            print(f"{'='*60}")
            
            batch_results = []
            batch_start_time = datetime.datetime.now()
            
            for i, question_data in enumerate(batch_data, 1):
                print(f"\n--- Question {i}/{len(batch_data)} (Global: {start_idx + i}/{len(data)}) ---")
                
                if use_fast_mode:
                    result = self.process_single_question_fast(question_data)
                else:
                    result = self.process_single_question(question_data)
                
                batch_results.append(result)
                
                # Progress indicator
                if i % 10 == 0 or i == len(batch_data):
                    correct_so_far = sum(1 for r in batch_results if r.get('is_correct', False))
                    accuracy_so_far = (correct_so_far / len(batch_results)) * 100
                    print(f"📊 Batch progress: {i}/{len(batch_data)} | Accuracy: {accuracy_so_far:.1f}%")
            
            # Save batch results
            batch_filepath = self.save_batch_progress(batch_num, batch_results, progress)
            
            batch_time = (datetime.datetime.now() - batch_start_time).total_seconds()
            correct_count = sum(1 for r in batch_results if r.get('is_correct', False))
            
            print(f"\n✅ BATCH {batch_num} COMPLETE")
            print(f"   Processed: {len(batch_results)} questions")
            print(f"   Correct: {correct_count}")
            print(f"   Accuracy: {(correct_count/len(batch_results)*100):.2f}%")
            print(f"   Time: {batch_time:.1f} seconds")
            print(f"   Results saved: {os.path.basename(batch_filepath)}")
            
            # Update progress for next iteration
            progress = self.load_progress()
        
        # Final summary
        final_progress = self.load_progress()
        cum_metrics = final_progress["cumulative_metrics"]
        
        print(f"\n{'='*60}")
        print(f"🎉 ALL BATCHES COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {cum_metrics['total_questions']} questions")
        print(f"Overall accuracy: {cum_metrics['overall_accuracy']:.2f}%")
        print(f"Overall approval rate: {cum_metrics['approval_rate']:.2f}%")
        print(f"Total time: {cum_metrics['total_time']:.1f} seconds")
        print(f"Average time per question: {cum_metrics['average_time']:.2f} seconds")
        
        return final_progress

def load_medmcqa_data(file_path: str) -> List[Dict]:
    """Load MedMCQA data from JSON file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
    return data

def main():
    """Main evaluation function with batch processing support"""
    import sys
    
    # Parse command line arguments
    use_fast_mode = "--fast" in sys.argv
    max_questions = None  # Default to all questions
    batch_size = 100  # Default batch size
    start_batch = None  # Start from next available batch
    
    if "--questions" in sys.argv:
        try:
            idx = sys.argv.index("--questions")
            max_questions = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid --questions argument, processing all questions")
    
    if "--batch_size" in sys.argv:
        try:
            idx = sys.argv.index("--batch_size")
            batch_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid --batch_size argument, using default of 100")
    
    if "--start_batch" in sys.argv:
        try:
            idx = sys.argv.index("--start_batch")
            start_batch = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid --start_batch argument, will auto-detect next batch")
    
    # Configuration
    DEV_DATA_PATH = r'C:\Users\User\Downloads\nusrat\medrag\data\medmcqa\dev_stratified_sample.json'
    
    # Initialize multi-agent system
    print("🤖 Initializing CrewAI Multi-Agent Medical QA System...")
    if use_fast_mode:
        multiagent_qa = CrewAIMultiAgentMedicalQA(
            performance_mode="fast"
        )
        print("⚡ Using FAST mode (direct API calls)")
    else:
        multiagent_qa = CrewAIMultiAgentMedicalQA(
            resident_model="thewindmom/llama3-med42-8b:latest",
            attending_model="thewindmom/llama3-med42-8b:latest",
            base_output_dir="evaluation_results"
        )
        print("🔄 Using STANDARD mode (CrewAI agents)")
    
    # Load data
    print("📂 Loading evaluation data...")
    try:
        # Use dev data for evaluation
        dev_data = load_medmcqa_data(DEV_DATA_PATH)
        print(f"📊 Loaded {len(dev_data)} questions from dev set")
        
        # Limit questions if specified
        if max_questions and max_questions < len(dev_data):
            dev_data = dev_data[:max_questions]
            print(f"🎯 Limited to first {max_questions} questions")
        
        # Check if using batch mode
        if max_questions and max_questions <= 20:
            # Small test run - use regular evaluation
            print(f"🧪 Running small test with {len(dev_data)} questions...")
            evaluation_results = multiagent_qa.evaluate_dataset(
                dev_data, 
                max_questions=None,
                use_fast_mode=use_fast_mode
            )
        else:
            # Batch processing mode
            print(f"📦 Running batch processing mode...")
            evaluation_results = multiagent_qa.evaluate_dataset_with_batches(
                dev_data,
                batch_size=batch_size,
                start_batch=start_batch,
                use_fast_mode=use_fast_mode
            )
        
        return evaluation_results
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data file - {e}")
        return None
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
