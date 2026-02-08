#!/usr/bin/env python3
"""
Script to generate synthetic reasoning dataset using DeepSeek R1 via OpenRouter API.
This script processes the train_reasoning_sample.json file and gene                se        if self.progress_file.exists():
            self.progress_file.unlink()
            print("Progress file deleted")
        
        self.current_batch = 0
        self.stats = {
            'total_processed': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'api_errors': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'current_batch': 0,
            'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir)
        }
        
        print("Progress reset - starting fresh")progress_data.get('current_batch', 0)
                self.stats.update(progress_data.get('stats', {}))
                
                print(f"Loaded progress: starting from batch {self.current_batch}")
                return True
            except Exception as e:
                print(f"Could not load progress: {e}")easoning traces
for each medical question.
"""

import json
import os
import time
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime
import re
from pathlib import Path

# Load environment variables from root directory
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / ".env")

# Setup logging
def setup_logging(output_dir: Path):
    """Setup logging with proper file path in output directory."""
    log_file = output_dir / 'deepseek_reasoning_generation.log'
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

class DeepSeekReasoningGenerator:
    def __init__(self, api_key: str, rate_limit_rpm: int = 60, batch_size: int = 10, start_batch: int = 0, output_dir: str = None):
        """
        Initialize the DeepSeek reasoning generator.
        
        Args:
            api_key: OpenRouter API key
            rate_limit_rpm: Rate limit in requests per minute
            batch_size: Number of samples per batch
            start_batch: Batch number to start from (for resuming)
            output_dir: Output directory for results
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-r1"
        self.rate_limit_rpm = rate_limit_rpm
        self.min_delay = 60.0 / rate_limit_rpm  # Minimum delay between requests
        self.last_request_time = 0
        self.batch_size = batch_size
        self.start_batch = start_batch
        
        # Setup output directory
        if output_dir is None:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir.parent.parent / "evaluation_results" / "reasoning_dataset_generation"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = setup_logging(self.output_dir)
        
        # Progress tracking
        self.progress_file = self.output_dir / "reasoning_progress.json"
        self.current_batch = start_batch
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'api_errors': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'current_batch': start_batch,
            'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir)
        }
    
    def format_prompt(self, sample: Dict) -> str:
        """Format medical question for DeepSeek R1"""
        question = sample.get('question', '')
        opa = sample.get('opa', '')
        opb = sample.get('opb', '')
        opc = sample.get('opc', '')
        opd = sample.get('opd', '')
        explanation = sample.get('exp', '') or ''
        
        # Enhanced system prompt with explanation integration
        system_prompt = """You are an expert medical AI assistant that provides detailed reasoning for medical questions. 
You must think step-by-step through each medical question, considering relevant medical knowledge, differential diagnoses, 
pathophysiology, clinical presentation, and diagnostic criteria before arriving at your final answer.

Please provide your reasoning enclosed in <Reasoning></Reasoning> tags, followed by your final answer as a single letter (A, B, C, or D).

In your reasoning, consider:
1. Key medical concepts and terminology
2. Pathophysiology and disease mechanisms
3. Clinical presentation and symptoms
4. Differential diagnoses
5. Diagnostic criteria and methods
6. Treatment implications if relevant
7. No self talk or unnecessary commentary"""
        
        # Format the question with options
        formatted_prompt = f"""{system_prompt}

Medical Question: {question}

Options:
A) {opa}
B) {opb}
C) {opc}
D) {opd}"""
        
        # Always include explanation to improve reasoning quality
        if explanation.strip():
            formatted_prompt += f"""

Reference Explanation: {explanation}

Use this explanation to guide your reasoning process, but provide your own step-by-step analysis."""
        
        formatted_prompt += "\n\nPlease provide your detailed medical reasoning in <Reasoning></Reasoning> tags and then state your final answer."
        
        return formatted_prompt
    
    async def rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def save_progress(self):
        """Save current progress to file."""
        progress_data = {
            'current_batch': self.current_batch,
            'stats': self.stats,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.logger.info(f"Progress saved: batch {self.current_batch}")
    
    def load_progress(self):
        """Load progress from file if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                self.current_batch = progress_data.get('current_batch', 0)
                self.stats.update(progress_data.get('stats', {}))
                
                self.logger.info(f"Loaded progress: starting from batch {self.current_batch}")
                return True
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")
        
        return False
    
    def delete_progress(self):
        """Delete progress file and reset to start fresh."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info("Progress file deleted")
        
        self.current_batch = 0
        self.stats = {
            'total_processed': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'api_errors': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'current_batch': 0,
            'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir)
        }
        
        self.logger.info("Progress reset - starting fresh")
    
    async def call_deepseek_api(self, session: aiohttp.ClientSession, prompt: str) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        Call DeepSeek R1 via OpenRouter API.
        
        Args:
            session: aiohttp session
            prompt: Formatted prompt
            
        Returns:
            Tuple of (content, reasoning, usage_stats) or (None, None, None) if error
        """
        await self.rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/abdullahmoosa/medrag-research",  # Required by OpenRouter
            "X-Title": "MedMCQA Reasoning Generation"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,  # Low temperature for consistent reasoning
            "top_p": 0.9
        }
        
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    message = data['choices'][0]['message']
                    
                    # DeepSeek R1 has reasoning in a separate field
                    content = message.get('content', '')
                    reasoning = message.get('reasoning', '')
                    usage = data.get('usage', {})
                    
                    # Update token statistics
                    self.stats['total_input_tokens'] += usage.get('prompt_tokens', 0)
                    self.stats['total_output_tokens'] += usage.get('completion_tokens', 0)
                    
                    self.logger.debug(f"API call successful. Input tokens: {usage.get('prompt_tokens', 0)}, "
                               f"Output tokens: {usage.get('completion_tokens', 0)}")
                    
                    return content, reasoning, usage
                
                else:
                    error_text = await response.text()
                    self.logger.error(f"API error {response.status}: {error_text}")
                    self.stats['api_errors'] += 1
                    return None, None, None
                    
        except asyncio.TimeoutError:
            self.logger.error("API request timeout")
            self.stats['api_errors'] += 1
            return None, None, None
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            self.stats['api_errors'] += 1
            return None, None, None
    
    def extract_reasoning_and_answer(self, content: str, raw_reasoning: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract structured reasoning and final answer from DeepSeek R1 response.
        
        Args:
            content: Main response content from DeepSeek R1
            raw_reasoning: Raw thinking process from API reasoning field
            
        Returns:
            Tuple of (structured_reasoning, raw_reasoning, final_answer)
        """
        # Extract structured reasoning from <Reasoning></Reasoning> tags
        reasoning_pattern = r'<Reasoning>(.*?)</Reasoning>'
        reasoning_match = re.search(reasoning_pattern, content, re.DOTALL | re.IGNORECASE)
        structured_reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        # Extract final answer from content (after reasoning tags or from the whole content)
        if reasoning_match:
            after_reasoning = content[reasoning_match.end():].strip()
        else:
            after_reasoning = content.strip()
        
        # Look for final answer (A, B, C, or D)
        answer_patterns = [
            r'\b([ABCD])\b',  # Single letter
            r'answer\s*:?\s*([ABCD])',  # "answer: A" or "answer A"
            r'final\s*answer\s*:?\s*([ABCD])',  # "final answer: A"
            r'option\s*([ABCD])',  # "option A"
            r'correct\s*answer\s*:?\s*([ABCD])',  # "correct answer: A"
        ]
        
        final_answer = None
        # First try to find answer in the content after reasoning
        for pattern in answer_patterns:
            match = re.search(pattern, after_reasoning, re.IGNORECASE)
            if match:
                final_answer = match.group(1).upper()
                break
        
        # If no answer found in content, look in the entire content
        if not final_answer:
            for pattern in answer_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).upper()
                    break
        
        return structured_reasoning, raw_reasoning, final_answer
    
    def is_answer_correct(self, predicted_answer: str, correct_option: int) -> bool:
        """
        Check if the predicted answer matches the correct option.
        
        Args:
            predicted_answer: Predicted answer (A, B, C, D)
            correct_option: Correct option number (1, 2, 3, 4)
            
        Returns:
            True if correct, False otherwise
        """
        if not predicted_answer:
            return False
        
        # Convert option number to letter
        option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        correct_letter = option_map.get(correct_option)
        
        return predicted_answer.upper() == correct_letter
    
    async def process_sample(self, session: aiohttp.ClientSession, sample: Dict) -> Dict:
        """
        Process a single medical question sample.
        
        Args:
            session: aiohttp session
            sample: Medical question sample
            
        Returns:
            Processed sample with reasoning and correctness info
        """
        sample_id = sample.get('id', 'unknown')
        self.logger.info(f"Processing sample {sample_id}")
        
        # Format prompt
        prompt = self.format_prompt(sample)
        
        # Call API
        content, raw_reasoning, usage = await self.call_deepseek_api(session, prompt)
        
        if content is None:
            self.logger.warning(f"Failed to get response for sample {sample_id}")
            return {
                'id': sample.get('id'),
                'question': sample.get('question'),
                'opa': sample.get('opa'),
                'opb': sample.get('opb'),
                'opc': sample.get('opc'), 
                'opd': sample.get('opd'),
                'cop': sample.get('cop'),
                'exp': sample.get('exp'),
                'subject_name': sample.get('subject_name'),
                'structured_reasoning': None,
                'raw_reasoning': None,
                'predicted_answer': None,
                'is_correct': False,
                'error': 'api_error'
            }
        
        # Extract structured reasoning and answer
        structured_reasoning, processed_raw_reasoning, predicted_answer = self.extract_reasoning_and_answer(content, raw_reasoning)
        
        # Check correctness
        correct_option = sample.get('cop')
        is_correct = self.is_answer_correct(predicted_answer, correct_option)
        
        # Update statistics
        self.stats['total_processed'] += 1
        if is_correct:
            self.stats['correct_answers'] += 1
        else:
            self.stats['incorrect_answers'] += 1
        
        self.logger.info(f"Sample {sample_id}: Predicted={predicted_answer}, "
                   f"Correct={correct_option}, Match={is_correct}")
        
        # Return simplified result with only essential fields and both reasoning types
        result = {
            'id': sample.get('id'),
            'question': sample.get('question'),
            'opa': sample.get('opa'),
            'opb': sample.get('opb'), 
            'opc': sample.get('opc'),
            'opd': sample.get('opd'),
            'cop': sample.get('cop'),
            'exp': sample.get('exp'),
            'subject_name': sample.get('subject_name'),
            'structured_reasoning': structured_reasoning,  # Extracted from <Reasoning> tags
            'raw_reasoning': processed_raw_reasoning,      # Raw thinking from API reasoning field
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        }
        
        # Only include full response in debug mode
        if self.logger.level <= logging.DEBUG:
            result['full_content'] = content
            result['usage_stats'] = usage
        
        return result
    
    async def process_dataset(self, input_file: str, output_file: str, resume: bool = True):
        """
        Process the entire dataset with batch management.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            resume: Whether to resume from previous progress
        """
        self.logger.info(f"Starting dataset processing: {input_file} -> {output_file}")
        
        # Load existing progress if resuming
        if resume:
            self.load_progress()
        
        # Load dataset
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        total_samples = len(samples)
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        self.stats['total_batches'] = total_batches
        
        self.logger.info(f"Loaded {total_samples} samples, {total_batches} batches of size {self.batch_size}")
        self.logger.info(f"Starting from batch {self.current_batch + 1}/{total_batches}")
        
        # Load existing results if resuming
        processed_samples = []
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            processed_samples.append(json.loads(line))
                self.logger.info(f"Loaded {len(processed_samples)} existing results")
            except Exception as e:
                self.logger.warning(f"Failed to load existing results: {e}")
                processed_samples = []
        
        # Process samples in batches
        async with aiohttp.ClientSession() as session:
            for batch_idx in range(self.current_batch, total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch = samples[start_idx:end_idx]
                
                self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                           f"(samples {start_idx + 1}-{end_idx})")
                
                # Process batch concurrently  
                tasks = [self.process_sample(session, sample) for sample in batch]
                batch_results = await asyncio.gather(*tasks)
                
                processed_samples.extend(batch_results)
                self.current_batch = batch_idx + 1
                self.stats['current_batch'] = self.current_batch
                
                # Save progress and results after each batch
                self.save_results(processed_samples, output_file)
                self.save_progress()
                self.print_statistics()
                
                # Add delay between batches to be respectful
                if batch_idx < total_batches - 1:
                    await asyncio.sleep(2)
        
        self.logger.info("Dataset processing completed")
        self.print_statistics()
        
        # Clean up progress file on completion
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            self.logger.info("Progress file cleaned up - processing complete")
        
        return processed_samples
    
    def save_results(self, processed_samples: List[Dict], output_file: str):
        """Save processed samples to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                f.write(json.dumps(sample) + '\n')
        
        self.logger.info(f"Saved {len(processed_samples)} processed samples to {output_file}")
    
    def print_statistics(self):
        """Print current processing statistics."""
        total = self.stats['total_processed']
        if total > 0:
            accuracy = (self.stats['correct_answers'] / total) * 100
            self.logger.info(f"Statistics: {total} processed, {accuracy:.1f}% accuracy, "
                       f"{self.stats['api_errors']} API errors")
            self.logger.info(f"Tokens: {self.stats['total_input_tokens']} input, "
                       f"{self.stats['total_output_tokens']} output")

async def main():
    """Main function to run the reasoning generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate reasoning data using DeepSeek R1')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size (default: 10)')
    parser.add_argument('--start_batch', type=int, default=0, help='Batch to start from (default: 0)')
    parser.add_argument('--rate_limit', type=int, default=50, help='Rate limit RPM (default: 50)')
    parser.add_argument('--clear_progress', action='store_true', help='Clear progress and start fresh')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from previous progress')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume, start fresh')
    
    args = parser.parse_args()
    
    # Handle resume logic
    resume = args.resume and not args.no_resume and not args.clear_progress
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("Please check your .env file in the root directory")
        return
    
    print(f"Configuration: batch_size={args.batch_size}, start_batch={args.start_batch}, "
          f"rate_limit={args.rate_limit}, resume={resume}")
    
    # File paths
    script_dir = Path(__file__).parent
    # Input file is in data/medmcqa/ directory relative to scripts
    input_file = script_dir.parent.parent / "data" / "medmcqa" / "train_reasoning_sample.json"
    
    # Initialize generator first to get output directory
    generator = DeepSeekReasoningGenerator(
        api_key=api_key,
        rate_limit_rpm=args.rate_limit,
        batch_size=args.batch_size,
        start_batch=args.start_batch
    )
    
    output_file = generator.output_dir / f"train_reasoning_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Use existing output file if resuming
    if resume:
        # Look for existing output files
        existing_files = list(generator.output_dir.glob("train_reasoning_generated_*.json"))
        if existing_files:
            # Use the most recent file
            output_file = max(existing_files, key=os.path.getctime)
            print(f"Resuming with existing output file: {output_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    # Clear progress if requested
    if args.clear_progress:
        generator.delete_progress()
        # Create new output file
        output_file = generator.output_dir / f"train_reasoning_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # Process dataset
        await generator.process_dataset(
            input_file=str(input_file),
            output_file=str(output_file),
            resume=resume
        )
        
        print(f"Final results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("Processing interrupted by user")
        print("Progress has been saved. Use --resume to continue from where you left off")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())