"""
Quick test of the multi-agent medical QA system with just 1 question
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent_medical_qa import MultiAgentMedicalQA, load_medmcqa_data

def test_single_question():
    """Test with just one question"""
    
    # Load one question
    dev_data_path = r'C:\Users\User\Downloads\nusrat\medrag\data\medmcqa\dev_stratified_sample.json'
    data = load_medmcqa_data(dev_data_path)
    
    if not data:
        print("No data loaded!")
        return
    
    sample_question = data[0]  # Just the first question
    print(f"Testing with question: {sample_question['question'][:100]}...")
    
    # Initialize system
    multiagent_qa = MultiAgentMedicalQA(
        resident_model="meditron:latest",
        attending_model="deepseek-r1:8b",
        base_output_dir="evaluation_results"
    )
    
    # Process one question
    result = multiagent_qa.process_single_question(sample_question, max_iterations=1)
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Question: {result['question']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print(f"Predicted Answer: {result['predicted_answer']}")
        print(f"Is Correct: {result['is_correct']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Iterations: {len(result['iterations'])}")
    
    return result

if __name__ == "__main__":
    test_single_question()
