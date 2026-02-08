# Multi-Agent Medical Question Answering System

## Overview
This project implements a multi-agent system for medical question answering that mimics real-world medical education scenarios. The system uses two AI agents representing:

1. **Senior Medical Resident** - Provides initial medical analysis
2. **Senior Attending Physician** - Reviews and validates the analysis

## System Architecture

### Real-World Parallel
The system mirrors the clinical case presentation workflow in teaching hospitals:
- Residents present patient cases and proposed diagnoses
- Attending physicians provide feedback and corrections  
- Residents refine their approach based on expert guidance

### Two Implementation Approaches

#### 1. Simple Multi-Agent System (`simple_multiagent_qa.py`)
- **Direct LLM calls** via Ollama API
- **Simplified workflow** without framework overhead
- **Fast execution** and easy debugging
- **Recommended for initial testing**

#### 2. CrewAI-Based System (`multiagent_medical_qa.py`)
- **Full CrewAI framework** implementation
- **Structured agent definitions** with roles and backstories
- **Task-based workflow** management
- **Enhanced collaboration features**

## Installation & Dependencies

### Required Software
- Python 3.9+ (3.10+ recommended for latest CrewAI)
- Ollama running locally on port 11434
- Required medical models: `meditron:latest`, `deepseek-r1:8b`

### Python Dependencies
```bash
pip install crewai==0.1.15  # For Python 3.9 compatibility
pip install requests
pip install pandas
```

## Model Configuration

### Available Models
- **Resident Agent**: `meditron:latest` (medical specialist model)
- **Attending Agent**: `deepseek-r1:8b` (reasoning-capable model)

### Alternative Models
You can configure different models by modifying the initialization:
```python
multiagent_qa = SimpleMultiAgentMedicalQA(
    resident_model="OussamaELALLAM/MedExpert:latest",
    attending_model="gemma3:12b-it-qat"
)
```

## Usage Examples

### Quick Test (Single Question)
```python
from simple_multiagent_qa import SimpleMultiAgentMedicalQA, load_medmcqa_data

# Load data and initialize system
data = load_medmcqa_data("path/to/dev_data.json")
qa_system = SimpleMultiAgentMedicalQA()

# Process one question
result = qa_system.process_single_question(data[0])
print(f"Accuracy: {result['is_correct']}")
```

### Full Evaluation
```python
# Run evaluation on dataset subset
results = qa_system.evaluate_dataset(data, max_questions=50)
print(f"Final Accuracy: {results['metadata']['final_accuracy']:.2f}%")
```

## Workflow Process

### 1. Initial Analysis
- Resident agent receives medical question with 4 options (A, B, C, D)
- Provides step-by-step clinical reasoning
- Selects initial answer

### 2. Attending Review
- Attending agent evaluates resident's analysis
- Identifies potential errors or improvements
- Either **APPROVES** or provides **constructive feedback**

### 3. Iterative Improvement (Optional)
- If not approved, resident revises analysis based on feedback
- Process can repeat up to `max_iterations` (default: 2)
- Tracks all iterations for analysis

## Output & Results

### Result Structure
```json
{
  "question_id": "unique_id",
  "question": "Medical question text",
  "correct_answer": "C",
  "predicted_answer": "C", 
  "is_correct": true,
  "initial_answer": "B",
  "final_approved": true,
  "iterations": [...],
  "processing_time": 45.2,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Evaluation Metrics
- **Accuracy**: Percentage of correct answers
- **Approval Rate**: Percentage of answers approved by attending
- **Processing Time**: Average time per question
- **Iteration Analysis**: Improvement patterns

### Saved Results
Results are automatically saved to `evaluation_results/` with:
- Detailed per-question results (JSON)
- Progress tracking during evaluation
- Summary statistics
- Timestamp-based organization

## Integration with Existing Evaluation Framework

### Directory Structure
```
evaluation_results/
├── simple_multiagent_evaluation/
│   └── 20250629_143052/
│       ├── simple_multiagent_evaluation_results.json
│       ├── evaluation_summary.json
│       └── progress.json
└── crewai_multiagent_evaluation/
    └── 20250629_143052/
        └── crewai_multiagent_evaluation_results.json
```

### Comparison with Single-Agent Results
The multi-agent results can be compared with your existing single-agent evaluations:
- `deepseek-r1-8b_zero_shot/`
- `meditron-latest_zero_shot/`
- `gemma3-12b-it-qat_zero_shot/`

## Benefits of Multi-Agent Approach

### 1. **Improved Accuracy**
- Second opinion mechanism catches errors
- Iterative refinement based on expert feedback
- Combines strengths of different models

### 2. **Educational Value**
- Transparent reasoning process
- Identifies common error patterns
- Provides learning insights

### 3. **Quality Assurance**
- Built-in validation mechanism
- Approval tracking
- Error categorization

### 4. **Realistic Workflow**
- Mirrors actual medical practice
- Structured peer review process
- Professional development simulation

## Current Status

✅ **Implemented**: Simple multi-agent system  
✅ **Implemented**: CrewAI-based system  
✅ **Tested**: Basic functionality with Ollama models  
✅ **Integrated**: Evaluation results saving  
🔄 **Running**: Initial 5-question evaluation test  

## Next Steps

1. **Complete initial evaluation** and analyze results
2. **Compare performance** with single-agent baselines
3. **Optimize prompts** for better agent collaboration
4. **Scale evaluation** to full dataset (500+ questions)
5. **Analyze iteration patterns** and approval rates
6. **Fine-tune model selection** based on performance
