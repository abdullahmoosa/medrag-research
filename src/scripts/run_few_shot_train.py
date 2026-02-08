from batch_evaluation import BatchEvaluator
import json

# Load training data
file_path = 'C:/Users/User/Downloads/nusrat/medrag/data/medmcqa/train_stratified_sample.json'
with open(file_path, 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# Initialize evaluator with few-shot mode
evaluator = BatchEvaluator(
    model_name="deepseek-r1:8b",
    use_thinking_model=True,
    few_shot_mode=True,
    num_examples=3  # Use 3 examples per question
)

evaluator.reset_evaluation()

# URL for the model API
url = "http://localhost:11434/api/generate"

# Run today's batch
evaluator.evaluate_batch(train_data, url, batch_size=1000)
