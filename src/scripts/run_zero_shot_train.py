from batch_evaluation import BatchEvaluator
import json

# Load training data
file_path = 'C:/Users/User/Downloads/nusrat/medrag/data/medmcqa/train_stratified_sample.json'
with open(file_path, 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# Initialize evaluator with specific model and mode
evaluator = BatchEvaluator(
    model_name="meditron:latest",
    use_thinking_model=False  # Set to False for standard evaluation
)

evaluator.reset_evaluation()

# URL for the model API
url = "http://localhost:11434/api/generate"

# Run today's batch
evaluator.evaluate_batch(train_data, url, batch_size=1000)