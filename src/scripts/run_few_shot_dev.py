from batch_evaluation import BatchEvaluator
import json

# Load development data
dev_file_path = 'C:/Users/User/Downloads/nusrat/medrag/data/medmcqa/dev_stratified_sample.json'
with open(dev_file_path, 'r', encoding='utf-8') as f:
    dev_data = [json.loads(line) for line in f]

# Initialize evaluator for dev set evaluation
evaluator = BatchEvaluator(
    model_name="meditron:latest",
    use_thinking_model=False,
    few_shot_mode=True,  # Set to True if you want to use few-shot learning
    num_examples=3,
    data_split="dev"  # Specify this is dev set evaluation
)

evaluator.reset_evaluation()

# URL for the model API
url = "http://localhost:11434/api/generate"

# Run evaluation on dev set
evaluator.evaluate_batch(dev_data, url, batch_size=1000)
