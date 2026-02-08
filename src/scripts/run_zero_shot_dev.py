from batch_evaluation import BatchEvaluator
import json

# Load dev data
file_path = 'C:/Users/User/Downloads/nusrat/medrag/data/medmcqa/dev_stratified_sample.json'
with open(file_path, 'r', encoding='utf-8') as f:
    dev_data = [json.loads(line) for line in f]

# Initialize evaluator with dev split
evaluator = BatchEvaluator(
    model_name="OussamaELALLAM/MedExpert:latest",
    use_thinking_model=False,  # Set to False for standard evaluation
    data_split="dev"  # Specify dev split
)

evaluator.reset_evaluation()

# URL for the model API
url = "http://localhost:11434/api/generate"

# Run evaluation on dev set
evaluator.evaluate_batch(dev_data, url, batch_size=1000)
