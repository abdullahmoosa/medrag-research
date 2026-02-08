import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_train_data(file_path):
    """Load and parse the training data JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            train_data = []
            for line in f:
                try:
                    train_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing line: {e}")
                    continue
        return train_data
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        raise

def create_stratified_sample(train_data, sample_percentage=0.12):
    """Create a stratified sample of the data"""
    # Get subject names for stratification
    subjects = [q['subject_name'] for q in train_data]
    total_samples = len(train_data)
    sample_size = int(sample_percentage * total_samples)
    
    # Convert data to numpy array for sklearn
    indices = np.arange(total_samples)
    
    # Initialize and perform stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1, 
        train_size=sample_size, 
        random_state=42
    )
    
    # Get sample indices and data
    for sample_idx, _ in splitter.split(indices, subjects):
        stratified_sample = [train_data[i] for i in sample_idx]
    
    return stratified_sample

def print_distribution_stats(train_data, sampled_data):
    """Print distribution statistics of original and sampled data"""
    orig_subjects = [q['subject_name'] for q in train_data]
    sampled_subjects = [q['subject_name'] for q in sampled_data]
    
    original_dist = Counter(orig_subjects)
    sample_dist = Counter(sampled_subjects)
    
    logging.info("\nSubject distribution:")
    logging.info(f"{'Subject':<30} {'Original':<10} {'Sampled':<10}")
    logging.info("-" * 50)
    
    for subject in original_dist:
        logging.info(f"{subject:<30} {original_dist[subject]:<10} {sample_dist.get(subject, 0):<10}")

def main():
    # Define paths
    base_path = Path("C:/Users/User/Downloads/nusrat/medrag")
    input_path = base_path / "data/medmcqa/dev.json"
    output_path = base_path / "data/medmcqa/dev_stratified_sample.json"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logging.info("Loading training data...")
        train_data = load_train_data(input_path)
        
        # Create stratified sample
        logging.info("Creating stratified sample...")
        sampled_data = create_stratified_sample(train_data)
        
        # Print statistics
        logging.info(f"Total samples in dataset: {len(train_data)}")
        logging.info(f"Number of samples in 1.1% stratified sample: {len(sampled_data)}")
        print_distribution_stats(train_data, sampled_data)
        
        # Save sampled data in JSONL format (one JSON object per line)
        logging.info(f"Saving sampled data to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                # Write each object on a single line without pretty printing
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info("Sampling completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
