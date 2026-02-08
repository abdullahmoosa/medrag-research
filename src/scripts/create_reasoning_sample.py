#!/usr/bin/env python3
"""
Script to create a balanced sample dataset from MedMCQA train.json
for synthetic reasoning dataset generation using DeepSeek API.

This script creates train_reasoning_sample.json with 10k samples,
ensuring equal representation from each subject_name.
"""

import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List
import argparse

def load_dataset(file_path: str) -> List[Dict]:
    """Load the MedMCQA dataset from JSON file."""
    print(f"Loading dataset from: {file_path}")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)
    
    print(f"Total samples loaded: {len(samples)}")
    return samples

def analyze_subjects(samples: List[Dict]) -> Dict[str, int]:
    """Analyze the distribution of subjects in the dataset."""
    subject_counts = Counter()
    
    for sample in samples:
        subject = sample.get('subject_name', 'Unknown')
        subject_counts[subject] += 1
    
    print(f"\nSubject distribution:")
    for subject, count in subject_counts.most_common():
        print(f"  {subject}: {count:,}")
    
    return dict(subject_counts)

def create_balanced_sample(samples: List[Dict], target_size: int = 10000) -> List[Dict]:
    """
    Create a balanced sample ensuring equal representation from each subject.
    
    Args:
        samples: List of all samples
        target_size: Target number of samples (default: 10000)
    
    Returns:
        List of balanced samples
    """
    # Group samples by subject
    subject_samples = defaultdict(list)
    for sample in samples:
        subject = sample.get('subject_name', 'Unknown')
        subject_samples[subject].append(sample)
    
    # Calculate samples per subject
    num_subjects = len(subject_samples)
    samples_per_subject = target_size // num_subjects
    remaining_samples = target_size % num_subjects
    
    print(f"\nCreating balanced sample:")
    print(f"Target size: {target_size:,}")
    print(f"Number of subjects: {num_subjects}")
    print(f"Base samples per subject: {samples_per_subject}")
    print(f"Remaining samples to distribute: {remaining_samples}")
    
    balanced_samples = []
    subject_allocation = {}
    
    # First, allocate base samples per subject
    for subject, subject_data in subject_samples.items():
        available_samples = len(subject_data)
        allocated_samples = min(samples_per_subject, available_samples)
        
        # Randomly sample from this subject
        if allocated_samples > 0:
            selected = random.sample(subject_data, allocated_samples)
            balanced_samples.extend(selected)
            subject_allocation[subject] = allocated_samples
        else:
            subject_allocation[subject] = 0
    
    # Distribute remaining samples to subjects with available data
    subjects_with_remaining = [(subject, len(data)) for subject, data in subject_samples.items() 
                              if len(data) > samples_per_subject]
    
    # Sort by available samples (descending) to prioritize subjects with more data
    subjects_with_remaining.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(remaining_samples):
        if i < len(subjects_with_remaining):
            subject = subjects_with_remaining[i][0]
            subject_data = subject_samples[subject]
            
            # Get one more sample from this subject (not already selected)
            already_selected = set(sample['id'] for sample in balanced_samples 
                                 if sample.get('subject_name') == subject)
            
            available = [s for s in subject_data if s['id'] not in already_selected]
            if available:
                selected = random.choice(available)
                balanced_samples.append(selected)
                subject_allocation[subject] += 1
    
    # Print final allocation
    print(f"\nFinal sample allocation:")
    total_allocated = 0
    for subject, count in sorted(subject_allocation.items()):
        print(f"  {subject}: {count:,}")
        total_allocated += count
    
    print(f"\nTotal samples in balanced dataset: {total_allocated:,}")
    
    return balanced_samples

def save_dataset(samples: List[Dict], output_path: str):
    """Save the balanced dataset to JSON file."""
    print(f"\nSaving balanced dataset to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Dataset saved successfully!")

def validate_sample(samples: List[Dict]):
    """Validate the created sample dataset."""
    print(f"\n--- Validation ---")
    
    # Check subject distribution
    subject_counts = Counter()
    for sample in samples:
        subject = sample.get('subject_name', 'Unknown')
        subject_counts[subject] += 1
    
    print(f"Subject distribution in sample:")
    for subject, count in subject_counts.most_common():
        print(f"  {subject}: {count:,}")
    
    # Check for duplicates
    ids = [sample['id'] for sample in samples]
    if len(ids) != len(set(ids)):
        print(f"WARNING: Found duplicate IDs in sample!")
    else:
        print(f"✓ No duplicate IDs found")
    
    # Check for required fields
    required_fields = ['id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'subject_name']
    missing_fields = set()
    
    for sample in samples:
        for field in required_fields:
            if field not in sample:
                missing_fields.add(field)
    
    if missing_fields:
        print(f"WARNING: Missing fields: {missing_fields}")
    else:
        print(f"✓ All required fields present")

def main():
    """Main function to create balanced sample dataset."""
    parser = argparse.ArgumentParser(description='Create balanced sample from MedMCQA train.json')
    parser.add_argument('--sample_size', type=int, default=10000, 
                       help='Target sample size (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # File paths
    train_file = "../../data/medmcqa/train.json"
    output_file = "train_reasoning_sample.json"
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, train_file)
    output_path = os.path.join(script_dir, output_file)
    
    print(f"Script directory: {script_dir}")
    print(f"Train file path: {train_path}")
    print(f"Output file path: {output_path}")
    
    # Check if train file exists
    if not os.path.exists(train_path):
        print(f"Error: Train file not found at {train_path}")
        return
    
    try:
        # Load dataset
        samples = load_dataset(train_path)
        
        # Analyze subjects
        subject_counts = analyze_subjects(samples)
        
        # Create balanced sample
        balanced_samples = create_balanced_sample(samples, args.sample_size)
        
        # Validate the sample
        validate_sample(balanced_samples)
        
        # Save the balanced dataset
        save_dataset(balanced_samples, output_path)
        
        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Original dataset size: {len(samples):,}")
        print(f"Sample dataset size: {len(balanced_samples):,}")
        print(f"Sample saved to: {output_file}")
        print(f"Random seed used: {args.seed}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
