#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick Medical AI Improvement Script
Start with the most promising single technique to get immediate results
"""

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Deep Learning
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from torch.utils.data import Dataset
    transformers_available = True
except ImportError:
    print("Transformers not available. Please install: pip install transformers torch")
    transformers_available = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if transformers_available:
    torch.manual_seed(RANDOM_SEED)

class OptimizedMedicalDataset(Dataset):
    """Optimized dataset for medical text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('//'):
                continue
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data

def enhanced_preprocess_text(text):
    """Enhanced text preprocessing for medical domain"""
    if not text or not isinstance(text, str):
        return ""
    
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Medical abbreviation expansion (key ones)
    medical_abbrevs = {
        'ecg': 'electrocardiogram', 'ekg': 'electrocardiogram',
        'mri': 'magnetic resonance imaging', 'ct': 'computed tomography',
        'bp': 'blood pressure', 'hr': 'heart rate',
        'htn': 'hypertension', 'dm': 'diabetes mellitus',
        'mi': 'myocardial infarction', 'copd': 'chronic obstructive pulmonary disease'
    }
    
    for abbr, full in medical_abbrevs.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text)
    
    return text

def compute_enhanced_metrics(eval_pred):
    """Enhanced metrics computation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def train_improved_medical_classifier(train_texts, train_labels, val_texts, val_labels, output_dir):
    """Train improved medical classifier with optimized settings"""
    
    print("Setting up improved medical classifier...")
    
    # Use the best medical model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    num_labels = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_labels}")
    print(f"Using model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = OptimizedMedicalDataset(train_texts, train_labels_encoded, tokenizer)
    val_dataset = OptimizedMedicalDataset(val_texts, val_labels_encoded, tokenizer)
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'improved_model'),
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=16,  # Optimal batch size
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,  # Warmup ratio
        weight_decay=0.01,
        learning_rate=2e-5,  # Optimal learning rate for medical domain
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=150,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
        # Advanced optimization
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # Early stopping patience
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_enhanced_metrics,
    )
    
    print("Training improved model...")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Train
    trainer.train()
    
    # Final evaluation
    eval_results = trainer.evaluate()
    
    # Get detailed predictions
    predictions = trainer.predict(val_dataset)
    y_pred_encoded = np.argmax(predictions.predictions, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, y_pred)
    report = classification_report(val_labels, y_pred, output_dict=True, zero_division=0)
    
    print(f"\n=== IMPROVED MODEL RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-weighted: {report['weighted avg']['f1-score']:.4f}")
    print(f"F1-macro: {report['macro avg']['f1-score']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'improved_model_results.txt'), 'w') as f:
        f.write(f"Model: {model_name} (Improved Settings)\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")
        f.write(f"Improvement over baseline (55.06%): {(accuracy - 0.5506) * 100:.2f} percentage points\n")
        f.write(f"Training details:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value}\n")
    
    # Save model
    trainer.save_model(os.path.join(output_dir, 'improved_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'improved_model'))
    
    # Sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    sample_indices = np.random.choice(len(val_texts), 5, replace=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    model.to(device)
    model.eval()
    
    for idx in sample_indices:
        question = val_texts[idx]
        true_subject = val_labels[idx]
        
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = prediction.argmax().item()
            predicted_subject = label_encoder.inverse_transform([predicted_class_id])[0]
            confidence = prediction.max().item()
        
        print(f"\nQuestion: {question[:100]}...")
        print(f"True: {true_subject}")
        print(f"Predicted: {predicted_subject}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Correct: {'✓' if predicted_subject == true_subject else '✗'}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'eval_results': eval_results
    }

def main():
    """Main function for quick improvement"""
    print("=== QUICK MEDICAL AI IMPROVEMENT ===")
    print("Starting with most promising single technique...")
    
    # Setup paths
    base_dir = Path(r"c:\Users\User\Downloads\nusrat\medrag")
    data_dir = base_dir / "data" / "medmcqa"
    output_dir = base_dir / "evaluation_results" / "quick_improvement"
    
    train_path = data_dir / "train.json"
    dev_path = data_dir / "dev.json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data = load_jsonl_data(train_path)
    dev_data = load_jsonl_data(dev_path)
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(dev_data)} development examples")
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)[['question', 'subject_name']].dropna()
    dev_df = pd.DataFrame(dev_data)[['question', 'subject_name']].dropna()
    
    # Enhanced preprocessing
    print("Preprocessing text...")
    train_df['processed_question'] = train_df['question'].apply(enhanced_preprocess_text)
    dev_df['processed_question'] = dev_df['question'].apply(enhanced_preprocess_text)
    
    # Filter subjects with sufficient examples
    min_examples = 20
    subject_counts = Counter(train_df['subject_name'])
    valid_subjects = [subject for subject, count in subject_counts.items() if count >= min_examples]
    
    train_df = train_df[train_df['subject_name'].isin(valid_subjects)]
    dev_df = dev_df[dev_df['subject_name'].isin(valid_subjects)]
    
    print(f"Using {len(valid_subjects)} subjects with >= {min_examples} examples")
    print(f"Final training set: {len(train_df)} examples")
    print(f"Final dev set: {len(dev_df)} examples")
    
    if not transformers_available:
        print("Transformers not available. Please install: pip install transformers torch")
        return
    
    # Use manageable sample size (adjust based on your GPU memory)
    sample_size = min(6000, len(train_df))
    train_sample = train_df.sample(sample_size, random_state=RANDOM_SEED)
    
    print(f"Using {len(train_sample)} training examples")
    
    # Train improved model
    results = train_improved_medical_classifier(
        train_sample['processed_question'].tolist(),
        train_sample['subject_name'].tolist(),
        dev_df['processed_question'].tolist(),
        dev_df['subject_name'].tolist(),
        str(output_dir)
    )
    
    # Final summary
    print(f"\n{'='*50}")
    print("QUICK IMPROVEMENT RESULTS")
    print(f"{'='*50}")
    print(f"✨ New Accuracy: {results['accuracy']:.4f}")
    print(f"📊 Previous Best: 0.5506 (55.06%)")
    print(f"📈 Improvement: +{(results['accuracy'] - 0.5506) * 100:.2f} percentage points")
    
    if results['accuracy'] > 0.60:
        print("🎉 EXCELLENT: Broke 60% accuracy barrier!")
    elif results['accuracy'] > 0.57:
        print("✅ GOOD: Solid improvement achieved!")
    else:
        print("⚠️ MODERATE: Some improvement, more techniques needed")
    
    print(f"\nNext steps:")
    print(f"1. Run 'advanced_medical_classifier.py' for multiple models")
    print(f"2. Run 'specialized_medical_ai.py' for advanced techniques")
    print(f"3. Run 'ultimate_ensemble.py' for best possible results")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n⏱️ Execution time: {time.time() - start_time:.2f} seconds")
