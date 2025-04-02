# data_processing.py - Module for dataset processing

import pandas as pd
import json
import os
import torch
from datasets import load_dataset

def process_dataset(file_path, output_dir="."):
    """Process a dataset file and split into train/validation sets"""
    try:
        # Determine file type and load
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            return False, None, "Unsupported file format. Please use CSV, JSON, or JSONL."
        
        # Check for required columns
        required_columns = ['description', 'essay']
        if not all(col in df.columns for col in required_columns):
            return False, None, f"Dataset must contain columns: {', '.join(required_columns)}"
            
        # Keep only relevant columns
        df = df[required_columns].dropna()
        
        # Train-Validation Split (80% Train, 20% Validation)
        df = df.sample(frac=1, random_state=42)  # Shuffle dataset
        train_size = int(0.8 * len(df))
        train_df, val_df = df[:train_size].copy(), df[train_size:].copy()
        
        # Preprocess dataset
        train_df = preprocess_for_llm(train_df)
        val_df = preprocess_for_llm(val_df)
        
        # Save processed datasets
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_dataset.json")
        val_path = os.path.join(output_dir, "val_dataset.json")
        
        train_df.to_json(train_path, orient="records", lines=True)
        val_df.to_json(val_path, orient="records", lines=True)
        
        result = {
            "train_path": train_path,
            "val_path": val_path,
            "train_size": len(train_df),
            "val_size": len(val_df)
        }
        
        return True, result, "Dataset processed successfully"
    except Exception as e:
        return False, None, f"Error processing dataset: {str(e)}"

def preprocess_for_llm(df):
    """Apply preprocessing to dataset for LLM training"""
    # Convert to lowercase
    df['description'] = df['description'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['essay'] = df['essay'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Truncate long texts
    max_length = 512
    df['description'] = df['description'].apply(lambda x: x[:max_length] if isinstance(x, str) else x)
    df['essay'] = df['essay'].apply(lambda x: x[:max_length] if isinstance(x, str) else x)
    
    return df

def load_processed_datasets(train_path, val_path):
    """Load processed datasets from disk using the datasets library"""
    try:
        dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
        return True, dataset, "Datasets loaded successfully"
    except Exception as e:
        return False, None, f"Error loading datasets: {str(e)}"

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize dataset for model training"""
    try:
        def tokenize_function(examples):
            return tokenizer(examples["essay"], truncation=True, padding="max_length", max_length=max_length)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["essay", "description"])
        return True, tokenized_dataset, "Dataset tokenized successfully"
    except Exception as e:
        return False, None, f"Error tokenizing dataset: {str(e)}"
