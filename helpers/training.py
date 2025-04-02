# training.py - Module for model training

import torch
import optuna
import math
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from evaluate import load

def get_training_recommendations(model_name):
    """Get recommended hyperparameters for a model"""
    recommendations = {
        "Llama 3": {"learning_rate": 2e-5, "batch_size": 8, "num_train_epochs": 3, "weight_decay": 0.01},
        "Mistral": {"learning_rate": 3e-5, "batch_size": 16, "num_train_epochs": 4, "weight_decay": 0.02},
        "SmolLM": {"learning_rate": 5e-5, "batch_size": 8, "num_train_epochs": 3, "weight_decay": 0.01},
    }
    return recommendations.get(model_name, recommendations["Llama 3"])

def setup_training(model, tokenizer, tokenized_dataset, params, output_dir="./fine_tuned_model"):
    """Set up training arguments and trainer"""
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            num_train_epochs=params["num_train_epochs"],
            weight_decay=params["weight_decay"],
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=100,
            fp16=True if torch.cuda.is_available() else False,
            report_to="none",
            gradient_checkpointing=True,
            gradient_accumulation_steps=4
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation", None),
            data_collator=data_collator,
        )
        
        return True, trainer, "Training setup complete"
    except Exception as e:
        return False, None, f"Error setting up training: {str(e)}"

def train_model(trainer, callback=None):
    """Train the model with optional progress callback"""
    try:
        # Start training
        training_results = trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        # Calculate perplexity
        eval_loss = eval_results["eval_loss"]
        perplexity = math.exp(eval_loss)
        
        results = {
            "training_time": training_results.metrics.get("train_runtime", 0),
            "eval_loss": eval_loss,
            "perplexity": perplexity
        }
        
        return True, results, "Model training complete"
    except Exception as e:
        return False, None, f"Error during training: {str(e)}"

def evaluate_model(model, tokenizer, val_data_path):
    """Evaluate model performance with BLEU, ROUGE, and token accuracy"""
    try:
        # Load metrics
        bleu_metric = load("bleu")
        rouge_metric = load("rouge")
        
        # Load validation data
        val_data = []
        with open(val_data_path, "r", encoding="utf-8") as f:
            for line in f:
                val_data.append(json.loads(line.strip()))
        
        # Use a sample for faster evaluation
        sample_size = min(100, len(val_data))
        sample_data = val_data[:sample_size]
        
        # Extract inputs and references
        sample_inputs = [item["essay"] for item in sample_data]
        references = [[item["description"]] for item in sample_data]
        
        # Generate predictions (this might take time)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        predictions = []
        
        for text in sample_inputs:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
        
        # Compute BLEU score
        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        
        # Compute ROUGE score
        rouge_score = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
        
        # Calculate token accuracy
        token_accuracy = calculate_token_accuracy(predictions, references, tokenizer)
        
        results = {
            "bleu": bleu_score["bleu"],
            "rouge": rouge_score,
            "token_accuracy": token_accuracy
        }
        
        return True, results, "Evaluation complete"
    except Exception as e:
        return False, None, f"Error during evaluation: {str(e)}"

def calculate_token_accuracy(predictions, references, tokenizer):
    """Calculate token-level accuracy between predictions and references"""
    total_tokens = 0
    correct_tokens = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenizer.tokenize(pred)
        ref_tokens = tokenizer.tokenize(ref[0])

        # Compute number of matching tokens
        matches = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)

        correct_tokens += matches
        total_tokens += len(ref_tokens)

    # Avoid division by zero
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0
    return accuracy
