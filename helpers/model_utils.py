# model_utils.py - Module for model loading and management

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfApi, create_repo

def authenticate_huggingface(token):
    """Authenticate with Hugging Face using provided token"""
    try:
        login(token)
        return True, "Authentication successful!"
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

def get_model_map():
    """Return mapping of model names to Hugging Face model IDs"""
    return {
        "Llama 3": "meta-llama/Llama-3.1-8B",
        "Mistral": "mistralai/Mistral-7B-v0.1",
        "SmolLM": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }

def load_model(model_name, token=None):
    """Load model and tokenizer from Hugging Face"""
    model_map = get_model_map()
    
    if model_name in model_map:
        model_id = model_map[model_name]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token is not None)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                use_auth_token=token is not None
            )
            return True, (model, tokenizer), f"{model_name} loaded successfully!"
        except Exception as e:
            return False, None, f"Error loading model: {str(e)}"
    else:
        return False, None, "Invalid model selection."

def create_huggingface_repo(token, repo_name):
    """Create a repository on Hugging Face Hub"""
    try:
        create_repo(repo_name, token=token, private=False, exist_ok=True, repo_type="model")
        return True, f"Repository '{repo_name}' created successfully!"
    except Exception as e:
        return False, f"Error creating repository: {str(e)}"

def push_to_hub(model, tokenizer, repo_name, token):
    """Push model and tokenizer to Hugging Face Hub"""
    try:
        model.push_to_hub(repo_name, token=token)
        tokenizer.push_to_hub(repo_name, token=token)
        return True, f"Model and tokenizer pushed to {repo_name} successfully!"
    except Exception as e:
        return False, f"Error pushing to hub: {str(e)}"

def generate_sample_text(model, tokenizer, input_text, max_new_tokens=50):
    """Generate text using the loaded model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        model.to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating text: {str(e)}"
