# Py-TunAIr: Fine-tuning as a Service

Py-TunAIr is a comprehensive platform for fine-tuning foundation models on decentralized computing infrastructure, providing a seamless experience from model selection to deployment.

## 🌟 Features

- **Intuitive UI**: Easy-to-use interface for configuring and monitoring fine-tuning jobs
- **Multiple Model Support**: Fine-tune Llama 3, Mistral, SmolLM, and other foundation models
- **Hyperparameter Optimization**: Automated or manual configuration of training parameters
- **Real-time Progress Tracking**: Monitor training metrics and progress in real-time
- **Comprehensive Evaluation**: Evaluate models with perplexity, BLEU, ROUGE, and other metrics
- **Flexible Deployment Options**: Deploy to Hugging Face Hub or as API endpoints

## 📋 Requirements

```
# Core dependencies
streamlit>=1.30.0
huggingface_hub>=0.22.0
transformers>=4.40.0
torch>=2.2.0
accelerate>=0.27.0
sentencepiece>=0.2.0
nltk>=3.8.1
absl-py>=2.0.0
rouge_score>=0.1.2
evaluate>=0.4.1
onnx>=1.15.0

# Optimization tools
optuna>=3.5.0
peft>=0.8.0
bitsandbytes>=0.42.0

# Data processing
datasets>=2.18.0
pandas>=2.1.0
tokenizers>=0.15.0
tqdm>=4.66.0

# Visualization & notebooks
matplotlib>=3.8.0
ipywidgets>=8.1.0
```

## 🚀 Getting Started

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the frontend application:
   ```bash
   streamlit run app.py
   ```

3. Start the backend notebook (for advanced options):
   ```bash
   jupyter notebook fine-tune-backend.ipynb
   ```

## 🔧 Using the Platform

### Step 1: Model Setup
- Select a foundation model (Llama 3, Mistral, etc.)
- Upload your training dataset
- Configure validation strategy

### Step 2: Training
- Set hyperparameters or use recommended defaults
- Monitor training progress and logs in real-time
- View loss curves and intermediate metrics

### Step 3: Evaluation
- Comprehensive evaluation with industry-standard metrics
- Compare performance with original base model
- Test inference with custom inputs

### Step 4: Deployment
- Push to Hugging Face Hub with customized model card
- Deploy as scalable API endpoint
- Export in various formats (PyTorch, ONNX, etc.)

## 🧠 Advanced Features

- **Hyperparameter Optimization**: Leverages Optuna for efficient hyperparameter search
- **Quantization Options**: Deploy with FP16, INT8, or INT4 quantization for efficiency
- **Parameter-Efficient Fine-Tuning**: Support for techniques like LoRA and QLoRA
- **Distributed Training**: Scale to multi-GPU setups for larger models

## 🔗 Integration

Py-TunAIr integrates seamlessly with:
- Hugging Face ecosystem for model sharing and discovery
- Decentralized GPU networks for cost-effective compute
- Various ML frameworks through standardized export formats

## 📊 Architecture

The platform consists of two main components:
1. **Streamlit Frontend**: Intuitive UI for configuring and monitoring fine-tuning jobs
2. **Fine-Tuning Backend**: Powerful notebook-based backend for advanced customization and monitoring

## 📝 License

MIT License

## 🙏 Acknowledgments

- Hugging Face for their transformative work on democratizing NLP
- The open-source ML community for developing the foundation models and tools
- Streamlit for enabling rapid development of ML applications

## 📧 Contact

For questions or support, please open an issue on our GitHub repository.