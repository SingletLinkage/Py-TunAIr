import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Model Fine-Tuning Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .stButton>button {
        background-color: #4285F4;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #333;
    }
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning {
        color: #FFC107;
        font-weight: bold;
    }
    .error {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 0  # 0: setup, 1: training, 2: evaluation, 3: deployment
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'deployed' not in st.session_state:
    st.session_state.deployed = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Application title and description
st.title("AI Model Fine-Tuning Pipeline")
st.markdown("Fine-tune foundation models on Spheron's decentralized GPU network")

# Sidebar for navigation and status
with st.sidebar:
    st.image("helpers/spheron_logo.jpeg", width=150)
    st.subheader("Pipeline Status")
    
    # Show progress based on current stage
    stages = ["Setup", "Training", "Evaluation", "Deployment"]
    current_stage = stages[st.session_state.stage]
    
    for i, stage in enumerate(stages):
        if i < st.session_state.stage:
            st.success(f"✅ {stage}")
        elif i == st.session_state.stage:
            st.info(f"🔄 {stage}")
        else:
            st.text(f"⏳ {stage}")
    
    # Reset button (only show when not in middle of a process)
    if st.session_state.progress == 0 or st.session_state.progress == 100:
        if st.button("Reset Pipeline"):
            st.session_state.stage = 0
            st.session_state.training_complete = False
            st.session_state.evaluation_complete = False
            st.session_state.deployed = False
            st.session_state.metrics = None
            st.session_state.model_path = None
            st.session_state.progress = 0
            st.rerun()

# Main content area
if st.session_state.stage == 0:  # Setup stage
    st.header("Model Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        
        # Hugging Face token input
        hf_token = st.text_input("Hugging Face Token", type="password", 
                                help="Your Hugging Face token for authentication")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["Large Language Model (LLM)", "Image Generation", "Audio Processing"]
        )
        
        # Dynamic model options based on type
        model_options = {
            "Large Language Model (LLM)": ["Llama 3", "Mistral 7B", "SmolLM", "Phi-3", "Gemma"],
            "Image Generation": ["Stable Diffusion XL", "Kandinsky", "PixArt-α"],
            "Audio Processing": ["Whisper", "MusicGen", "AudioLDM"]
        }
        
        selected_model = st.selectbox("Select Model", model_options[model_type])
        
        # Model configuration
        model_size = st.select_slider(
            "Model Size",
            options=["Small", "Medium", "Large"],
            value="Medium",
            help="Larger models may have better performance but require more resources"
        )
        
    with col2:
        st.subheader("Dataset Configuration")
        
        # Dataset upload
        dataset_upload = st.file_uploader(
            "Upload Training Dataset",
            type=["csv", "json", "jsonl", "txt"],
            help="Upload your dataset in CSV, JSON, JSONL, or TXT format"
        )
        
        # Validation set
        validation_method = st.radio(
            "Validation Method",
            ["Split from training data", "Upload separate validation set"]
        )
        
        if validation_method == "Upload separate validation set":
            validation_upload = st.file_uploader(
                "Upload Validation Dataset",
                type=["csv", "json", "jsonl", "txt"]
            )
        else:
            validation_split = st.slider(
                "Training/Validation Split",
                min_value=10, max_value=30, value=20,
                help="Percentage of training data to use for validation"
            )
    
    st.subheader("Fine-Tuning Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3,
                                help="Number of complete passes through the training dataset")
        batch_size = st.select_slider(
            "Batch Size",
            options=[1, 2, 4, 8, 16, 32, 64, 128],
            value=8,
            help="Number of samples processed before model update"
        )
    
    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.0001,
            format_func=lambda x: f"{x:.5f}",
            help="Step size for gradient updates"
        )
        optimizer = st.selectbox(
            "Optimizer",
            ["AdamW", "SGD", "Adam", "RMSProp"],
            help="Algorithm used to update model weights"
        )
    
    with col3:
        advanced = st.checkbox("Show Advanced Options")
        
        if advanced:
            weight_decay = st.slider(
                "Weight Decay",
                min_value=0.0, max_value=0.1, value=0.01, step=0.01,
                help="L2 regularization to prevent overfitting"
            )
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                help="Fraction of neurons randomly set to zero during training"
            )
        else:
            weight_decay = 0.01
            dropout = 0.1
    
    # # Resource allocation
    # st.subheader("Resource Allocation")
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     gpu_type = st.selectbox(
    #         "GPU Type",
    #         ["NVIDIA A100", "NVIDIA V100", "NVIDIA T4", "Multi-GPU Cluster"]
    #     )
    
    # with col2:
    #     priority = st.select_slider(
    #         "Job Priority",
    #         options=["Low", "Standard", "High", "Urgent"],
    #         value="Standard",
    #         help="Higher priority jobs may start sooner but cost more credits"
    #     )
    
    # Validation and submission
    st.subheader("Validation & Submission")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ready = all([
            hf_token,
            dataset_upload is not None,
            (validation_method == "Split from training data" or validation_upload is not None)
        ])
        
        if not ready:
            st.warning("Please complete all required fields before proceeding")
        else:
            st.success("Configuration complete - ready to begin fine-tuning")
    
    with col2:
        start_button = st.button("Start Fine-Tuning", disabled=not ready, use_container_width=True)
        
        if start_button:
            st.session_state.stage = 1
            st.session_state.progress = 0
            st.rerun()

elif st.session_state.stage == 1:  # Training stage
    st.header("Model Training")
    
    # Display selected configuration
    with st.expander("Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model:** " + st.session_state.get('selected_model', 'Llama 3'))
            st.write("**Epochs:** " + str(st.session_state.get('epochs', 3)))
            st.write("**Batch Size:** " + str(st.session_state.get('batch_size', 8)))
            st.write("**Learning Rate:** " + str(st.session_state.get('learning_rate', 0.0001)))
        with col2:
            st.write("**Optimizer:** " + st.session_state.get('optimizer', 'AdamW'))
            st.write("**GPU Type:** " + st.session_state.get('gpu_type', 'NVIDIA A100'))
            st.write("**Priority:** " + st.session_state.get('priority', 'Standard'))
    
    # Training progress
    st.subheader("Training Progress")
    
    progress_bar = st.progress(int(st.session_state.progress))
    status_text = st.empty()
    
    # Training logs
    log_container = st.container()
    
    # Simulated training progress
    if st.session_state.progress < 100:
        # Generate some realistic logs
        logs = []
        current_epoch = int(st.session_state.progress / 33) + 1
        max_epochs = 3
        
        for epoch in range(1, current_epoch + 1):
            loss = 0.5 - (0.1 * epoch) + random.uniform(-0.05, 0.05)
            logs.append(f"Epoch {epoch}/{max_epochs} - loss: {loss:.4f} - val_loss: {(loss + random.uniform(0.01, 0.1)):.4f}")
            
            # Add batch logs for current epoch
            if epoch == current_epoch:
                total_batches = 50
                completed_batches = int((st.session_state.progress % 33) / 33 * total_batches)
                for batch in range(1, completed_batches + 1):
                    batch_loss = loss - (0.001 * batch) + random.uniform(-0.01, 0.01)
                    logs.append(f"Batch {batch}/{total_batches} - loss: {batch_loss:.4f}")
        
        with log_container:
            log_area = st.text_area("Training Logs", value="\n".join(logs), height=200)
        
        # Update progress
        increment = random.randint(2, 5)
        new_progress = min(st.session_state.progress + increment, 100)
        
        if st.session_state.progress < 33:
            status_text.info("Training: Epoch 1 in progress...")
        elif st.session_state.progress < 66:
            status_text.info("Training: Epoch 2 in progress...")
        else:
            status_text.info("Training: Epoch 3 in progress...")
        
        st.session_state.progress = new_progress
        progress_bar.progress(int(new_progress))
        
        if new_progress < 100:
            time.sleep(1)  # Simulate training time
            st.rerun()
        else:
            st.session_state.training_complete = True
            st.session_state.model_path = f"fine-tuned-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            status_text.success("Training complete! Model saved successfully.")
            
            # Generate metrics for evaluation
            st.session_state.metrics = {
                'train_loss': [0.532, 0.423, 0.387],
                'val_loss': [0.587, 0.512, 0.452],
                'perplexity': 4.23,
                'bleu': 0.78,
                'rouge': {'rouge-1': 0.82, 'rouge-2': 0.51, 'rouge-l': 0.79},
                'token_accuracy': 0.91
            }
            
            st.button("Proceed to Evaluation", on_click=lambda: setattr(st.session_state, 'stage', 2))

elif st.session_state.stage == 2:  # Evaluation stage
    st.header("Model Evaluation")
    
    # Display metrics
    metrics = st.session_state.metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Loss", f"{metrics['train_loss'][-1]:.4f}", 
                 delta=f"-{metrics['train_loss'][0] - metrics['train_loss'][-1]:.3f}")
    with col2:
        st.metric("Perplexity", f"{metrics['perplexity']:.2f}")
    with col3:
        st.metric("BLEU Score", f"{metrics['bleu']:.2f}")
    with col4:
        st.metric("Token Accuracy", f"{metrics['token_accuracy']:.2%}")
    
    st.subheader("Training Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss curve
        fig, ax = plt.subplots()
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        ax.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    with col2:
        # ROUGE scores
        rouge_data = pd.DataFrame({
            'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
            'Score': [metrics['rouge']['rouge-1'], metrics['rouge']['rouge-2'], metrics['rouge']['rouge-l']]
        })
        
        fig, ax = plt.subplots()
        ax.bar(rouge_data['Metric'], rouge_data['Score'], color=['#4285F4', '#FBBC05', '#34A853'])
        ax.set_ylim(0, 1)
        ax.set_title('ROUGE Metrics')
        ax.set_ylabel('Score')
        for i, v in enumerate(rouge_data['Score']):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
        st.pyplot(fig)
    
    # Sample outputs
    st.subheader("Sample Outputs")
    
    with st.expander("View Sample Predictions", expanded=True):
        samples = [
            {"input": "What are the main applications of artificial intelligence?", 
             "output": "Artificial intelligence has numerous applications across various sectors including healthcare (disease diagnosis, drug discovery), finance (fraud detection, algorithmic trading), transportation (autonomous vehicles), customer service (chatbots), manufacturing (predictive maintenance), and entertainment (recommendation systems). These applications leverage AI's ability to process large amounts of data, recognize patterns, and make decisions with minimal human intervention."},
            {"input": "Explain the concept of neural networks.", 
             "output": "Neural networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information. The basic architecture includes an input layer that receives data, hidden layers that perform computations, and an output layer that produces results. Each connection between neurons has a weight that adjusts during training through a process called backpropagation, allowing the network to learn patterns and make predictions from data."}
        ]
        
        for i, sample in enumerate(samples):
            st.markdown(f"**Example {i+1}:**")
            st.markdown(f"**Input:** {sample['input']}")
            st.markdown(f"**Model Output:** {sample['output']}")
            st.markdown("---")
    
    # Performance comparison
    st.subheader("Comparison with Base Model")
    
    comparison_data = pd.DataFrame({
        'Metric': ['Perplexity', 'BLEU Score', 'ROUGE-L', 'Inference Time (ms)'],
        'Base Model': [7.85, 0.65, 0.62, 125],
        'Fine-tuned Model': [4.23, 0.78, 0.79, 130]
    })
    
    st.table(comparison_data)
    
    # Inference testing
    st.subheader("Live Inference Testing")
    
    test_input = st.text_area("Enter text to test the model", 
                             "Explain the process of fine-tuning language models.")
    
    if st.button("Run Inference"):
        with st.spinner("Generating response..."):
            time.sleep(2)  # Simulate inference time
            
            # Sample response
            response = """Fine-tuning language models involves taking a pre-trained foundation model and further training it on a specific dataset to adapt it for particular tasks or domains. The process typically includes:

1. Preparing a high-quality dataset relevant to the target application
2. Configuring hyperparameters like learning rate, batch size, and training epochs
3. Implementing training techniques such as parameter-efficient methods (LoRA, QLoRA)
4. Monitoring training metrics to prevent overfitting
5. Evaluating the fine-tuned model against appropriate benchmarks

This approach is more efficient than training from scratch as it leverages the knowledge already captured in the pre-trained weights while adapting the model to specialized requirements."""
            
            st.markdown("### Model Response:")
            st.markdown(response)
    
    # Proceed to deployment
    st.button("Proceed to Deployment", on_click=lambda: setattr(st.session_state, 'stage', 3))

elif st.session_state.stage == 3:  # Deployment stage
    st.header("Model Deployment")
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Model ID:** `{st.session_state.model_path}`")
        st.markdown("**Base Model:** Llama 3")
        st.markdown("**Size:** 2.1 GB")
        st.markdown("**Framework:** PyTorch")
    
    with col2:
        st.markdown("**Training Completed:** ✅")
        st.markdown("**Evaluation Completed:** ✅")
        st.markdown("**Fine-tuning Time:** 4m 32s")
        st.markdown("**Status:** Ready for deployment")
    
    # Deployment options
    st.subheader("Deployment Options")
    
    tab1, tab2, tab3 = st.tabs(["Hugging Face", "API Endpoint", "Export"])
    
    with tab1:
        st.markdown("### Push to Hugging Face Hub")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            hf_repo_name = st.text_input("Repository Name", 
                                        f"my-fine-tuned-{st.session_state.get('selected_model', 'llama3').lower().replace(' ', '-')}")
            hf_visibility = st.radio("Visibility", ["Private", "Public"])
            hf_description = st.text_area("Description", 
                                         "Fine-tuned language model for specialized tasks.")
        
        with col2:
            if st.button("Push to Hub", use_container_width=True):
                with st.spinner("Uploading model to Hugging Face..."):
                    # Simulate upload process
                    progress_bar = st.progress(0)
                    for i in range(101):
                        time.sleep(0.05)
                        progress_bar.progress(i)
                    
                    st.session_state.deployed = True
                    st.success(f"Model successfully uploaded to Hugging Face Hub at `{hf_repo_name}`")
                    
                    # Display model card
                    st.markdown("### Model Card Preview")
                    st.code(f"""---
language: en
license: mit
datasets:
  - custom_dataset
tags:
  - {st.session_state.get('selected_model', 'llama3').lower()}
  - fine-tuned
---

# {hf_repo_name}

{hf_description}

## Model Details

- **Base Model:** {st.session_state.get('selected_model', 'Llama 3')}
- **Training Platform:** Spheron decentralized GPU network
- **Fine-tuning Method:** Full parameter fine-tuning

## Performance

- **Perplexity:** {st.session_state.metrics['perplexity']:.2f}
- **BLEU Score:** {st.session_state.metrics['bleu']:.2f}
- **ROUGE-L:** {st.session_state.metrics['rouge']['rouge-l']:.2f}
""", language="yaml")
    
    with tab2:
        st.markdown("### Deploy as API Endpoint")
        
        col1, col2 = st.columns(2)
        
        with col1:
            endpoint_name = st.text_input("Endpoint Name", "my-model-api")
            compute_type = st.selectbox(
                "Compute Type",
                ["Standard CPU", "GPU Accelerated", "Enterprise Cluster"]
            )
        
        with col2:
            scaling = st.selectbox(
                "Scaling Mode",
                ["Single Instance", "Auto Scaling"]
            )
            
            if scaling == "Auto Scaling":
                min_instances = st.number_input("Minimum Instances", min_value=1, value=1)
                max_instances = st.number_input("Maximum Instances", min_value=1, value=3)
        
        if st.button("Deploy API"):
            with st.spinner("Deploying model to API endpoint..."):
                # Simulate deployment process
                time.sleep(3)
                
                st.success(f"API endpoint deployed successfully!")
                st.code("""curl -X POST \\
  https://api.spheron.ai/v1/models/{endpoint_name}/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": "Your prompt here"}'""".replace("{endpoint_name}", endpoint_name), language="bash")
    
    with tab3:
        st.markdown("### Export Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["PyTorch (native)", "ONNX", "TensorFlow SavedModel", "TorchScript"]
            )
            
            quantization = st.selectbox(
                "Quantization",
                ["None (FP32)", "FP16", "INT8", "INT4"]
            )
        
        with col2:
            optimization = st.multiselect(
                "Optimizations",
                ["Pruning", "Knowledge Distillation", "Weight Merging", "None"],
                default=["None"]
            )
        
        if st.button("Export Model"):
            with st.spinner("Preparing model for export..."):
                # Simulate export process
                time.sleep(2)
                
                st.success(f"Model exported successfully in {export_format} format")
                
                # Download button
                st.download_button(
                    label="Download Model",
                    data=b"This would be the actual model file",
                    file_name=f"fine-tuned-model-{export_format.lower().replace(' ', '-')}.zip",
                    mime="application/zip"
                )
    
    # Next steps
    st.subheader("Next Steps")
    
    st.markdown("""
    - **Integrate** your fine-tuned model into your application
    - **Monitor** performance and collect user feedback
    - **Iterate** with additional fine-tuning as needed
    - **Share** your model with collaborators
    """)
    
    # Reset pipeline
    if st.button("Start New Fine-Tuning Pipeline"):
        st.session_state.stage = 0
        st.session_state.training_complete = False
        st.session_state.evaluation_complete = False
        st.session_state.deployed = False
        st.session_state.metrics = None
        st.session_state.model_path = None
        st.session_state.progress = 0
        st.rerun()
