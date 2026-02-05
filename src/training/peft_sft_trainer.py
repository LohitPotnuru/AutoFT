"""PEFT SFT Trainer

This module provides functions for fine-tuning models using Parameter-Efficient Fine-Tuning (PEFT)
with LoRA adapters for Supervised Fine-Tuning (SFT) on customer support ticket data.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from src.utils import load_yaml_config

# Load global config for company_id
GLOBAL_CONFIG_PATH = "configs/global.yaml"
global_config = load_yaml_config(GLOBAL_CONFIG_PATH, required=True)
company_id = global_config.get("company", {}).get("company_id", None)
if not company_id:
    raise ValueError("company_id is not set in global.yaml. Update configs/global.yaml with the correct company_id.")

# Load model config
MODEL_CONFIG_PATH = "configs/model/model_config.yaml"
model_config = load_yaml_config(MODEL_CONFIG_PATH, required=True)
model_name = model_config.get("model", {}).get("model_name", None)
if not model_name:
    raise ValueError("model_name is not set in model_config.yaml")


def train_sft(
    sft_data_path: Optional[str] = None,
    adapter_output_path: Optional[str] = None,
    model_name_override: Optional[str] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    training_args: Optional[Dict[str, Any]] = None
) -> None:
    """Train a LoRA adapter for Supervised Fine-Tuning (SFT) on customer support tickets.
    
    Loads a model in float16 precision, applies LoRA adapters, and fine-tunes on SFT data
    from the processed JSONL file. The adapter is saved for later use.
    
    Args:
        sft_data_path: Path to SFT training data JSONL file. 
                       Defaults to f"data/processed/{company_id}/sft.jsonl"
        adapter_output_path: Path to save the LoRA adapter.
                            Defaults to f"lora_adapters/{company_id}/sft_adapter"
        model_name_override: Optional model name override. Uses model_config.yaml if None.
        lora_config: Optional dictionary to override default LoRA configuration.
        training_args: Optional dictionary to override default training arguments.
        
    Raises:
        FileNotFoundError: If SFT data file doesn't exist
        ValueError: If company_id or model_name is not set
    """
    # Use company_id from global config
    if not company_id:
        raise ValueError("company_id is not set in global.yaml")
    
    # Get model name
    model = model_name_override or model_name
    if not model:
        raise ValueError("model_name is not set in model_config.yaml")
    
    # Set default paths
    if sft_data_path is None:
        sft_data_path = f"data/processed/{company_id}/sft.jsonl"
    
    if adapter_output_path is None:
        adapter_output_path = f"lora_adapters/{company_id}/sft_adapter"
    
    # Check if data file exists
    data_path = Path(sft_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"SFT data file not found: {sft_data_path}")
    
    # Create output directory
    adapter_path = Path(adapter_output_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model}")
    print(f"Loading SFT data from: {sft_data_path}")
    print(f"Saving adapter to: {adapter_output_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in float16 (Apple Silicon compatible)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Configure LoRA
    default_lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    if lora_config:
        default_lora_config.update(lora_config)
    
    peft_config = LoraConfig(**default_lora_config)
    
    # Load dataset
    dataset = load_dataset("json", data_files=sft_data_path, split="train")
    
    # Default training arguments
    default_training_args = {
        "output_dir": str(adapter_path / "checkpoints"),
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "optim": "adamw_torch",  # Changed from paged_adamw_32bit for Apple Silicon compatibility
        "logging_steps": 10,
        "save_steps": 100,
        "learning_rate": 2e-4,
        "fp16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "report_to": "none",
    }
    
    if training_args:
        default_training_args.update(training_args)
    
    training_arguments = TrainingArguments(**default_training_args)

    sft_config = SFTConfig(
        **default_training_args,
        max_length=512,
        packing=False,
        dataset_text_field="messages"
    )
    
    # Create SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class = tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving adapter to {adapter_output_path}...")
    model.save_pretrained(adapter_output_path)
    tokenizer.save_pretrained(adapter_output_path)
    
    print("Training complete!")
