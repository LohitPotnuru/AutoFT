"""Data Ingestion Pipeline

This module processes raw synthetic ticket data and converts it into training formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

from src.utils import load_yaml_config

GLOBAL_CONFIG_PATH = "configs/global.yaml"
global_config = load_yaml_config(GLOBAL_CONFIG_PATH, required=True)
company_id = global_config.get("company", {}).get("company_id", None)
if not company_id:
    raise ValueError("company_id is not set in global.yaml. Update configs/global.yaml with the correct company_id.")


def process_success(ticket: Dict[str, Any], system_message: Optional[str] = None) -> Dict[str, Any]:
    """Process a single success ticket into the training format.
    
    Converts a success ticket from the JSONL format into a messages format suitable
    for fine-tuning, with system, user, and assistant roles.
    
    Args:
        ticket: Dictionary containing a success ticket with:
            - customer_message: The customer's inquiry
            - initial_response: The correct support agent response
            - category: Should be "success"
        system_message: Optional custom system message. Defaults to standard support agent message.
        
    Returns:
        Dictionary with "messages" key containing:
            - system: System message defining the agent role
            - user: Customer's message
            - assistant: Support agent's correct response
            
    Raises:
        ValueError: If ticket is not a success ticket or missing required fields
    """
    # Validate ticket is a success ticket
    if ticket.get("category") != "success":
        raise ValueError(f"Expected success ticket, got category: {ticket.get('category')}")
    
    # Validate required fields
    if "customer_message" not in ticket:
        raise ValueError("Ticket missing required field: customer_message")
    if "initial_response" not in ticket:
        raise ValueError("Ticket missing required field: initial_response")
    
    # Default system message
    if system_message is None:
        system_message = "You are a helpful support agent for a SaaS company named Techflow Solutions."
    
    # Build the messages format
    result = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": ticket["customer_message"]
            },
            {
                "role": "assistant",
                "content": ticket["initial_response"]
            }
        ]
    }
    
    return result


def process_correction(ticket: Dict[str, Any], system_message: Optional[str] = None) -> Dict[str, Any]:
    """Process a single correction ticket into the DPO/RLHF training format.
    
    Converts a correction ticket from the JSONL format into a format suitable for
    Direct Preference Optimization (DPO) or Reinforcement Learning from Human Feedback (RLHF),
    with prompt, chosen (corrected response), and rejected (initial incorrect response).
    
    Args:
        ticket: Dictionary containing a correction ticket with:
            - customer_message: The customer's inquiry
            - initial_response: The incorrect support agent response (rejected)
            - corrected_response: The correct support agent response (chosen)
            - category: Should be "correction"
        system_message: Optional custom system message. Defaults to standard support agent message.
        
    Returns:
        Dictionary with:
            - prompt: List containing system and user messages
            - chosen: List containing the corrected (good) assistant response
            - rejected: List containing the initial (bad) assistant response
            
    Raises:
        ValueError: If ticket is not a correction ticket or missing required fields
    """
    # Validate ticket is a correction ticket
    if ticket.get("category") != "correction":
        raise ValueError(f"Expected correction ticket, got category: {ticket.get('category')}")
    
    # Validate required fields
    if "customer_message" not in ticket:
        raise ValueError("Ticket missing required field: customer_message")
    if "initial_response" not in ticket:
        raise ValueError("Ticket missing required field: initial_response")
    if "corrected_response" not in ticket:
        raise ValueError("Ticket missing required field: corrected_response")
    
    # Default system message
    if system_message is None:
        system_message = "You are a helpful support agent for a SaaS company named Techflow Solutions."
    
    # Build the DPO/RLHF format
    result = {
        "prompt": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": ticket["customer_message"]
            }
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": ticket["corrected_response"]
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": ticket["initial_response"]
            }
        ]
    }
    
    return result


def ingest_data(
    input_file: Optional[str] = None,
    sft_output_file: Optional[str] = None,
    dpo_output_file: Optional[str] = None,
    system_message: Optional[str] = None
):
    """Read and process tickets from JSONL file, writing to SFT and DPO output files.
    
    Processes tickets line by line from the input JSONL file:
    - Success tickets are processed and written to SFT output file
    - Correction tickets are processed and written to DPO output file
    - Failed tickets are skipped
    
    Args:
        input_file: Path to input JSONL file. Defaults to f"data/synthetic_raw/{company_id}.jsonl"
        sft_output_file: Path to SFT output file. Defaults to f"data/processed/{company_id}/sft.jsonl"
        dpo_output_file: Path to DPO output file. Defaults to f"data/processed/{company_id}/dpo.jsonl"
        system_message: Optional custom system message for processing. Uses default if None.
        
    Returns:
        None
            
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If company_id is not set
    """
    # Use company_id from global config
    if not company_id:
        raise ValueError("company_id is not set in global.yaml")
    
    # Set default file paths
    if input_file is None:
        input_file = f"data/synthetic_raw/{company_id}.jsonl"
    
    if sft_output_file is None:
        sft_output_file = f"data/processed/{company_id}/sft.jsonl"
    
    if dpo_output_file is None:
        dpo_output_file = f"data/processed/{company_id}/dpo.jsonl"
    
    # Create output directories if they don't exist
    sft_path = Path(sft_output_file)
    dpo_path = Path(dpo_output_file)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    dpo_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    
    # Process tickets line by line
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(sft_path, 'a', encoding='utf-8') as sft_file, \
         open(dpo_path, 'a', encoding='utf-8') as dpo_file:
        
        for line in tqdm(infile, desc="Processing tickets"):
            line = line.strip()
            if not line:
                continue
            
            try:
                ticket = json.loads(line)
                category = ticket.get("category")
                
                if category == "success":
                    # Process success ticket for SFT
                    processed = process_success(ticket, system_message=system_message)
                    sft_file.write(json.dumps(processed) + '\n')
                    
                elif category == "correction":
                    # Process correction ticket for DPO
                    processed = process_correction(ticket, system_message=system_message)
                    dpo_file.write(json.dumps(processed) + '\n')
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON {e}")
            except ValueError as e:
                print(f"Warning: Failed to process ticket {e}")
            except Exception as e:
                print(f"Warning: Unexpected error processing ticket {e}")
    
