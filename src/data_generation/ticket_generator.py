"""Ticket Generator Module

This module provides functions to load configuration files for ticket generation:
- Company knowledge base (products, processes, issues, etc.)
- Generator configuration (ticket counts, settings, etc.)
- Ticket JSON template specification
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pprint
import json
import os
import random
import string
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None

from src.utils import load_yaml_config

# Configuration file paths (relative to project root)
KNOWLEDGE_BASE_PATH = "configs/company_knowledge_base/techflow_solutions.yaml"
GENERATOR_CONFIG_PATH = "configs/data_generation/generator_config.yaml"
TICKET_TEMPLATE_PATH = "configs/data_generation/ticket_json_template.yaml"

# Load configuration files
knowledge_base = load_yaml_config(KNOWLEDGE_BASE_PATH, required=True)
generator_config = load_yaml_config(GENERATOR_CONFIG_PATH, required=True)
ticket_template = load_yaml_config(TICKET_TEMPLATE_PATH, required=True)

load_dotenv()

def get_configs():
    return knowledge_base, generator_config, ticket_template

# Pretty print for debugging
def print_configs():
    pp = pprint.PrettyPrinter(indent=2, width=120, depth=None)
    
    print("=" * 120)
    print("KNOWLEDGE BASE")
    print("=" * 120)
    pp.pprint(knowledge_base)
    
    print("\n" + "=" * 120)
    print("GENERATOR CONFIG")
    print("=" * 120)
    pp.pprint(generator_config)
    
    print("\n" + "=" * 120)
    print("TICKET TEMPLATE")
    print("=" * 120)
    pp.pprint(ticket_template)


def _generate_ticket_id() -> str:
    """Generate a unique ticket ID in the format TKT-{timestamp}-{random_suffix}."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"TKT-{timestamp}-{random_suffix}"


def _build_llm_prompt(
    knowledge_base: Dict[str, Any],
    ticket_template: Dict[str, Any],
    issue: Dict[str, Any],
    category: str
) -> str:
    """Build a comprehensive prompt for the LLM to generate a ticket."""
    
    # Get the template for the specific category
    category_template = ticket_template.get(f"{category}_template", {})
    base_template = ticket_template.get("base_template", {})
    example = ticket_template.get("examples", {}).get(category, {})
    
    # Extract company information
    company_name = knowledge_base.get("company", {}).get("name", "the company")
    products = knowledge_base.get("products", [])
    pricing_tiers = knowledge_base.get("pricing_tiers", [])
    internal_processes = knowledge_base.get("internal_processes", {})
    
    prompt = f"""You are generating a customer support ticket for {company_name}.

TASK: Generate a realistic customer support ticket based on the following specifications.

ISSUE DETAILS:
- Issue ID: {issue.get('issue_id', 'UNKNOWN')}
- Title: {issue.get('title', '')}
- Description: {issue.get('description', '')}
- Products Affected: {', '.join(issue.get('context', {}).get('products_affected', []))}
- Tier Affected: {', '.join(issue.get('context', {}).get('tier_affected', []))}
- Common Scenario: {issue.get('context', {}).get('common_scenario', '')}

COMPANY KNOWLEDGE:
Company: {company_name}
Products: {json.dumps([p.get('name') for p in products], indent=2)}
Pricing Tiers: {json.dumps([t.get('name') for t in pricing_tiers], indent=2)}
Internal Processes: {json.dumps(list(internal_processes.keys()), indent=2)}

REQUIRED KNOWLEDGE TO RESOLVE:
{chr(10).join(f"- {k}" for k in issue.get('required_knowledge', []))}

TICKET CATEGORY: {category.upper()}

For {category} tickets:
"""
    
    if category == "success":
        prompt += f"""
- The initial_response should CORRECTLY resolve the issue using company-specific knowledge
- Use the correct resolution steps: {json.dumps(issue.get('correct_resolution', {}).get('steps', []), indent=2)}
- Use the response template as guidance: {issue.get('correct_resolution', {}).get('response_template', '')}
- requires_intervention must be false
- resolution_status must be "resolved"
"""
    elif category == "correction":
        prompt += f"""
- The initial_response should be INCORRECT or INCOMPLETE (use one of these incorrect responses as inspiration: {json.dumps(issue.get('incorrect_responses', []), indent=2)})
- The corrected_response should CORRECTLY resolve the issue using company-specific knowledge
- Use the correct resolution steps: {json.dumps(issue.get('correct_resolution', {}).get('steps', []), indent=2)}
- Use the response template as guidance: {issue.get('correct_resolution', {}).get('response_template', '')}
- requires_intervention must be true
- resolution_status must be "resolved"
"""
    elif category == "failed":
        prompt += f"""
- The initial_response should be WRONG and fail to resolve the issue
- Use one of these incorrect responses as inspiration: {json.dumps(issue.get('incorrect_responses', []), indent=2)}
- The response should NOT use the correct resolution knowledge
- requires_intervention must be true
- resolution_status must be "failed"
- follow_up_required should be true
"""
    
    prompt += f"""

TICKET TEMPLATE REQUIREMENTS:
{json.dumps(base_template, indent=2)}

EXAMPLE {category.upper()} TICKET:
{json.dumps(example, indent=2)}

INSTRUCTIONS:
1. Generate a realistic customer_message that a customer would write about this issue
2. Generate an initial_response according to the category requirements above
3. For correction category, also generate a corrected_response
4. Ensure all required fields are present according to the template
5. Make the ticket realistic and use company-specific terminology, product names, and processes
6. The ticket must reference specific company knowledge that a foundational model wouldn't know

OUTPUT FORMAT: Return ONLY valid JSON matching the template structure. Do not include any markdown formatting or code blocks.
"""
    
    return prompt


def generate_ticket(
    knowledge_base: Dict[str, Any],
    ticket_template: Dict[str, Any],
    issue: Dict[str, Any],
    category: str,
    model: str = "gpt-4",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a single ticket using an LLM.
    
    Args:
        knowledge_base: Company knowledge base dictionary
        ticket_template: Ticket template specification dictionary
        issue: Specific issue dictionary from knowledge_base['specific_issues']
        category: Ticket category - one of "success", "correction", or "failed"
        model: OpenAI model to use (default: "gpt-4")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        
    Returns:
        Dictionary containing the generated ticket matching the template structure
        
    Raises:
        ValueError: If category is invalid or required dependencies are missing
        RuntimeError: If LLM API call fails
    """
    if category not in ["success", "correction", "failed"]:
        raise ValueError(f"Invalid category: {category}. Must be one of: success, correction, failed")
    
    if openai is None:
        raise RuntimeError(
            "OpenAI library not installed. Install it with: pip install openai"
        )
    
    # Get API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter"
        )
    
    # Build prompt
    prompt = _build_llm_prompt(knowledge_base, ticket_template, issue, category)
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at generating realistic customer support tickets. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract response content
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        ticket = json.loads(content)
        
        # Generate ticket_id if not present
        if "ticket_id" not in ticket or not ticket["ticket_id"]:
            ticket["ticket_id"] = _generate_ticket_id()
        
        # Ensure required fields are set
        ticket["category"] = category
        ticket["issue_type"] = issue.get("issue_id", "UNKNOWN")
        
        if category == "success":
            ticket["resolution_status"] = "resolved"
            ticket["requires_intervention"] = False
        elif category == "correction":
            ticket["resolution_status"] = "resolved"
            ticket["requires_intervention"] = True
            if "corrected_response" not in ticket:
                raise ValueError("correction category requires corrected_response field")
        elif category == "failed":
            ticket["resolution_status"] = "failed"
            ticket["requires_intervention"] = True
            if "follow_up_required" not in ticket:
                ticket["follow_up_required"] = True
        
        return ticket
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nResponse: {content[:500]}")
    except Exception as e:
        raise RuntimeError(f"Error calling LLM API: {e}")


def create_ticket_data() -> None:

    output_dir = generator_config.get("output", {}).get("output_dir", None)
    if not output_dir:
        raise ValueError(
            "output_dir is not set in generator_config.yaml. Update configs/data_generation/generator_config.yaml with the correct output directory."
        )
    ticket_counts = generator_config.get("generation", {}).get("ticket_counts", {})
    if not ticket_counts:
        raise ValueError(
            "Ticket counts are not set in generator_config.yaml. Update configs/data_generation/generator_config.yaml with the correct ticket counts."
        )
    model = generator_config.get("generation", {}).get("model", None)
    if not model:
        model = "gpt-4"
    file_name = knowledge_base.get("company", {}).get("name", "the company").strip().lower().replace(" ", "_")
    
    
    for issue in knowledge_base.get("specific_issues", []):
        issue_id = issue.get("issue_id", "UNKNOWN")
        if issue_id not in ["ISSUE-019", "ISSUE-020"]:
            continue
        for category, category_count in ticket_counts.items():
            for _ in tqdm(range(category_count), desc=f"Generating {category} tickets for {issue.get('issue_id', 'UNKNOWN')}"):
                ticket = generate_ticket(
                    knowledge_base=knowledge_base, 
                    ticket_template=ticket_template, 
                    issue=issue, 
                    category=category, 
                    model=model)
                with open(os.path.join(output_dir, f"{file_name}.jsonl"), "a") as f:
                    f.write(json.dumps(ticket) + "\n")