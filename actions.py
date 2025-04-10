# guardrails/actions.py
from typing import Dict, Any, List
import re

def validate_context(context: List[str]) -> Dict[str, Any]:
    """
    Validate if the retrieved context is relevant and safe
    """
    risky_patterns = [
        r'(?i)(?:how\s+to|steps\s+for|instructions\s+on)\s+(?:hack|attack|exploit)',
        r'(?i)(?:create|make|build)\s+(?:bomb|explosive|weapon)',
        r'(?i)(?:synthesize|produce)\s+(?:illegal\s+drug|narcotic)'
    ]
    
    # Check for risky content in retrieved passages
    for passage in context:
        for pattern in risky_patterns:
            if re.search(pattern, passage):
                return {
                    "success": False,
                    "message": "Retrieved context contains potentially unsafe information"
                }
    
    # Check for minimum context relevance
    if not context or all(len(c.strip()) < 20 for c in context):
        return {
            "success": False, 
            "message": "Insufficient context retrieved"
        }
        
    return {"success": True, "context": context}

def check_response_safety(response: str) -> Dict[str, Any]:
    """
    Check if the generated response contains unsafe content
    """
    unsafe_patterns = [
        r'(?i)(?:steps|instructions|guide)\s+(?:to|for)\s+(?:create|make|hack)',
        r'(?i)(?:how\s+to|method\s+for)\s+(?:bypass|circumvent|break)',
        r'(?i)(?:recipe|formula)\s+(?:for|to\s+make)\s+(?:explosive|drug)'
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, response):
            return {
                "success": False,
                "message": "Response contains potentially unsafe content"
            }
    
    # Check for disclaimer/citation
    if len(response) > 100 and not any(phrase in response.lower() for phrase in [
        "based on the document", "according to the pdf", "the text states"
    ]):
        return {
            "success": False,
            "message": "Response does not properly cite the source document"
        }
        
    return {"success": True, "response": response}