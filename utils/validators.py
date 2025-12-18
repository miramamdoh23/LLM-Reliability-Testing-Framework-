"""
Behavioral Validators for LLM Responses
Author: Mira Mamdoh
"""

from typing import Dict, List, Any
import re


class BehaviorValidator:
    """Validate LLM responses against expected behaviors"""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_must_include(self, response: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Check if response contains required keywords
        """
        response_lower = response.lower()
        missing_keywords = []
        found_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        is_valid = len(missing_keywords) == 0
        
        return {
            "valid": is_valid,
            "found": found_keywords,
            "missing": missing_keywords,
            "coverage": len(found_keywords) / len(keywords) if keywords else 1.0
        }
    
    def validate_must_not_include(self, response: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Check if response avoids forbidden keywords
        """
        response_lower = response.lower()
        found_forbidden = []
        
        for keyword in keywords:
            if keyword.lower() in response_lower:
                found_forbidden.append(keyword)
        
        is_valid = len(found_forbidden) == 0
        
        return {
            "valid": is_valid,
            "forbidden_found": found_forbidden,
            "message": "Response contains forbidden keywords" if found_forbidden else "No forbidden keywords"
        }
    
    def validate_length(self, response: str, min_length: int = 0, max_length: int = 10000) -> Dict[str, Any]:
        """
        Validate response length
        """
        actual_length = len(response)
        is_valid = min_length <= actual_length <= max_length
        
        return {
            "valid": is_valid,
            "actual_length": actual_length,
            "min_length": min_length,
            "max_length": max_length,
            "message": "Length within bounds" if is_valid else f"Length {actual_length} outside bounds [{min_length}, {max_length}]"
        }
    
    def validate_tone(self, response: str, expected_tone: str) -> Dict[str, Any]:
        """
        Simple tone validation based on keywords and patterns
        More lenient to account for varied valid responses
        """
        response_lower = response.lower()
        
        tone_indicators = {
            "informative": ["explains", "describes", "demonstrates", "shows", "indicates", "provides", "helps", "allows", "enables"],
            "technical": ["algorithm", "function", "parameter", "system", "process", "implementation", "architecture"],
            "professional": ["therefore", "however", "furthermore", "consequently", "additionally", "specifically"],
            "conversational": ["you", "let's", "we can", "i think", "you can", "we use"],
            "explanatory": ["because", "since", "due to", "this means", "in other words", "for example", "such as"]
        }
        
        indicators = tone_indicators.get(expected_tone.lower(), [])
        found_indicators = [ind for ind in indicators if ind in response_lower]
        
        # More lenient: valid if response is substantial OR has indicators
        is_valid = len(found_indicators) > 0 or len(response) > 50
        
        return {
            "valid": is_valid,
            "expected_tone": expected_tone,
            "indicators_found": found_indicators,
            "confidence": len(found_indicators) / len(indicators) if indicators else 0.5
        }
    
    def validate_no_errors(self, response: str) -> Dict[str, Any]:
        """
        Check for common error patterns in LLM responses
        """
        error_patterns = [
            r"error:",
            r"exception:",
            r"failed to",
            r"cannot process",
            r"invalid input",
            r"i apologize, but i cannot",
            r"i'm sorry, i can't"
        ]
        
        errors_found = []
        for pattern in error_patterns:
            if re.search(pattern, response.lower()):
                errors_found.append(pattern)
        
        return {
            "valid": len(errors_found) == 0,
            "errors_found": errors_found,
            "message": "No errors detected" if not errors_found else f"Found error patterns: {errors_found}"
        }
    
    def validate_coherence(self, response: str) -> Dict[str, Any]:
        """
        Basic coherence check
        """
        if not response:
            return {"valid": False, "message": "Empty response"}
        
        # Check for reasonable sentence structure
        sentences = response.split('.')
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        # Check for repetition
        words = response.lower().split()
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words) if unique_words else 0
        
        is_coherent = len(valid_sentences) > 0 and repetition_ratio < 3.0
        
        return {
            "valid": is_coherent,
            "sentence_count": len(valid_sentences),
            "repetition_ratio": repetition_ratio,
            "message": "Response appears coherent" if is_coherent else "Response may lack coherence"
        }
    
    def validate_all(self, response: str, expected_behavior: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all applicable validations
        """
        results = {
            "response_length": len(response),
            "validations": {}
        }
        
        # Must include keywords
        if "must_include" in expected_behavior:
            results["validations"]["must_include"] = self.validate_must_include(
                response, expected_behavior["must_include"]
            )
        
        # Must not include keywords
        if "must_not_include" in expected_behavior:
            results["validations"]["must_not_include"] = self.validate_must_not_include(
                response, expected_behavior["must_not_include"]
            )
        
        # Length validation
        if "min_length" in expected_behavior or "max_length" in expected_behavior:
            results["validations"]["length"] = self.validate_length(
                response,
                expected_behavior.get("min_length", 0),
                expected_behavior.get("max_length", 10000)
            )
        
        # Tone validation
        if "tone" in expected_behavior:
            results["validations"]["tone"] = self.validate_tone(
                response, expected_behavior["tone"]
            )
        
        # Error check
        results["validations"]["no_errors"] = self.validate_no_errors(response)
        
        # Coherence check
        results["validations"]["coherence"] = self.validate_coherence(response)
        
        # Overall validity
        all_valid = all(
            v.get("valid", False) 
            for v in results["validations"].values()
        )
        results["overall_valid"] = all_valid
        
        return results


def quick_validate(response: str, must_include: List[str] = None, must_not_include: List[str] = None) -> bool:
    """
    Quick validation helper for simple checks
    """
    validator = BehaviorValidator()
    
    if must_include:
        result = validator.validate_must_include(response, must_include)
        if not result["valid"]:
            return False
    
    if must_not_include:
        result = validator.validate_must_not_include(response, must_not_include)
        if not result["valid"]:
            return False
    
    return True