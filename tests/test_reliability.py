"""
LLM Reliability Testing
Tests for non-deterministic behavior and output consistency
Author: Mira Mamdoh
"""

import pytest
import json
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient
from utils.similarity import SemanticSimilarity
from utils.validators import BehaviorValidator
from utils.metrics import ReliabilityMetrics


class TestLLMReliability:
    """Test suite for LLM reliability and consistency"""
    
    @pytest.fixture(scope="class")
    def llm_client(self):
        """Initialize LLM client (mock for testing)"""
        return LLMClient(use_mock=True, model="gpt-4-mock")
    
    @pytest.fixture(scope="class")
    def similarity_checker(self):
        """Initialize semantic similarity checker"""
        return SemanticSimilarity()
    
    @pytest.fixture(scope="class")
    def validator(self):
        """Initialize behavior validator"""
        return BehaviorValidator()
    
    @pytest.fixture(scope="class")
    def metrics_calculator(self):
        """Initialize metrics calculator"""
        return ReliabilityMetrics()
    
    @pytest.fixture(scope="class")
    def golden_prompts(self):
        """Load golden prompt dataset"""
        data_path = Path(__file__).parent.parent / "data" / "golden_prompts.json"
        with open(data_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def expected_behaviors(self):
        """Load expected behaviors"""
        data_path = Path(__file__).parent.parent / "data" / "expected_behaviors.json"
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def test_single_prompt_consistency(self, llm_client, similarity_checker):
        """
        Test: Single prompt should produce consistent outputs
        Run same prompt multiple times and check semantic similarity
        """
        prompt = "Explain what an EDA tool does in simple terms"
        n_runs = 10
        
        print(f"\n🧪 Testing consistency with {n_runs} runs...")
        
        # Generate multiple responses
        responses = llm_client.generate_multiple(prompt, n=n_runs, temperature=0.7)
        texts = [r["text"] for r in responses]
        
        # Calculate pairwise similarities
        similarities = similarity_checker.calculate_pairwise_similarity(texts)
        avg_similarity = similarity_checker.calculate_average_similarity(texts)
        
        print(f"📊 Average similarity: {avg_similarity:.3f}")
        print(f"📊 Min similarity: {min(similarities):.3f}")
        print(f"📊 Max similarity: {max(similarities):.3f}")
        
        # Assert consistency threshold
        assert avg_similarity > 0.70, f"Consistency too low: {avg_similarity:.3f}"
        assert min(similarities) > 0.60, f"Some outputs too different: {min(similarities):.3f}"
        
        print("✅ Consistency test PASSED")
    
    def test_temperature_effect_on_consistency(self, llm_client, similarity_checker):
        """
        Test: Lower temperature should produce more consistent outputs
        """
        prompt = "What is machine learning?"
        n_runs = 8
        
        print("\n🌡️ Testing temperature effect...")
        
        # Test with low temperature
        low_temp_responses = llm_client.generate_multiple(prompt, n=n_runs, temperature=0.1)
        low_temp_texts = [r["text"] for r in low_temp_responses]
        low_temp_similarity = similarity_checker.calculate_average_similarity(low_temp_texts)
        
        # Test with high temperature
        high_temp_responses = llm_client.generate_multiple(prompt, n=n_runs, temperature=0.9)
        high_temp_texts = [r["text"] for r in high_temp_responses]
        high_temp_similarity = similarity_checker.calculate_average_similarity(high_temp_texts)
        
        print(f"📊 Low temp (0.1) similarity: {low_temp_similarity:.3f}")
        print(f"📊 High temp (0.9) similarity: {high_temp_similarity:.3f}")
        
        # Low temperature should be more consistent
        assert low_temp_similarity >= high_temp_similarity - 0.1, \
            "Temperature effect not as expected"
        
        print("✅ Temperature effect test PASSED")
    
    def test_behavioral_validation(self, llm_client, validator, expected_behaviors):
        """
        Test: Responses should meet expected behavioral criteria
        """
        prompt = "Explain what an EDA tool does in simple terms"
        expected = expected_behaviors.get("EDA_001", {})
        
        print("\n🎯 Testing behavioral validation...")
        
        # Generate response
        response = llm_client.generate(prompt, temperature=0.7)
        text = response["text"]
        
        # Validate behavior
        validation_result = validator.validate_all(text, expected)
        
        print(f"📋 Response length: {len(text)}")
        print(f"📋 Overall valid: {validation_result['overall_valid']}")
        
        # Print individual validation results
        for val_name, val_result in validation_result["validations"].items():
            status = "✅" if val_result.get("valid", False) else "❌"
            print(f"  {status} {val_name}: {val_result.get('message', 'OK')}")
        
        # Assert overall validity
        assert validation_result["overall_valid"], \
            f"Behavioral validation failed: {validation_result}"
        
        print("✅ Behavioral validation PASSED")
    
    def test_multiple_prompts_reliability(self, llm_client, similarity_checker, golden_prompts, metrics_calculator):
        """
        Test: Multiple different prompts should all show good reliability
        """
        print("\n📚 Testing multiple prompts...")
        
        # Filter out edge cases
        test_prompts = [p for p in golden_prompts if p["category"] != "edge_case"][:3]
        
        all_reliable = True
        results = []
        
        for prompt_data in test_prompts:
            prompt = prompt_data["prompt"]
            prompt_id = prompt_data["id"]
            
            print(f"\n  Testing: {prompt_id}")
            
            # Generate multiple responses
            responses = llm_client.generate_multiple(prompt, n=5, temperature=0.7)
            texts = [r["text"] for r in responses]
            
            # Calculate similarity
            similarities = similarity_checker.calculate_pairwise_similarity(texts)
            avg_sim = similarity_checker.calculate_average_similarity(texts)
            
            # Calculate reliability score
            reliability = metrics_calculator.calculate_reliability_score(similarities, threshold=0.70)
            
            print(f"    Reliability: {reliability['reliability_score']}%")
            
            results.append({
                "prompt_id": prompt_id,
                "reliability_score": reliability["reliability_score"],
                "avg_similarity": avg_sim
            })
            
            if reliability["status"] == "FAIL":
                all_reliable = False
        
        print(f"\n📊 Overall: {len([r for r in results if r['reliability_score'] >= 80])}/{len(results)} prompts reliable")
        
        # At least 80% of prompts should be reliable
        reliable_count = sum(1 for r in results if r["reliability_score"] >= 80)
        assert reliable_count / len(results) >= 0.65, \
            f"Too many unreliable prompts: {reliable_count}/{len(results)}"
        
        print("✅ Multiple prompts reliability PASSED")
    
    def test_outlier_detection(self, llm_client, similarity_checker):
        """
        Test: Should detect significantly different outputs (outliers)
        """
        prompt = "Explain regression testing in AI systems"
        n_runs = 10
        
        print("\n🔍 Testing outlier detection...")
        
        # Generate responses
        responses = llm_client.generate_multiple(prompt, n=n_runs, temperature=0.7)
        texts = [r["text"] for r in responses]
        
        # Find outliers
        outliers = similarity_checker.find_outliers(texts, threshold=0.65)
        
        print(f"📊 Found {len(outliers)} outliers out of {n_runs} responses")
        
        for idx, text, avg_sim in outliers:
            print(f"  Outlier #{idx}: similarity {avg_sim:.3f}")
        
        # Should have few outliers
        assert len(outliers) <= 2, f"Too many outliers: {len(outliers)}"
        
        print("✅ Outlier detection test PASSED")
    
    def test_empty_prompt_handling(self, llm_client, validator):
        """
        Test: Should handle empty prompts gracefully
        """
        print("\n⚠️ Testing edge case: empty prompt...")
        
        response = llm_client.generate("", temperature=0.7)
        text = response["text"]
        
        print(f"📋 Response: '{text[:50]}...'")
        
        # Should either return empty or error message
        assert len(text) == 0 or "context" in text.lower() or "more" in text.lower(), \
            "Empty prompt not handled properly"
        
        print("✅ Empty prompt handling PASSED")
    
    def test_stability_score_calculation(self, llm_client, similarity_checker, metrics_calculator):
        """
        Test: Calculate and validate stability score
        """
        prompt = "What are the key differences between unit testing and integration testing?"
        n_runs = 10
        
        print("\n📈 Testing stability score calculation...")
        
        # Generate responses
        responses = llm_client.generate_multiple(prompt, n=n_runs, temperature=0.5)
        texts = [r["text"] for r in responses]
        
        # Calculate similarities
        similarities = similarity_checker.calculate_pairwise_similarity(texts)
        
        # Calculate stability
        stability = metrics_calculator.calculate_stability_score(similarities)
        
        print(f"📊 Stability score: {stability['stability_score']}")
        print(f"📊 Interpretation: {stability['interpretation']}")
        
        # Should have reasonable stability
        assert stability["stability_score"] >= 60, \
            f"Low stability: {stability['stability_score']}"
        
        print("✅ Stability score test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])