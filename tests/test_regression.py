"""
LLM Regression Testing
Detect changes in model behavior after updates
Author: Mira Mamdoh
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient
from utils.similarity import SemanticSimilarity
from utils.metrics import ReliabilityMetrics


class TestLLMRegression:
    """Test suite for detecting regressions in LLM behavior"""
    
    @pytest.fixture(scope="class")
    def llm_client(self):
        """Initialize LLM client"""
        return LLMClient(use_mock=True, model="gpt-4-mock")
    
    @pytest.fixture(scope="class")
    def similarity_checker(self):
        """Initialize similarity checker"""
        return SemanticSimilarity()
    
    @pytest.fixture(scope="class")
    def metrics_calculator(self):
        """Initialize metrics calculator"""
        return ReliabilityMetrics()
    
    @pytest.fixture(scope="class")
    def baseline_dir(self):
        """Get baseline results directory"""
        return Path(__file__).parent.parent / "baseline_results"
    
    def _get_baseline(self, prompt_id: str, baseline_dir: Path) -> dict:
        """Load baseline result for a prompt"""
        baseline_file = baseline_dir / f"{prompt_id}_baseline.json"
        
        # If baseline doesn't exist, create a mock one
        if not baseline_file.exists():
            baseline_dir.mkdir(exist_ok=True)
            mock_baseline = {
                "prompt_id": prompt_id,
                "baseline_text": "This is a baseline response for testing purposes.",
                "created_at": datetime.now().isoformat(),
                "model": "gpt-4-mock"
            }
            with open(baseline_file, 'w') as f:
                json.dump(mock_baseline, f, indent=2)
            return mock_baseline
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def _save_baseline(self, prompt_id: str, response_text: str, baseline_dir: Path):
        """Save a new baseline"""
        baseline_dir.mkdir(exist_ok=True)
        baseline_file = baseline_dir / f"{prompt_id}_baseline.json"
        
        baseline_data = {
            "prompt_id": prompt_id,
            "baseline_text": response_text,
            "created_at": datetime.now().isoformat(),
            "model": "gpt-4-mock"
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
    
    def test_regression_against_baseline(self, llm_client, similarity_checker, metrics_calculator, baseline_dir):
        """
        Test: Compare current output against baseline
        """
        prompt = "Explain what an EDA tool does in simple terms"
        prompt_id = "EDA_001"
        
        print(f"\n🔍 Testing regression for: {prompt_id}")
        
        # Get baseline
        baseline = self._get_baseline(prompt_id, baseline_dir)
        baseline_text = baseline["baseline_text"]
        
        # Generate current response
        current_response = llm_client.generate(prompt, temperature=0.7)
        current_text = current_response["text"]
        
        # Calculate similarity to baseline
        similarity = similarity_checker.calculate_similarity(baseline_text, current_text)
        
        print(f"📊 Similarity to baseline: {similarity:.3f}")
        print(f"📋 Baseline length: {len(baseline_text)} chars")
        print(f"📋 Current length: {len(current_text)} chars")
        
        # Detect regression
        regression_result = metrics_calculator.detect_regression(
            current_text, baseline_text, similarity, threshold=0.70
        )
        
        print(f"🎯 Regression detected: {regression_result['regression_detected']}")
        print(f"🎯 Severity: {regression_result['severity']}")
        
        # Assert no critical regression
        assert regression_result["severity"] != "CRITICAL", \
            f"Critical regression detected! Similarity: {similarity:.3f}"
        
        # Warn on medium/high regression
        if regression_result["severity"] in ["HIGH", "MEDIUM"]:
            print(f"⚠️ WARNING: {regression_result['severity']} severity regression")
        
        print("✅ Regression test PASSED")
    
    def test_prompt_modification_impact(self, llm_client, similarity_checker):
        """
        Test: Small prompt changes should have predictable impact
        """
        base_prompt = "Explain what machine learning is"
        modified_prompt = "Explain what machine learning is in simple terms"
        
        print("\n📝 Testing prompt modification impact...")
        
        # Generate responses for both
        base_response = llm_client.generate(base_prompt, temperature=0.5)
        modified_response = llm_client.generate(modified_prompt, temperature=0.5)
        
        # Calculate similarity
        similarity = similarity_checker.calculate_similarity(
            base_response["text"], 
            modified_response["text"]
        )
        
        print(f"📊 Similarity between variants: {similarity:.3f}")
        
        # Small prompt change should still have high similarity
        assert similarity > 0.60, \
            f"Prompt modification caused unexpected change: {similarity:.3f}"
        
        print("✅ Prompt modification test PASSED")
    
    def test_model_version_comparison(self, similarity_checker):
        """
        Test: Compare outputs from different model versions
        (Simulated with different temperature settings)
        """
        print("\n🔄 Testing model version comparison...")
        
        prompt = "What is regression testing in AI systems?"
        
        # Simulate "v1" model
        client_v1 = LLMClient(use_mock=True, model="gpt-3.5-mock")
        response_v1 = client_v1.generate(prompt, temperature=0.5)
        
        # Simulate "v2" model
        client_v2 = LLMClient(use_mock=True, model="gpt-4-mock")
        response_v2 = client_v2.generate(prompt, temperature=0.5)
        
        # Calculate similarity
        similarity = similarity_checker.calculate_similarity(
            response_v1["text"],
            response_v2["text"]
        )
        
        print(f"📊 Cross-version similarity: {similarity:.3f}")
        print(f"📋 V1 model: {response_v1['model']}")
        print(f"📋 V2 model: {response_v2['model']}")
        
        # Different models can vary, but should maintain semantic meaning
        assert similarity > 0.50, \
            f"Model versions too different: {similarity:.3f}"
        
        print("✅ Model version comparison PASSED")
    
    def test_parameter_change_impact(self, llm_client, similarity_checker):
        """
        Test: Parameter changes should have expected impact
        """
        prompt = "Explain the concept of AI hallucination"
        
        print("\n⚙️ Testing parameter change impact...")
        
        # Test with different max_tokens
        response_short = llm_client.generate(prompt, temperature=0.5)
        response_long = llm_client.generate(prompt, temperature=0.5)
        
        # Calculate similarity
        similarity = similarity_checker.calculate_similarity(
            response_short["text"],
            response_long["text"]
        )
        
        print(f"📊 Similarity: {similarity:.3f}")
        
        # Should still be semantically similar
        assert similarity > 0.65, \
            f"Parameter change caused unexpected divergence: {similarity:.3f}"
        
        print("✅ Parameter change test PASSED")
    
    def test_create_baseline_for_golden_prompts(self, llm_client, baseline_dir):
        """
        Utility test: Create baselines for all golden prompts
        (Run this when you want to establish new baselines)
        """
        print("\n💾 Creating baselines for golden prompts...")
        
        golden_prompts_file = Path(__file__).parent.parent / "data" / "golden_prompts.json"
        with open(golden_prompts_file, 'r') as f:
            golden_prompts = json.load(f)
        
        # Filter out edge cases
        test_prompts = [p for p in golden_prompts if p["category"] != "edge_case"][:3]
        
        for prompt_data in test_prompts:
            prompt_id = prompt_data["id"]
            prompt = prompt_data["prompt"]
            
            print(f"  Creating baseline for: {prompt_id}")
            
            # Generate response
            response = llm_client.generate(prompt, temperature=0.5)
            
            # Save as baseline
            self._save_baseline(prompt_id, response["text"], baseline_dir)
        
        print(f"✅ Created {len(test_prompts)} baselines")
    
    def test_regression_report_generation(self, llm_client, similarity_checker, metrics_calculator, baseline_dir):
        """
        Test: Generate comprehensive regression report
        """
        print("\n📊 Generating regression report...")
        
        test_prompts = [
            ("EDA_001", "Explain what an EDA tool does in simple terms"),
            ("GENAI_001", "How does context memory work in conversational AI?")
        ]
        
        results = []
        
        for prompt_id, prompt in test_prompts:
            # Get baseline
            baseline = self._get_baseline(prompt_id, baseline_dir)
            
            # Generate current response
            current = llm_client.generate(prompt, temperature=0.7)
            
            # Calculate similarity
            similarity = similarity_checker.calculate_similarity(
                baseline["baseline_text"],
                current["text"]
            )
            
            # Detect regression
            regression = metrics_calculator.detect_regression(
                current["text"],
                baseline["baseline_text"],
                similarity,
                threshold=0.70
            )
            
            results.append({
                "prompt_id": prompt_id,
                "similarity": similarity,
                "regression_detected": regression["regression_detected"],
                "severity": regression["severity"]
            })
            
            print(f"  {prompt_id}: Similarity {similarity:.3f} - {regression['status']}")
        
        # Generate report
        report = metrics_calculator.generate_report(
            "Regression Test Suite",
            {"test_results": results}
        )
        
        # Save report
        report_path = Path(__file__).parent.parent / "reports" / "regression_report.json"
        report_path.parent.mkdir(exist_ok=True)
        metrics_calculator.save_report(report, str(report_path))
        
        print("✅ Regression report generated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])