"""
Metrics Calculator for LLM Reliability Testing
Author: Mira Mamdoh
"""

import numpy as np
from typing import List, Dict, Any
import json
from datetime import datetime


class ReliabilityMetrics:
    """Calculate reliability and consistency metrics for LLM outputs"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_reliability_score(self, similarities: List[float], threshold: float = 0.75) -> Dict[str, Any]:
        """
        Calculate overall reliability score
        Score = percentage of outputs above similarity threshold
        """
        if not similarities:
            return {"score": 0, "message": "No data"}
        
        above_threshold = sum(1 for s in similarities if s >= threshold)
        score = (above_threshold / len(similarities)) * 100
        
        return {
            "reliability_score": round(score, 2),
            "threshold": threshold,
            "above_threshold": above_threshold,
            "total_comparisons": len(similarities),
            "status": "PASS" if score >= 80 else "FAIL"
        }
    
    def calculate_consistency_metrics(self, similarities: List[float]) -> Dict[str, Any]:
        """
        Calculate statistical metrics for consistency
        """
        if not similarities:
            return {}
        
        return {
            "mean_similarity": round(np.mean(similarities), 4),
            "median_similarity": round(np.median(similarities), 4),
            "std_deviation": round(np.std(similarities), 4),
            "min_similarity": round(min(similarities), 4),
            "max_similarity": round(max(similarities), 4),
            "variance": round(np.var(similarities), 4),
            "range": round(max(similarities) - min(similarities), 4)
        }
    
    def calculate_stability_score(self, similarities: List[float]) -> Dict[str, Any]:
        """
        Stability = how consistent are the outputs
        Lower variance = higher stability
        """
        if not similarities or len(similarities) < 2:
            return {"stability_score": 100, "message": "Insufficient data"}
        
        variance = np.var(similarities)
        std_dev = np.std(similarities)
        
        # Stability score: 100 - (std_dev * 100)
        # Lower std_dev = higher stability
        stability_score = max(0, 100 - (std_dev * 100))
        
        return {
            "stability_score": round(stability_score, 2),
            "std_deviation": round(std_dev, 4),
            "variance": round(variance, 4),
            "interpretation": self._interpret_stability(stability_score)
        }
    
    def _interpret_stability(self, score: float) -> str:
        """Interpret stability score"""
        if score >= 90:
            return "Excellent - Very consistent outputs"
        elif score >= 75:
            return "Good - Acceptable consistency"
        elif score >= 60:
            return "Fair - Some variation in outputs"
        else:
            return "Poor - High variation in outputs"
    
    def detect_regression(self, current_output: str, baseline_output: str, similarity_score: float, threshold: float = 0.75) -> Dict[str, Any]:
        """
        Detect if there's a regression compared to baseline
        """
        is_regression = similarity_score < threshold
        
        return {
            "regression_detected": is_regression,
            "similarity_to_baseline": round(similarity_score, 4),
            "threshold": threshold,
            "severity": self._calculate_severity(similarity_score, threshold),
            "status": "REGRESSION" if is_regression else "PASS",
            "current_length": len(current_output),
            "baseline_length": len(baseline_output),
            "length_diff": abs(len(current_output) - len(baseline_output))
        }
    
    def _calculate_severity(self, similarity: float, threshold: float) -> str:
        """Calculate regression severity"""
        if similarity >= threshold:
            return "NONE"
        
        diff = threshold - similarity
        
        if diff >= 0.3:
            return "CRITICAL"
        elif diff >= 0.15:
            return "HIGH"
        elif diff >= 0.05:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_performance_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics (latency, tokens, etc.)
        """
        if not responses:
            return {}
        
        latencies = [r.get("latency_ms", 0) for r in responses]
        tokens = [r.get("tokens_used", 0) for r in responses]
        
        return {
            "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_tokens": round(np.mean(tokens), 2) if tokens else 0,
            "total_tokens": sum(tokens),
            "response_count": len(responses)
        }
    
    def generate_report(self, test_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive test report
        """
        report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        return report
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary"""
        summary = {
            "overall_status": "PASS",
            "key_findings": []
        }
        
        # Check reliability
        if "reliability_score" in results:
            score = results["reliability_score"].get("reliability_score", 0)
            if score < 80:
                summary["overall_status"] = "FAIL"
                summary["key_findings"].append(f"Low reliability score: {score}%")
            else:
                summary["key_findings"].append(f"Good reliability: {score}%")
        
        # Check regression
        if "regression" in results and results["regression"].get("regression_detected"):
            summary["overall_status"] = "REGRESSION"
            summary["key_findings"].append("Regression detected")
        
        return summary
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(report, indent=2, fp=f)
        print(f"✅ Report saved to: {filepath}")


def calculate_quick_metrics(similarities: List[float]) -> str:
    """Quick metrics summary for console output"""
    if not similarities:
        return "No data"
    
    avg = np.mean(similarities)
    min_sim = min(similarities)
    max_sim = max(similarities)
    
    return f"Avg: {avg:.3f} | Min: {min_sim:.3f} | Max: {max_sim:.3f}"