"""
Semantic Similarity Utilities
Uses sentence transformers for embedding-based similarity
Author: Mira Mamdoh
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Tuple


class SemanticSimilarity:
    """Calculate semantic similarity between texts using embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with sentence transformer model
        all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
        """
        print(f"Loading similarity model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded successfully")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        Returns: float between 0 and 1 (1 = identical meaning)
        """
        if not text1 or not text2:
            return 0.0
        
        # Generate embeddings
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(emb1, emb2).item()
        
        return similarity
    
    def calculate_pairwise_similarity(self, texts: List[str]) -> List[float]:
        """
        Calculate all pairwise similarities in a list of texts
        Returns: List of similarity scores
        """
        if len(texts) < 2:
            return [1.0]
        
        similarities = []
        
        # Generate all embeddings at once (more efficient)
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Calculate pairwise similarities
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                similarities.append(sim)
        
        return similarities
    
    def calculate_average_similarity(self, texts: List[str]) -> float:
        """
        Calculate average similarity across all pairs
        Useful for measuring consistency
        """
        similarities = self.calculate_pairwise_similarity(texts)
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_variance(self, texts: List[str]) -> float:
        """
        Calculate variance in similarity scores
        Lower variance = more consistent outputs
        """
        similarities = self.calculate_pairwise_similarity(texts)
        return np.var(similarities) if len(similarities) > 1 else 0.0
    
    def find_outliers(self, texts: List[str], threshold: float = 0.7) -> List[Tuple[int, str, float]]:
        """
        Find texts that are significantly different from others
        Returns: List of (index, text, avg_similarity)
        """
        if len(texts) < 3:
            return []
        
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        outliers = []
        
        for i, text in enumerate(texts):
            # Calculate average similarity to all other texts
            similarities = []
            for j in range(len(texts)):
                if i != j:
                    sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            
            # If below threshold, it's an outlier
            if avg_sim < threshold:
                outliers.append((i, text, avg_sim))
        
        return outliers
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Calculate full similarity matrix
        Useful for detailed analysis
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        return similarity_matrix


def simple_text_similarity(text1: str, text2: str) -> float:
    """
    Simple word-overlap based similarity (fallback)
    Returns: Jaccard similarity coefficient
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)