"""
Face Matching Module
Matches detected face embeddings against database of known subjects
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from src.detect_embed import cosine_similarity, euclidean_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceMatcher:
    """
    Matches face embeddings against a database of known subjects
    Supports multiple similarity metrics and configurable thresholds
    """
    
    def __init__(self, similarity_threshold: float = 0.4, metric: str = 'cosine'):
        """
        Initialize the matcher
        
        Args:
            similarity_threshold: Minimum similarity score for a match (0-1)
            metric: Similarity metric ('cosine' or 'euclidean')
        """
        self.similarity_threshold = similarity_threshold
        self.metric = metric
        logger.info(f"Matcher initialized with threshold={similarity_threshold}, metric={metric}")
    
    def find_best_match(self, query_embedding: np.ndarray, 
                        database_embeddings: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching subject from database
        
        Args:
            query_embedding: Embedding vector to match
            database_embeddings: List of dicts with 'embedding', 'subject_id', 'name', 'crime'
            
        Returns:
            Dictionary with match info or None if no match above threshold
        """
        if not database_embeddings:
            logger.warning("Database is empty")
            return None
        
        best_score = -1 if self.metric == 'cosine' else float('inf')
        best_match = None
        
        for db_entry in database_embeddings:
            db_embedding = db_entry['embedding']
            
            # Calculate similarity
            if self.metric == 'cosine':
                score = cosine_similarity(query_embedding, db_embedding)
                is_better = score > best_score
            else:  # euclidean
                score = euclidean_distance(query_embedding, db_embedding)
                is_better = score < best_score
            
            if is_better:
                best_score = score
                best_match = {
                    'subject_id': db_entry['subject_id'],
                    'name': db_entry['name'],
                    'crime': db_entry['crime'],
                    'similarity_score': float(score),
                    'embedding': db_embedding
                }
        
        # Check if best match exceeds threshold
        if self.metric == 'cosine':
            is_match = best_score >= self.similarity_threshold
        else:  # euclidean - lower is better, typical threshold is ~1.0
            is_match = best_score <= (2.0 - self.similarity_threshold)
        
        if is_match:
            logger.info(f"Match found: {best_match['name']} (score={best_score:.3f})")
            return best_match
        else:
            logger.info(f"No match above threshold (best score={best_score:.3f})")
            return None
    
    def find_top_k_matches(self, query_embedding: np.ndarray,
                           database_embeddings: List[Dict],
                           k: int = 5) -> List[Dict]:
        """
        Find top-k most similar subjects
        
        Args:
            query_embedding: Embedding vector to match
            database_embeddings: Database entries
            k: Number of top matches to return
            
        Returns:
            List of top k matches sorted by similarity
        """
        if not database_embeddings:
            return []
        
        matches = []
        
        for db_entry in database_embeddings:
            db_embedding = db_entry['embedding']
            
            if self.metric == 'cosine':
                score = cosine_similarity(query_embedding, db_embedding)
            else:
                score = euclidean_distance(query_embedding, db_embedding)
            
            matches.append({
                'subject_id': db_entry['subject_id'],
                'name': db_entry['name'],
                'crime': db_entry['crime'],
                'similarity_score': float(score)
            })
        
        # Sort by score
        if self.metric == 'cosine':
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        else:
            matches.sort(key=lambda x: x['similarity_score'])
        
        return matches[:k]
    
    def batch_match(self, query_embeddings: List[np.ndarray],
                    database_embeddings: List[Dict]) -> List[Optional[Dict]]:
        """
        Match multiple query embeddings against database
        
        Args:
            query_embeddings: List of embedding vectors
            database_embeddings: Database entries
            
        Returns:
            List of match results (one per query embedding)
        """
        results = []
        for i, query_emb in enumerate(query_embeddings):
            match = self.find_best_match(query_emb, database_embeddings)
            results.append(match)
            logger.info(f"Batch match {i+1}/{len(query_embeddings)}: "
                       f"{'MATCH' if match else 'NO MATCH'}")
        return results
    
    def verify_identity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
        """
        Verify if two embeddings belong to the same person (1:1 verification)
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        if self.metric == 'cosine':
            score = cosine_similarity(embedding1, embedding2)
            is_same = score >= self.similarity_threshold
        else:
            score = euclidean_distance(embedding1, embedding2)
            is_same = score <= (2.0 - self.similarity_threshold)
        
        return is_same, float(score)
    
    def calculate_confidence(self, similarity_score: float) -> str:
        """
        Convert similarity score to confidence level
        
        Args:
            similarity_score: Similarity score from matching
            
        Returns:
            Confidence level string
        """
        if self.metric == 'cosine':
            if similarity_score >= 0.7:
                return "HIGH"
            elif similarity_score >= 0.5:
                return "MEDIUM"
            elif similarity_score >= self.similarity_threshold:
                return "LOW"
            else:
                return "NO MATCH"
        else:  # euclidean
            if similarity_score <= 0.6:
                return "HIGH"
            elif similarity_score <= 1.0:
                return "MEDIUM"
            elif similarity_score <= 1.4:
                return "LOW"
            else:
                return "NO MATCH"
    
    def generate_match_report(self, match_result: Optional[Dict],
                             query_info: Dict = None) -> Dict:
        """
        Generate a comprehensive match report
        
        Args:
            match_result: Result from find_best_match
            query_info: Optional additional info about the query
            
        Returns:
            Formatted match report
        """
        timestamp = datetime.now().isoformat()
        
        if match_result is None:
            return {
                'timestamp': timestamp,
                'status': 'NO_MATCH',
                'message': 'No matching subject found in database',
                'query_info': query_info or {}
            }
        
        confidence = self.calculate_confidence(match_result['similarity_score'])
        
        return {
            'timestamp': timestamp,
            'status': 'MATCH',
            'subject_id': match_result['subject_id'],
            'name': match_result['name'],
            'crime': match_result['crime'],
            'similarity_score': match_result['similarity_score'],
            'confidence_level': confidence,
            'metric': self.metric,
            'threshold': self.similarity_threshold,
            'query_info': query_info or {}
        }


class MultiModalMatcher(FaceMatcher):
    """
    Advanced matcher with multi-modal fusion capabilities
    Can combine multiple embeddings of the same person for better accuracy
    """
    
    def __init__(self, similarity_threshold: float = 0.4, metric: str = 'cosine'):
        super().__init__(similarity_threshold, metric)
    
    def fuse_embeddings(self, embeddings: List[np.ndarray], method: str = 'average') -> np.ndarray:
        """
        Fuse multiple embeddings of the same person
        
        Args:
            embeddings: List of embedding vectors
            method: Fusion method ('average', 'weighted', 'max')
            
        Returns:
            Fused embedding vector
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot fuse empty embedding list")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        embeddings_array = np.array(embeddings)
        
        if method == 'average':
            fused = np.mean(embeddings_array, axis=0)
        elif method == 'max':
            fused = np.max(embeddings_array, axis=0)
        elif method == 'weighted':
            # Weight by embedding confidence (could be extended)
            weights = np.ones(len(embeddings)) / len(embeddings)
            fused = np.average(embeddings_array, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Normalize fused embedding
        fused = fused / np.linalg.norm(fused)
        
        return fused
    
    def match_with_quality_score(self, query_embedding: np.ndarray,
                                 database_embeddings: List[Dict]) -> Optional[Dict]:
        """
        Match with additional quality scoring
        
        Args:
            query_embedding: Query embedding
            database_embeddings: Database entries
            
        Returns:
            Match result with quality score
        """
        match = self.find_best_match(query_embedding, database_embeddings)
        
        if match:
            # Add quality score based on embedding characteristics
            embedding_norm = np.linalg.norm(query_embedding)
            quality_score = min(1.0, embedding_norm)  # Normalized embeddings should be ~1.0
            match['quality_score'] = float(quality_score)
        
        return match


# Example usage
if __name__ == "__main__":
    # Create sample embeddings (in real use, these come from FaceDetectorEmbedder)
    query_emb = np.random.randn(512)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    # Sample database
    database = [
        {
            'subject_id': 'CRIM001',
            'name': 'John Doe',
            'crime': 'Theft',
            'embedding': np.random.randn(512)
        },
        {
            'subject_id': 'CRIM002',
            'name': 'Jane Smith',
            'crime': 'Fraud',
            'embedding': query_emb + np.random.randn(512) * 0.1  # Similar to query
        }
    ]
    
    # Normalize database embeddings
    for entry in database:
        entry['embedding'] = entry['embedding'] / np.linalg.norm(entry['embedding'])
    
    # Initialize matcher
    matcher = FaceMatcher(similarity_threshold=0.4, metric='cosine')
    
    # Find match
    match = matcher.find_best_match(query_emb, database)
    
    # Generate report
    report = matcher.generate_match_report(match)
    print("\nMatch Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")