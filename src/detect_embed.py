"""
Face Detection and Embedding Module
Uses MTCNN for face detection and FaceNet (InceptionResnetV1) for feature extraction
"""

import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetectorEmbedder:
    """
    Unified class for face detection and embedding extraction
    Uses MTCNN for detection and InceptionResnetV1 (FaceNet) for embeddings
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the detector and embedder models
        
        Args:
            device: 'cuda' or 'cpu'. If None, automatically detects GPU availability
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing models on device: {self.device}")
        
        # Initialize MTCNN for face detection
        # Parameters tuned for accuracy and speed
        self.mtcnn = MTCNN(
            image_size=160,  # Standard size for FaceNet
            margin=20,       # Margin around detected face
            min_face_size=40,  # Minimum face size to detect
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds for P-Net, R-Net, O-Net
            factor=0.709,    # Scale factor for image pyramid
            post_process=True,  # Normalize output
            device=self.device,
            keep_all=True  # Keep all detected faces
        )
        
        # Initialize InceptionResnetV1 (FaceNet) for embeddings
        # Pretrained on VGGFace2 dataset (state-of-the-art)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        logger.info("Models initialized successfully")
    
    def detect_faces(self, image_path: str) -> Tuple[Optional[torch.Tensor], Optional[List], Optional[List]]:
        """
        Detect faces in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (face_tensors, bounding_boxes, probabilities)
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Detect faces
            # Returns: face tensors, probabilities, bounding boxes
            faces, probs, boxes = self.mtcnn(img, return_prob=True)
            
            if faces is None:
                logger.warning(f"No faces detected in {image_path}")
                return None, None, None
            
            logger.info(f"Detected {len(faces)} face(s) in {image_path}")
            return faces, boxes, probs
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return None, None, None
    
    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embedding from a face tensor
        
        Args:
            face_tensor: Preprocessed face tensor from MTCNN
            
        Returns:
            512-dimensional embedding vector (normalized)
        """
        try:
            # Ensure correct dimensions
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # Move to device
            face_tensor = face_tensor.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
            
            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def process_image(self, image_path: str) -> List[Dict]:
        """
        Complete pipeline: detect faces and extract embeddings
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of dictionaries containing face data:
            [{'embedding': np.ndarray, 'box': list, 'confidence': float}, ...]
        """
        results = []
        
        # Detect faces
        faces, boxes, probs = self.detect_faces(image_path)
        
        if faces is None:
            return results
        
        # Process each detected face
        for i, (face, box, prob) in enumerate(zip(faces, boxes, probs)):
            # Extract embedding
            embedding = self.get_embedding(face)
            
            if embedding is not None:
                results.append({
                    'embedding': embedding,
                    'box': box.tolist() if isinstance(box, torch.Tensor) else box,
                    'confidence': float(prob)
                })
                logger.info(f"Face {i+1}: Confidence={prob:.3f}, Embedding shape={embedding.shape}")
        
        return results
    
    def process_video_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single video frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of face data dictionaries
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Detect faces
            faces, probs, boxes = self.mtcnn(img, return_prob=True)
            
            if faces is None:
                return []
            
            results = []
            for face, box, prob in zip(faces, boxes, probs):
                embedding = self.get_embedding(face)
                if embedding is not None:
                    results.append({
                        'embedding': embedding,
                        'box': box.tolist() if isinstance(box, torch.Tensor) else box,
                        'confidence': float(prob)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            return []
    
    def draw_boxes(self, image_path: str, boxes: List, labels: List[str], 
                    output_path: str = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image_path: Input image path
            boxes: List of bounding boxes [x1, y1, x2, y2]
            labels: List of labels for each box
            output_path: Optional path to save annotated image
            
        Returns:
            Annotated image as numpy array
        """
        try:
            img = cv2.imread(image_path)
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            if output_path:
                cv2.imwrite(output_path, img)
                logger.info(f"Annotated image saved to {output_path}")
            
            return img
            
        except Exception as e:
            logger.error(f"Error drawing boxes: {str(e)}")
            return None


# Utility functions
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between -1 and 1 (higher is more similar)
    """
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Distance (lower is more similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FaceDetectorEmbedder()
    
    # Process an image
    results = detector.process_image("test_image.jpg")
    
    for i, result in enumerate(results):
        print(f"Face {i+1}:")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Bounding Box: {result['box']}")
        print(f"  Embedding Shape: {result['embedding'].shape}")