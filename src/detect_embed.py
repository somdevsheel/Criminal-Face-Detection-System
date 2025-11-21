"""
Face Detection and Embedding Module
Uses MTCNN for face detection and FaceNet (InceptionResnetV1) for feature extraction
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
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
        # Parameters tuned for better detection (very lenient for video)
        self.mtcnn = MTCNN(
            image_size=160,  # Standard size for FaceNet
            margin=20,       # Margin around detected face
            min_face_size=20,  # Minimum face size to detect
            thresholds=[0.3, 0.4, 0.4],  # Very lenient thresholds for P-Net, R-Net, O-Net
            factor=0.709,    # Scale factor for image pyramid
            post_process=True,  # Normalize output
            device=self.device,
            keep_all=True,  # Keep all detected faces
            selection_method='largest'  # Select largest face if multiple
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
            
            # Enhance image quality for better detection
            from PIL import ImageEnhance
            
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Increase brightness slightly if image is too dark
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            # Detect faces - handle both return formats
            try:
                result = self.mtcnn(img, return_prob=True)
                
                # Debug logging
                logger.info(f"MTCNN returned type: {type(result)}")
                if isinstance(result, tuple):
                    logger.info(f"MTCNN returned tuple length: {len(result)}")
                
                # Check what was returned
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    logger.warning(f"No faces detected in {image_path}")
                    return None, None, None
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        faces, probs, boxes = result
                    elif len(result) == 2:
                        faces, probs = result
                        boxes = None
                    else:
                        faces = result[0]
                        probs = None
                        boxes = None
                else:
                    faces = result
                    probs = None
                    boxes = None
                
                # If we got faces but no boxes, we need to get them separately
                if faces is not None and boxes is None:
                    # Detect again to get boxes
                    boxes, probs = self.mtcnn.detect(img)
                    if boxes is None:
                        logger.warning(f"Faces detected but no bounding boxes in {image_path}")
                        return None, None, None
                
                logger.info(f"Detected {len(faces) if faces is not None else 0} face(s) in {image_path}")
                return faces, boxes, probs
                
            except Exception as mtcnn_error:
                logger.error(f"MTCNN error: {str(mtcnn_error)}")
                return None, None, None
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
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
            
            # Enhance image quality for better detection
            from PIL import ImageEnhance
            
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Increase brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            # Detect faces - handle both return formats
            try:
                result = self.mtcnn(img, return_prob=True)
                
                # Check what was returned
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    return []
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        faces, probs, boxes = result
                    elif len(result) == 2:
                        faces, probs = result
                        boxes = None
                    else:
                        faces = result[0]
                        probs = None
                        boxes = None
                else:
                    faces = result
                    probs = None
                    boxes = None
                
                # If we got faces but no boxes, we need to get them separately
                if faces is not None and boxes is None:
                    # Detect again to get boxes
                    boxes, probs = self.mtcnn.detect(img)
                    if boxes is None:
                        return []
                
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
                
            except Exception as mtcnn_error:
                logger.error(f"Error processing video frame: {str(mtcnn_error)}")
                return []
            
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