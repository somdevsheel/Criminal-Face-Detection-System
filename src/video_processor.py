"""
Video Processing Module
Process CCTV footage frame by frame for face detection and recognition
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import os
from tqdm import tqdm

from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process video files (CCTV footage) frame by frame for face detection
    """
    
    def __init__(self, db_path: str = "criminal_detection.db", 
                 similarity_threshold: float = 0.4):
        """
        Initialize video processor
        
        Args:
            db_path: Path to database
            similarity_threshold: Threshold for face matching
        """
        self.detector = FaceDetectorEmbedder()
        self.matcher = FaceMatcher(similarity_threshold=similarity_threshold)
        self.db = Database(db_path)
        logger.info("Video processor initialized")
    
    def extract_frames(self, video_path: str, fps: int = 1, 
                      max_frames: int = None) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract (1 = 1 frame per second)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        logger.info(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Video info: {video_fps} FPS, {total_frames} frames, {duration:.2f} seconds")
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at interval
            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def process_video(self, video_path: str, 
                     output_dir: str = "data/processed/video_results",
                     fps: int = 1,
                     save_detections: bool = True,
                     create_report: bool = True) -> Dict:
        """
        Process entire video for face detection and recognition
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            fps: Frames per second to process
            save_detections: Save frames with detected faces
            create_report: Create detailed report
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(video_path, fps=fps)
        
        if not frames:
            return {'error': 'No frames extracted'}
        
        # Get database embeddings
        database_embeddings = self.db.get_all_embeddings()
        
        # Process results
        results = {
            'video_path': video_path,
            'total_frames': len(frames),
            'frames_processed': 0,
            'total_faces_detected': 0,
            'matches_found': [],
            'detection_timeline': [],
            'unique_subjects': set()
        }
        
        # Process each frame
        logger.info(f"Processing {len(frames)} frames...")
        
        for frame_num, frame in tqdm(frames, desc="Processing frames"):
            # Save frame temporarily
            temp_frame_path = os.path.join(output_dir, f"temp_frame_{frame_num}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect faces in frame
            detections = self.detector.process_video_frame(frame)
            
            results['frames_processed'] += 1
            results['total_faces_detected'] += len(detections)
            
            # Process each detected face
            for i, detection in enumerate(detections):
                embedding = detection['embedding']
                box = detection['box']
                confidence = detection['confidence']
                
                # Match against database
                match = self.matcher.find_best_match(embedding, database_embeddings)
                
                if match:
                    match_info = {
                        'frame_number': frame_num,
                        'timestamp': frame_num / fps,  # Approximate timestamp
                        'subject_id': match['subject_id'],
                        'name': match['name'],
                        'crime': match['crime'],
                        'similarity_score': match['similarity_score'],
                        'detection_confidence': confidence,
                        'bounding_box': box
                    }
                    
                    results['matches_found'].append(match_info)
                    results['unique_subjects'].add(match['subject_id'])
                    
                    # Log detection
                    self.db.log_event(
                        'VIDEO_MATCH',
                        match['subject_id'],
                        match['similarity_score'],
                        {
                            'video': video_path,
                            'frame': frame_num,
                            'timestamp': frame_num / fps
                        }
                    )
                    
                    # Save frame with detection
                    if save_detections:
                        self._save_detection_frame(
                            frame, box, match, 
                            output_dir, frame_num, i
                        )
                
                results['detection_timeline'].append({
                    'frame': frame_num,
                    'timestamp': frame_num / fps,
                    'faces': len(detections),
                    'matched': match is not None,
                    'subject_id': match['subject_id'] if match else None
                })
            
            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        # Convert set to list for JSON serialization
        results['unique_subjects'] = list(results['unique_subjects'])
        
        # Create report
        if create_report:
            self._create_report(results, output_dir)
        
        logger.info(f"Video processing complete: {len(results['matches_found'])} matches found")
        return results
    
    def _save_detection_frame(self, frame: np.ndarray, box: List, 
                             match: Dict, output_dir: str, 
                             frame_num: int, face_num: int):
        """Save frame with bounding box and label"""
        frame_copy = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label
        label = f"{match['name']} ({match['similarity_score']:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame_copy, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, label, (x1, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # Save frame
        filename = f"detection_frame{frame_num}_face{face_num}_{match['subject_id']}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame_copy)
    
    def _create_report(self, results: Dict, output_dir: str):
        """Create detailed text report"""
        report_path = os.path.join(output_dir, "detection_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("VIDEO PROCESSING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Video: {results['video_path']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Processed: {results['frames_processed']}\n")
            f.write(f"Total Faces Detected: {results['total_faces_detected']}\n")
            f.write(f"Total Matches Found: {len(results['matches_found'])}\n")
            f.write(f"Unique Subjects Detected: {len(results['unique_subjects'])}\n\n")
            
            if results['unique_subjects']:
                f.write("DETECTED SUBJECTS:\n")
                f.write("-" * 70 + "\n")
                for subject_id in results['unique_subjects']:
                    f.write(f"  • {subject_id}\n")
                f.write("\n")
            
            if results['matches_found']:
                f.write("DETAILED MATCHES:\n")
                f.write("-" * 70 + "\n")
                
                for match in results['matches_found']:
                    f.write(f"\nFrame: {match['frame_number']}\n")
                    f.write(f"Timestamp: {match['timestamp']:.2f}s\n")
                    f.write(f"Subject ID: {match['subject_id']}\n")
                    f.write(f"Name: {match['name']}\n")
                    f.write(f"Crime: {match['crime']}\n")
                    f.write(f"Similarity Score: {match['similarity_score']:.4f}\n")
                    f.write(f"Detection Confidence: {match['detection_confidence']:.4f}\n")
                    f.write("-" * 70 + "\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def process_live_stream(self, stream_url: str, duration: int = 60):
        """
        Process live CCTV stream
        
        Args:
            stream_url: RTSP or HTTP stream URL
            duration: Duration to process in seconds
        """
        logger.info(f"Processing live stream: {stream_url}")
        
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"Cannot open stream: {stream_url}")
            return
        
        database_embeddings = self.db.get_all_embeddings()
        start_time = datetime.now()
        frame_count = 0
        
        while (datetime.now() - start_time).seconds < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame to reduce load
            if frame_count % 30 == 0:  # Process 1 frame per second at 30fps
                detections = self.detector.process_video_frame(frame)
                
                for detection in detections:
                    match = self.matcher.find_best_match(
                        detection['embedding'], 
                        database_embeddings
                    )
                    
                    if match:
                        logger.info(f"MATCH: {match['name']} ({match['similarity_score']:.2f})")
                        self.db.log_event(
                            'LIVE_STREAM_MATCH',
                            match['subject_id'],
                            match['similarity_score'],
                            {'stream': stream_url}
                        )
            
            frame_count += 1
        
        cap.release()
        logger.info("Live stream processing complete")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process CCTV video footage")
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to process')
    parser.add_argument('--output', default='data/processed/video_results', 
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.4, 
                       help='Similarity threshold')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(similarity_threshold=args.threshold)
    results = processor.process_video(
        args.video,
        output_dir=args.output,
        fps=args.fps
    )
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Total faces detected: {results['total_faces_detected']}")
    print(f"Matches found: {len(results['matches_found'])}")
    print(f"Unique subjects: {len(results['unique_subjects'])}")