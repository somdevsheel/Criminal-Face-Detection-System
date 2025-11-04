"""
Video Processing Module
Real-time face detection and recognition from video streams (webcam, CCTV, video files)
"""

import cv2
import numpy as np
from datetime import datetime
import logging
import time
from typing import List, Dict, Optional, Callable
import threading
import queue

from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process video streams for face detection and recognition
    Supports webcam, CCTV streams, and video files
    """
    
    def __init__(self, db_path: str = "criminal_detection.db", 
                 similarity_threshold: float = 0.4):
        """
        Initialize video processor
        
        Args:
            db_path: Path to database
            similarity_threshold: Matching threshold
        """
        self.detector = FaceDetectorEmbedder()
        self.matcher = FaceMatcher(similarity_threshold=similarity_threshold)
        self.db = Database(db_path)
        
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Performance settings
        self.process_every_n_frames = 5  # Process every 5th frame
        self.frame_count = 0
        
        # Detection cache to avoid repeated processing
        self.detection_cache = {}
        self.cache_timeout = 3.0  # seconds
        
        logger.info("Video processor initialized")
    
    def process_webcam(self, camera_id: int = 0, 
                      display: bool = True,
                      save_matches: bool = True,
                      output_dir: str = "data/processed/video"):
        """
        Process live webcam feed
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            display: Show video window
            save_matches: Save frames with matches
            output_dir: Directory to save matched frames
        """
        logger.info(f"Starting webcam processing (camera {camera_id})...")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                self.frame_count += 1
                
                # Only process every N frames for performance
                if self.frame_count % self.process_every_n_frames == 0:
                    processed_frame, matches = self._process_frame(frame)
                    
                    # Handle matches
                    if matches:
                        for match in matches:
                            logger.info(f"MATCH: {match['name']} ({match['similarity_score']:.2f})")
                            
                            # Log to database
                            self.db.log_event(
                                'MATCH',
                                match['subject_id'],
                                match['similarity_score'],
                                {'source': 'webcam', 'camera_id': camera_id},
                                location='Webcam'
                            )
                            
                            # Save frame if requested
                            if save_matches:
                                self._save_match_frame(processed_frame, match, output_dir)
                    
                    display_frame = processed_frame
                else:
                    display_frame = frame
                
                # Display
                if display:
                    # Add info overlay
                    self._add_overlay(display_frame, self.frame_count)
                    
                    cv2.imshow('Criminal Face Detection - Live Feed', display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(f"{output_dir}/{filename}", display_frame)
                        logger.info(f"Saved snapshot: {filename}")
                
        finally:
            self.is_running = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info("Webcam processing stopped")
    
    def process_video_file(self, video_path: str,
                          display: bool = True,
                          save_output: bool = True,
                          output_path: str = None):
        """
        Process video file (e.g., CCTV footage)
        
        Args:
            video_path: Path to video file
            display: Show processing window
            save_output: Save annotated video
            output_path: Output video path
        """
        logger.info(f"Processing video file: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video writer
        writer = None
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"data/processed/video/output_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        self.is_running = True
        frame_num = 0
        matches_found = []
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_num += 1
                
                # Process every Nth frame
                if frame_num % self.process_every_n_frames == 0:
                    processed_frame, matches = self._process_frame(frame)
                    
                    # Record matches
                    if matches:
                        for match in matches:
                            match['frame_number'] = frame_num
                            match['timestamp'] = frame_num / fps
                            matches_found.append(match)
                            
                            logger.info(f"Frame {frame_num}: MATCH - {match['name']} "
                                      f"({match['similarity_score']:.2f})")
                            
                            # Log to database
                            self.db.log_event(
                                'MATCH',
                                match['subject_id'],
                                match['similarity_score'],
                                {
                                    'source': 'video_file',
                                    'video_path': video_path,
                                    'frame_number': frame_num,
                                    'timestamp': match['timestamp']
                                }
                            )
                    
                    display_frame = processed_frame
                else:
                    display_frame = frame
                
                # Add progress overlay
                progress = (frame_num / total_frames) * 100
                self._add_overlay(display_frame, frame_num, 
                                f"Progress: {progress:.1f}%")
                
                # Save to output video
                if writer:
                    writer.write(display_frame)
                
                # Display
                if display:
                    cv2.imshow('Video Processing', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                
                # Progress update
                if frame_num % 100 == 0:
                    logger.info(f"Processed {frame_num}/{total_frames} frames "
                              f"({progress:.1f}%)")
        
        finally:
            self.is_running = False
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Summary
            logger.info(f"\nProcessing complete!")
            logger.info(f"Total frames processed: {frame_num}")
            logger.info(f"Matches found: {len(matches_found)}")
            
            # Generate report
            self._generate_video_report(matches_found, video_path, output_path)
    
    def process_rtsp_stream(self, rtsp_url: str,
                           display: bool = True,
                           save_matches: bool = True):
        """
        Process RTSP stream (IP camera/CCTV)
        
        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://username:password@ip:port/stream)
            display: Show video window
            save_matches: Save frames with matches
        """
        logger.info(f"Connecting to RTSP stream...")
        
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            logger.error("Failed to connect to RTSP stream")
            return
        
        logger.info("Connected to RTSP stream")
        
        # Use same logic as webcam
        self.process_webcam(camera_id=None, display=display, save_matches=save_matches)
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame for face detection and recognition
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, matches_list)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.process_video_frame(frame)
        
        matches = []
        annotated_frame = frame.copy()
        
        if results:
            # Get database embeddings
            database_embeddings = self.db.get_all_embeddings()
            
            if database_embeddings:
                for i, result in enumerate(results):
                    embedding = result['embedding']
                    box = result['box']
                    confidence = result['confidence']
                    
                    # Match against database
                    match = self.matcher.find_best_match(embedding, database_embeddings)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    
                    if match:
                        # Match found
                        label = f"{match['name']} ({match['similarity_score']:.2f})"
                        color = (0, 255, 0)  # Green for match
                        matches.append(match)
                        
                        # Draw additional info
                        crime_label = f"Crime: {match['crime']}"
                        cv2.putText(annotated_frame, crime_label, 
                                  (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
                    else:
                        # No match
                        label = "Unknown"
                        color = (0, 165, 255)  # Orange
                    
                    # Draw rectangle and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # No database, just draw detected faces
                for result in results:
                    box = result['box']
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, "Face Detected", (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return annotated_frame, matches
    
    def _add_overlay(self, frame: np.ndarray, frame_count: int, 
                     extra_info: str = ""):
        """Add information overlay to frame"""
        # Background for overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {frame_count}", (300, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Extra info
        if extra_info:
            cv2.putText(frame, extra_info, (500, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 255, 0), -1)
    
    def _save_match_frame(self, frame: np.ndarray, match: Dict, output_dir: str):
        """Save frame with detected match"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"match_{match['subject_id']}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved match frame: {filename}")
    
    def _generate_video_report(self, matches: List[Dict], 
                              video_path: str, output_path: str):
        """Generate processing report"""
        report_path = output_path.replace('.mp4', '_report.txt') if output_path else 'video_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VIDEO PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input Video: {video_path}\n")
            f.write(f"Output Video: {output_path}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Matches: {len(matches)}\n\n")
            
            if matches:
                f.write("MATCHES FOUND:\n")
                f.write("-" * 60 + "\n")
                
                # Group by subject
                from collections import defaultdict
                by_subject = defaultdict(list)
                for match in matches:
                    by_subject[match['subject_id']].append(match)
                
                for subject_id, subject_matches in by_subject.items():
                    f.write(f"\nSubject ID: {subject_id}\n")
                    f.write(f"Name: {subject_matches[0]['name']}\n")
                    f.write(f"Crime: {subject_matches[0]['crime']}\n")
                    f.write(f"Appearances: {len(subject_matches)}\n")
                    f.write(f"Timestamps: ")
                    for m in subject_matches[:5]:  # First 5
                        f.write(f"{m['timestamp']:.2f}s, ")
                    if len(subject_matches) > 5:
                        f.write("...")
                    f.write("\n")
        
        logger.info(f"Report saved: {report_path}")
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        logger.info("Stopping video processor...")


# Standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Face Detection")
    parser.add_argument('--mode', choices=['webcam', 'video', 'rtsp'], 
                       required=True, help='Processing mode')
    parser.add_argument('--source', help='Video file path or RTSP URL')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for webcam')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--threshold', type=float, default=0.4, 
                       help='Similarity threshold')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(similarity_threshold=args.threshold)
    
    try:
        if args.mode == 'webcam':
            processor.process_webcam(camera_id=args.camera, 
                                    display=not args.no_display)
        
        elif args.mode == 'video':
            if not args.source:
                print("Error: --source required for video mode")
            else:
                processor.process_video_file(args.source, 
                                            display=not args.no_display)
        
        elif args.mode == 'rtsp':
            if not args.source:
                print("Error: --source (RTSP URL) required for rtsp mode")
            else:
                processor.process_rtsp_stream(args.source, 
                                             display=not args.no_display)
    
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()