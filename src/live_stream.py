"""
Live CCTV Stream Monitoring Module
Real-time face detection from RTSP/HTTP streams with WebSocket support
"""

import cv2
import numpy as np
from threading import Thread, Event
import logging
from datetime import datetime
import time
from queue import Queue
import base64

from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveStreamMonitor:
    """
    Monitor live CCTV streams in real-time with frame emission
    Supports RTSP, HTTP, and local camera streams
    """
    
    def __init__(self, db_path: str = "criminal_detection.db",
                 similarity_threshold: float = 0.4):
        """Initialize live stream monitor"""
        self.detector = FaceDetectorEmbedder()
        self.matcher = FaceMatcher(similarity_threshold=similarity_threshold)
        self.db = Database(db_path)
        
        self.is_monitoring = False
        self.stop_event = Event()
        self.detection_queue = Queue()
        self.alert_callbacks = []
        
        logger.info("Live stream monitor initialized")
    
    def add_alert_callback(self, callback):
        """Add callback function for alerts and frame updates"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, stream_url: str, process_fps: float = 1):
        """
        Start monitoring a live stream
        
        Args:
            stream_url: RTSP URL (e.g., rtsp://camera-ip:554/stream)
                       or camera index (e.g., 0 for webcam)
            process_fps: Process N frames per second
        """
        if self.is_monitoring:
            logger.warning("Already monitoring a stream")
            return False
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start monitoring thread
        monitor_thread = Thread(
            target=self._monitor_stream,
            args=(stream_url, process_fps),
            daemon=True
        )
        monitor_thread.start()
        
        logger.info(f"Started monitoring stream: {stream_url}")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.stop_event.set()
        self.is_monitoring = False
        logger.info("Stopped monitoring")
    
    def _emit_frame(self, frame, face_count, detections=None, matches=None):
        """Emit frame to frontend via WebSocket with bounding boxes"""
        try:
            # Resize frame for faster transmission
            height, width = frame.shape[:2]
            max_width = 640
            if width > max_width:
                ratio = max_width / width
                frame_resized = cv2.resize(frame, (max_width, int(height * ratio)))
                scale_factor = ratio
            else:
                frame_resized = frame
                scale_factor = 1.0
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare bounding box data
            boxes = []
            if detections and matches:
                logger.info(f"📦 Preparing {len(detections)} bounding boxes")
                for i, (detection, match) in enumerate(zip(detections, matches)):
                    box_coords = detection['box']
                    # Scale coordinates if frame was resized
                    # scaled_coords = [
                    #     int(box_coords[0] * scale_factor),
                    #     int(box_coords[1] * scale_factor),
                    #     int(box_coords[2] * scale_factor),
                    #     int(box_coords[3] * scale_factor)
                    # ]
                    frame_height, frame_width = frame.shape[:2]
                    scaled_coords = [
                        float(box_coords[0] / frame_width),
                        float(box_coords[1] / frame_height),
                        float(box_coords[2] / frame_width),
                        float(box_coords[3] / frame_height)
                    ]
                    
                    box_data = {
                        'coordinates': scaled_coords,
                        'matched': match is not None,
                        'name': match['name'] if match else 'Unknown',
                        'similarity_score': float(match['similarity_score']) if match else 0.0
                    }
                    boxes.append(box_data)
                    
                    if match:
                        logger.info(f"  Box {i}: 🔴 MATCH - {match['name']} ({match['similarity_score']:.3f})")
                    else:
                        logger.info(f"  Box {i}: 🟢 Face detected (no match)")
            
            # Emit via callback
            for callback in self.alert_callbacks:
                if hasattr(callback, 'emit_frame'):
                    callback.emit_frame({
                        'frame': frame_base64,
                        'faces': face_count,
                        'boxes': boxes
                    })
                    
            if boxes:
                logger.info(f"✅ Emitted frame with {len(boxes)} boxes")
                    
        except Exception as e:
            logger.error(f"Error emitting frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _emit_stats(self, frames_processed, total_detections):
        """Emit statistics update"""
        try:
            for callback in self.alert_callbacks:
                if hasattr(callback, 'emit_stats'):
                    callback.emit_stats({
                        'frames_processed': frames_processed,
                        'total_detections': total_detections
                    })
        except Exception as e:
            logger.error(f"Error emitting stats: {e}")
    
    def _monitor_stream(self, stream_url, process_fps):
        """Internal monitoring loop with frame emission"""
        try:
            # Convert '0' string to integer for webcam
            if stream_url == '0':
                stream_url = 0
            
            # Open stream
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                logger.error(f"Cannot open stream: {stream_url}")
                self.is_monitoring = False
                return
            
            # Get stream info
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = int(fps / process_fps) if process_fps > 0 else 1
            
            logger.info(f"Stream FPS: {fps}, processing every {frame_interval} frames")
            
            # Get database embeddings
            database_embeddings = self.db.get_all_embeddings()
            
            frame_count = 0
            last_detection_time = {}  # Track last detection per subject
            total_detections = 0
            frames_processed_count = 0
            
            # For frame emission throttling
            last_frame_emit_time = time.time()
            frame_emit_interval = 0.1  # Emit frames every 100ms (10 FPS max)
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    time.sleep(5)
                    cap = cv2.VideoCapture(stream_url)
                    continue
                
                # Emit frame to frontend (throttled)
                current_time = time.time()
                if current_time - last_frame_emit_time >= frame_emit_interval:
                    self._emit_frame(frame, 0, None, None)
                    last_frame_emit_time = current_time
                
                # Process every Nth frame
                if frame_count % frame_interval == 0:
                    frames_processed_count += 1
                    detections = self.detector.process_video_frame(frame)
                    total_detections += len(detections)
                    
                    # Emit stats update
                    if frames_processed_count % 5 == 0:  # Every 5 processed frames
                        self._emit_stats(frames_processed_count, total_detections)
                    
                    # Match detections and emit frame with bounding boxes
                    if detections:
                        matches = []
                        for detection in detections:
                            match = self.matcher.find_best_match(
                                detection['embedding'],
                                database_embeddings
                            )
                            matches.append(match)
                        
                        # Emit frame with bounding boxes
                        self._emit_frame(frame, len(detections), detections, matches)
                        
                        # Process matches for alerts
                        for detection, match in zip(detections, matches):
                            if match:
                                subject_id = match['subject_id']
                                current_time = time.time()
                                
                                # Avoid duplicate alerts (30 second cooldown)
                                if subject_id not in last_detection_time or \
                                   current_time - last_detection_time[subject_id] > 30:
                                    
                                    last_detection_time[subject_id] = current_time
                                    
                                    # Create alert
                                    alert_data = {
                                        'timestamp': datetime.now().isoformat(),
                                        'subject_id': match['subject_id'],
                                        'name': match['name'],
                                        'crime': match['crime'],
                                        'similarity_score': match['similarity_score'],
                                        'confidence': detection['confidence'],
                                        'stream_url': str(stream_url),
                                        'frame': frame.copy(),
                                        'box': detection['box']
                                    }
                                    
                                    # Log to database
                                    self.db.log_event(
                                        'LIVE_STREAM_DETECTION',
                                        subject_id,
                                        match['similarity_score'],
                                        {
                                            'stream': str(stream_url),
                                            'confidence': detection['confidence']
                                        }
                                    )
                                    
                                    # Trigger alerts
                                    self._trigger_alerts(alert_data)
                                    
                                    logger.warning(f"🚨 ALERT: {match['name']} detected!")
                
                frame_count += 1
            
            cap.release()
            logger.info("Monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_monitoring = False
    
    def _trigger_alerts(self, alert_data):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                # Call the main callback (criminal detection)
                if callable(callback):
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


# Example usage
if __name__ == "__main__":
    monitor = LiveStreamMonitor()
    
    # Example: Monitor webcam
    print("Starting webcam monitoring...")
    print("Press Ctrl+C to stop")
    
    monitor.start_monitoring(0, process_fps=1)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop_monitoring()