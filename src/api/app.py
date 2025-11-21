
"""
Flask API Application with Live Stream Support
Complete backend for Criminal Face Detection System
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import base64

from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database
from src.live_stream import LiveStreamMonitor

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
project_root = Path(__file__).parent.parent.parent

app = Flask(
    __name__,
    template_folder=str(project_root / "web" / "templates"),
    static_folder=str(project_root / "web" / "static")
)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB for videos
app.config["UPLOAD_FOLDER"] = "data/raw"
app.config["PROCESSED_FOLDER"] = "data/processed"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp", "mp4", "mov", "avi", "mkv"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize components
detector = FaceDetectorEmbedder()
matcher = FaceMatcher(similarity_threshold=0.4, metric="cosine")
db = Database("criminal_detection.db")

# Global monitor instance
live_monitor = None

logger.info("Flask backend initialized successfully")


# NumPy JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def save_uploaded_file(file, subfolder=None):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = app.config["UPLOAD_FOLDER"]
        if subfolder:
            folder = os.path.join(folder, subfolder)
            os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{ts}_{filename}")
        file.save(path)
        return path
    return None


# WebSocket Callback Class
class WebSocketCallback:
    """Callback handler for live stream events"""
    
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
    
    def __call__(self, alert_data):
        """Handle criminal detection alerts"""
        self.socketio.emit('criminal_detected', {
            'timestamp': alert_data['timestamp'],
            'subject_id': alert_data['subject_id'],
            'name': alert_data['name'],
            'crime': alert_data['crime'],
            'similarity_score': float(alert_data['similarity_score']),
            'confidence': float(alert_data['confidence'])
        })
        logger.info(f"🚨 Alert emitted: {alert_data['name']}")
    
    def emit_frame(self, data):
        """Emit video frame to frontend"""
        self.socketio.emit('frame_update', data)
    
    def emit_stats(self, data):
        """Emit statistics update"""
        self.socketio.emit('stats_update', data)


# HTML Routes
@app.route("/")
def index():
    return render_template("index.html")


# API Routes
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/add_subject", methods=["POST"])
def add_subject():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files["image"]
        subject_id = request.form.get("subject_id", "").strip()
        name = request.form.get("name", "").strip()
        crime = request.form.get("crime", "").strip()

        if not all([subject_id, name, crime]):
            return jsonify({"success": False, "error": "subject_id, name and crime are required"}), 400

        filepath = save_uploaded_file(file, subfolder="subjects")
        if not filepath:
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        logger.info(f"Processing new subject {subject_id} - {name}")
        results = detector.process_image(filepath)
        
        if not results:
            os.remove(filepath)
            return jsonify({"success": False, "error": "No face detected in image"}), 400

        embedding = results[0]["embedding"]
        confidence = float(results[0].get("confidence", 0.0))
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        success = db.add_subject(
            subject_id, name, crime, embedding, filepath, 
            {"detection_confidence": confidence}
        )
        
        if not success:
            return jsonify({"success": False, "error": "Subject already exists or DB error"}), 400

        db.log_event("ADD_SUBJECT", subject_id, confidence, {"name": name, "crime": crime})
        
        return jsonify({
            "success": True, 
            "data": {
                "subject_id": subject_id, 
                "name": name, 
                "crime": crime, 
                "confidence": confidence, 
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.exception("Error adding subject")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recognize", methods=["POST"])
def recognize():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        filepath = save_uploaded_file(file, subfolder="recognize")
        if not filepath:
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        threshold = float(request.form.get("threshold", 0.4))
        top_k = int(request.form.get("top_k", 5))

        logger.info(f"Recognition request for {filepath}")
        results = detector.process_image(filepath)
        
        if not results:
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({
                "success": True, 
                "data": {
                    "faces_detected": 0, 
                    "matches": [], 
                    "message": "No faces detected"
                }
            })

        database_embeddings = db.get_all_embeddings()
        if not database_embeddings:
            return jsonify({
                "success": True, 
                "data": {
                    "faces_detected": len(results), 
                    "matches": [], 
                    "message": "Database is empty"
                }
            })

        matcher.similarity_threshold = threshold
        matches = []
        
        for i, r in enumerate(results):
            embedding = r["embedding"]
            box = r.get("box")
            det_conf = float(r.get("confidence", 0.0))

            match = matcher.find_best_match(embedding, database_embeddings)
            
            if match:
                top_matches = matcher.find_top_k_matches(embedding, database_embeddings, top_k)
                match_data = {
                    "face_id": i + 1,
                    "bounding_box": box,
                    "detection_confidence": det_conf,
                    "status": "MATCH",
                    "best_match": {
                        "subject_id": match["subject_id"],
                        "name": match["name"],
                        "crime": match["crime"],
                        "similarity_score": float(match["similarity_score"]),
                        "confidence_level": matcher.calculate_confidence(match["similarity_score"]),
                    },
                    "top_matches": top_matches,
                }
                db.log_event(
                    "MATCH", match["subject_id"], match["similarity_score"], 
                    {"name": match["name"], "crime": match["crime"]}, filepath
                )
            else:
                match_data = {
                    "face_id": i + 1, 
                    "bounding_box": box, 
                    "detection_confidence": det_conf, 
                    "status": "NO_MATCH", 
                    "message": "No matching subject found"
                }
                db.log_event("NO_MATCH", None, None, {}, filepath)
            
            matches.append(match_data)

        response = {
            "success": True, 
            "data": {
                "faces_detected": len(results), 
                "matches": matches, 
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return app.response_class(
            json.dumps(response, cls=NumpyEncoder), 
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.exception("Error in recognition")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/process_video", methods=["POST"])
def process_video():
    try:
        from src.video_processor import VideoProcessor
        
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        video_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"{timestamp}_{video_filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)
        
        fps = float(request.form.get('fps', 1))
        save_detections = request.form.get('save_detections', 'true').lower() == 'true'
        
        logger.info(f"Processing video: {video_filename}")
        
        processor = VideoProcessor()
        results = processor.process_video(
            video_path=video_path,
            fps=fps,
            save_detections=save_detections
        )
        
        if 'error' in results:
            return jsonify({'success': False, 'error': results['error']}), 400
        
        # Convert numpy arrays to JSON-safe format
        json_safe_results = {
            'video_path': str(results['video_path']),
            'total_frames': int(results['total_frames']),
            'frames_processed': int(results['frames_processed']),
            'total_faces_detected': int(results['total_faces_detected']),
            'unique_subjects': results['unique_subjects'],
            'matches_found': [],
            'detection_timeline': []
        }
        
        for match in results['matches_found']:
            json_safe_results['matches_found'].append({
                'frame_number': int(match['frame_number']),
                'timestamp': float(match['timestamp']),
                'subject_id': str(match['subject_id']),
                'name': str(match['name']),
                'crime': str(match['crime']),
                'similarity_score': float(match['similarity_score']),
                'detection_confidence': float(match['detection_confidence']),
                'bounding_box': [float(x) for x in match['bounding_box']]
            })
        
        logger.info(f"Video processing complete: {len(json_safe_results['matches_found'])} matches")
        
        return jsonify({'success': True, 'data': json_safe_results})
    
    except Exception as e:
        logger.exception("Error processing video")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        stats = db.get_statistics()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        logger.exception("Error getting stats")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/subjects", methods=["GET"])
def get_subjects():
    try:
        subjects = db.get_all_subjects()
        for s in subjects:
            s.pop("embedding", None)
        return jsonify({"success": True, "data": subjects, "count": len(subjects)})
    except Exception as e:
        logger.exception("Error getting subjects")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/events", methods=["GET"])
def get_events():
    try:
        limit = int(request.args.get("limit", 20))
        events = db.get_events(limit=limit)
        return jsonify({"success": True, "data": events, "count": len(events)})
    except Exception as e:
        logger.exception("Error getting events")
        return jsonify({"success": False, "error": str(e)}), 500


# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'success'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('start_monitoring')
def handle_start_monitoring(data):
    """Start live stream monitoring"""
    global live_monitor
    
    stream_url = data.get('stream_url')
    process_fps = float(data.get('fps', 1))
    threshold = float(data.get('threshold', 0.65))  # Get threshold from frontend
    
    if not stream_url:
        emit('error', {'message': 'Stream URL required'})
        return
    
    try:
        # Stop existing monitor if any
        if live_monitor:
            live_monitor.stop_monitoring()
        
        # Create new monitor with custom threshold
        live_monitor = LiveStreamMonitor(similarity_threshold=threshold)
        callback = WebSocketCallback(socketio)
        live_monitor.add_alert_callback(callback)
        
        # Start monitoring
        success = live_monitor.start_monitoring(stream_url, process_fps)
        
        if success:
            emit('monitoring_started', {'status': 'success'})
            logger.info(f"✅ Started monitoring: {stream_url} (threshold: {threshold})")
        else:
            emit('error', {'message': 'Failed to start monitoring'})
            
    except Exception as e:
        logger.exception("Error starting monitoring")
        emit('error', {'message': str(e)})


@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop live stream monitoring"""
    global live_monitor
    
    if live_monitor:
        live_monitor.stop_monitoring()
        live_monitor = None
        emit('monitoring_stopped', {'status': 'success'})
        logger.info("⏹️ Stopped monitoring")


# Error handlers
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": f"Endpoint {request.path} not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "error": "File too large"}), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500


# Run application
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True  # For development only
    )
