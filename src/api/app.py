# """
# Flask API Application
# REST API for criminal face detection and recognition
# """

# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# import os
# import logging
# from datetime import datetime
# # from werkzeug.serving import WSGIRequestHandler
# import base64
# import io
# import json
# import numpy as np
# from PIL import Image

# from src.detect_embed import FaceDetectorEmbedder
# from src.matcher import FaceMatcher
# from src.db import Database

# # -------------------------------------
# # NumPy-safe JSON encoder
# # -------------------------------------
# class NumpyEncoder(json.JSONEncoder):
#     """Safely convert NumPy data types to JSON serializable ones"""
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, (np.float32, np.float64, np.float_)):
#             return float(obj)
#         if isinstance(obj, (np.int32, np.int64, np.integer)):
#             return int(obj)
#         return super().default(obj)

# # -------------------------------------
# # Logging Configuration
# # -------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # -------------------------------------
# # Initialize Flask app
# # -------------------------------------
# from pathlib import Path
# project_root = Path(__file__).parent.parent.parent

# app = Flask(
#     __name__,
#     template_folder=str(project_root / "web" / "templates"),
#     static_folder=str(project_root / "web" / "static")
# )
# CORS(app)

# # -------------------------------------
# # Configuration
# # -------------------------------------
# app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
# app.config["UPLOAD_FOLDER"] = "data/raw"
# app.config["PROCESSED_FOLDER"] = "data/processed"
# app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp"}

# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)
# os.makedirs("logs", exist_ok=True)

# # -------------------------------------
# # Initialize components
# # -------------------------------------
# detector = FaceDetectorEmbedder()
# matcher = FaceMatcher(similarity_threshold=0.4, metric="cosine")
# db = Database("criminal_detection.db")

# logger.info("Flask app initialized successfully")


# def allowed_file(filename):
#     """Check if file extension is allowed"""
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


# def save_uploaded_file(file):
#     """Save uploaded file and return path"""
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{timestamp}_{filename}"
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)
#         return filepath
#     return None


# @app.route("/")
# def index():
#     """Serve main HTML page"""
#     return render_template("index.html")


# @app.route("/api/health", methods=["GET"])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": "1.0.0"
#     })


# # ----------------------------------------------------------
# # Add new subject
# # ----------------------------------------------------------
# @app.route("/api/add_subject", methods=["POST"])
# def add_subject():
#     """Add a new subject to the database"""
#     try:
#         if "image" not in request.files:
#             return jsonify({"success": False, "error": "No image file provided"}), 400

#         file = request.files["image"]
#         subject_id = request.form.get("subject_id", "").strip()
#         name = request.form.get("name", "").strip()
#         crime = request.form.get("crime", "").strip()

#         if not all([subject_id, name, crime]):
#             return jsonify({"success": False, "error": "subject_id, name, and crime are required"}), 400

#         if file.filename == "":
#             return jsonify({"success": False, "error": "No file selected"}), 400

#         filepath = save_uploaded_file(file)
#         if not filepath:
#             return jsonify({"success": False, "error": "Invalid file type"}), 400

#         logger.info(f"Processing new subject: {subject_id} - {name}")
#         results = detector.process_image(filepath)

#         if not results:
#             os.remove(filepath)
#             return jsonify({"success": False, "error": "No face detected in image"}), 400

#         if len(results) > 1:
#             logger.warning("Multiple faces detected, using first face")

#         embedding = results[0]["embedding"]
#         confidence = results[0]["confidence"]

#         # Convert NumPy array to list before DB storage
#         if isinstance(embedding, np.ndarray):
#             embedding = embedding.tolist()

#         additional_info = {}
#         if request.form.get("additional_info"):
#             try:
#                 additional_info = json.loads(request.form.get("additional_info"))
#             except json.JSONDecodeError:
#                 pass
#         additional_info["detection_confidence"] = confidence

#         success = db.add_subject(
#             subject_id=subject_id,
#             name=name,
#             crime=crime,
#             embedding=embedding,
#             image_path=filepath,
#             additional_info=additional_info
#         )

#         if not success:
#             return jsonify({"success": False, "error": "Subject already exists or database error"}), 400

#         db.log_event("ADD_SUBJECT", subject_id, confidence, {"name": name, "crime": crime})
#         logger.info(f"Successfully added subject: {subject_id}")

#         return jsonify({
#             "success": True,
#             "data": {
#                 "subject_id": subject_id,
#                 "name": name,
#                 "crime": crime,
#                 "confidence": confidence,
#                 "timestamp": datetime.now().isoformat()
#             }
#         })

#     except Exception as e:
#         logger.error(f"Error adding subject: {e}")
#         return jsonify({"success": False, "error": str(e)}), 500


# # ----------------------------------------------------------
# # Face Recognition
# # ----------------------------------------------------------
# @app.route("/api/recognize", methods=["POST"])
# def recognize():
#     """Recognize faces in uploaded image"""
#     try:
#         if "image" not in request.files:
#             return jsonify({"success": False, "error": "No image file provided"}), 400

#         file = request.files["image"]
#         if file.filename == "":
#             return jsonify({"success": False, "error": "No file selected"}), 400

#         threshold = float(request.form.get("threshold", 0.4))
#         top_k = int(request.form.get("top_k", 5))
#         filepath = save_uploaded_file(file)
#         if not filepath:
#             return jsonify({"success": False, "error": "Invalid file type"}), 400

#         logger.info(f"Processing recognition request: {filepath}")
#         results = detector.process_image(filepath)

#         if not results:
#             os.remove(filepath)
#             return jsonify({
#                 "success": True,
#                 "data": {"faces_detected": 0, "matches": [], "message": "No faces detected in image"}
#             })

#         logger.info(f"Detected {len(results)} face(s)")
#         database_embeddings = db.get_all_embeddings()
#         if not database_embeddings:
#             return jsonify({
#                 "success": True,
#                 "data": {"faces_detected": len(results), "matches": [], "message": "Database is empty"}
#             })

#         matcher.similarity_threshold = threshold
#         matches = []
#         for i, result in enumerate(results):
#             embedding = result["embedding"]
#             box = result["box"]
#             det_confidence = result["confidence"]

#             match = matcher.find_best_match(embedding, database_embeddings)
#             if match:
#                 top_matches = matcher.find_top_k_matches(embedding, database_embeddings, top_k)
#                 match_data = {
#                     "face_id": i + 1,
#                     "bounding_box": np.array(box).tolist() if isinstance(box, np.ndarray) else box,
#                     "detection_confidence": float(det_confidence),
#                     "status": "MATCH",
#                     "best_match": {
#                         "subject_id": match["subject_id"],
#                         "name": match["name"],
#                         "crime": match["crime"],
#                         "similarity_score": float(match["similarity_score"]),
#                         "confidence_level": matcher.calculate_confidence(match["similarity_score"])
#                     },
#                     "top_matches": top_matches
#                 }
#                 db.log_event("MATCH", match["subject_id"], match["similarity_score"],
#                              {"name": match["name"], "crime": match["crime"]}, filepath)
#             else:
#                 match_data = {
#                     "face_id": i + 1,
#                     "bounding_box": np.array(box).tolist() if isinstance(box, np.ndarray) else box,
#                     "detection_confidence": float(det_confidence),
#                     "status": "NO_MATCH",
#                     "message": "No matching subject found"
#                 }
#                 db.log_event("NO_MATCH", None, None, {}, filepath)
#             matches.append(match_data)

#         logger.info(f"Recognition complete: {len(matches)} face(s) processed")

#         response_data = {
#             "success": True,
#             "data": {
#                 "faces_detected": len(results),
#                 "matches": matches,
#                 "timestamp": datetime.now().isoformat()
#             }
#         }
#         return app.response_class(
#             json.dumps(response_data, cls=NumpyEncoder),
#             mimetype="application/json"
#         )

#     except Exception as e:
#         logger.error(f"Error in recognition: {e}")
#         return jsonify({"success": False, "error": str(e)}), 500


# # # Silence Chrome devtools noise
# # logging.getLogger('werkzeug').setLevel(logging.WARNING)
# # WSGIRequestHandler.protocol_version = "HTTP/1.1"

# # ----------------------------------------------------------
# # Additional endpoints for dashboard stats
# # ----------------------------------------------------------
# @app.route("/api/stats", methods=["GET"])
# def get_stats():
#     """Return dashboard statistics"""
#     try:
#         stats = db.get_statistics()
#         return jsonify({"success": True, "data": stats})
#     except Exception as e:
#         logger.error(f"Error getting stats: {e}")
#         return jsonify({"success": False, "error": str(e)}), 500


# @app.route("/api/subjects", methods=["GET"])
# def get_subjects():
#     """Return all registered subjects"""
#     try:
#         subjects = db.get_all_subjects()
#         for s in subjects:
#             s.pop("embedding", None)
#         return jsonify({"success": True, "data": subjects, "count": len(subjects)})
#     except Exception as e:
#         logger.error(f"Error getting subjects: {e}")
#         return jsonify({"success": False, "error": str(e)}), 500


# @app.route("/api/events", methods=["GET"])
# def get_events():
#     """Return recent system events"""
#     try:
#         limit = int(request.args.get("limit", 20))
#         events = db.get_events(limit=limit)
#         return jsonify({"success": True, "data": events, "count": len(events)})
#     except Exception as e:
#         logger.error(f"Error getting events: {e}")
#         return jsonify({"success": False, "error": str(e)}), 500


# # ----------------------------------------------------------
# # Error handlers
# # ----------------------------------------------------------
# @app.errorhandler(404)
# def not_found(error):
#     """Return JSON 404 for API routes"""
#     if request.path.startswith("/api/"):
#         return jsonify({"success": False, "error": f"Endpoint {request.path} not found"}), 404
#     return render_template("index.html"), 404


# @app.errorhandler(413)
# def request_entity_too_large(error):
#     return jsonify({"success": False, "error": "File too large. Maximum size is 16MB"}), 413


# @app.errorhandler(500)
# def internal_error(error):
#     logger.error(f"Internal error: {error}")
#     return jsonify({"success": False, "error": "Internal server error"}), 500


# # ----------------------------------------------------------
# # Run Flask App
# # ----------------------------------------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)




"""
Flask API Application
REST API for criminal face detection, recognition, and admin security
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path

# Core modules
from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database
from src.security import ip_whitelist, get_security_status  # ✅ security integration


# ----------------------------------------------------------
# NumPy-safe JSON encoder
# ----------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        return super().default(obj)


# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CriminalFaceDetection")


# ----------------------------------------------------------
# Flask App setup
# ----------------------------------------------------------
project_root = Path(__file__).parent.parent.parent
app = Flask(
    __name__,
    template_folder=str(project_root / "web" / "templates"),
    static_folder=str(project_root / "web" / "static")
)
CORS(app)

app.config["UPLOAD_FOLDER"] = "data/raw"
app.config["PROCESSED_FOLDER"] = "data/processed"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ----------------------------------------------------------
# Initialize ML and Database
# ----------------------------------------------------------
detector = FaceDetectorEmbedder()
matcher = FaceMatcher(similarity_threshold=0.4, metric="cosine")
db = Database("criminal_detection.db")

logger.info("Flask backend initialized successfully")


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
        file.save(filepath)
        return filepath
    return None


# ----------------------------------------------------------
# HTML Routes
# ----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")


# ----------------------------------------------------------
# API Routes
# ----------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# -------------------------- ADD SUBJECT --------------------------
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
            return jsonify({"success": False, "error": "subject_id, name, and crime are required"}), 400

        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        logger.info(f"Processing new subject: {subject_id} - {name}")
        results = detector.process_image(filepath)

        if not results:
            os.remove(filepath)
            return jsonify({"success": False, "error": "No face detected in image"}), 400

        if len(results) > 1:
            logger.warning("Multiple faces detected, using first face")

        embedding = results[0]["embedding"]
        confidence = results[0]["confidence"]

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        success = db.add_subject(subject_id, name, crime, embedding, filepath, {"detection_confidence": confidence})
        if not success:
            return jsonify({"success": False, "error": "Subject already exists"}), 400

        db.log_event("ADD_SUBJECT", subject_id, confidence, {"name": name, "crime": crime})
        logger.info(f"Successfully added subject: {subject_id}")

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
        logger.error(f"Error adding subject: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------- RECOGNIZE FACE --------------------------
@app.route("/api/recognize", methods=["POST"])
def recognize():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        threshold = float(request.form.get("threshold", 0.4))
        top_k = int(request.form.get("top_k", 5))

        logger.info(f"Processing recognition request: {filepath}")
        results = detector.process_image(filepath)

        if not results:
            return jsonify({"success": True, "data": {"faces_detected": 0, "matches": [], "message": "No faces detected"}})

        database_embeddings = db.get_all_embeddings()
        if not database_embeddings:
            return jsonify({"success": True, "data": {"faces_detected": len(results), "matches": [], "message": "Database is empty"}})

        matcher.similarity_threshold = threshold
        matches = []

        for i, result in enumerate(results):
            embedding = result["embedding"]
            box = result["box"]
            det_confidence = result["confidence"]

            match = matcher.find_best_match(embedding, database_embeddings)
            if match:
                top_matches = matcher.find_top_k_matches(embedding, database_embeddings, top_k)
                match_data = {
                    "face_id": i + 1,
                    "bounding_box": np.array(box).tolist() if isinstance(box, np.ndarray) else box,
                    "detection_confidence": float(det_confidence),
                    "status": "MATCH",
                    "best_match": {
                        "subject_id": match["subject_id"],
                        "name": match["name"],
                        "crime": match["crime"],
                        "similarity_score": float(match["similarity_score"]),
                        "confidence_level": matcher.calculate_confidence(match["similarity_score"])
                    },
                    "top_matches": top_matches
                }
                db.log_event("MATCH", match["subject_id"], match["similarity_score"],
                             {"name": match["name"], "crime": match["crime"]}, filepath)
            else:
                match_data = {
                    "face_id": i + 1,
                    "bounding_box": np.array(box).tolist() if isinstance(box, np.ndarray) else box,
                    "detection_confidence": float(det_confidence),
                    "status": "NO_MATCH",
                    "message": "No matching subject found"
                }
                db.log_event("NO_MATCH", None, None, {}, filepath)
            matches.append(match_data)

        logger.info(f"Recognition complete: {len(matches)} face(s) processed")
        response_data = {"success": True, "data": {"faces_detected": len(results), "matches": matches}}
        return app.response_class(json.dumps(response_data, cls=NumpyEncoder), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error in recognition: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------- STATS & EVENTS --------------------------
@app.route("/api/stats")
def get_stats():
    try:
        stats = db.get_statistics()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/subjects")
def get_subjects():
    try:
        subjects = db.get_all_subjects()
        for s in subjects:
            s.pop("embedding", None)
        return jsonify({"success": True, "data": subjects})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/events")
def get_events():
    try:
        limit = int(request.args.get("limit", 20))
        events = db.get_events(limit)
        return jsonify({"success": True, "data": events})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------- SECURITY API --------------------------
@app.route("/api/security/status")
def security_status():
    try:
        return jsonify({"success": True, "data": get_security_status()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/security/whitelist/add", methods=["POST"])
def add_whitelist():
    data = request.get_json()
    ip = data.get("ip")
    if not ip:
        return jsonify({"success": False, "error": "Missing IP"}), 400
    if ip_whitelist.add_to_whitelist(ip):
        return jsonify({"success": True, "message": f"{ip} added"})
    return jsonify({"success": False, "message": f"{ip} already exists"})


@app.route("/api/security/whitelist/remove", methods=["POST"])
def remove_whitelist():
    data = request.get_json()
    ip = data.get("ip")
    if not ip:
        return jsonify({"success": False, "error": "Missing IP"}), 400
    if ip_whitelist.remove_from_whitelist(ip):
        return jsonify({"success": True, "message": f"{ip} removed"})
    return jsonify({"success": False, "message": f"{ip} not found"})


@app.route("/api/security/blacklist/add", methods=["POST"])
def add_blacklist():
    data = request.get_json()
    ip = data.get("ip")
    if not ip:
        return jsonify({"success": False, "error": "Missing IP"}), 400
    if ip_whitelist.add_to_blacklist(ip):
        return jsonify({"success": True, "message": f"{ip} added to blacklist"})
    return jsonify({"success": False, "message": f"{ip} already exists"})


@app.route("/api/security/blacklist/remove", methods=["POST"])
def remove_blacklist():
    data = request.get_json()
    ip = data.get("ip")
    if not ip:
        return jsonify({"success": False, "error": "Missing IP"}), 400
    if ip_whitelist.remove_from_blacklist(ip):
        return jsonify({"success": True, "message": f"{ip} removed from blacklist"})
    return jsonify({"success": False, "message": f"{ip} not found"})


# ----------------------------------------------------------
# Error Handlers
# ----------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": f"Endpoint {request.path} not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
