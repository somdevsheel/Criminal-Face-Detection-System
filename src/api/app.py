"""
Flask API Application
REST API for criminal face detection and recognition
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import base64
import io
from PIL import Image
import numpy as np

from src.detect_embed import FaceDetectorEmbedder
from src.matcher import FaceMatcher
from src.db import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='../../web/templates',
           static_folder='../../web/static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['PROCESSED_FOLDER'] = 'data/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize components
detector = FaceDetectorEmbedder()
matcher = FaceMatcher(similarity_threshold=0.4, metric='cosine')
db = Database('criminal_detection.db')

logger.info("Flask app initialized successfully")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_uploaded_file(file):
    """Save uploaded file and return path"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None


@app.route('/')
def index():
    """Serve main HTML page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/add_subject', methods=['POST'])
def add_subject():
    """
    Add a new subject to the database
    
    Request:
        - multipart/form-data with:
          - image: Image file
          - subject_id: Unique ID
          - name: Subject name
          - crime: Crime description
          - additional_info: JSON string (optional)
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        subject_id = request.form.get('subject_id', '').strip()
        name = request.form.get('name', '').strip()
        crime = request.form.get('crime', '').strip()
        
        if not all([subject_id, name, crime]):
            return jsonify({
                'success': False,
                'error': 'subject_id, name, and crime are required'
            }), 400
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded file
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400
        
        logger.info(f"Processing new subject: {subject_id} - {name}")
        
        # Detect faces and extract embeddings
        results = detector.process_image(filepath)
        
        if not results:
            os.remove(filepath)  # Clean up
            return jsonify({
                'success': False,
                'error': 'No face detected in image'
            }), 400
        
        if len(results) > 1:
            logger.warning(f"Multiple faces detected, using first face")
        
        # Use first detected face
        embedding = results[0]['embedding']
        confidence = results[0]['confidence']
        
        # Parse additional info
        additional_info = {}
        if request.form.get('additional_info'):
            import json
            try:
                additional_info = json.loads(request.form.get('additional_info'))
            except json.JSONDecodeError:
                pass
        
        additional_info['detection_confidence'] = confidence
        
        # Add to database
        success = db.add_subject(
            subject_id=subject_id,
            name=name,
            crime=crime,
            embedding=embedding,
            image_path=filepath,
            additional_info=additional_info
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Subject already exists or database error'
            }), 400
        
        # Log event
        db.log_event('ADD_SUBJECT', subject_id, confidence, 
                    {'name': name, 'crime': crime})
        
        logger.info(f"Successfully added subject: {subject_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'subject_id': subject_id,
                'name': name,
                'crime': crime,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Error adding subject: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Recognize faces in uploaded image
    
    Request:
        - multipart/form-data with:
          - image: Image file
          - threshold: Similarity threshold (optional)
          - top_k: Number of top matches to return (optional)
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get parameters
        threshold = float(request.form.get('threshold', 0.4))
        top_k = int(request.form.get('top_k', 5))
        
        # Save uploaded file
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400
        
        logger.info(f"Processing recognition request: {filepath}")
        
        # Detect faces
        results = detector.process_image(filepath)
        
        if not results:
            os.remove(filepath)
            return jsonify({
                'success': True,
                'data': {
                    'faces_detected': 0,
                    'matches': [],
                    'message': 'No faces detected in image'
                }
            })
        
        logger.info(f"Detected {len(results)} face(s)")
        
        # Get all database embeddings
        database_embeddings = db.get_all_embeddings()
        
        if not database_embeddings:
            return jsonify({
                'success': True,
                'data': {
                    'faces_detected': len(results),
                    'matches': [],
                    'message': 'Database is empty'
                }
            })
        
        # Update matcher threshold
        matcher.similarity_threshold = threshold
        
        # Match each detected face
        matches = []
        for i, result in enumerate(results):
            embedding = result['embedding']
            box = result['box']
            det_confidence = result['confidence']
            
            # Find best match
            match = matcher.find_best_match(embedding, database_embeddings)
            
            if match:
                # Get top-k matches for this face
                top_matches = matcher.find_top_k_matches(embedding, database_embeddings, top_k)
                
                match_data = {
                    'face_id': i + 1,
                    'bounding_box': box,
                    'detection_confidence': det_confidence,
                    'status': 'MATCH',
                    'best_match': {
                        'subject_id': match['subject_id'],
                        'name': match['name'],
                        'crime': match['crime'],
                        'similarity_score': match['similarity_score'],
                        'confidence_level': matcher.calculate_confidence(
                            match['similarity_score']
                        )
                    },
                    'top_matches': top_matches
                }
                
                # Log match event
                db.log_event('MATCH', match['subject_id'], 
                           match['similarity_score'],
                           {'name': match['name'], 'crime': match['crime']},
                           filepath)
            else:
                match_data = {
                    'face_id': i + 1,
                    'bounding_box': box,
                    'detection_confidence': det_confidence,
                    'status': 'NO_MATCH',
                    'message': 'No matching subject found'
                }
                
                # Log no-match event
                db.log_event('NO_MATCH', None, None, {}, filepath)
            
            matches.append(match_data)
        
        logger.info(f"Recognition complete: {len(matches)} face(s) processed")
        
        return jsonify({
            'success': True,
            'data': {
                'faces_detected': len(results),
                'matches': matches,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Error in recognition: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get all subjects or search subjects"""
    try:
        query = request.args.get('q', '').strip()
        
        if query:
            subjects = db.search_subjects(query)
        else:
            subjects = db.get_all_subjects()
        
        # Remove embeddings from response (too large)
        subjects_data = []
        for subject in subjects:
            subject_copy = subject.copy()
            subject_copy.pop('embedding', None)
            subjects_data.append(subject_copy)
        
        return jsonify({
            'success': True,
            'data': subjects_data,
            'count': len(subjects_data)
        })
    
    except Exception as e:
        logger.error(f"Error getting subjects: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/subjects/<subject_id>', methods=['GET'])
def get_subject(subject_id):
    """Get specific subject details"""
    try:
        subject = db.get_subject(subject_id)
        
        if not subject:
            return jsonify({
                'success': False,
                'error': 'Subject not found'
            }), 404
        
        # Remove embedding
        subject.pop('embedding', None)
        
        return jsonify({
            'success': True,
            'data': subject
        })
    
    except Exception as e:
        logger.error(f"Error getting subject: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/subjects/<subject_id>', methods=['DELETE'])
def delete_subject(subject_id):
    """Delete a subject"""
    try:
        success = db.delete_subject(subject_id, soft_delete=True)
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Subject not found'
            }), 404
        
        db.log_event('DELETE_SUBJECT', subject_id)
        
        return jsonify({
            'success': True,
            'message': f'Subject {subject_id} deleted'
        })
    
    except Exception as e:
        logger.error(f"Error deleting subject: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get system events/logs"""
    try:
        limit = int(request.args.get('limit', 100))
        event_type = request.args.get('type')
        subject_id = request.args.get('subject_id')
        
        events = db.get_events(limit, event_type, subject_id)
        
        return jsonify({
            'success': True,
            'data': events,
            'count': len(events)
        })
    
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )