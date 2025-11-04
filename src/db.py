"""
Database Module
Handles SQLite database operations for storing subjects and events
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import os
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for criminal face detection system
    Stores subject information and recognition events
    """
    
    def __init__(self, db_path: str = "criminal_detection.db"):
        """
        Initialize database connection
        """
        self.db_path = db_path
        self.init_database()
        logger.info(f"Database initialized at {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Subjects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    crime TEXT NOT NULL,
                    added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB NOT NULL,
                    image_path TEXT,
                    additional_info TEXT,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    subject_id TEXT,
                    score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra TEXT,
                    image_path TEXT,
                    location TEXT
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject_id ON subjects(subject_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_timestamp ON events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_subject ON events(subject_id)')
            
            logger.info("Database tables created/verified")
    
    def embedding_to_blob(self, embedding: np.ndarray) -> bytes:
        """Convert numpy embedding to binary blob"""
        return embedding.tobytes()
    
    def blob_to_embedding(self, blob: bytes) -> np.ndarray:
        """Convert binary blob to numpy embedding"""
        return np.frombuffer(blob, dtype=np.float32)
    
    def add_subject(self, subject_id: str, name: str, crime: str,
                   embedding: np.ndarray, image_path: str = None,
                   additional_info: Dict = None) -> bool:
        """
        Add a new subject to the database
        """
        try:
            import numpy as np

            # ✅ FIX: Convert list embeddings back to NumPy arrays if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = embedding.astype(np.float32)

            embedding_blob = self.embedding_to_blob(embedding)

            # Convert additional_info to JSON
            info_json = json.dumps(additional_info) if additional_info else None

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO subjects (subject_id, name, crime, embedding, 
                                        image_path, additional_info)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (subject_id, name, crime, embedding_blob, image_path, info_json))

                logger.info(f"Added subject: {subject_id} - {name}")
                return True

        except sqlite3.IntegrityError:
            logger.error(f"Subject {subject_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding subject: {e}")
            return False
    
    def get_subject(self, subject_id: str) -> Optional[Dict]:
        """Retrieve a subject by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM subjects WHERE subject_id = ? AND is_active = 1', (subject_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_subject_dict(row)
        except Exception as e:
            logger.error(f"Error retrieving subject: {e}")
            return None
    
    def get_all_subjects(self, active_only: bool = True) -> List[Dict]:
        """Get all subjects from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if active_only:
                    cursor.execute('SELECT * FROM subjects WHERE is_active = 1')
                else:
                    cursor.execute('SELECT * FROM subjects')
                rows = cursor.fetchall()
                return [self._row_to_subject_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving subjects: {e}")
            return []
    
    def get_all_embeddings(self) -> List[Dict]:
        """Get all embeddings for matching"""
        subjects = self.get_all_subjects()
        return [{
            'subject_id': s['subject_id'],
            'name': s['name'],
            'crime': s['crime'],
            'embedding': s['embedding']
        } for s in subjects]
    
    def update_subject(self, subject_id: str, **kwargs) -> bool:
        """Update subject information"""
        try:
            valid_fields = ['name', 'crime', 'embedding', 'image_path', 'additional_info', 'is_active']
            updates = []
            values = []

            for key, value in kwargs.items():
                if key in valid_fields:
                    if key == 'embedding':
                        if isinstance(value, list):
                            value = np.array(value, dtype=np.float32)
                        value = self.embedding_to_blob(value.astype(np.float32))
                    elif key == 'additional_info':
                        value = json.dumps(value)
                    updates.append(f"{key} = ?")
                    values.append(value)

            if not updates:
                return False

            values.append(subject_id)
            query = f"UPDATE subjects SET {', '.join(updates)} WHERE subject_id = ?"

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                logger.info(f"Updated subject: {subject_id}")
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating subject: {e}")
            return False
    
    def delete_subject(self, subject_id: str, soft_delete: bool = True) -> bool:
        """Delete a subject (soft or hard delete)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if soft_delete:
                    cursor.execute('UPDATE subjects SET is_active = 0 WHERE subject_id = ?', (subject_id,))
                else:
                    cursor.execute('DELETE FROM subjects WHERE subject_id = ?', (subject_id,))
                logger.info(f"Deleted subject: {subject_id} (soft={soft_delete})")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting subject: {e}")
            return False
    
    def log_event(self, event_type: str, subject_id: str = None,
                 score: float = None, extra: Dict = None,
                 image_path: str = None, location: str = None) -> bool:
        """Log a recognition event"""
        try:
            extra_json = json.dumps(extra) if extra else None
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (event_type, subject_id, score, extra, 
                                      image_path, location)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (event_type, subject_id, score, extra_json, image_path, location))
                logger.info(f"Logged event: {event_type}")
                return True
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False
    
    def get_events(self, limit: int = 100, event_type: str = None,
                  subject_id: str = None) -> List[Dict]:
        """Retrieve events with optional filtering"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = 'SELECT * FROM events WHERE 1=1'
                params = []
                if event_type:
                    query += ' AND event_type = ?'
                    params.append(event_type)
                if subject_id:
                    query += ' AND subject_id = ?'
                    params.append(subject_id)
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                events = []
                for row in rows:
                    event = dict(row)
                    if event['extra']:
                        event['extra'] = json.loads(event['extra'])
                    events.append(event)
                return events
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM subjects WHERE is_active = 1')
                total_subjects = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM events')
                total_events = cursor.fetchone()[0]
                cursor.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type')
                events_by_type = {row[0]: row[1] for row in cursor.fetchall()}
                cursor.execute('''
                    SELECT COUNT(*) FROM events 
                    WHERE event_type = 'MATCH' 
                    AND timestamp >= datetime('now', '-7 days')
                ''')
                recent_matches = cursor.fetchone()[0]
                return {
                    'total_subjects': total_subjects,
                    'total_events': total_events,
                    'events_by_type': events_by_type,
                    'recent_matches': recent_matches
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _row_to_subject_dict(self, row: sqlite3.Row) -> Dict:
        """Convert database row to subject dictionary"""
        subject = dict(row)
        subject['embedding'] = self.blob_to_embedding(subject['embedding'])
        if subject['additional_info']:
            subject['additional_info'] = json.loads(subject['additional_info'])
        return subject
    
    def search_subjects(self, query: str) -> List[Dict]:
        """Search subjects by name, ID, or crime"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM subjects 
                    WHERE (name LIKE ? OR subject_id LIKE ? OR crime LIKE ?)
                    AND is_active = 1
                ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
                rows = cursor.fetchall()
                return [self._row_to_subject_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching subjects: {e}")
            return []


# Example usage
if __name__ == "__main__":
    db = Database("test.db")
    sample_embedding = np.random.randn(512).astype(np.float32)
    success = db.add_subject(
        subject_id="TEST001",
        name="Test Subject",
        crime="Testing Database",
        embedding=sample_embedding,
        additional_info={"age": 30, "height": 180}
    )
    subject = db.get_subject("TEST001")
    print(f"Retrieved subject: {subject['name']}")
    db.log_event("MATCH", "TEST001", 0.85, {"confidence": "HIGH"})
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")
