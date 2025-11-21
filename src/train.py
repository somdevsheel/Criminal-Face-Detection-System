"""
Training and Batch Processing Module
Processes multiple images and adds them to the database
"""

import os
import glob
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import csv
import json
from tqdm import tqdm

from src.detect_embed import FaceDetectorEmbedder
from src.db import Database
from src.matcher import FaceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes multiple images and adds subjects to database
    Supports CSV input for bulk operations
    """
    
    def __init__(self, db_path: str = "criminal_detection.db"):
        """
        Initialize batch processor
        
        Args:
            db_path: Path to database
        """
        self.detector = FaceDetectorEmbedder()
        self.db = Database(db_path)
        logger.info("Batch processor initialized")
    
    def process_single_image(self, image_path: str, subject_id: str,
                            name: str, crime: str,
                            additional_info: Dict = None) -> bool:
        """
        Process a single image and add to database
        
        Args:
            image_path: Path to image file
            subject_id: Unique subject ID
            name: Subject name
            crime: Crime description
            additional_info: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Processing: {subject_id} - {name}")
            
            # Detect faces and extract embeddings
            results = self.detector.process_image(image_path)
            
            if not results:
                logger.warning(f"No faces detected in {image_path}")
                return False
            
            if len(results) > 1:
                logger.warning(f"Multiple faces detected, using first face")
            
            # Use first detected face
            embedding = results[0]['embedding']
            confidence = results[0]['confidence']
            
            # Add to database
            success = self.db.add_subject(
                subject_id=subject_id,
                name=name,
                crime=crime,
                embedding=embedding,
                image_path=image_path,
                additional_info={
                    **(additional_info or {}),
                    'detection_confidence': confidence
                }
            )
            
            if success:
                logger.info(f"✓ Added {subject_id} with confidence {confidence:.3f}")
            else:
                logger.error(f"✗ Failed to add {subject_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_directory(self, directory: str, pattern: str = "*.jpg",
                         id_prefix: str = "SUBJ") -> Dict:
        """
        Process all images in a directory
        Uses filename as subject name (assumes format: ID_Name.jpg)
        
        Args:
            directory: Directory containing images
            pattern: File pattern to match
            id_prefix: Prefix for auto-generated IDs
            
        Returns:
            Dictionary with processing statistics
        """
        image_paths = glob.glob(os.path.join(directory, pattern))
        
        if not image_paths:
            logger.warning(f"No images found in {directory} with pattern {pattern}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        stats = {'success': 0, 'failed': 0, 'total': len(image_paths)}
        
        logger.info(f"Processing {stats['total']} images from {directory}")
        
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            filename = Path(image_path).stem
            
            # Try to parse filename (ID_Name format)
            if '_' in filename:
                parts = filename.split('_', 1)
                subject_id = parts[0]
                name = parts[1].replace('_', ' ')
            else:
                subject_id = f"{id_prefix}{idx+1:04d}"
                name = filename.replace('_', ' ')
            
            # Process with default crime as "Unknown"
            success = self.process_single_image(
                image_path=image_path,
                subject_id=subject_id,
                name=name,
                crime="Unknown - Update Required"
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Batch processing complete: {stats['success']} success, "
                   f"{stats['failed']} failed")
        return stats
    
    def process_csv(self, csv_path: str, image_dir: str = None) -> Dict:
        """
        Process subjects from CSV file
        
        CSV format:
        subject_id,name,crime,image_filename,additional_info
        
        Args:
            csv_path: Path to CSV file
            image_dir: Directory containing images (if not absolute paths in CSV)
            
        Returns:
            Processing statistics
        """
        stats = {'success': 0, 'failed': 0, 'total': 0}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                stats['total'] = len(rows)
                
                logger.info(f"Processing {stats['total']} entries from CSV")
                
                for row in tqdm(rows, desc="Processing CSV"):
                    subject_id = row.get('subject_id', '').strip()
                    name = row.get('name', '').strip()
                    crime = row.get('crime', '').strip()
                    image_filename = row.get('image_filename', '').strip()
                    
                    if not all([subject_id, name, crime, image_filename]):
                        logger.warning(f"Skipping row with missing data: {row}")
                        stats['failed'] += 1
                        continue
                    
                    # Construct image path
                    if image_dir and not os.path.isabs(image_filename):
                        image_path = os.path.join(image_dir, image_filename)
                    else:
                        image_path = image_filename
                    
                    if not os.path.exists(image_path):
                        logger.warning(f"Image not found: {image_path}")
                        stats['failed'] += 1
                        continue
                    
                    # Parse additional info if present
                    additional_info = None
                    if 'additional_info' in row and row['additional_info']:
                        try:
                            additional_info = json.loads(row['additional_info'])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in additional_info for {subject_id}")
                    
                    # Process image
                    success = self.process_single_image(
                        image_path=image_path,
                        subject_id=subject_id,
                        name=name,
                        crime=crime,
                        additional_info=additional_info
                    )
                    
                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
        
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
        
        logger.info(f"CSV processing complete: {stats['success']} success, "
                   f"{stats['failed']} failed")
        return stats
    
    def export_database_to_csv(self, output_path: str = "database_export.csv"):
        """
        Export database to CSV file
        
        Args:
            output_path: Output CSV file path
        """
        try:
            subjects = self.db.get_all_subjects()
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['subject_id', 'name', 'crime', 'added_on', 
                            'image_path', 'additional_info']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for subject in subjects:
                    writer.writerow({
                        'subject_id': subject['subject_id'],
                        'name': subject['name'],
                        'crime': subject['crime'],
                        'added_on': subject['added_on'],
                        'image_path': subject.get('image_path', ''),
                        'additional_info': json.dumps(subject.get('additional_info', {}))
                    })
            
            logger.info(f"Exported {len(subjects)} subjects to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def validate_database(self) -> Dict:
        """
        Validate database integrity
        
        Returns:
            Validation report
        """
        logger.info("Validating database...")
        
        subjects = self.db.get_all_subjects()
        report = {
            'total_subjects': len(subjects),
            'valid_embeddings': 0,
            'invalid_embeddings': 0,
            'missing_images': 0,
            'duplicate_ids': 0
        }
        
        seen_ids = set()
        
        for subject in subjects:
            # Check for duplicates
            if subject['subject_id'] in seen_ids:
                report['duplicate_ids'] += 1
            seen_ids.add(subject['subject_id'])
            
            # Validate embedding
            try:
                embedding = subject['embedding']
                if embedding.shape == (512,) and np.isfinite(embedding).all():
                    report['valid_embeddings'] += 1
                else:
                    report['invalid_embeddings'] += 1
                    logger.warning(f"Invalid embedding for {subject['subject_id']}")
            except:
                report['invalid_embeddings'] += 1
            
            # Check image existence
            if subject.get('image_path'):
                if not os.path.exists(subject['image_path']):
                    report['missing_images'] += 1
        
        logger.info(f"Validation complete: {report}")
        return report
    
    def benchmark_matching(self, num_queries: int = 10) -> Dict:
        """
        Benchmark matching performance
        
        Args:
            num_queries: Number of test queries
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running matching benchmark with {num_queries} queries...")
        
        import time
        
        # Get all embeddings
        embeddings = self.db.get_all_embeddings()
        
        if not embeddings:
            logger.warning("No embeddings in database")
            return {}
        
        matcher = FaceMatcher(similarity_threshold=0.4, metric='cosine')
        
        # Generate random query embeddings
        times = []
        for i in range(num_queries):
            query = np.random.randn(512).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            start = time.time()
            match = matcher.find_best_match(query, embeddings)
            elapsed = time.time() - start
            times.append(elapsed)
        
        results = {
            'num_database_subjects': len(embeddings),
            'num_queries': num_queries,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'throughput_qps': 1.0 / np.mean(times)
        }
        
        logger.info(f"Benchmark results: {results}")
        return results


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process criminal face images")
    parser.add_argument('--mode', choices=['single', 'directory', 'csv', 'validate', 'benchmark'],
                       required=True, help='Processing mode')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--directory', help='Directory containing images')
    parser.add_argument('--csv', help='Path to CSV file')
    parser.add_argument('--image-dir', help='Directory for images referenced in CSV')
    parser.add_argument('--subject-id', help='Subject ID for single image')
    parser.add_argument('--name', help='Subject name for single image')
    parser.add_argument('--crime', help='Crime description for single image')
    parser.add_argument('--db', default='criminal_detection.db', help='Database path')
    
    args = parser.parse_args()
    
    processor = BatchProcessor(args.db)
    
    if args.mode == 'single':
        if not all([args.image, args.subject_id, args.name, args.crime]):
            print("Error: single mode requires --image, --subject-id, --name, --crime")
        else:
            processor.process_single_image(args.image, args.subject_id, 
                                          args.name, args.crime)
    
    elif args.mode == 'directory':
        if not args.directory:
            print("Error: directory mode requires --directory")
        else:
            processor.process_directory(args.directory)
    
    elif args.mode == 'csv':
        if not args.csv:
            print("Error: csv mode requires --csv")
        else:
            processor.process_csv(args.csv, args.image_dir)
    
    elif args.mode == 'validate':
        processor.validate_database()
    
    elif args.mode == 'benchmark':
        processor.benchmark_matching()