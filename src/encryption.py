"""
Encryption Module
Provides AES-256 encryption for sensitive data including face embeddings
"""

import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AESEncryption:
    """
    AES-256 encryption/decryption for face embeddings and sensitive data
    Uses PBKDF2 for key derivation from password
    """
    
    def __init__(self, password: str = None):
        """
        Initialize encryption with password
        
        Args:
            password: Encryption password (if None, uses environment variable)
        """
        if password is None:
            password = os.environ.get('ENCRYPTION_KEY', 'default-key-change-in-production')
        
        self.password = password.encode()
        logger.info("Encryption initialized")
    
    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2
        
        Args:
            salt: Random salt for key derivation
            
        Returns:
            32-byte encryption key
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,  # NIST recommended minimum
            backend=default_backend()
        )
        return kdf.derive(self.password)
    
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Encrypt a face embedding vector
        
        Args:
            embedding: Numpy array to encrypt
            
        Returns:
            Encrypted data as bytes (includes salt and IV)
        """
        try:
            # Convert embedding to bytes
            embedding_bytes = embedding.tobytes()
            
            # Generate random salt and IV
            salt = os.urandom(16)
            iv = os.urandom(16)
            
            # Derive key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Pad data to AES block size (16 bytes)
            padded_data = self._pad(embedding_bytes)
            
            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine salt + IV + encrypted_data
            result = salt + iv + encrypted_data
            
            return result
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_embedding(self, encrypted_data: bytes, dtype=np.float32) -> np.ndarray:
        """
        Decrypt an encrypted embedding
        
        Args:
            encrypted_data: Encrypted bytes (includes salt and IV)
            dtype: Data type of original array
            
        Returns:
            Decrypted numpy array
        """
        try:
            # Extract salt, IV, and encrypted data
            salt = encrypted_data[:16]
            iv = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Derive key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            data = self._unpad(padded_data)
            
            # Convert back to numpy array
            embedding = np.frombuffer(data, dtype=dtype)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def encrypt_text(self, text: str) -> str:
        """
        Encrypt text data (e.g., names, crimes)
        
        Args:
            text: String to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        try:
            # Generate salt and IV
            salt = os.urandom(16)
            iv = os.urandom(16)
            
            # Derive key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Pad and encrypt
            padded_data = self._pad(text.encode('utf-8'))
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine and encode
            result = salt + iv + encrypted_data
            return base64.b64encode(result).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Text encryption error: {e}")
            raise
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt encrypted text
        
        Args:
            encrypted_text: Base64-encoded encrypted string
            
        Returns:
            Decrypted string
        """
        try:
            # Decode
            encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
            
            # Extract components
            salt = encrypted_data[:16]
            iv = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Derive key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt and unpad
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            data = self._unpad(padded_data)
            
            return data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Text decryption error: {e}")
            raise
    
    def _pad(self, data: bytes) -> bytes:
        """
        Apply PKCS7 padding to data
        
        Args:
            data: Data to pad
            
        Returns:
            Padded data
        """
        block_size = 16  # AES block size
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad(self, padded_data: bytes) -> bytes:
        """
        Remove PKCS7 padding from data
        
        Args:
            padded_data: Padded data
            
        Returns:
            Unpadded data
        """
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    @staticmethod
    def generate_key() -> str:
        """
        Generate a secure random encryption key
        
        Returns:
            Base64-encoded random key
        """
        key = os.urandom(32)
        return base64.b64encode(key).decode('utf-8')


class SecureDatabase:
    """
    Wrapper for Database class with encryption support
    """
    
    def __init__(self, db, encryption_password: str = None):
        """
        Initialize secure database wrapper
        
        Args:
            db: Database instance
            encryption_password: Password for encryption
        """
        self.db = db
        self.encryptor = AESEncryption(encryption_password)
        logger.info("Secure database initialized")
    
    def add_subject_encrypted(self, subject_id: str, name: str, crime: str,
                             embedding: np.ndarray, **kwargs) -> bool:
        """
        Add subject with encrypted embedding
        
        Args:
            subject_id: Subject ID
            name: Subject name
            crime: Crime description
            embedding: Face embedding
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        try:
            # Encrypt embedding
            encrypted_embedding = self.encryptor.encrypt_embedding(embedding)
            
            # Store as blob (already encrypted)
            return self.db.add_subject(
                subject_id, name, crime,
                np.frombuffer(encrypted_embedding, dtype=np.uint8),
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error adding encrypted subject: {e}")
            return False
    
    def get_subject_decrypted(self, subject_id: str) -> dict:
        """
        Get subject with decrypted embedding
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Subject dictionary with decrypted embedding
        """
        try:
            subject = self.db.get_subject(subject_id)
            if subject:
                # Decrypt embedding
                encrypted_bytes = subject['embedding'].tobytes()
                subject['embedding'] = self.encryptor.decrypt_embedding(encrypted_bytes)
            return subject
            
        except Exception as e:
            logger.error(f"Error getting decrypted subject: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Generate secure key
    secure_key = AESEncryption.generate_key()
    print(f"Generated key: {secure_key}")
    print("IMPORTANT: Store this key securely (environment variable or secrets manager)")
    
    # Initialize encryption
    encryptor = AESEncryption("test-password-123")
    
    # Test embedding encryption
    original_embedding = np.random.randn(512).astype(np.float32)
    print(f"\nOriginal embedding shape: {original_embedding.shape}")
    
    # Encrypt
    encrypted = encryptor.encrypt_embedding(original_embedding)
    print(f"Encrypted size: {len(encrypted)} bytes")
    
    # Decrypt
    decrypted_embedding = encryptor.decrypt_embedding(encrypted)
    print(f"Decrypted embedding shape: {decrypted_embedding.shape}")
    
    # Verify
    is_equal = np.allclose(original_embedding, decrypted_embedding)
    print(f"Encryption/Decryption successful: {is_equal}")
    
    # Test text encryption
    original_text = "Sensitive criminal data"
    encrypted_text = encryptor.encrypt_text(original_text)
    decrypted_text = encryptor.decrypt_text(encrypted_text)
    print(f"\nText encryption test: {original_text == decrypted_text}")