# """
# Security Module - IP Whitelist & Authentication
# Implements IP-based access control and authentication
# """

# import logging
# from functools import wraps
# from flask import request, jsonify, render_template_string
# from datetime import datetime, timedelta
# import json
# import os
# import hashlib
# import secrets
# from typing import Optional
# import ipaddress

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class IPWhitelist:
#     """IP-based access control system"""

#     def __init__(self, whitelist_file: str = 'config/ip_whitelist.json'):
#         self.whitelist_file = whitelist_file
#         self.whitelist = []
#         self.blacklist = []
#         self.failed_attempts = {}
#         self.max_failed_attempts = 5
#         self.lockout_duration = 3600  # 1 hour

#         os.makedirs('config', exist_ok=True)
#         self.load_whitelist()
#         logger.info(f"IP Whitelist initialized with {len(self.whitelist)} allowed IPs")

#     def load_whitelist(self):
#         """Load whitelist from file"""
#         try:
#             if os.path.exists(self.whitelist_file):
#                 with open(self.whitelist_file, 'r') as file:
#                     data = json.load(file)
#                     self.whitelist = data.get('whitelist', [])
#                     self.blacklist = data.get('blacklist', [])
#                     if not self.whitelist:
#                         self.whitelist = [
#                             '127.0.0.1',
#                             'localhost',
#                             '::1'
#                         ]
#                         self.save_whitelist()
#             else:
#                 self.whitelist = [
#                     '127.0.0.1',
#                     'localhost',
#                     '::1',
#                     '192.168.1.0/24'
#                 ]
#                 self.save_whitelist()
#             logger.info(f"Loaded {len(self.whitelist)} whitelisted IPs")
#         except Exception as e:
#             logger.error(f"Error loading whitelist: {e}")
#             self.whitelist = ['127.0.0.1', 'localhost', '::1']

#     def save_whitelist(self):
#         """Save whitelist to file"""
#         try:
#             with open(self.whitelist_file, 'w') as file:
#                 json.dump({
#                     'whitelist': self.whitelist,
#                     'blacklist': self.blacklist,
#                     'last_updated': datetime.now().isoformat()
#                 }, file, indent=2)
#             logger.info("Whitelist saved")
#         except Exception as e:
#             logger.error(f"Error saving whitelist: {e}")

#     def is_ip_allowed(self, ip: str) -> bool:
#         """Check if IP address is allowed"""
#         if ip in self.blacklist:
#             logger.warning(f"Blocked blacklisted IP: {ip}")
#             return False
#         if self.is_locked_out(ip):
#             logger.warning(f"Blocked locked-out IP: {ip}")
#             return False

#         for allowed in self.whitelist:
#             try:
#                 if '/' in allowed:
#                     network = ipaddress.ip_network(allowed, strict=False)
#                     if ipaddress.ip_address(ip) in network:
#                         logger.info(f"Allowed IP {ip} (matches {allowed})")
#                         return True
#                 elif ip == allowed or (allowed == 'localhost' and ip in ['127.0.0.1', '::1']):
#                     logger.info(f"Allowed IP: {ip}")
#                     return True
#             except ValueError:
#                 continue

#         logger.warning(f"Denied IP (not in whitelist): {ip}")
#         self.record_failed_attempt(ip)
#         return False

#     def record_failed_attempt(self, ip: str):
#         """Record failed access attempt"""
#         self.failed_attempts.setdefault(ip, [])
#         self.failed_attempts[ip].append(datetime.now())

#         cutoff = datetime.now() - timedelta(seconds=self.lockout_duration)
#         self.failed_attempts[ip] = [a for a in self.failed_attempts[ip] if a > cutoff]

#         if len(self.failed_attempts[ip]) >= self.max_failed_attempts:
#             self.add_to_blacklist(ip)
#             logger.warning(f"IP {ip} auto-blacklisted after too many attempts")

#     def is_locked_out(self, ip: str) -> bool:
#         """Check if IP is temporarily locked out"""
#         if ip not in self.failed_attempts:
#             return False
#         cutoff = datetime.now() - timedelta(seconds=self.lockout_duration)
#         self.failed_attempts[ip] = [a for a in self.failed_attempts[ip] if a > cutoff]
#         return len(self.failed_attempts[ip]) >= self.max_failed_attempts

#     def add_to_whitelist(self, ip: str) -> bool:
#         if ip not in self.whitelist:
#             self.whitelist.append(ip)
#             self.save_whitelist()
#             logger.info(f"Added to whitelist: {ip}")
#             return True
#         return False

#     def add_to_blacklist(self, ip: str) -> bool:
#         if ip not in self.blacklist:
#             self.blacklist.append(ip)
#             self.save_whitelist()
#             logger.info(f"Added to blacklist: {ip}")
#             return True
#         return False

#     def get_client_ip(self, request_obj) -> str:
#         """Get client IP address from Flask request"""
#         if request_obj.headers.get('X-Forwarded-For'):
#             return request_obj.headers.get('X-Forwarded-For').split(',')[0].strip()
#         elif request_obj.headers.get('X-Real-IP'):
#             return request_obj.headers.get('X-Real-IP')
#         return request_obj.remote_addr


# class AuthenticationSystem:
#     """Simple authentication for admin access"""

#     def __init__(self, auth_file: str = 'config/auth.json'):
#         self.auth_file = auth_file
#         self.users = {}
#         self.sessions = {}
#         self.session_duration = 3600
#         os.makedirs('config', exist_ok=True)
#         self.load_users()

#     def load_users(self):
#         try:
#             if os.path.exists(self.auth_file):
#                 with open(self.auth_file, 'r') as file:
#                     self.users = json.load(file)
#             else:
#                 default_password = self.hash_password('admin123')
#                 self.users = {
#                     'admin': {
#                         'password': default_password,
#                         'role': 'admin',
#                         'created': datetime.now().isoformat()
#                     }
#                 }
#                 self.save_users()
#                 logger.warning("Created default admin user (username: admin, password: admin123)")
#         except Exception as e:
#             logger.error(f"Error loading users: {e}")

#     def save_users(self):
#         with open(self.auth_file, 'w') as file:
#             json.dump(self.users, file, indent=2)

#     def hash_password(self, password: str) -> str:
#         return hashlib.sha256(password.encode()).hexdigest()

#     def verify_password(self, username: str, password: str) -> bool:
#         if username not in self.users:
#             return False
#         return self.users[username]['password'] == self.hash_password(password)

#     def create_session(self, username: str) -> str:
#         token = secrets.token_urlsafe(32)
#         self.sessions[token] = {
#             'username': username,
#             'created': datetime.now(),
#             'expires': datetime.now() + timedelta(seconds=self.session_duration)
#         }
#         return token

#     def verify_session(self, token: str) -> Optional[str]:
#         session = self.sessions.get(token)
#         if not session or datetime.now() > session['expires']:
#             self.sessions.pop(token, None)
#             return None
#         return session['username']


# # Global instances
# ip_whitelist = IPWhitelist()
# auth_system = AuthenticationSystem()


# def require_ip_whitelist(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         client_ip = ip_whitelist.get_client_ip(request)
#         if not ip_whitelist.is_ip_allowed(client_ip):
#             logger.warning(f"Access denied for IP: {client_ip}")
#             return render_template_string(
#                 "<h1>Access Denied</h1><p>Your IP {{ ip }} is not authorized.</p>", ip=client_ip
#             ), 403
#         return f(*args, **kwargs)
#     return decorated


# def require_authentication(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.cookies.get('session_token')
#         if not token or not auth_system.verify_session(token):
#             return jsonify({'success': False, 'error': 'Authentication required'}), 401
#         return f(*args, **kwargs)
#     return decorated


# def log_access(f):
#     """Decorator to log access attempts"""
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         client_ip = ip_whitelist.get_client_ip(request)
#         log_entry = {
#             'timestamp': datetime.now().isoformat(),
#             'ip': client_ip,
#             'method': request.method,
#             'path': request.path,
#             'user_agent': request.user_agent.string
#         }
#         os.makedirs('logs', exist_ok=True)
#         log_path = 'logs/access.log'
#         with open(log_path, 'a') as log_file:
#             log_file.write(json.dumps(log_entry) + '\n')
#         return f(*args, **kwargs)
#     return decorated


# def get_security_status() -> dict:
#     return {
#         'whitelist_count': len(ip_whitelist.whitelist),
#         'blacklist_count': len(ip_whitelist.blacklist),
#         'active_sessions': len(auth_system.sessions),
#         'failed_attempts': sum(len(v) for v in ip_whitelist.failed_attempts.values()),
#         'whitelist': ip_whitelist.whitelist,
#         'blacklist': ip_whitelist.blacklist
#     }


# if __name__ == "__main__":
#     print("Testing IP Whitelist Security System")
#     print("=" * 60)
#     test_ips = ['127.0.0.1', '192.168.1.100', '8.8.8.8']
#     for ip in test_ips:
#         print(f"{ip}: {'✓ Allowed' if ip_whitelist.is_ip_allowed(ip) else '✗ Denied'}")
#     print("\nWhitelist:", ip_whitelist.whitelist)
#     print("Blacklist:", ip_whitelist.blacklist)
#     print("\nAuthentication test:", auth_system.verify_password('admin', 'admin123'))


"""
Security Module - IP Whitelist & Authentication
Implements IP-based access control and authentication
"""

import logging
from functools import wraps
from flask import request, jsonify, render_template_string
from datetime import datetime, timedelta
import json
import os
import hashlib
import secrets
from typing import List, Optional
import ipaddress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPWhitelist:
    """
    IP-based access control system
    Allows only whitelisted IP addresses to access the application
    """
    
    def __init__(self, whitelist_file: str = 'config/ip_whitelist.json'):
        self.whitelist_file = whitelist_file
        self.whitelist = []
        self.blacklist = []
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 3600  # seconds
        os.makedirs('config', exist_ok=True)
        self.load_whitelist()
        logger.info(f"IP Whitelist initialized with {len(self.whitelist)} allowed IPs")

    def load_whitelist(self):
        """Load whitelist from file"""
        try:
            if os.path.exists(self.whitelist_file):
                with open(self.whitelist_file, 'r') as f:
                    data = json.load(f)
                    self.whitelist = data.get('whitelist', [])
                    self.blacklist = data.get('blacklist', [])
            else:
                self.whitelist = ['127.0.0.1', 'localhost', '::1', '192.168.1.0/24']
                self.save_whitelist()
        except Exception as e:
            logger.error(f"Error loading whitelist: {e}")
            self.whitelist = ['127.0.0.1']

    def save_whitelist(self):
        """Save whitelist and blacklist"""
        try:
            with open(self.whitelist_file, 'w') as f:
                json.dump({
                    'whitelist': self.whitelist,
                    'blacklist': self.blacklist,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
            logger.info("Whitelist saved")
        except Exception as e:
            logger.error(f"Error saving whitelist: {e}")

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed"""
        if ip in self.blacklist:
            logger.warning(f"Blocked blacklisted IP: {ip}")
            return False
        for allowed in self.whitelist:
            try:
                if '/' in allowed:
                    if ipaddress.ip_address(ip) in ipaddress.ip_network(allowed, strict=False):
                        return True
                elif ip == allowed or (allowed == 'localhost' and ip in ['127.0.0.1', '::1']):
                    return True
            except ValueError:
                continue
        logger.warning(f"Denied IP (not in whitelist): {ip}")
        return False

    def add_to_whitelist(self, ip: str) -> bool:
        """Add IP"""
        if ip not in self.whitelist:
            self.whitelist.append(ip)
            self.save_whitelist()
            return True
        return False

    def remove_from_whitelist(self, ip: str) -> bool:
        """Remove IP"""
        if ip in self.whitelist:
            self.whitelist.remove(ip)
            self.save_whitelist()
            return True
        return False

    def add_to_blacklist(self, ip: str) -> bool:
        """Add IP to blacklist"""
        if ip not in self.blacklist:
            self.blacklist.append(ip)
            self.save_whitelist()
            return True
        return False

    def remove_from_blacklist(self, ip: str) -> bool:
        """Remove IP from blacklist"""
        if ip in self.blacklist:
            self.blacklist.remove(ip)
            self.save_whitelist()
            return True
        return False

    def get_client_ip(self, request_obj):
        """Get real client IP"""
        if request_obj.headers.get('X-Forwarded-For'):
            return request_obj.headers.get('X-Forwarded-For').split(',')[0].strip()
        return request_obj.remote_addr


class AuthenticationSystem:
    """Simple authentication for admin access"""
    def __init__(self, auth_file: str = 'config/auth.json'):
        self.auth_file = auth_file
        self.users = {}
        self.sessions = {}
        self.session_duration = 3600
        os.makedirs('config', exist_ok=True)
        self.load_users()

    def load_users(self):
        if os.path.exists(self.auth_file):
            with open(self.auth_file, 'r') as f:
                self.users = json.load(f)
        else:
            default_password = self.hash_password('admin123')
            self.users = {'admin': {'password': default_password, 'role': 'admin'}}
            self.save_users()
            logger.warning("Default admin user created (admin/admin123)")

    def save_users(self):
        with open(self.auth_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def hash_password(self, password: str):
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, username: str, password: str):
        if username not in self.users:
            return False
        return self.users[username]['password'] == self.hash_password(password)

    def create_session(self, username: str):
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            'username': username,
            'expires': datetime.now() + timedelta(seconds=self.session_duration)
        }
        return token

    def verify_session(self, token: str):
        if token not in self.sessions:
            return None
        if datetime.now() > self.sessions[token]['expires']:
            del self.sessions[token]
            return None
        return self.sessions[token]['username']


# Global instances
ip_whitelist = IPWhitelist()
auth_system = AuthenticationSystem()


def get_security_status():
    """Return current whitelist/blacklist info"""
    return {
        'whitelist': ip_whitelist.whitelist,
        'blacklist': ip_whitelist.blacklist,
        'whitelist_count': len(ip_whitelist.whitelist),
        'blacklist_count': len(ip_whitelist.blacklist)
    }
