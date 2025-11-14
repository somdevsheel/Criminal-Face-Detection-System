"""
Security Setup Script
Quick configuration of IP whitelist and authentication
"""

import sys
import os
import json
import hashlib

sys.path.insert(0, 'src')

from src.security import ip_whitelist, auth_system

def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")

def main():
    print_header("🔒 SECURITY SETUP - CRIMINAL FACE DETECTION SYSTEM")
    
    print("This wizard will help you configure security settings.")
    print()
    
    # Step 1: Change Admin Password
    print_header("STEP 1: Change Admin Password")
    print("⚠️  Default password is 'admin123' - CHANGE IT NOW!")
    print()
    
    change_pwd = input("Change admin password now? (y/n): ").strip().lower()
    
    if change_pwd == 'y':
        while True:
            new_pwd = input("Enter new admin password: ").strip()
            if len(new_pwd) < 8:
                print("❌ Password must be at least 8 characters")
                continue
            
            confirm_pwd = input("Confirm password: ").strip()
            if new_pwd != confirm_pwd:
                print("❌ Passwords don't match")
                continue
            
            # Update password
            auth_system.users['admin']['password'] = auth_system.hash_password(new_pwd)
            auth_system.save_users()
            print("✅ Admin password updated successfully!")
            break
    else:
        print("⚠️  Remember to change it later!")
    
    # Step 2: Add Your IP
    print_header("STEP 2: Add Your IP to Whitelist")
    print("Current whitelisted IPs:")
    for ip in ip_whitelist.whitelist:
        print(f"  ✅ {ip}")
    print()
    
    # Try to detect current IP
    try:
        import requests
        current_ip = requests.get('https://api.ipify.org').text
        print(f"Your detected IP: {current_ip}")
        print()
        
        add_current = input(f"Add {current_ip} to whitelist? (y/n): ").strip().lower()
        if add_current == 'y':
            ip_whitelist.add_to_whitelist(current_ip)
            print(f"✅ Added {current_ip} to whitelist")
    except:
        print("⚠️  Could not detect your IP automatically")
    
    # Manual IP entry
    while True:
        add_more = input("\nAdd another IP to whitelist? (y/n): ").strip().lower()
        if add_more != 'y':
            break
        
        ip = input("Enter IP address or CIDR range (e.g., 192.168.1.100 or 192.168.1.0/24): ").strip()
        if ip:
            ip_whitelist.add_to_whitelist(ip)
            print(f"✅ Added {ip} to whitelist")
    
    # Step 3: Network Configuration
    print_header("STEP 3: Network Configuration")
    print("Common setups:")
    print()
    print("1. Office Network - Add your office subnet")
    print("2. Home Network - Add your home router IP")
    print("3. VPN Access - Add VPN subnet")
    print("4. Skip - Configure later")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == '1':
        subnet = input("Enter office subnet (e.g., 192.168.1.0/24): ").strip()
        if subnet:
            ip_whitelist.add_to_whitelist(subnet)
            print(f"✅ Added office network: {subnet}")
    
    elif choice == '2':
        home_ip = input("Enter home IP address: ").strip()
        if home_ip:
            ip_whitelist.add_to_whitelist(home_ip)
            print(f"✅ Added home IP: {home_ip}")
    
    elif choice == '3':
        vpn_subnet = input("Enter VPN subnet (e.g., 10.8.0.0/24): ").strip()
        if vpn_subnet:
            ip_whitelist.add_to_whitelist(vpn_subnet)
            print(f"✅ Added VPN network: {vpn_subnet}")
    
    # Step 4: Summary
    print_header("SETUP COMPLETE!")
    
    print("✅ Security Configuration Summary:")
    print()
    print(f"📝 Admin password: {'Changed' if change_pwd == 'y' else '⚠️  Still default'}")
    print(f"📊 Whitelisted IPs: {len(ip_whitelist.whitelist)}")
    print(f"📊 Blacklisted IPs: {len(ip_whitelist.blacklist)}")
    print()
    
    print("Current Whitelist:")
    for ip in ip_whitelist.whitelist:
        print(f"  ✅ {ip}")
    print()
    
    print("Next Steps:")
    print("1. Start the server: python -m src.api.app")
    print("2. Access admin panel: http://localhost:5000/admin")
    print("3. Test access from whitelisted IPs")
    print("4. Review logs: logs/access.log")
    print()
    
    print("Security Files Created:")
    print("  📄 config/ip_whitelist.json - IP whitelist/blacklist")
    print("  📄 config/auth.json - User authentication")
    print("  📄 logs/access.log - Access logs (created on first access)")
    print()
    
    print("⚠️  Important:")
    print("  • Only whitelisted IPs can access the system")
    print("  • All access attempts are logged")
    print("  • 5 failed attempts = automatic blacklist")
    print("  • Backup your config files regularly")
    print()
    
    print("🔒 Your system is now secured!")
    print()
    
    # Option to start server
    start_now = input("Start the server now? (y/n): ").strip().lower()
    if start_now == 'y':
        print("\n🚀 Starting server...")
        print("Access: http://localhost:5000")
        print("Admin: http://localhost:5000/admin")
        print("\nPress Ctrl+C to stop\n")
        
        import subprocess
        subprocess.run([sys.executable, "-m", "src.api.app"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()