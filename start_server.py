#!/usr/bin/env python3
"""
Railway startup script for poker range classifier
Handles PORT environment variable properly
"""
import os
import sys
import subprocess

def main():
    # Get port from environment variable
    port = os.environ.get('PORT', '3000')
    
    print(f"Starting poker range classifier on port {port}")
    print(f"Environment variables: PORT={port}")
    
    # Validate port is a number
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            raise ValueError(f"Port {port_int} is out of valid range (1-65535)")
    except ValueError as e:
        print(f"Error: Invalid port '{port}': {e}")
        print("Using default port 3000")
        port = '3000'
    
    # Build gunicorn command
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '1',
        '--timeout', '120',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'app:app'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Start the server
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)

if __name__ == '__main__':
    main()
