#!/usr/bin/env python3
"""
Poker Range Classification Web Application Runner
===============================================

This script starts the Flask web application for the Poker Range Classification System.
"""

import os
import sys
import webbrowser
from threading import Timer

def open_browser():
    """Open the web browser to the application"""
    webbrowser.open('http://localhost:5000')

def main():
    """Main function to run the web application"""
    
    print("=" * 60)
    print("POKER RANGE CLASSIFICATION WEB APPLICATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('poker_range_classifier.pkl'):
        print("⚠️  No trained model found!")
        print("   The web app will work, but you'll need to train a model first.")
        print("   You can train a model through the web interface or run:")
        print("   python main.py --generate-data --train")
        print()
    
    # Import and run Flask app
    try:
        from app import app
        
        print("🚀 Starting web application...")
        print("📱 The app will be available at: http://localhost:5000")
        print("🔄 Press Ctrl+C to stop the server")
        print()
        
        # Open browser after a short delay
        Timer(1.5, open_browser).start()
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to avoid duplicate browser windows
        )
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Web application stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
