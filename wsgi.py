#!/usr/bin/env python3
"""
WSGI entry point for production deployment
"""

from app import app

# This is what Render needs - the app object
application = app
app = application

if __name__ == "__main__":
    app.run()
