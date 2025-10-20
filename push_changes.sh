#!/bin/bash
echo "Pushing Railway deployment fixes to GitHub..."

# Add the key files
git add app.py
git add start_server.py  
git add Procfile
git add railway.json

# Commit changes
git commit -m "Fix Railway deployment: Add Python startup script for PORT handling

- Create start_server.py to properly handle PORT environment variable
- Update Procfile to use Python startup script instead of direct gunicorn
- Update railway.json to use Python startup script
- This should resolve the '$PORT' is not a valid port number error"

# Push to GitHub
git push origin main

echo "Changes pushed successfully!"
