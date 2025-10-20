#!/bin/bash
# Railway startup script
echo "Starting application on port ${PORT:-3000}"
exec gunicorn --bind 0.0.0.0:${PORT:-3000} --workers 1 --timeout 120 app:app
