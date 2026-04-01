#!/bin/bash

# Configuration
PORT=5001
WORKERS=1
LOG_FILE="app.log"
PID_FILE="app.pid"

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "=== Starting Chrono-Trader Web Server ==="

# 1. Install Gunicorn if not present
if ! python3 -c "import gunicorn" &> /dev/null; then
    echo "Installing gunicorn..."
    pip install gunicorn
fi

# 2. Kill existing processes (app.py or gunicorn)
echo "Stopping existing server processes..."
pkill -f "python3 app.py"
pkill -f "gunicorn.*app:app"

# Wait a moment for ports to free up
sleep 2

# 3. Start Gunicorn
echo "Starting Gunicorn on port $PORT with $WORKERS workers..."
# -w: workers
# -b: bind address
# --access-logfile: access logs
# --error-logfile: error logs
# --daemon: run in background (deprecated in some versions, better to use nohup)
# --timeout: worker timeout (increase for long ML tasks if running synchronously, though tasks are async via TaskRunner)

nohup gunicorn -k eventlet -w 1 -b 0.0.0.0:$PORT \
    --access-logfile $LOG_FILE \
    --error-logfile $LOG_FILE \
    --timeout 300 \
    app:app > /dev/null 2>&1 &

NEW_PID=$!
echo $NEW_PID > $PID_FILE

echo "Server started successfully (PID: $NEW_PID)."
echo "Logs are being written to $LOG_FILE"

# 4. Restart Nginx (if installed)
if command -v nginx &> /dev/null; then
    echo "Restarting Nginx reverse proxy..."
    sudo systemctl restart nginx
    echo "Nginx restarted."
fi
