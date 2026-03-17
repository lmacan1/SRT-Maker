#!/bin/bash
# SRT-Maker - Start script

cd "$(dirname "$0")"

PORT=${PORT:-8765}
HOST=${HOST:-0.0.0.0}

echo "Starting SRT-Maker on http://$HOST:$PORT"

# Check if --tunnel flag is passed for quick Cloudflare tunnel
if [[ "$1" == "--tunnel" ]]; then
    echo "Starting with Cloudflare Quick Tunnel..."
    # Start uvicorn in background
    ./venv/bin/python -m uvicorn main:app --host "$HOST" --port "$PORT" &
    UVICORN_PID=$!
    sleep 2

    # Start cloudflared tunnel
    cloudflared tunnel --url "http://localhost:$PORT"

    # Cleanup on exit
    kill $UVICORN_PID 2>/dev/null
else
    # Standard local run
    ./venv/bin/python -m uvicorn main:app --host "$HOST" --port "$PORT"
fi
