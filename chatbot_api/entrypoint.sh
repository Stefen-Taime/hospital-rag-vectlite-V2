#!/bin/bash
set -e

echo "Starting chatbot API..."
uvicorn chatbot_api.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --reload
