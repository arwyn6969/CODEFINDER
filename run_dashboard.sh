#!/bin/bash

# Research Dashboard - One-Click Launcher
# =======================================

echo "🚀 Prophetic Research Dashboard Launcher"
echo "========================================"

# 1. Check Python Environment
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Running setup..."
    python3 scripts/setup_env.py
fi

# 2. Start Backend (Background)
echo "💎 Starting Backend API (Port 8000)..."
source .venv/bin/activate
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 3. Start Frontend
echo "🎨 Starting Frontend (Port 3000)..."
cd frontend
# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Dependencies..."
    npm install
fi
npm start &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

echo "✅ Systems are GO!"
echo "👉 Dashboard: http://localhost:3000"
echo "👉 API Docs:  http://localhost:8000/api/docs"
echo "Press CTRL+C to stop all services."

wait
