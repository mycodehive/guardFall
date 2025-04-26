#!/bin/bash

# 현재 실행 중인 Streamlit 앱 프로세스 확인
EXISTING_PID=$(pgrep -f "uv run -- streamlit run streamlit_app.py")

if [ -n "$EXISTING_PID" ]; then
    echo "Streamlit app is already running (PID: $EXISTING_PID)"
    exit 1
fi

# Streamlit 앱을 백그라운드로 실행
echo "Starting the Streamlit app in the background..."
nohup uv run -- streamlit run streamlit_app.py > output.log 2>&1 &

# 실행 안정화를 위한 대기
sleep 1
