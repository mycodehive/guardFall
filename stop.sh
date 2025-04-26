#!/bin/bash

# 현재 실행 중인 Streamlit 앱 프로세스 확인
if [ -z "$PID" ]; then
    echo "There are no Streamlit apps running."
else
    echo "Streamlit app is closing... (PID: $PID)"
    kill "$PID"
    echo "The Streamlit app has been terminated."
    sleep 0.03
fi
