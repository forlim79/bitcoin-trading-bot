#!/bin/bash
echo "🛑 비트코인 자동매매 봇 중지..."

# PID 파일에서 프로세스 ID 읽기
if [ -f bot.pid ]; then
    PID=$(cat bot.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✅ 봇이 중지되었습니다. (PID: $PID)"
    else
        echo "⚠️ 해당 프로세스가 실행 중이 아닙니다."
    fi
    rm -f bot.pid
else
    # 일반적인 방법으로 중지
    pkill -f "python3 main.py"
    echo "✅ 모든 관련 프로세스를 중지했습니다."
fi

echo "📊 현재 실행 중인 Python 프로세스:"
ps aux | grep python3 | grep -v grep