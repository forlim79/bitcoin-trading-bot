#!/bin/bash
echo "🚀 비트코인 자동매매 봇 설치 시작..."

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python과 pip 설치
sudo apt install -y python3 python3-pip python3-venv git

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 필요한 디렉토리 생성
mkdir -p logs data backups

# 환경변수 파일 생성 (템플릿)
if [ ! -f .env ]; then
    cat > .env << EOF
# 바이낸스 API 설정
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here

# 실행 환경
ENVIRONMENT=sandbox
EOF
    echo "📝 .env 파일이 생성되었습니다. API 키를 설정하세요!"
fi

# 실행 권한 부여
chmod +x run.sh stop.sh monitor.sh

echo "✅ 설치 완료!"
echo ""
echo "📋 다음 단계:"
echo "1. nano .env        # API 키 설정"
echo "2. ./run.sh         # 봇 실행"
echo "3. ./monitor.sh     # 상태 확인"