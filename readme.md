# 🤖 Bitcoin Auto Trading Bot

AI 기반 비트코인 자동매매 봇입니다.

## 🚀 빠른 시작

1. 설치
\`\`\`bash
git clone https://github.com/your-username/bitcoin-trading-bot.git
cd bitcoin-trading-bot
./setup.sh
\`\`\`

2. API 키 설정
\`\`\`bash
nano .env
# BINANCE_API_KEY와 BINANCE_SECRET 입력
\`\`\`

3. 실행
\`\`\`bash
./run.sh
\`\`\`

## 📋 주요 명령어

- \`./run.sh\` - 봇 시작
- \`./stop.sh\` - 봇 중지  
- \`./monitor.sh\` - 상태 확인
- \`tail -f logs/trading_*.log\` - 실시간 로그

## ⚠️ 주의사항

- 처음엔 반드시 SANDBOX=True로 테스트
- 소액으로 시작하여 성능 확인 후 확장
- API 키는 절대 공개하지 마세요