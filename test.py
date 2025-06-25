#!/usr/bin/env python3
"""
연결 테스트 스크립트
각 서비스별 연결 상태를 확인합니다.
"""

import os
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """환경변수 확인"""
    logger.info("=== 환경변수 확인 ===")
    
    required_vars = [
        'UPBIT_ACCESS_KEY',
        'UPBIT_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID',
        'GOOGLE_CREDENTIALS_JSON',
        'GOOGLE_SPREADSHEET_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == 'GOOGLE_CREDENTIALS_JSON':
                logger.info(f"✓ {var}: {'설정됨 (길이: ' + str(len(value)) + ')'}")
                # JSON 형태 확인
                try:
                    if isinstance(value, str):
                        json.loads(value)
                        logger.info(f"  - JSON 형태: 문자열 (파싱 가능)")
                    else:
                        logger.info(f"  - JSON 형태: {type(value).__name__}")
                except json.JSONDecodeError as e:
                    logger.error(f"  - JSON 파싱 오류: {e}")
            else:
                logger.info(f"✓ {var}: 설정됨")
        else:
            logger.error(f"✗ {var}: 설정되지 않음")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def test_upbit_connection():
    """업비트 연결 테스트"""
    logger.info("=== 업비트 연결 테스트 ===")
    
    try:
        import pyupbit
        
        access_key = os.getenv('UPBIT_ACCESS_KEY')
        secret_key = os.getenv('UPBIT_SECRET_KEY')
        
        if not access_key or not secret_key:
            logger.error("업비트 API 키가 설정되지 않았습니다.")
            return False
        
        # 업비트 객체 생성
        upbit = pyupbit.Upbit(access_key, secret_key)
        
        # 잔고 조회 테스트
        balances = upbit.get_balances()
        logger.info(f"✓ 업비트 연결 성공 - 보유 자산 수: {len(balances) if balances else 0}")
        
        # 현재 비트코인 가격 조회
        btc_price = pyupbit.get_current_price("KRW-BTC")
        logger.info(f"✓ 현재 BTC 가격: {btc_price:,} KRW")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 업비트 연결 실패: {e}")
        return False

def test_telegram_connection():
    """텔레그램 연결 테스트"""
    logger.info("=== 텔레그램 연결 테스트 ===")
    
    try:
        import requests
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.error("텔레그램 설정이 완료되지 않았습니다.")
            return False
        
        # 봇 정보 확인
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            logger.info(f"✓ 텔레그램 봇 연결 성공 - 봇명: {bot_info['result']['username']}")
            
            # 테스트 메시지 발송
            test_message = f"🔧 연결 테스트 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            response = requests.post(send_url, json={
                'chat_id': chat_id,
                'text': test_message
            }, timeout=10)
            
            if response.status_code == 200:
                logger.info("✓ 텔레그램 메시지 발송 성공")
                return True
            else:
                logger.error(f"✗ 텔레그램 메시지 발송 실패: {response.text}")
                return False
        else:
            logger.error(f"✗ 텔레그램 봇 정보 조회 실패: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"✗ 텔레그램 연결 실패: {e}")
        return False

def test_google_sheets_connection():
    """구글 시트 연결 테스트"""
    logger.info("=== 구글 시트 연결 테스트 ===")
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # 환경변수에서 구글 인증 정보 가져오기
        google_creds_str = os.getenv('GOOGLE_CREDENTIALS_JSON')
        spreadsheet_id = os.getenv('GOOGLE_SPREADSHEET_ID')
        
        if not google_creds_str or not spreadsheet_id:
            logger.error("구글 시트 설정이 완료되지 않았습니다.")
            return False
        
        # JSON 파싱 (문자열인지 딕셔너리인지 확인)
        if isinstance(google_creds_str, str):
            google_creds = json.loads(google_creds_str)
            logger.info("구글 인증 정보: 문자열에서 파싱됨")
        else:
            google_creds = google_creds_str
            logger.info(f"구글 인증 정보: {type(google_creds_str).__name__} 타입")
        
        # 인증 설정
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_info(google_creds, scopes=scope)
        gc = gspread.authorize(credentials)
        
        # 스프레드시트 열기
        spreadsheet = gc.open_by_key(spreadsheet_id)
        logger.info(f"✓ 구글 시트 연결 성공 - 제목: {spreadsheet.title}")
        
        # 워크시트 목록 확인
        worksheets = spreadsheet.worksheets()
        logger.info(f"✓ 워크시트 수: {len(worksheets)}")
        for ws in worksheets:
            logger.info(f"  - {ws.title}")
        
        # 테스트 데이터 쓰기 (첫 번째 워크시트)
        if worksheets:
            ws = worksheets[0]
            test_data = [
                "연결 테스트",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "성공"
            ]
            ws.append_row(test_data)
            logger.info("✓ 테스트 데이터 쓰기 성공")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 구글 시트 연결 실패: {e}")
        return False

def test_aws_connection():
    """AWS 연결 테스트"""
    logger.info("=== AWS 연결 테스트 ===")
    
    try:
        import boto3
        
        # S3 클라이언트 생성
        s3_client = boto3.client('s3')
        
        # 버킷 목록 조회
        response = s3_client.list_buckets()
        bucket_count = len(response['Buckets'])
        
        logger.info(f"✓ AWS S3 연결 성공 - 버킷 수: {bucket_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AWS 연결 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 비트코인 트레이딩 봇 연결 테스트 시작")
    logger.info("=" * 50)
    
    results = {}
    
    # 각 테스트 실행
    results['환경변수'] = test_environment_variables()
    results['업비트'] = test_upbit_connection()
    results['텔레그램'] = test_telegram_connection()
    results['구글시트'] = test_google_sheets_connection()
    results['AWS'] = test_aws_connection()
    
    # 결과 요약
    logger.info("=" * 50)
    logger.info("📊 연결 테스트 결과 요약")
    logger.info("=" * 50)
    
    for service, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        logger.info(f"{service}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\n전체 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        logger.info("🎉 모든 연결 테스트가 성공했습니다!")
    else:
        logger.warning("⚠️  일부 연결에 문제가 있습니다. 위의 오류 메시지를 확인하세요.")

if __name__ == "__main__":
    main()
