import os

class Config:
    # 환경변수에서 API 키 읽기 (보안)
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_SECRET', '')
    
    # 거래소 설정
    EXCHANGE = 'binance'
    SYMBOL = 'BTC/USDT'
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']
    
    # 거래 설정 (처음엔 보수적으로)
    POSITION_SIZE = 0.05     # 5%만 사용
    STOP_LOSS = 0.02         # 2% 손절
    TAKE_PROFIT = 0.04       # 4% 익절
    MIN_CONFIDENCE = 0.7     # 높은 신뢰도만
    
    # 데이터 설정
    LOOKBACK_DAYS = 30
    PREDICTION_MINUTES = 15
    
    # 안전 설정
    SANDBOX = True           # 처음엔 반드시 True!
    MAX_DAILY_TRADES = 10    # 일일 최대 거래 횟수
