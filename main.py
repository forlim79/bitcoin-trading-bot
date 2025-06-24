import json
import os
import time
import hashlib
import hmac
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import tweepy
import praw
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class EnhancedBitcoinTradingBot:
    def __init__(self):
        # API 키 설정
        self.upbit_access_key = os.environ['UPBIT_ACCESS_KEY']
        self.upbit_secret_key = os.environ['UPBIT_SECRET_KEY']
        self.google_credentials_json = os.environ['GOOGLE_SERVICE_ACCOUNT_JSON']
        self.spreadsheet_id = os.environ['GOOGLE_SPREADSHEET_ID']
        
        # 소셜 미디어 API 키
        self.twitter_bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        
        # 설정 - 개선된 매매 파라미터
        self.symbol = "KRW-BTC"
        self.lookback_days = 60
        self.lstm_sequence_length = 10
        
        # 수수료 및 매매 설정
        self.trading_fee = 0.0005  # 0.05% 수수료
        self.min_trade_amount = 5000  # 최소 거래 금액 (업비트 기준)
        self.max_portfolio_ratio = 0.1  # 한 번에 투자할 수 있는 최대 비율 (10%)
        self.min_portfolio_ratio = 0.02  # 최소 투자 비율 (2%)
        self.take_profit_ratio = 0.3  # 수익 실현 비율 (30%)
        self.stop_loss_ratio = 0.5  # 손절 비율 (50%)
        
        # 최소 수익률 임계값 (수수료 고려)
        self.min_profit_threshold = self.trading_fee * 3  # 수수료의 3배 (0.15%)
        
        # 모델 가중치 (앙상블용)
        self.model_weights = {
            'technical': 0.4,
            'lstm': 0.3,
            'sentiment': 0.2,
            'market': 0.1
        }
        
        self.setup_google_sheets()
        self.setup_social_media_clients()
    
    def setup_google_sheets(self):
        """Google Sheets API 클라이언트 설정"""
        credentials_info = json.loads(self.google_credentials_json)
        credentials = Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.sheets_service = build('sheets', 'v4', credentials=credentials)
    
    def setup_social_media_clients(self):
        """소셜 미디어 클라이언트 설정"""
        if self.twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
        
        if self.reddit_client_id and self.reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent="bitcoin_bot/1.0"
            )
    
    def get_upbit_headers(self, query_string=''):
        """업비트 API 인증 헤더 생성"""
        payload = {
            'access_key': self.upbit_access_key,
            'nonce': str(int(time.time() * 1000))
        }
        
        if query_string:
            payload['query_hash'] = hashlib.sha512(query_string.encode()).hexdigest()
            payload['query_hash_alg'] = 'SHA512'
        
        jwt_token = base64.b64encode(json.dumps(payload).encode()).decode()
        signature = hmac.new(
            self.upbit_secret_key.encode(),
            jwt_token.encode(),
            hashlib.sha512
        ).hexdigest()
        
        return {
            'Authorization': f'Bearer {jwt_token}.{signature}',
            'Content-Type': 'application/json'
        }
    
    def get_market_data(self, interval='days'):
        """시장 데이터 수집 (다양한 시간프레임)"""
        if interval == 'days':
            url = "https://api.upbit.com/v1/candles/days"
            count = self.lookback_days
        elif interval == 'hours':
            url = "https://api.upbit.com/v1/candles/minutes/60"
            count = self.lookback_days * 24
        else:
            url = "https://api.upbit.com/v1/candles/minutes/15"
            count = self.lookback_days * 96
        
        params = {'market': self.symbol, 'count': count}
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['candle_date_time_kst'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def calculate_portfolio_value(self):
        """현재 포트폴리오 가치 계산"""
        try:
            # 잔고 조회
            balance_url = "https://api.upbit.com/v1/accounts"
            balance_response = requests.get(balance_url, headers=self.get_upbit_headers())
            balances = balance_response.json()
            
            # 현재 가격 조회
            ticker_url = "https://api.upbit.com/v1/ticker"
            ticker_response = requests.get(ticker_url, params={'markets': self.symbol})
            current_price = ticker_response.json()[0]['trade_price']
            
            krw_balance = 0
            btc_balance = 0
            
            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_balance = float(balance['balance'])
                elif balance['currency'] == 'BTC':
                    btc_balance = float(balance['balance'])
            
            btc_value = btc_balance * current_price
            total_value = krw_balance + btc_value
            
            return {
                'krw_balance': krw_balance,
                'btc_balance': btc_balance,
                'btc_value': btc_value,
                'total_value': total_value,
                'current_price': current_price
            }
        
        except Exception as e:
            print(f"포트폴리오 조회 오류: {e}")
            return None
    
    def calculate_dynamic_trade_amount(self, portfolio_data, signal_strength):
        """신호 강도와 포트폴리오 비율에 따른 동적 거래 금액 계산"""
        total_value = portfolio_data['total_value']
        
        # 신호 강도에 따른 비율 조정 (1-10 범위를 0.02-0.1로 매핑)
        ratio = self.min_portfolio_ratio + (signal_strength - 1) * (self.max_portfolio_ratio - self.min_portfolio_ratio) / 9
        
        trade_amount = total_value * ratio
        
        # 최소 거래 금액 확인
        if trade_amount < self.min_trade_amount:
            trade_amount = self.min_trade_amount
        
        return trade_amount
    
    def get_market_dominance_data(self):
        """전체 시장 데이터 및 비트코인 도미넌스 수집"""
        try:
            # CoinGecko API 사용 (무료)
            dominance_url = "https://api.coingecko.com/api/v3/global"
            market_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            
            # 도미넌스 데이터
            dom_response = requests.get(dominance_url)
            dom_data = dom_response.json()
            btc_dominance = dom_data['data']['market_cap_percentage']['btc']
            total_market_cap = dom_data['data']['total_market_cap']['usd']
            
            # 시장 공포/탐욕 지수 (Alternative.me API)
            fear_greed_url = "https://api.alternative.me/fng/"
            fg_response = requests.get(fear_greed_url)
            fg_data = fg_response.json()
            fear_greed_index = int(fg_data['data'][0]['value'])
            
            return {
                'btc_dominance': btc_dominance,
                'total_market_cap': total_market_cap,
                'fear_greed_index': fear_greed_index
            }
        except Exception as e:
            print(f"시장 데이터 수집 오류: {e}")
            return {'btc_dominance': 50, 'total_market_cap': 2000000000000, 'fear_greed_index': 50}
    
    def create_advanced_technical_indicators(self, df):
        """고급 기술적 지표 생성"""
        # 기본 이동평균
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['trade_price'].rolling(period).mean()
        
        # 지수이동평균
        for period in [12, 26]:
            df[f'ema_{period}'] = df['trade_price'].ewm(span=period).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI (14일)
        delta = df['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 스토캐스틱
        low_14 = df['low_price'].rolling(14).min()
        high_14 = df['high_price'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['trade_price'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # 볼린저 밴드
        df['bb_middle'] = df['trade_price'].rolling(20).mean()
        bb_std = df['trade_price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['trade_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['tr1'] = df['high_price'] - df['low_price']
        df['tr2'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['tr3'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # 거래량 지표
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(20).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # 가격 변화율
        for period in [1, 3, 7, 14]:
            df[f'price_change_{period}'] = df['trade_price'].pct_change(period)
        
        return df
    
    def get_social_sentiment(self):
        """소셜 미디어 감정 분석"""
        sentiment_scores = []
        
        try:
            # 트위터 감정 분석
            if hasattr(self, 'twitter_client'):
                tweets = self.twitter_client.search_recent_tweets(
                    query="bitcoin OR BTC -is:retweet lang:en",
                    max_results=100
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        blob = TextBlob(tweet.text)
                        sentiment_scores.append(blob.sentiment.polarity)
            
            # 레딧 감정 분석
            if hasattr(self, 'reddit'):
                subreddit = self.reddit.subreddit('Bitcoin')
                hot_posts = subreddit.hot(limit=50)
                
                for post in hot_posts:
                    blob = TextBlob(post.title + " " + post.selftext)
                    sentiment_scores.append(blob.sentiment.polarity)
        
        except Exception as e:
            print(f"소셜 미디어 분석 오류: {e}")
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_volatility': sentiment_volatility,
                'sample_size': len(sentiment_scores)
            }
        else:
            return {'sentiment_score': 0, 'sentiment_volatility': 0, 'sample_size': 0}
    
    def prepare_lstm_data(self, df, target_col='trade_price'):
        """LSTM 모델용 데이터 준비"""
        # 특성 선택
        feature_cols = [
            'trade_price', 'ma_5', 'ma_20', 'rsi', 'macd', 'stoch_k', 
            'bb_position', 'atr', 'volume_ratio'
        ]
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols].dropna())
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(self.lstm_sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.lstm_sequence_length:i])
            y.append(scaled_data[i, 0])  # 가격 예측
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape):
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_models(self, df, market_data, sentiment_data):
        """다중 모델 훈련"""
        predictions = {}
        
        # 1. 기술적 분석 모델 (Random Forest)
        tech_features = [
            'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal', 
            'stoch_k', 'stoch_d', 'bb_position', 'bb_width', 'atr', 'volume_ratio'
        ]
        
        df_clean = df.dropna()
        if len(df_clean) > 30:
            X_tech = df_clean[tech_features].iloc[:-1]
            y_tech = df_clean['trade_price'].iloc[1:]
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_tech, y_tech)
            
            # 최신 데이터로 예측
            latest_features = df_clean[tech_features].iloc[-1:].values
            tech_prediction = rf_model.predict(latest_features)[0]
            predictions['technical'] = tech_prediction
        
        # 2. LSTM 모델
        try:
            X_lstm, y_lstm, scaler = self.prepare_lstm_data(df_clean)
            if len(X_lstm) > 20:
                lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
                lstm_model.fit(X_lstm[:-5], y_lstm[:-5], epochs=50, batch_size=32, verbose=0)
                
                # 예측
                last_sequence = X_lstm[-1].reshape(1, X_lstm.shape[1], X_lstm.shape[2])
                lstm_pred_scaled = lstm_model.predict(last_sequence)[0][0]
                
                # 스케일 복원
                dummy_array = np.zeros((1, X_lstm.shape[2]))
                dummy_array[0, 0] = lstm_pred_scaled
                lstm_prediction = scaler.inverse_transform(dummy_array)[0, 0]
                predictions['lstm'] = lstm_prediction
        except Exception as e:
            print(f"LSTM 모델 오류: {e}")
            predictions['lstm'] = df_clean['trade_price'].iloc[-1]
        
        # 3. 감정 분석 기반 예측
        current_price = df_clean['trade_price'].iloc[-1]
        sentiment_multiplier = 1 + (sentiment_data['sentiment_score'] * 0.05)  # 5% 최대 영향
        predictions['sentiment'] = current_price * sentiment_multiplier
        
        # 4. 시장 데이터 기반 예측
        market_multiplier = 1 + ((market_data['fear_greed_index'] - 50) / 1000)  # 시장 공포/탐욕 지수 반영
        predictions['market'] = current_price * market_multiplier
        
        return predictions
    
    def make_ensemble_prediction(self, predictions):
        """앙상블 예측"""
        if not predictions:
            return None
        
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                weighted_sum += prediction * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return None
    
    def generate_trading_signal(self, df, predictions, market_data, sentiment_data, portfolio_data):
        """개선된 거래 신호 생성"""
        current_price = df['trade_price'].iloc[-1]
        ensemble_prediction = self.make_ensemble_prediction(predictions)
        
        if ensemble_prediction is None:
            return "HOLD", 0
        
        # 예측 가격 변화율
        price_change_pct = (ensemble_prediction - current_price) / current_price
        
        # 수수료를 고려한 최소 수익률 확인
        if abs(price_change_pct) < self.min_profit_threshold:
            return "HOLD", 0
        
        # 추가 조건들
        rsi = df['rsi'].iloc[-1]
        macd_histogram = df['macd_histogram'].iloc[-1]
        bb_position = df['bb_position'].iloc[-1]
        fear_greed = market_data['fear_greed_index']
        sentiment_score = sentiment_data['sentiment_score']
        
        # 포트폴리오 비율 확인
        btc_ratio = portfolio_data['btc_value'] / portfolio_data['total_value'] if portfolio_data['total_value'] > 0 else 0
        
        # 매수 조건 (수수료 고려하여 더 엄격하게)
        buy_conditions = [
            price_change_pct > self.min_profit_threshold * 2,  # 수수료의 2배 이상 상승 예측
            rsi < 65,  # RSI 과매수 아님 (더 보수적)
            bb_position < 0.75,  # 볼린저 밴드 상단 근처 아님
            fear_greed < 75,  # 극도의 탐욕 아님
            sentiment_score > -0.2,  # 극도로 부정적이지 않음
            btc_ratio < 0.7,  # BTC 비중이 70% 미만
            portfolio_data['krw_balance'] >= self.min_trade_amount  # 충분한 KRW 잔고
        ]
        
        # 매도 조건 (부분 매도 고려)
        sell_conditions = [
            price_change_pct < -self.min_profit_threshold * 2,  # 수수료의 2배 이상 하락 예측
            rsi > 35,  # RSI 과매도 아님 (더 보수적)
            bb_position > 0.25,  # 볼린저 밴드 하단 근처 아님
            fear_greed > 25,  # 극도의 공포 아님  
            sentiment_score < 0.2,  # 극도로 긍정적이지 않음
            btc_ratio > 0.1,  # BTC 보유 중
            portfolio_data['btc_balance'] > 0  # BTC 잔고 보유
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # 신뢰도 계산 (수수료 고려)
        confidence = min(abs(price_change_pct) * 50, 10)  # 0-10 범위
        
        if buy_score >= 6 and confidence > 2:
            return "BUY", confidence
        elif sell_score >= 6 and confidence > 2:
            return "SELL", confidence
        else:
            return "HOLD", confidence
    
    def execute_trade(self, signal, confidence, portfolio_data):
        """개선된 거래 실행"""
        if signal == "HOLD":
            return {"status": "hold", "message": "거래 신호 없음"}
        
        try:
            current_price = portfolio_data['current_price']
            
            if signal == "BUY":
                # 동적 거래 금액 계산
                trade_amount = self.calculate_dynamic_trade_amount(portfolio_data, confidence)
                
                # 수수료 고려한 실제 투자 금액
                trade_amount_after_fee = trade_amount * (1 - self.trading_fee)
                
                if portfolio_data['krw_balance'] >= trade_amount:
                    # 매수 주문
                    order_data = {
                        'market': self.symbol,
                        'side': 'bid',
                        'price': str(int(trade_amount)),
                        'ord_type': 'price'
                    }
                    
                    query_string = '&'.join([f"{k}={v}" for k, v in order_data.items()])
                    order_url = "https://api.upbit.com/v1/orders"
                    
                    order_response = requests.post(
                        order_url,
                        json=order_data,
                        headers=self.get_upbit_headers(query_string)
                    )
                    
                    result = order_response.json()
                    return {
                        "status": "buy_executed",
                        "order_id": result.get('uuid'),
                        "price": current_price,
                        "amount": trade_amount,
                        "amount_after_fee": trade_amount_after_fee,
                        "confidence": confidence,
                        "portfolio_ratio": trade_amount / portfolio_data['total_value']
                    }
                else:
                    return {"status": "insufficient_krw_balance", "message": f"KRW 잔고 부족: {portfolio_data['krw_balance']:.0f} < {trade_amount:.0f}"}
            
            elif signal == "SELL":
                # 부분 매도 (신뢰도에 따라 비율 조정)
                sell_ratio = min(confidence / 10 * self.take_profit_ratio, self.take_profit_ratio)
                sell_volume = portfolio_data['btc_balance'] * sell_ratio
                
                if sell_volume * current_price >= self.min_trade_amount:
                    # 매도 주문
                    order_data = {
                        'market': self.symbol,
                        'side': 'ask',
                        'volume': f"{sell_volume:.8f}",
                        'ord_type': 'market'
                    }
                    
                    query_string = '&'.join([f"{k}={v}" for k, v in order_data.items()])
                    order_url = "https://api.upbit.com/v1/orders"
                    
                    order_response = requests.post(
                        order_url,
                        json=order_data,
                        headers=self.get_upbit_headers(query_string)
                    )
                    
                    result = order_response.json()
                    expected_amount = sell_volume * current_price
                    expected_amount_after_fee = expected_amount * (1 - self.trading_fee)
                    
                    return {
                        "status": "sell_executed",
                        "order_id": result.get('uuid'),
                        "price": current_price,
                        "volume": sell_volume,
                        "sell_ratio": sell_ratio,
                        "expected_amount": expected_amount,
                        "expected_amount_after_fee": expected_amount_after_fee,
                        "confidence": confidence
                    }
                else:
                    return {"status": "insufficient_btc_balance", "message": f"BTC 잔고 부족 또는 최소 거래 금액 미달"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_portfolio_return(self):
        """포트폴리오 수익률 계산"""
        try:
            portfolio_data = self.calculate_portfolio_value()
            if not portfolio_data:
                return {"error": "포트폴리오 데이터 조회 실패"}
            
            # 초기 투자금 (환경 변수에서 설정)
            initial_investment = float(os.environ.get('INITIAL_INVESTMENT', 1000000))
            total_value = portfolio_data['total_value']
            return_rate = ((total_value - initial_investment) / initial_investment) * 100
            
            return {
                'total_value': total_value,
                'initial_investment': initial_investment,
                'return_rate': return_rate,
                'profit_loss': total_value - initial_investment,
                'krw_balance': portfolio_data['krw_balance'],
                'btc_balance': portfolio_data['btc_balance'],
                'btc_value': portfolio_data['btc_value'],
                'btc_ratio': portfolio_data['btc_value'] / total_value if total_value > 0 else 0
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def log_to_sheets(self, trade_result, portfolio_return, predictions, market_data, sentiment_data):
        """구글 시트에 결과 기록 (수수료 정보 포함)"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            values = [[
                timestamp,
                trade_result.get('status', 'N/A'),
                trade_result.get('price', 0),
                trade_result.get('amount', 0),
                trade_result.get('amount_after_fee', 0) if 'amount_after_fee' in trade_result else trade_result.get('expected_amount_after_fee', 0),
                trade_result.get('confidence', 0),
                trade_result.get('portfolio_ratio', 0) if 'portfolio_ratio' in trade_result else trade_result.get('sell_ratio', 0),
                portfolio_return.get('total_value', 0),
                portfolio_return.get('return_rate', 0),
                portfolio_return.get('btc_ratio', 0),
                predictions.get('technical', 0),
                predictions.get('lstm', 0),
                predictions.get('sentiment', 0),
                predictions.get('market', 0),
                market_data.get('btc_dominance', 0),
                market_data.get('fear_greed_index', 0),
                sentiment_data.get('sentiment_score', 0),
                self.trading_fee  # 수수료 기록
            ]]
            
            body = {'values': values}
            
            self.sheets_service.spreadsheets().values().append(
