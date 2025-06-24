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
import praw # Reddit만 사용
from textblob import TextBlob
import warnings
from urllib.parse import urlencode
import logging
import sys
import boto3 # boto3 임포트
from botocore.exceptions import ClientError # ClientError 임포트

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler("bitcoin_trading_bot.log")
                    ])

# EnhancedBitcoinTradingBot 클래스 정의
class EnhancedBitcoinTradingBot:
    def __init__(self):
        logging.info("EnhancedBitcoinTradingBot을 초기화합니다.")

        # AWS Secrets Manager에서 비밀 정보 로드
        self.secret_name = "bitcoin-trading-bot-secrets" # Secrets Manager에 저장한 이름
        self.region_name = "ap-northeast-2" # 귀하의 AWS 리전

        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=self.region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=self.secret_name
            )
        except ClientError as e:
            logging.error(f"Secrets Manager에서 비밀 정보를 가져오는 중 오류 발생: {e}")
            # 특정 오류 처리 (예: SecretNotFoundException, AccessDeniedException)
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logging.critical(f"Secrets Manager '{self.secret_name}'을(를) 찾을 수 없습니다. 이름 및 리전을 확인하세요.")
            elif e.response['Error']['Code'] == 'AccessDeniedException':
                logging.critical("EC2 인스턴스에 Secrets Manager 접근 권한이 없습니다. IAM 역할을 확인하세요.")
            raise e # 복구 불가능한 오류는 프로그램 종료

        # SecretString에서 JSON 데이터를 로드
        secret = get_secret_value_response['SecretString']
        secrets_dict = json.loads(secret)

        # 로드된 값으로 봇 설정
        self.upbit_access_key = secrets_dict.get('UPBIT_ACCESS_KEY')
        self.upbit_secret_key = secrets_dict.get('UPBIT_SECRET_KEY')
        self.google_credentials_json_str = secrets_dict.get('GOOGLE_SERVICE_ACCOUNT_JSON') # JSON 문자열 형태
        self.spreadsheet_id = secrets_dict.get('GOOGLE_SPREADSHEET_ID')
        self.initial_investment = float(secrets_dict.get('INITIAL_INVESTMENT', 1000000))
        
        # 트위터는 사용하지 않으므로 관련 키는 로드하지 않음
        self.twitter_bearer_token = None 
        
        self.reddit_client_id = secrets_dict.get('REDDIT_CLIENT_ID')
        self.reddit_client_secret = secrets_dict.get('REDDIT_CLIENT_SECRET')
        
        # 필수 API 키가 설정되었는지 확인
        if not all([self.upbit_access_key, self.upbit_secret_key, 
                      self.google_credentials_json_str, self.spreadsheet_id]):
            logging.error("Secrets Manager에서 필수 API 키를 로드하는 데 실패했습니다. 누락된 키가 있습니다.")
            raise ValueError("필수 환경 변수가 Secrets Manager에 올바르게 설정되지 않았습니다.")

        # Google 서비스 계정 JSON 문자열을 실제 JSON 객체로 파싱
        try:
            self.google_credentials_json = json.loads(self.google_credentials_json_str)
            logging.info("Google 서비스 계정 JSON이 성공적으로 파싱되었습니다.")
        except json.JSONDecodeError as e:
            logging.critical(f"Google 서비스 계정 JSON 파싱 오류: {e}. Secrets Manager에 저장된 JSON 문자열 형식을 확인하세요.")
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON 형식이 올바르지 않습니다. Secrets Manager의 값을 확인하세요.")

        # 나머지 설정은 기존과 동일
        self.symbol = "KRW-BTC"
        self.lookback_days = 60
        self.lstm_sequence_length = 10
        self.fee_rate = 0.005
        self.min_profit_threshold = 0.012 
        self.max_position_ratio = 0.8
        self.base_trade_ratio = 0.1
        self.volatility_adjustment = True
        self.split_levels = 3
        self.partial_sell_ratio = 0.3
        
        self.model_weights = {
            'technical': 0.4,
            'lstm': 0.3,
            'sentiment': 0.2,
            'market': 0.1
        }
        
        # Google Sheets 및 소셜 미디어 클라이언트 설정
        self.setup_google_sheets()
        self.setup_social_media_clients()
        
    def setup_google_sheets(self):
        """Google Sheets API 클라이언트 설정"""
        try:
            # self.google_credentials_json은 이미 파싱된 JSON 객체이므로 바로 사용
            credentials = Credentials.from_service_account_info(
                self.google_credentials_json, # 여기를 변경
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            logging.info("Google Sheets API 클라이언트가 성공적으로 설정되었습니다.")
        except Exception as e:
            logging.error(f"Google Sheets API 클라이언트 설정 오류: {e}")
            self.sheets_service = None

    def setup_social_media_clients(self):
        """소셜 미디어 클라이언트 설정 (트위터 제외)"""
        try:
            # 트위터 클라이언트 설정 로직 제거 (twitter_bearer_token이 None이므로 자동 스킵)
            # if self.twitter_bearer_token:
            #     self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
            #     logging.info("트위터 클라이언트가 성공적으로 설정되었습니다.")
            # else:
            #     logging.warning("트위터 Bearer 토큰이 없어 트위터 클라이언트를 설정하지 않습니다.")
            
            if self.reddit_client_id and self.reddit_client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent="bitcoin_bot/1.0" # User-Agent는 필수입니다. 실제 사용자에 맞게 수정하세요.
                )
                logging.info("레딧 클라이언트가 성공적으로 설정되었습니다.")
            else:
                logging.warning("레딧 클라이언트 ID 또는 시크릿이 없어 레딧 클라이언트를 설정하지 않습니다.")
        except Exception as e:
            logging.error(f"소셜 미디어 클라이언트 설정 오류: {e}")

    def get_upbit_headers(self, params=None, json_data=None):
        """
        업비트 API 인증 헤더 생성.
        GET 요청의 경우 'params'를 사용하여 쿼리 문자열 해시를 생성하고,
        POST 요청의 경우 'json_data'를 사용하여 JSON 본문 해시를 생성합니다.
        """
        payload = {
            'access_key': self.upbit_access_key,
            'nonce': str(int(time.time() * 1000))
        }

        if params:
            query_string = urlencode(sorted(params.items()), doseq=True).encode('utf-8')
            payload['query_hash'] = hashlib.sha512(query_string).hexdigest()
            payload['query_hash_alg'] = 'SHA512'
        elif json_data:
            json_string_for_hash = json.dumps(json_data, sort_keys=True).encode('utf-8')
            payload['query_hash'] = hashlib.sha512(json_string_for_hash).hexdigest()
            payload['query_hash_alg'] = 'SHA512'

        jwt_token = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('utf-8')
        
        signature = hmac.new(
            self.upbit_secret_key.encode('utf-8'),
            jwt_token.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()

        headers = {
            'Authorization': f'Bearer {jwt_token}.{signature}',
        }
        if json_data:
            headers['Content-Type'] = 'application/json'
        return headers
    
    def _make_api_request(self, method, url, params=None, json_data=None, headers=None, max_retries=3):
        """
        API 요청을 수행하고 재시도 로직을 포함합니다.
        """
        for retry_count in range(max_retries):
            try:
                if headers is None:
                    headers = self.get_upbit_headers(params=params, json_data=json_data)

                if method == 'GET':
                    response = requests.get(url, params=params, headers=headers)
                elif method == 'POST':
                    response = requests.post(url, json=json_data, headers=headers)
                else:
                    raise ValueError(f"지원하지 않는 HTTP 메소드: {method}")

                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                error_msg = f"HTTP 오류 발생 ({url}, 재시도 {retry_count + 1}/{max_retries}): {http_err}"
                try:
                    error_details = http_err.response.json()
                    error_msg += f" (세부: {error_details})"
                except json.JSONDecodeError:
                    pass
                logging.error(error_msg)
                if retry_count < max_retries - 1 and http_err.response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(2 ** retry_count)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logging.error(f"API 요청 오류 ({url}, 재시도 {retry_count + 1}/{max_retries}): {e}")
                if retry_count < max_retries - 1:
                    time.sleep(2 ** retry_count)
                else:
                    raise
            except Exception as e:
                logging.error(f"예상치 못한 오류 발생 ({url}): {e}")
                raise
        return None

    def get_market_data(self, interval='days'):
        """시장 데이터 수집 (일, 시간, 15분 단위)"""
        url = ""
        count = 0
        if interval == 'days':
            url = "https://api.upbit.com/v1/candles/days"
            count = self.lookback_days
        elif interval == 'hours':
            url = "https://api.upbit.com/v1/candles/minutes/60"
            count = self.lookback_days * 24
        elif interval == 'minutes_15':
            url = "https://api.upbit.com/v1/candles/minutes/15"
            count = self.lookback_days * 24 * 4
        else:
            logging.warning(f"지원하지 않는 시간프레임: {interval}. 'days'로 기본 설정합니다.")
            url = "https://api.upbit.com/v1/candles/days"
            count = self.lookback_days
        
        params = {'market': self.symbol, 'count': count}
        try:
            data = self._make_api_request('GET', url, params=params)
            if data:
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['candle_date_time_kst'])
                df = df.sort_values('datetime').reset_index(drop=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"시장 데이터 수집 오류 ({interval}): {e}")
            return pd.DataFrame()
    
    def get_market_dominance_data(self):
        """전체 시장 데이터 및 비트코인 도미넌스, 공포/탐욕 지수 수집"""
        try:
            dominance_url = "https://api.coingecko.com/api/v3/global"
            dom_data = self._make_api_request('GET', dominance_url)
            btc_dominance = dom_data['data']['market_cap_percentage']['btc'] if dom_data and 'data' in dom_data and 'market_cap_percentage' in dom_data['data'] and 'btc' in dom_data['data']['market_cap_percentage'] else 50
            total_market_cap = dom_data['data']['total_market_cap']['usd'] if dom_data and 'data' in dom_data and 'total_market_cap' in dom_data['data'] and 'usd' in dom_data['data']['total_market_cap'] else 2000000000000
            
            fear_greed_url = "https://api.alternative.me/fng/"
            fg_data = self._make_api_request('GET', fear_greed_url)
            fear_greed_index = int(fg_data['data'][0]['value']) if fg_data and 'data' in fg_data and len(fg_data['data']) > 0 and 'value' in fg_data['data'][0] else 50

            return {
                'btc_dominance': btc_dominance,
                'total_market_cap': total_market_cap,
                'fear_greed_index': fear_greed_index
            }
        except Exception as e:
            logging.error(f"시장 지표 데이터 수집 오류: {e}. 기본값으로 대체합니다.")
            return {'btc_dominance': 50, 'total_market_cap': 2000000000000, 'fear_greed_index': 50}

    def calculate_volatility(self, df, period=14):
        """주어진 데이터프레임의 변동성 (표준편차 기반)을 계산합니다."""
        if 'trade_price' not in df.columns or len(df) < period:
            return 0.02
        returns = df['trade_price'].pct_change()
        volatility = returns.rolling(period).std() * np.sqrt(period)
        return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.02
        
    def create_advanced_technical_indicators(self, df):
        """
        주어진 데이터프레임에 다양한 기술적 분석 지표를 추가합니다.
        (이동평균, MACD, RSI, 스토캐스틱, 볼린저 밴드, ATR, 거래량 지표, 가격 변화율)
        """
        if df.empty:
            return df

        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['trade_price'].rolling(window=period, min_periods=1).mean()
        
        for period in [12, 26]:
            df[f'ema_{period}'] = df['trade_price'].ewm(span=period, adjust=False, min_periods=1).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        delta = df['trade_price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50, inplace=True)
        
        low_14 = df['low_price'].rolling(window=14, min_periods=1).min()
        high_14 = df['high_price'].rolling(window=14, min_periods=1).max()
        denominator = (high_14 - low_14)
        df['stoch_k'] = 100 * ((df['trade_price'] - low_14) / denominator.replace(0, np.nan))
        df['stoch_k'].fillna(50, inplace=True)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
        
        df['bb_middle'] = df['trade_price'].rolling(window=20, min_periods=1).mean()
        bb_std = df['trade_price'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        denominator_bb = (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = (df['trade_price'] - df['bb_lower']) / denominator_bb.replace(0, np.nan)
        df['bb_position'].fillna(0.5, inplace=True)
        
        df['tr1'] = df['high_price'] - df['low_price']
        df['tr2'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['tr3'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14, min_periods=1).mean()
        
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma'].replace(0, np.nan)
        df['volume_ratio'].fillna(1, inplace=True)

        for period in [1, 3, 7, 14]:
            df[f'price_change_{period}'] = df['trade_price'].pct_change(period)
            df[f'price_change_{period}'].fillna(0, inplace=True)
            
        return df

    def get_social_sentiment(self):
        """소셜 미디어 (레딧만)에서 비트코인 관련 게시글의 감정을 분석합니다."""
        sentiment_scores = []
        
        try:
            # 트위터 감정 분석 로직 제거
            
            # 레딧 감정 분석
            if hasattr(self, 'reddit') and self.reddit: # reddit 객체가 성공적으로 초기화되었는지 확인
                logging.info("레딧 감정 분석 시작...")
                subreddit = self.reddit.subreddit('Bitcoin')
                hot_posts = subreddit.hot(limit=50)
                
                post_count = 0
                for post in hot_posts:
                    text_content = post.title
                    if post.selftext:
                        text_content += " " + post.selftext
                    
                    if text_content.strip():
                        blob = TextBlob(text_content)
                        sentiment_scores.append(blob.sentiment.polarity)
                        post_count += 1
                logging.info(f"레딧에서 {post_count}개의 게시글을 분석했습니다.")
            else:
                logging.warning("레딧 클라이언트가 설정되지 않아 레딧 감정 분석을 건너뜁니다.")
            
        except Exception as e:
            logging.error(f"소셜 미디어 분석 중 오류 발생: {e}")
        
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
        """LSTM 모델 학습을 위한 데이터를 준비합니다."""
        if df.empty or len(df) < self.lstm_sequence_length + 1:
            logging.warning(f"LSTM 데이터 준비 실패: 데이터프레임이 비었거나 시퀀스 길이({self.lstm_sequence_length})보다 짧습니다.")
            return np.array([]), np.array([]), MinMaxScaler()
            
        feature_cols = [
            'trade_price', 'ma_5', 'ma_20', 'rsi', 'macd', 'stoch_k',  
            'bb_position', 'atr', 'volume_ratio', 'price_change_1'
        ]
        
        df_selected = df[feature_cols].dropna()
        
        if df_selected.empty:
            logging.warning("LSTM 데이터 준비 실패: 선택된 특성 컬럼에 NaN 값이 너무 많아 유효한 데이터가 없습니다.")
            return np.array([]), np.array([]), MinMaxScaler()
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_selected)
        
        X, y = [], []
        for i in range(self.lstm_sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.lstm_sequence_length:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler

    def build_lstm_model(self, input_shape):
        """LSTM 모델을 구축합니다."""
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
        """
        기술적, LSTM, 감정, 시장 데이터를 기반으로 다중 모델을 훈련하고 예측을 수행합니다.
        """
        predictions = {}
        
        # 1. 기술적 분석 모델 (Random Forest)
        tech_features = [
            'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal',  
            'stoch_k', 'stoch_d', 'bb_position', 'bb_width', 'atr', 'volume_ratio',
            'price_change_1', 'price_change_3', 'price_change_7', 'price_change_14'
        ]
        
        df_clean = df.dropna(subset=tech_features + ['trade_price'])

        if len(df_clean) > 30:
            X_tech = df_clean[tech_features].iloc[:-1]
            y_tech = df_clean['trade_price'].iloc[1:]
            
            if len(X_tech) == len(y_tech):
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_tech, y_tech)
                
                latest_features = df_clean[tech_features].iloc[-1:]
                tech_prediction = rf_model.predict(latest_features)[0]
                predictions['technical'] = tech_prediction
                logging.info(f"기술적 분석 모델 예측: {tech_prediction:,.0f}")
            else:
                logging.warning("기술적 분석 모델 훈련 데이터 길이가 일치하지 않아 건너뜁니다.")
        else:
            logging.warning("기술적 분석 모델 훈련을 위한 데이터가 충분하지 않습니다.")
            
        # 2. LSTM 모델
        try:
            X_lstm, y_lstm, scaler = self.prepare_lstm_data(df_clean)
            if X_lstm.shape[0] > 20:
                lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
                lstm_model.fit(X_lstm[:-5], y_lstm[:-5], epochs=50, batch_size=32, verbose=0)
                
                last_sequence = X_lstm[-1].reshape(1, X_lstm.shape[1], X_lstm.shape[2])
                lstm_pred_scaled = lstm_model.predict(last_sequence)[0][0]
                
                dummy_array = np.zeros((1, scaler.n_features_in_))
                dummy_array[0, 0] = lstm_pred_scaled
                lstm_prediction = scaler.inverse_transform(dummy_array)[0, 0]
                predictions['lstm'] = lstm_prediction
                logging.info(f"LSTM 모델 예측: {lstm_prediction:,.0f}")
            else:
                logging.warning("LSTM 모델 훈련을 위한 데이터가 충분하지 않습니다.")
                if not df_clean.empty:
                    predictions['lstm'] = df_clean['trade_price'].iloc[-1]
        except Exception as e:
            logging.error(f"LSTM 모델 훈련 또는 예측 중 오류 발생: {e}")
            if not df_clean.empty:
                predictions['lstm'] = df_clean['trade_price'].iloc[-1]

        # 3. 감정 분석 기반 예측
        current_price = df_clean['trade_price'].iloc[-1] if not df_clean.empty else 0
        sentiment_multiplier = 1 + (sentiment_data['sentiment_score'] * 0.05)
        predictions['sentiment'] = current_price * sentiment_multiplier
        logging.info(f"감정 분석 예측: {predictions['sentiment']:,.0f} (감정 점수: {sentiment_data['sentiment_score']:.2f})")
        
        # 4. 시장 데이터 기반 예측
        market_multiplier = 1 + ((market_data['fear_greed_index'] - 50) / 50) * 0.05
        predictions['market'] = current_price * market_multiplier
        logging.info(f"시장 데이터 예측: {predictions['market']:,.0f} (공포/탐욕 지수: {market_data['fear_greed_index']})")
        
        return predictions
    
    def make_ensemble_prediction(self, predictions):
        """
        다중 모델의 예측을 가중 평균하여 최종 앙상블 예측을 생성합니다.
        """
        if not predictions:
            logging.warning("앙상블 예측을 위한 모델 예측이 없습니다.")
            return None
        
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if model_name in self.model_weights and prediction is not None:
                weight = self.model_weights[model_name]
                weighted_sum += prediction * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = weighted_sum / total_weight
            logging.info(f"최종 앙상블 예측: {ensemble_pred:,.0f}")
            return ensemble_pred
        logging.warning("모든 모델의 예측 가중치가 0이거나 예측값이 없습니다.")
        return None

    def calculate_dynamic_trade_amount(self, total_portfolio_value, volatility, confidence):
        """
        총 포트폴리오 가치, 시장 변동성, 예측 신뢰도에 기반하여
        동적인 매매 금액을 계산합니다 (비례 매매).
        """
        base_ratio = self.base_trade_ratio
        
        if self.volatility_adjustment:
            volatility_factor = max(0.5, 1 - (volatility / 0.05))
            base_ratio *= volatility_factor
        
        confidence_normalized = confidence / 10.0
        confidence_factor = 0.75 + (confidence_normalized * 0.75)
        base_ratio *= confidence_factor
        
        final_ratio = np.clip(base_ratio, 0.02, 0.20)
        
        trade_amount = total_portfolio_value * final_ratio
        logging.info(f"동적 매매 금액 계산: 포트폴리오 {total_portfolio_value:,.0f} KRW, 변동성 {volatility:.2f}, 신뢰도 {confidence:.1f}")
        logging.info(f"최종 매매 비율: {final_ratio:.2%}, 매매 금액: {trade_amount:,.0f} KRW")
        return trade_amount

    def generate_trading_signal(self, df, predictions, market_data, sentiment_data):
        """
        다양한 지표와 모델 예측을 기반으로 거래 신호(BUY/SELL/HOLD)를 생성합니다.
        수수료를 고려한 최소 수익률 조건을 적용합니다.
        """
        current_price = df['trade_price'].iloc[-1] if not df.empty else 0
        ensemble_prediction = self.make_ensemble_prediction(predictions)
        
        if ensemble_prediction is None or current_price == 0:
            logging.warning("거래 신호 생성 불가: 예측값 또는 현재 가격 정보 부족.")
            return "HOLD", 0, 0
        
        price_change_pct = (ensemble_prediction - current_price) / current_price
        required_return_pct = self.min_profit_threshold 
        
        logging.info(f"예측 가격 변화율: {price_change_pct:.2%}, 요구 수익률: {required_return_pct:.2%}")

        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
        macd_histogram = df['macd_histogram'].iloc[-1] if 'macd_histogram' in df.columns and not pd.isna(df['macd_histogram'].iloc[-1]) else 0
        bb_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns and not pd.isna(df['bb_position'].iloc[-1]) else 0.5
        
        fear_greed = market_data.get('fear_greed_index', 50)
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        
        volatility = self.calculate_volatility(df)
        
        # 매수 조건
        buy_conditions = [
            price_change_pct > required_return_pct,
            rsi < 70,
            bb_position < 0.8,
            fear_greed < 80,
            sentiment_score > -0.1,
            volatility < 0.08,
            macd_histogram > 0
        ]
        
        # 매도 조건
        sell_conditions = [
            price_change_pct < -required_return_pct,
            rsi > 30,
            bb_position > 0.2,
            fear_greed > 20,
            sentiment_score < 0.1,
            volatility < 0.08,
            macd_histogram < 0
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        confidence = min(10, abs(price_change_pct) * 100 * 2)
        
        signal = "HOLD"
        if buy_score >= 5 and confidence >= 5:
            signal = "BUY"
        elif sell_score >= 5 and confidence >= 5:
            signal = "SELL"

        logging.info(f"최종 거래 신호: {signal}, 신뢰도: {confidence:.1f}, 변동성: {volatility:.4f}")
        return signal, confidence, volatility
    
    def execute_trade(self, signal, confidence, volatility):
        """
        생성된 거래 신호에 따라 실제 거래를 실행합니다.
        비례 매매 및 분할 매도 전략을 적용합니다.
        """
        if signal == "HOLD":
            return {"status": "hold", "message": "거래 신호 없음, 대기 중"}
        
        try:
            # 1. 현재 잔고 조회 (KRW, BTC)
            balances = self._make_api_request('GET', "https://api.upbit.com/v1/accounts")
            
            krw_balance = 0.0
            btc_balance = 0.0
            
            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_balance = float(balance['balance'])
                elif balance['currency'] == 'BTC':
                    btc_balance = float(balance['balance'])
            
            # 2. 현재 비트코인 가격 조회
            ticker_data = self._make_api_request('GET', "https://api.upbit.com/v1/ticker", params={'markets': self.symbol})
            current_price = ticker_data[0]['trade_price'] if ticker_data else 0
            
            # 3. 총 포트폴리오 가치 계산
            total_portfolio_value = krw_balance + (btc_balance * current_price)
            logging.info(f"현재 총 자산: {total_portfolio_value:,.0f} KRW (KRW: {krw_balance:,.0f}, BTC: {btc_balance:.8f} @ {current_price:,.0f})")
            
            # 4. 동적 매매 금액 계산 (비례 매매)
            dynamic_trade_amount = self.calculate_dynamic_trade_amount(
                total_portfolio_value, volatility, confidence
            )
            
            # 매수 주문 (BUY Signal)
            if signal == "BUY":
                if krw_balance >= dynamic_trade_amount:
                    current_btc_value = btc_balance * current_price
                    current_position_ratio = current_btc_value / total_portfolio_value if total_portfolio_value > 0 else 0

                    if current_position_ratio + (dynamic_trade_amount / total_portfolio_value) > self.max_position_ratio:
                        logging.info(f"매수 보류: 최대 포지션 비율 ({self.max_position_ratio:.0%}) 초과 예상. 현재: {current_position_ratio:.2%}")
                        return {"status": "skipped_max_position", "message": "최대 포지션 비율 초과 예상"}

                    if dynamic_trade_amount < 5000:
                        logging.info(f"매수 보류: 최소 주문 금액 (5,000 KRW) 미만. 계산된 금액: {dynamic_trade_amount:,.0f} KRW")
                        return {"status": "skipped_min_amount", "message": "최소 주문 금액 미만"}
                    
                    order_data = {
                        'market': self.symbol,
                        'side': 'bid',
                        'price': str(int(dynamic_trade_amount)),
                        'ord_type': 'price'
                    }
                    logging.info(f"매수 주문 생성: {dynamic_trade_amount:,.0f} KRW 상당의 비트코인")
                    
                    result = self._make_api_request('POST', "https://api.upbit.com/v1/orders", json_data=order_data)
                    
                    fee = dynamic_trade_amount * self.fee_rate
                    
                    return {
                        "status": "buy_executed",
                        "order_id": result.get('uuid'),
                        "price": current_price,
                        "amount": dynamic_trade_amount,
                        "fee": fee,
                        "confidence": confidence,
                        "portfolio_ratio_after_trade": (current_btc_value + dynamic_trade_amount) / total_portfolio_value,
                        "volatility": volatility,
                        "message": f"매수 주문 성공: {dynamic_trade_amount:,.0f} KRW"
                    }
                else:
                    return {
                        "status": "insufficient_krw_balance",
                        "message": f"매수 보류: KRW 잔고 부족. 필요: {dynamic_trade_amount:,.0f} KRW, 현재: {krw_balance:,.0f} KRW"
                    }
            
            # 매도 주문 (SELL Signal)
            elif signal == "SELL":
                if btc_balance > 0:
                    sell_ratio = self.partial_sell_ratio
                    if confidence > 7:
                        sell_ratio = min(0.6, sell_ratio * 1.5) 
                    
                    sell_volume = btc_balance * sell_ratio
                    
                    min_sell_volume_krw_equivalent = 5000 / current_price
                    if sell_volume * current_price < 5000 or sell_volume < 0.00000001:
                         logging.info(f"매도 보류: 최소 주문 금액 (5,000 KRW) 미만. 계산된 매도 금액: {sell_volume * current_price:,.0f} KRW")
                         return {"status": "skipped_min_amount", "message": "최소 주문 금액 미만"}
                    
                    order_data = {
                        'market': self.symbol,
                        'side': 'ask',
                        'volume': str(sell_volume),
                        'ord_type': 'market'
                    }
                    logging.info(f"매도 주문 생성: {sell_volume:.8f} BTC")

                    result = self._make_api_request('POST', "https://api.upbit.com/v1/orders", json_data=order_data)
                    
                    actual_sell_amount = sell_volume * current_price
                    fee = actual_sell_amount * self.fee_rate
                    
                    return {
                        "status": "sell_executed",
                        "order_id": result.get('uuid'),
                        "price": current_price,
                        "volume": sell_volume,
                        "amount": actual_sell_amount,
                        "fee": fee,
                        "confidence": confidence,
                        "sell_ratio_applied": sell_ratio,
                        "remaining_btc": btc_balance - sell_volume,
                        "volatility": volatility,
                        "message": f"매도 주문 성공: {sell_volume:.8f} BTC"
                    }
                else:
                    return {
                        "status": "insufficient_btc_balance",
                        "message": "매도 보류: BTC 잔고 부족."
                    }
            else:
                return {"status": "error", "message": "알 수 없는 거래 신호"}
                
        except Exception as e:
            logging.error(f"거래 실행 중 예상치 못한 오류 발생: {e}")
            return {"status": "error", "message": str(e)}

    def calculate_portfolio_return(self):
        """
        현재 포트폴리오 가치와 초기 투자금을 기반으로 수익률을 계산합니다.
        수수료는 이미 거래 시점에 반영되므로 별도로 계산하지 않습니다.
        """
        try:
            balances = self._make_api_request('GET', "https://api.upbit.com/v1/accounts")
            ticker_data = self._make_api_request('GET', "https://api.upbit.com/v1/ticker", params={'markets': self.symbol})
            current_price = ticker_data[0]['trade_price'] if ticker_data else 0
            
            total_value = 0.0
            krw_value = 0.0
            btc_value = 0.0

            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_value = float(balance['balance'])
                    total_value += krw_value
                elif balance['currency'] == 'BTC':
                    btc_value = float(balance['balance']) * current_price
                    total_value += btc_value
            
            # initial_investment는 Secrets Manager에서 로드된 값 사용
            initial_investment = self.initial_investment
            
            profit_loss = total_value - initial_investment
            return_rate = (profit_loss / initial_investment) * 100 if initial_investment > 0 else 0
            
            position_ratio = btc_value / total_value if total_value > 0 else 0
            
            return {
                'total_value': total_value,
                'initial_investment': initial_investment,
                'return_rate': return_rate,
                'profit_loss': profit_loss,
                'position_ratio': position_ratio,
                'krw_value': krw_value,
                'btc_value': btc_value,
                'current_btc_price': current_price
            }
        except Exception as e:
            logging.error(f"포트폴리오 수익률 계산 오류: {e}")
            return {"error": str(e)}

    def log_to_sheets(self, trade_result, portfolio_return, predictions, market_data, sentiment_data):
        """
        구글 시트에 거래 결과, 포트폴리오 현황, 예측 정보, 시장 및 감정 데이터를 기록합니다.
        """
        if not self.sheets_service:
            logging.warning("Google Sheets 서비스가 설정되지 않아 기록할 수 없습니다.")
            return

        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            values = [[
                timestamp,
                trade_result.get('status', 'N/A'),
                trade_result.get('price', 0),
                trade_result.get('amount', 0),
                trade_result.get('fee', 0),
                trade_result.get('confidence', 0),
                trade_result.get('volatility', 0),
                portfolio_return.get('total_value', 0),
                portfolio_return.get('initial_investment', 0),
                portfolio_return.get('profit_loss', 0),
                portfolio_return.get('return_rate', 0),
                portfolio_return.get('position_ratio', 0),
                predictions.get('technical', 0),
                predictions.get('lstm', 0),
                predictions.get('sentiment', 0),
                predictions.get('market', 0),
                market_data.get('btc_dominance', 0),
                market_data.get('total_market_cap', 0),
                market_data.get('fear_greed_index', 0),
                sentiment_data.get('sentiment_score', 0),
                sentiment_data.get('sentiment_volatility', 0),
                sentiment_data.get('sample_size', 0)
            ]]
            
            body = {'values': values}
            
            result = self.sheets_service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Sheet1!A:V',
                valueInputOption='RAW',
                body=body
            ).execute()
            logging.info(f"Google Sheets에 데이터 기록 성공. 업데이트된 셀: {result.get('updates').get('updatedCells')}")
        except Exception as e:
            logging.error(f"Google Sheets에 데이터 기록 오류: {e}")

    def run(self):
        """
        자동매매 봇의 메인 실행 루프.
        일정 간격으로 데이터를 수집, 분석, 거래 신호 생성, 거래 실행 및 결과 기록을 수행합니다.
        """
        logging.info("Bitcoin 자동매매 봇을 시작합니다.")
        while True:
            try:
                logging.info("\n--- 새로운 주기 시작 ---")
                
                df_daily = self.get_market_data(interval='days')
                df_15min = self.get_market_data(interval='minutes_15')

                if df_daily.empty or df_15min.empty:
                    logging.error("시장 데이터 수집에 실패했습니다. 다음 주기까지 대기합니다.")
                    time.sleep(300)
                    continue

                df_daily = self.create_advanced_technical_indicators(df_daily)
                df_15min = self.create_advanced_technical_indicators(df_15min)

                market_data = self.get_market_dominance_data()
                sentiment_data = self.get_social_sentiment()

                predictions = self.train_models(df_15min, market_data, sentiment_data)
                
                signal, confidence, volatility = self.generate_trading_signal(
                    df_15min, predictions, market_data, sentiment_data
                )
                logging.info(f"생성된 거래 신호: {signal} (신뢰도: {confidence:.1f}, 변동성: {volatility:.4f})")
                
                trade_result = self.execute_trade(signal, confidence, volatility)
                logging.info(f"거래 실행 결과: {trade_result.get('message', trade_result.get('status'))}")

                portfolio_return = self.calculate_portfolio_return()
                if not portfolio_return.get('error'):
                    logging.info(f"현재 포트폴리오 총 가치: {portfolio_return['total_value']:,.0f} KRW")
                    logging.info(f"수익률: {portfolio_return['return_rate']:.2f}%")
                    logging.info(f"BTC 포지션 비율: {portfolio_return['position_ratio']:.2%}")
                else:
                    logging.error(f"포트폴리오 수익률 계산 오류: {portfolio_return['error']}")

                self.log_to_sheets(trade_result, portfolio_return, predictions, market_data, sentiment_data)
                
                logging.info("--- 주기 완료 ---")
                time.sleep(300)

            except Exception as e:
                logging.critical(f"봇 실행 중 치명적인 오류 발생: {e}", exc_info=True)
                logging.info("5분 후 재시도합니다...")
                time.sleep(300)

if __name__ == "__main__":
    try:
        bot = EnhancedBitcoinTradingBot()
        bot.run()
    except ValueError as ve:
        logging.error(f"설정 오류: {ve}")
    except Exception as e:
        logging.critical(f"봇 초기화 중 오류 발생: {e}", exc_info=True)

