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
        
        # --- 수정된 부분 시작 ---
        # GOOGLE_SERVICE_ACCOUNT_JSON 처리 (문자열 또는 딕셔너리 형태 모두 지원)
        google_service_account_data = secrets_dict.get('GOOGLE_SERVICE_ACCOUNT_JSON')
        
        if google_service_account_data:
            try:
                # 이미 딕셔너리인 경우와 문자열인 경우를 모두 처리
                if isinstance(google_service_account_data, dict):
                    self.google_credentials_json = google_service_account_data
                    logging.info("Google 서비스 계정 JSON이 딕셔너리 형태로 로드되었습니다.")
                elif isinstance(google_service_account_data, str):
                    self.google_credentials_json = json.loads(google_service_account_data)
                    logging.info("Google 서비스 계정 JSON이 문자열에서 파싱되었습니다.")
                else:
                    raise ValueError(f"GOOGLE_SERVICE_ACCOUNT_JSON의 타입이 예상과 다릅니다: {type(google_service_account_data)}")
                    
            except json.JSONDecodeError as e:
                logging.error(f"Google 서비스 계정 JSON 파싱 오류: {e}")
                logging.error(f"JSON 데이터: {str(google_service_account_data)[:100]}...")  # 처음 100자만 로그
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON 형식이 올바르지 않습니다.")
            except Exception as e:
                logging.error(f"Google 서비스 계정 JSON 처리 중 예상치 못한 오류: {e}")
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON 처리에 실패했습니다.")
        else:
            logging.error("GOOGLE_SERVICE_ACCOUNT_JSON이 Secrets Manager에서 찾을 수 없습니다.")
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON이 설정되지 않았습니다.")
        # --- 수정된 부분 끝 ---
        
        self.spreadsheet_id = secrets_dict.get('GOOGLE_SPREADSHEET_ID')
        self.initial_investment = float(secrets_dict.get('INITIAL_INVESTMENT', 1000000))
        
        # 트위터는 사용하지 않으므로 관련 키는 로드하지 않음
        self.twitter_bearer_token = None
        
        self.reddit_client_id = secrets_dict.get('REDDIT_CLIENT_ID')
        self.reddit_client_secret = secrets_dict.get('REDDIT_CLIENT_SECRET')
        
        # 필수 API 키가 설정되었는지 확인
        if not all([self.upbit_access_key, self.upbit_secret_key, self.spreadsheet_id]):
            logging.error("Secrets Manager에서 필수 API 키를 로드하는 데 실패했습니다. 누락된 키가 있습니다.")
            raise ValueError("필수 환경 변수가 Secrets Manager에 올바르게 설정되지 않았습니다.")

        # Google 서비스 계정 JSON 유효성 검사
        required_keys = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        if not all(key in self.google_credentials_json for key in required_keys):
            logging.critical("Google 서비스 계정 JSON에 필수 키가 누락되었습니다.")
            raise ValueError("Google 서비스 계정 JSON 형식이 올바르지 않습니다.")
            
        logging.info("모든 필수 설정이 성공적으로 로드되었습니다.")

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
                self.google_credentials_json,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            logging.info("Google Sheets API 클라이언트가 성공적으로 설정되었습니다.")
        except Exception as e:
            logging.error(f"Google Sheets API 클라이언트 설정 오류: {e}")
            self.sheets_service = None
            
    def setup_social_media_clients(self):
        """소셜 미디어 클라이언트 설정 (Reddit만 사용)"""
        try:
            if self.reddit_client_id and self.reddit_client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent='bitcoin_trading_bot'
                )
                logging.info("Reddit 클라이언트가 성공적으로 설정되었습니다.")
            else:
                logging.warning("Reddit 클라이언트 설정을 위한 필수 정보가 누락되었습니다.")
                self.reddit = None
        except Exception as e:
            logging.error(f"Reddit 클라이언트 설정 오류: {e}")
            self.reddit = None
