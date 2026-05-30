"""
config.py
프로젝트 전역 설정 파일
"""
import os

ECOS_API_KEY = "N22PGRJO19YN53G31JK3"

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── 데이터 수집 기간 ─────────────────────────────────────────────────────────
DATA_START_YEAR  = "1995"   # 수집 시작 연도
DATA_START_MONTH = "01"
DATA_END_YEAR    = "2026"
DATA_END_MONTH   = "12"

# ECOS API 월별 포맷: YYYYMM
ECOS_START = DATA_START_YEAR + DATA_START_MONTH   # "199501"
ECOS_END   = DATA_END_YEAR   + DATA_END_MONTH     # "202612"

# yfinance 날짜 포맷: YYYY-MM-DD
MARKET_START = f"{DATA_START_YEAR}-{DATA_START_MONTH}-01"
MARKET_END   = f"{DATA_END_YEAR}-{DATA_END_MONTH}-31"

# ── FSI 임계값 ───────────────────────────────────────────────────────────────
FSI_NORMAL = 0.3   # 0.0 ~ 0.3  : 정상
FSI_WARN   = 0.6   # 0.3 ~ 0.6  : 주의
#                  # 0.6 ~      : 경계

# ── 가계부채 DSR 기준 ────────────────────────────────────────────────────────
DSR_LOW    = 20    # DSR < 20%  : 저위험
DSR_MID    = 40    # DSR 20~40% : 중위험
#                  # DSR > 40%  : 고위험

# ── 모델 파일명 ──────────────────────────────────────────────────────────────
FSI_CALC_FILE  = os.path.join(MODEL_DIR, "fsi_calculator.pkl")
RISK_CLF_FILE  = os.path.join(MODEL_DIR, "risk_classifier.pkl")
SCALER_FILE    = os.path.join(MODEL_DIR, "scaler.pkl")

# ── 피처 목록 ────────────────────────────────────────────────────────────────
HOUSEHOLD_FEATURES = [
    "dsr",              # 총부채원리금상환비율 (핵심)
    "lti",              # 소득 대비 총부채 비율
    "var_rate_pct",     # 변동금리 비중
    "mortgage_pct",     # 주담대 비중
    "dsr_stress",       # 금리 +1%p 적용 스트레스 DSR
    "annual_income",    # 연간 소득
    "total_debt",       # 총부채
    "age",              # 가구주 연령
    "household_size",   # 가구원 수
    "asset_total",      # 총자산
    "fin_asset",        # 금융자산
    "real_estate",      # 부동산 자산
]

# 디렉터리 자동 생성
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
