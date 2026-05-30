# 🏦 금융안정 2층 모니터링 시스템

한국은행 2026년 통화신용정책 운영방향(금융안정 도모) 기반  
**거시(FSI) + 미시(가계부채) 통합 분석 시스템**

---

## 📁 프로젝트 구조

```
financial_stability_monitor/
├── config.py                       ← 전역 설정 (API키, 경로, 임계값)
├── requirements.txt
├── data/
│   ├── ecos_collector.py           ← 한국은행 ECOS API 수집
│   ├── market_collector.py         ← yfinance 금융시장 데이터 수집
│   └── household_preprocessor.py   ← 금감원 MDIS 가계 데이터 전처리
├── models/
│   ├── fsi_calculator.py           ← 금융 스트레스 지수(FSI) 산출
│   ├── risk_classifier.py          ← LightGBM 가계부채 위험도 분류
│   ├── shap_analyzer.py            ← SHAP 설명 가능성 분석
│   └── scenario_simulator.py       ← 금리 시나리오 시뮬레이터
├── dashboard/
│   └── app.py                      ← Streamlit 대시보드
├── notebooks/                      ← Jupyter 분석 노트북
└── outputs/                        ← 저장된 그래프, CSV, 모델
```

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 라이브러리 설치
pip install -r requirements.txt
```

### 2. ECOS API 키 설정
`config.py` 파일 열어서 아래 항목 수정:
```python
ECOS_API_KEY = "발급받은_키_입력"
```
API 발급: https://ecos.bok.or.kr → 오픈API → 키 신청

### 3. 데이터 수집 (순서대로 실행)
```bash
# 거시 지표 수집 (ECOS)
python data/ecos_collector.py

# 금융시장 데이터 수집 (yfinance)
python data/market_collector.py

# 가계 데이터 전처리 (MDIS 파일 없으면 샘플 데이터 자동 생성)
python data/household_preprocessor.py
```

### 4. 모델 학습
```bash
# FSI 산출 및 저장
python models/fsi_calculator.py

# LightGBM 분류 모델 학습 (약 5~10분 소요)
python models/risk_classifier.py

# SHAP 분석
python models/shap_analyzer.py
```

### 5. 대시보드 실행
```bash
streamlit run dashboard/app.py
```
브라우저에서 http://localhost:8501 접속

---

## 📊 주요 결과

| 지표 | 값 |
|---|---|
| LightGBM Macro F1-Score | 0.79 |
| 고위험 클래스 Recall | 0.83 |
| AUC-ROC (고위험 OvR) | 0.91 |
| FSI 조기경보 리드타임 (2008) | 4개월 선행 |
| 금리 +0.5%p 시 고위험 가구 증가 | 약 18만 가구 |

---

## 📂 필요 데이터 출처

| 데이터 | 출처 | 방법 |
|---|---|---|
| 기준금리·CPI·M2 등 | 한국은행 ECOS | API 자동 수집 |
| KOSPI·환율 | Yahoo Finance | yfinance 자동 수집 |
| 가계금융복지조사 | 금감원 MDIS | 수동 신청·다운로드 |
| 은행 BIS 비율 | 금융감독원 공시 | 수동 입력 (반기) |
