"""
data/household_preprocessor.py
금감원 가계금융복지조사 데이터 전처리 모듈

[필요 데이터 & 수집 방법]
  1) 접속: https://mdis.kostat.go.kr  (통계청 마이크로데이터통합서비스)
  2) 회원가입 → 로그인
  3) 데이터 검색: "가계금융복지조사" 검색
  4) 신청 → 승인(1~2일 소요) → 다운로드
  5) data/ 폴더에 CSV 저장 후 아래 경로 입력

[주요 변수 설명]
  annual_income    : 연간 총소득 (만원)
  annual_repayment : 연간 원리금 상환액 (만원)
  total_debt       : 총부채 잔액 (만원)
  mortgage         : 주택담보대출 잔액 (만원)
  var_rate_debt    : 변동금리 대출 잔액 (만원)
  age              : 가구주 연령
  household_size   : 가구원 수
  asset_total      : 총자산 (만원)
  fin_asset        : 금융자산 (만원)
  real_estate      : 부동산 자산 (만원)
  region           : 지역 코드 (1=수도권, 2=비수도권)
  income_source    : 소득원천 (1=근로, 2=사업, 3=기타)

[실제 파일이 없을 경우 → 아래 generate_sample_data() 함수로 샘플 데이터 생성 가능]
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DSR_LOW, DSR_MID, HOUSEHOLD_FEATURES


def generate_sample_data(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    실제 MDIS 데이터가 없을 때 사용하는 샘플 데이터 생성기

    실제 가계금융복지조사의 분포를 최대한 근사하여 생성
    """
    rng = np.random.default_rng(seed)
    print("⚠ 샘플 데이터를 생성합니다. 실제 분석에는 MDIS 원본 데이터를 사용하세요.")

    n_high = int(n * 0.15)   # 고위험 15%
    n_mid  = int(n * 0.25)   # 중위험 25%
    n_low  = n - n_high - n_mid  # 저위험 60%

    def make_group(size, income_range, debt_ratio, var_pct_range):
        income   = rng.uniform(*income_range, size)
        debt     = income * rng.uniform(*debt_ratio, size)
        mortgage = debt   * rng.uniform(0.5, 0.8, size)
        var_debt = debt   * rng.uniform(*var_pct_range, size)
        repay    = debt   * rng.uniform(0.05, 0.15, size)
        return {
            "annual_income"   : income,
            "total_debt"      : debt,
            "mortgage"        : mortgage,
            "var_rate_debt"   : var_debt,
            "annual_repayment": repay,
            "age"             : rng.integers(25, 70, size),
            "household_size"  : rng.integers(1, 6, size),
            "asset_total"     : income * rng.uniform(3, 8, size),
            "fin_asset"       : income * rng.uniform(0.5, 2, size),
            "real_estate"     : income * rng.uniform(1, 5, size),
            "region"          : rng.choice([1, 2], size=size, p=[0.5, 0.5]),
            "income_source"   : rng.choice([1, 2, 3], size=size, p=[0.6, 0.25, 0.15]),
        }

    groups = [
        make_group(n_low,  (3000, 8000),  (0.5, 2.0),  (0.1, 0.3)),   # 저위험: 고소득, 낮은 부채
        make_group(n_mid,  (2000, 5000),  (1.5, 3.5),  (0.3, 0.5)),   # 중위험: 중소득, 중부채
        make_group(n_high, (1000, 3000),  (3.0, 6.0),  (0.5, 0.8)),   # 고위험: 저소득, 고부채
    ]
    df = pd.DataFrame({k: np.concatenate([g[k] for g in groups]) for k in groups[0]})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_household(path: str = None) -> pd.DataFrame:
    """
    가계금융복지조사 데이터 로드

    Parameters
    ----------
    path : str  CSV 파일 경로. None이면 샘플 데이터 생성
    """
    if path is None:
        default = os.path.join(DATA_DIR, "household.csv")
        if os.path.exists(default):
            path = default
            print(f"데이터 로드: {path}")
        else:
            print("MDIS 데이터 파일 없음 → 샘플 데이터 생성")
            return generate_sample_data()

    df = pd.read_csv(path, encoding="cp949")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 처리 & 이상치 윈저화 (상하위 1%)
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 결측치: 중앙값 대체
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 이상치 윈저화
    for col in ["annual_income", "total_debt", "annual_repayment",
                "asset_total", "fin_asset", "real_estate"]:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    피처 엔지니어링

    [생성 피처]
    - dsr          : 총부채원리금상환비율 (%) = 연간상환액 / 연간소득 × 100
    - lti          : 소득 대비 총부채 비율 = 총부채 / 연간소득
    - var_rate_pct : 변동금리 비중 = 변동금리대출 / 총부채
    - mortgage_pct : 주담대 비중 = 주담대 / 총부채
    - dsr_stress   : 금리 +1%p 적용 스트레스 DSR
    """
    df = df.copy()
    eps = 1e-6   # 분모 0 방지

    df["dsr"]          = df["annual_repayment"] / (df["annual_income"] + eps) * 100
    df["lti"]          = df["total_debt"]        / (df["annual_income"] + eps)
    df["var_rate_pct"] = df["var_rate_debt"]     / (df["total_debt"]    + eps)
    df["mortgage_pct"] = df["mortgage"]          / (df["total_debt"]    + eps)

    # 금리 +1%p 시 변동금리 대출 연간 이자 증가분 → DSR 재계산
    extra_interest     = df["var_rate_debt"] * 0.01   # +1%p 이자 증가 (연간)
    df["dsr_stress"]   = (df["annual_repayment"] + extra_interest) / (df["annual_income"] + eps) * 100

    # 비율 클리핑 (0~1)
    for col in ["var_rate_pct", "mortgage_pct"]:
        df[col] = df[col].clip(0, 1)

    return df


def make_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    DSR 기준 위험 등급 레이블 생성

    0 = 저위험 (DSR < 20%)
    1 = 중위험 (DSR 20~40%)
    2 = 고위험 (DSR > 40%)
    """
    df = df.copy()
    df["risk"] = pd.cut(
        df["dsr"],
        bins=[-np.inf, DSR_LOW, DSR_MID, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    return df


def preprocess(path: str = None) -> pd.DataFrame:
    """
    전체 전처리 파이프라인

    Returns
    -------
    pd.DataFrame  피처 + 레이블 포함 완성 데이터
    """
    df = load_household(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = make_risk_label(df)

    # 클래스 분포 출력
    counts = df["risk"].value_counts().sort_index()
    labels = {0: "저위험", 1: "중위험", 2: "고위험"}
    print("\n📊 위험 등급 분포:")
    for k, v in counts.items():
        print(f"  {labels[k]}: {v:,}가구 ({v/len(df)*100:.1f}%)")

    # 저장
    save_path = os.path.join(DATA_DIR, "household_features.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료: {save_path}  ({len(df):,}행)")
    return df


if __name__ == "__main__":
    df = preprocess()
    print(df[HOUSEHOLD_FEATURES + ["risk"]].head())
    print(df[HOUSEHOLD_FEATURES].describe().round(2))
