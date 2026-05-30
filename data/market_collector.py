"""
data/market_collector.py
금융시장 데이터 수집 모듈 (yfinance)

[필요 데이터 & 출처]
  - KOSPI        : yfinance 티커 "^KS11"
  - 원달러 환율  : yfinance 티커 "KRW=X"
  - BIS 비율     : 금융감독원 은행경영통계 (수동 입력 또는 금감원 OpenAPI)
                   https://www.fss.or.kr/fss/kr/promo/bodobbs_list.jsp

[설치]
  pip install yfinance pandas numpy
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKET_START, MARKET_END, DATA_DIR


# ── 주요 티커 정의 ────────────────────────────────────────────────────────────
TICKERS = {
    "kospi"   : "^KS11",    # KOSPI 종합지수
    "usdkrw"  : "KRW=X",    # 원달러 환율
    "sp500"   : "^GSPC",    # S&P500 (글로벌 리스크 참고용)
    "vix"     : "^VIX",     # VIX 공포지수 (글로벌 변동성 참고용)
}


def download_prices(start: str = MARKET_START,
                    end:   str = MARKET_END) -> pd.DataFrame:
    """
    yfinance로 가격 데이터 일별 수집

    Returns
    -------
    pd.DataFrame  index=날짜, columns=[kospi, usdkrw, sp500, vix]
    """
    print("yfinance 가격 데이터 수집 중...")
    frames = {}
    for name, ticker in TICKERS.items():
        print(f"  - {ticker} 수집 중...")
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  ⚠ {ticker} 데이터 없음")
                continue
            frames[name] = raw["Close"]
        except Exception as e:
            print(f"  ⚠ {ticker} 오류: {e}")

    if not frames:
        return pd.DataFrame()

    #df = pd.DataFrame(frames)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.columns = list(frames.keys())
    df.index.name = "date"
    return df.dropna(how="all")


def compute_volatility(df_prices: pd.DataFrame,
                       window_short: int = 20,
                       window_long:  int = 60) -> pd.DataFrame:
    """
    변동성 지표 계산

    - 일별 수익률의 롤링 표준편차 (연율화)
    - 단기(20일), 장기(60일) 비교

    Returns
    -------
    pd.DataFrame  columns=[kospi_vol_20d, kospi_vol_60d,
                            usdkrw_vol_20d, usdkrw_vol_60d]
    """
    result = pd.DataFrame(index=df_prices.index)

    for col in ["kospi", "usdkrw"]:
        if col not in df_prices.columns:
            continue
        ret = df_prices[col].pct_change()
        ann = np.sqrt(252)   # 연율화 계수
        result[f"{col}_vol_{window_short}d"] = ret.rolling(window_short).std() * ann
        result[f"{col}_vol_{window_long}d"]  = ret.rolling(window_long).std()  * ann

    return result.dropna(how="all")


def resample_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    일별 → 월별 집계
      - 가격  : 월말 값 (last)
      - 변동성: 월 평균 (mean)
    """
    df_m = df_daily.resample("MS").agg({
        col: ("mean" if "vol" in col else "last")
        for col in df_daily.columns
        if col in df_daily.columns
    })
    df_m.index.name = "date"
    return df_m.reset_index()


def add_bis_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    은행 BIS 비율 추가 (금감원 은행경영통계 수동 입력)

    [데이터 출처]
      금융감독원 → 통계 → 은행 → 은행경영통계 → BIS 자기자본비율
      URL: https://www.fss.or.kr/fss/kr/promo/bodobbs_list.jsp
      → 반기별 공시 데이터를 수동으로 입력 후 월별 forward-fill

    아래는 예시 데이터 (실제 값으로 교체 필요)
    """
    bis_data = {
        "2015-01-01": 14.01,
        "2016-01-01": 14.55,
        "2017-01-01": 14.83,
        "2018-01-01": 15.48,
        "2019-01-01": 15.86,
        "2020-01-01": 15.60,
        "2021-01-01": 16.30,
        "2022-01-01": 16.15,
        "2023-01-01": 16.40,
        "2024-01-01": 16.52,
        "2025-01-01": 16.67,
        "2026-01-01": 16.70,
    }
    bis_s = pd.Series(bis_data, name="bis_ratio")
    bis_s.index = pd.to_datetime(bis_s.index)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["bis_ratio"] = bis_s.reindex(df.index, method="ffill")
    return df.reset_index()


def get_market_data() -> pd.DataFrame:
    """
    금융시장 데이터 전체 파이프라인

    Returns
    -------
    pd.DataFrame  월별 금융시장 지표 (변동성, BIS 등)
    """
    # 1) 가격 수집
    df_prices = download_prices()
    if df_prices.empty:
        raise RuntimeError("yfinance 데이터 수집 실패")

    # 2) 변동성 계산
    df_vol = compute_volatility(df_prices)

    # 3) 가격 + 변동성 합치기
    df_all = pd.concat([df_prices, df_vol], axis=1)

    # 4) 월별 집계
    df_m = resample_monthly(df_all)

    # 5) BIS 비율 추가
    df_m = add_bis_ratio(df_m)

    # 저장
    save_path = os.path.join(DATA_DIR, "market_data.csv")
    df_m.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {save_path}  ({len(df_m)}행)")
    return df_m


def merge_macro_market() -> pd.DataFrame:
    """
    ECOS 거시 데이터 + 시장 데이터 병합

    [선행 조건] macro_ecos.csv, market_data.csv 존재해야 함
    """
    ecos_path   = os.path.join(DATA_DIR, "macro_ecos.csv")
    market_path = os.path.join(DATA_DIR, "market_data.csv")

    df_ecos   = pd.read_csv(ecos_path,   parse_dates=["date"])
    df_market = pd.read_csv(market_path, parse_dates=["date"])

    df = pd.merge(df_ecos, df_market, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    save_path = os.path.join(DATA_DIR, "macro_merged.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 병합 저장: {save_path}  ({len(df)}행)")
    return df


if __name__ == "__main__":
    df_market = get_market_data()
    print(df_market.tail())
    df_merged = merge_macro_market()
    print(df_merged.columns.tolist())
