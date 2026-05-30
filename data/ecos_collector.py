"""
data/ecos_collector.py
한국은행 ECOS API 데이터 수집 모듈

[필요 데이터 & 출처]
  - 기준금리     : ECOS → 통화/금융 → 금리 → 한국은행 기준금리 (통계코드: 722Y001)
  - 소비자물가   : ECOS → 물가 → 소비자물가지수 (통계코드: 901Y009)
  - GDP 성장률   : ECOS → 국민계정 → 실질 GDP 성장률 (통계코드: 200Y102)
  - M2 광의통화  : ECOS → 통화/금융 → 통화량 M2 (통계코드: 101Y004)
  - 가계대출     : ECOS → 금융 → 예금은행 가계대출 (통계코드: 104Y014)
  - 회사채 금리  : ECOS → 금리 → 회사채(AA-) 3년 (통계코드: 817Y002)
  - 국고채 금리  : ECOS → 금리 → 국고채 3년 (통계코드: 817Y002)

[ECOS API 발급]
  1) https://ecos.bok.or.kr 접속
  2) 회원가입 → API 키 신청 (무료, 즉시 발급)
  3) config.py의 ECOS_API_KEY에 입력
"""

"""
data/ecos_collector.py
한국은행 ECOS API 데이터 수집 모듈
"""

import requests
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ECOS_API_KEY, ECOS_START, ECOS_END, DATA_DIR


# ── ECOS 통계 코드 정의 ──────────────────────────────────────────────────────
STAT_CODES = {
    "base_rate" : ("722Y001", "0101000", "M"),  # 한국은행 기준금리
    "cpi"       : ("901Y009", "0",       "M"),  # 소비자물가지수
    "m2"        : ("161Y006", "BBHA00",  "M"),  # M2 광의통화
    "hh_loan"   : ("151Y002", "1110000", "M"),  # 예금취급기관 가계대출
    "corp_bond" : ("721Y001", "1040000", "M"),  # 회사채(AA-) 3년
    "gov_bond"  : ("721Y001", "5090000", "M"),  # 국고채 3년
}


def fetch_ecos(stat_code: str, item_code: str, cycle: str = "M",
               start: str = ECOS_START, end: str = ECOS_END) -> pd.DataFrame:
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{ECOS_API_KEY}/json/kr/1/10000"
        f"/{stat_code}/{cycle}/{start}/{end}/{item_code}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "StatisticSearch" not in data:
            print(f"  ⚠ 데이터 없음: {data}")
            return pd.DataFrame(columns=["date", "value"])

        rows = data["StatisticSearch"]["row"]
        df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"]  = pd.to_datetime(df["date"], format="%Y%m")
        df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
        return df

    except requests.RequestException as e:
        print(f"[ECOS] 네트워크 오류: {e}")
        return pd.DataFrame(columns=["date", "value"])


def get_all_macro() -> pd.DataFrame:
    print("ECOS API 데이터 수집 중...")

    frames = {}
    for name, (stat, item, cycle) in STAT_CODES.items():
        print(f"  - {name} 수집 중...")
        df_tmp = fetch_ecos(stat, item, cycle)
        if df_tmp.empty:
            print(f"  ⚠ {name} 데이터 없음 - 건너뜀")
            continue
        frames[name] = df_tmp.set_index("date")["value"]

    idx = pd.date_range(
        start=f"{ECOS_START[:4]}-{ECOS_START[4:]}-01",
        end  =f"{ECOS_END[:4]}-{ECOS_END[4:]}-01",
        freq ="MS"
    )
    result = pd.DataFrame(index=idx)

    for name, series in frames.items():
        result[name] = series.reindex(idx, method="ffill")

    if "corp_bond" in result.columns and "gov_bond" in result.columns:
        result["credit_spread"] = result["corp_bond"] - result["gov_bond"]

    if "hh_loan" in result.columns:
        result["hh_loan_yoy"] = result["hh_loan"].pct_change(12) * 100

    result = result.dropna(how="all").reset_index().rename(columns={"index": "date"})

    save_path = os.path.join(DATA_DIR, "macro_ecos.csv")
    result.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {save_path}  ({len(result)}행)")
    return result


if __name__ == "__main__":
    df = get_all_macro()
    print(df.tail())
    print(df.dtypes)