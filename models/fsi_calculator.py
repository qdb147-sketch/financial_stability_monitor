"""
models/fsi_calculator.py
금융 스트레스 지수(FSI: Financial Stress Index) 산출 모듈

[방법론]
  한국은행 금융안정보고서의 금융취약성 지수(FVI) 방법론 참고.
  5개 지표를 Z-Score 정규화 후 가중 합산.

[구성 지표 & 가중치]
  - KOSPI 변동성       (20d)  : 25%  → 주식시장 불안
  - 회사채-국채 스프레드     : 25%  → 신용 위험
  - 원달러 환율 변동성 (20d)  : 20%  → 외환시장 불안
  - 가계대출 전년비 증가율    : 15%  → 부채 취약성
  - 은행 BIS 비율(역수)       : 15%  → 은행 건전성
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, FSI_NORMAL, FSI_WARN, OUTPUT_DIR


class FSICalculator:
    """
    금융 스트레스 지수(FSI) 산출기

    사용 순서:
        calc = FSICalculator()
        calc.fit(df_train)          # 정규화 기준값(평균·표준편차) 학습
        fsi  = calc.transform(df)   # FSI 시계열 산출
        lbl  = calc.label(fsi)      # 경보 수준 분류
    """

    # 지표명 : 가중치 (합계 1.0)
    COMPONENTS = {
        "kospi_vol_20d" : 0.25,
        "credit_spread" : 0.25,
        "usdkrw_vol_20d": 0.20,
        "hh_loan_yoy"   : 0.15,
        "bis_ratio_inv" : 0.15,   # BIS의 역수 (낮을수록 위험)
    }

    def __init__(self):
        self.means_  = None
        self.stds_   = None
        self._fitted = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """BIS 역수 변환 등 전처리"""
        df = df.copy()
        if "bis_ratio" in df.columns:
            # BIS 역수: 높을수록 건전 → 역수로 변환해 높을수록 위험하게
            df["bis_ratio_inv"] = 1.0 / (df["bis_ratio"] + 1e-6)
        return df

    def fit(self, df: pd.DataFrame) -> "FSICalculator":
        """
        정규화 기준값 학습 (평균, 표준편차)

        Parameters
        ----------
        df : pd.DataFrame  학습용 거시 지표 데이터 (주로 위기 이전 평온기 데이터)
        """
        df = self._prepare(df)
        available = [c for c in self.COMPONENTS if c in df.columns]

        self.means_    = df[available].mean()
        self.stds_     = df[available].std().replace(0, 1e-6)
        self._fitted   = True
        self._avail    = available
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        FSI 시계열 산출

        Returns
        -------
        pd.Series  FSI 값 (0 이상, 클수록 금융 불안)
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        df = self._prepare(df)
        z  = (df[self._avail] - self.means_) / self.stds_

        # 가중 합산
        weights = pd.Series({c: self.COMPONENTS[c] for c in self._avail})
        fsi = (z * weights).sum(axis=1)

        # 음수 방지 (0 이상)
        fsi = fsi.clip(lower=0)

        if "date" in df.columns:
            fsi.index = pd.to_datetime(df["date"])

        return fsi

    def label(self, fsi: pd.Series) -> pd.Series:
        """
        FSI 수준 → 경보 등급 분류

        정상 (FSI < 0.3)
        주의 (0.3 ≤ FSI < 0.6)
        경계 (FSI ≥ 0.6)
        """
        return pd.cut(
            fsi,
            bins=[-np.inf, FSI_NORMAL, FSI_WARN, np.inf],
            labels=["정상", "주의", "경계"]
        )

    def get_contribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 구성 지표의 FSI 기여도 계산

        Returns
        -------
        pd.DataFrame  지표별 기여도 (같은 행 합계 = FSI)
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        df = self._prepare(df)
        z  = (df[self._avail] - self.means_) / self.stds_
        weights = pd.Series({c: self.COMPONENTS[c] for c in self._avail})
        contrib = z * weights

        if "date" in df.columns:
            contrib.index = pd.to_datetime(df["date"])

        return contrib

    def crisis_leadtime(self, fsi: pd.Series,
                        crisis_periods: dict,
                        level: str = "경계") -> pd.DataFrame:
        """
        과거 위기 대비 조기경보 리드타임 분석

        Parameters
        ----------
        fsi            : FSI 시계열
        crisis_periods : {"위기명": ("시작", "종료")} 딕셔너리
        level          : 경보 수준 ("주의" or "경계")

        Returns
        -------
        pd.DataFrame  위기별 리드타임(개월) 결과표
        """
        threshold = FSI_WARN if level == "경계" else FSI_NORMAL
        records = []

        for name, (start, end) in crisis_periods.items():
            crisis_start = pd.Timestamp(start)

            # 위기 발생 전 최초 임계값 초과 시점
            pre_crisis = fsi[fsi.index < crisis_start]
            exceed = pre_crisis[pre_crisis >= threshold]

            if exceed.empty:
                lead_months = None
            else:
                first_exceed = exceed.index[-1]
                lead_months  = round(
                    (crisis_start - first_exceed).days / 30.4
                )

            records.append({
                "위기": name,
                "위기 시작": start,
                f"{level} 수준 초과 최초일": first_exceed.strftime("%Y-%m") if not exceed.empty else "없음",
                "리드타임(개월)": lead_months,
            })

        return pd.DataFrame(records)

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "fsi_calculator.pkl")
        joblib.dump(self, path)
        print(f"✅ FSICalculator 저장: {path}")

    @staticmethod
    def load(path: str = None) -> "FSICalculator":
        if path is None:
            path = os.path.join(MODEL_DIR, "fsi_calculator.pkl")
        return joblib.load(path)


# ── 과거 금융 위기 기간 정의 ─────────────────────────────────────────────────
CRISIS_PERIODS = {
    "1997 외환위기"    : ("1997-11-01", "1998-12-01"),
    "2003 카드대란"    : ("2003-03-01", "2003-09-01"),
    "2008 금융위기"    : ("2008-09-01", "2009-06-01"),
    "2020 코로나 쇼크" : ("2020-03-01", "2020-06-01"),
    "2022 레고랜드"    : ("2022-09-01", "2022-12-01"),
}


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rc("font", family="Malgun Gothic")
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 병합 데이터 로드
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "macro_merged.csv"
    )
    if not os.path.exists(data_path):
        print("macro_merged.csv 없음. market_collector.py → merge_macro_market() 먼저 실행.")
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

        # 학습(2015 이전) / 전체 분리
        df_train = df[df["date"] < "2015-01-01"]
        calc = FSICalculator().fit(df_train)
        fsi  = calc.transform(df)
        lbl  = calc.label(fsi)

        # 리드타임 분석
        lt = calc.crisis_leadtime(fsi, CRISIS_PERIODS)
        print(lt)

        # 시각화
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(fsi.index, fsi.values, color="steelblue", linewidth=1.5, label="FSI")
        ax.fill_between(fsi.index, FSI_NORMAL, fsi.values,
                        where=fsi.values >= FSI_NORMAL,
                        alpha=0.2, color="orange", label="주의 구간")
        ax.fill_between(fsi.index, FSI_WARN, fsi.values,
                        where=fsi.values >= FSI_WARN,
                        alpha=0.3, color="red", label="경계 구간")
        for name, (s, e) in CRISIS_PERIODS.items():
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.15, color="crimson")
            ax.text(pd.Timestamp(s), ax.get_ylim()[1] * 0.9,
                    name, fontsize=7, rotation=45)
        ax.axhline(FSI_NORMAL, color="orange", linestyle="--", alpha=0.7)
        ax.axhline(FSI_WARN,   color="red",    linestyle="--", alpha=0.7)
        ax.set_title("금융 스트레스 지수(FSI) 시계열")
        ax.set_xlabel("날짜"); ax.set_ylabel("FSI")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fsi_timeseries.png"), dpi=150)
        plt.show()

        calc.save()
