"""
models/scenario_simulator.py
기준금리 변동 시나리오 시뮬레이터

[시나리오 정의]
  동결   : 기준금리 유지 (2.5%)
  인하   : -0.5%p (→ 2.0%)
  인상   : +0.5%p (→ 3.0%)

[두 층 연결 구조]
  금리 변동 → 1층(FSI 변화) + 2층(가계 DSR 재계산 → 위험 등급 변화)
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_classifier import RiskClassifier
from fsi_calculator import FSICalculator
from config import HOUSEHOLD_FEATURES, OUTPUT_DIR


# ── 시나리오 정의 ─────────────────────────────────────────────────────────────
SCENARIOS: Dict[str, float] = {
    "동결 (2.5%)"    :  0.0,
    "인하 (-0.5%p)"  : -0.5,
    "인상 (+0.5%p)"  : +0.5,
}

# FSI에 대한 금리 변동 영향 추정치 (회귀 기반 근사)
# 금리 +1%p → FSI +0.15p (회사채 스프레드 확대 효과 등)
FSI_RATE_SENSITIVITY = 0.15   # FSI 변화 / 금리 변화(%p)


class ScenarioSimulator:
    """
    금리 시나리오별 FSI·가계 위험도 동시 시뮬레이터

    사용 순서:
        sim = ScenarioSimulator(classifier, fsi_calculator)
        result = sim.run_all(df_hh, X_hh, current_fsi)
        mc     = sim.monte_carlo(df_hh, X_hh, delta=0.5)
    """

    def __init__(self, classifier, fsi_calculator):
        """
        Parameters
        ----------
        classifier     : RiskClassifier 또는 LGBMClassifier
        fsi_calculator : FSICalculator
        """
        self.clf = classifier
        self.fsi = fsi_calculator

    # ── 가계 DSR 재계산 ──────────────────────────────────────────────────────
    def apply_rate_shock(self, df: pd.DataFrame, delta: float) -> pd.DataFrame:
        """
        금리 변동(delta %p)을 가계 데이터에 반영

        [로직]
          변동금리 대출 이자 증가분 = 변동금리 대출 잔액 × delta/100
          신규 DSR = (기존 원리금 + 이자 증가분) / 연간소득 × 100

        Parameters
        ----------
        df    : 가계 피처 DataFrame (household_features.csv)
        delta : 기준금리 변동폭 (%p, 양수=인상, 음수=인하)
        """
        df = df.copy()
        eps = 1e-6

        # 변동금리 대출 이자 변화분 (연간, 만원 단위)
        if "var_rate_debt" in df.columns:
            extra_interest = df["var_rate_debt"] * (delta / 100)
        elif "var_rate_pct" in df.columns and "total_debt" in df.columns:
            extra_interest = df["total_debt"] * df["var_rate_pct"] * (delta / 100)
        else:
            extra_interest = pd.Series(0, index=df.index)

        # DSR 재계산
        if "annual_repayment" in df.columns and "annual_income" in df.columns:
            new_repayment  = df["annual_repayment"] + extra_interest
            df["dsr"]      = new_repayment / (df["annual_income"] + eps) * 100
            df["dsr_stress"] = df["dsr"] * (1 + df.get("var_rate_pct", 0) * 0.01)
        else:
            df["dsr"] = df["dsr"] + df.get("var_rate_pct", 0) * delta * 100

        df["dsr"] = df["dsr"].clip(0, 200)
        return df

    # ── 단일 시나리오 실행 ────────────────────────────────────────────────────
    def run_single(self, df: pd.DataFrame, X: np.ndarray,
                   delta: float, current_fsi: float) -> dict:
        """
        단일 금리 시나리오 실행

        Returns
        -------
        dict  {n_high, n_mid, n_low, fsi_new, delta_high, delta_fsi}
        """
        # 가계 시뮬레이션
        df_s = self.apply_rate_shock(df, delta)
        X_s  = df_s[HOUSEHOLD_FEATURES].values
        pred = self.clf.predict(X_s)

        n_high = int((pred == 2).sum())
        n_mid  = int((pred == 1).sum())
        n_low  = int((pred == 0).sum())

        # FSI 변화 추정
        fsi_new = current_fsi + delta * FSI_RATE_SENSITIVITY

        # 기준(delta=0) 대비 증감
        pred_base   = self.clf.predict(X)
        base_high   = int((pred_base == 2).sum())
        delta_high  = n_high - base_high

        return {
            "n_high"     : n_high,
            "n_mid"      : n_mid,
            "n_low"      : n_low,
            "fsi_new"    : round(fsi_new, 4),
            "delta_high" : delta_high,
            "delta_fsi"  : round(delta * FSI_RATE_SENSITIVITY, 4),
        }

    # ── 전체 시나리오 비교 ────────────────────────────────────────────────────
    def run_all(self, df: pd.DataFrame, X: np.ndarray,
                current_fsi: float) -> pd.DataFrame:
        """
        사전 정의된 모든 시나리오 실행 후 결과 테이블 반환

        Returns
        -------
        pd.DataFrame  시나리오별 결과 비교표
        """
        records = []
        for name, delta in SCENARIOS.items():
            res = self.run_single(df, X, delta, current_fsi)
            records.append({
                "시나리오"       : name,
                "금리 변동(%p)"  : delta,
                "고위험 가구 수" : f"{res['n_high']:,}",
                "증감"           : f"{res['delta_high']:+,}",
                "예상 FSI"       : res["fsi_new"],
                "FSI 변화"       : f"{res['delta_fsi']:+.3f}",
            })

        df_result = pd.DataFrame(records)
        print("\n📊 금리 시나리오별 시뮬레이션 결과")
        print(df_result.to_string(index=False))

        save_path = os.path.join(OUTPUT_DIR, "scenario_result.csv")
        df_result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장: {save_path}")
        return df_result

    # ── Monte Carlo 불확실성 분석 ─────────────────────────────────────────────
    def monte_carlo(self, df: pd.DataFrame, X: np.ndarray,
                    delta: float,
                    n_iter: int = 1000,
                    income_noise: float = 0.02,
                    asset_noise:  float = 0.05,
                    seed: int = 42) -> dict:
        """
        Monte Carlo 시뮬레이션으로 고위험 가구 수 95% 신뢰구간 추정

        Parameters
        ----------
        delta        : 금리 변동폭 (%p)
        n_iter       : 반복 횟수 (기본 1,000)
        income_noise : 소득 불확실성 표준편차 (기본 ±2%)
        asset_noise  : 자산 불확실성 표준편차 (기본 ±5%)
        """
        rng     = np.random.default_rng(seed)
        results = []

        for i in range(n_iter):
            df_s = self.apply_rate_shock(df, delta)

            # 소득·자산 불확실성 반영
            if "annual_income" in df_s.columns:
                noise = rng.normal(1.0, income_noise, size=len(df_s))
                df_s["annual_income"] = (df_s["annual_income"] * noise).clip(100)
                eps = 1e-6
                if "annual_repayment" in df_s.columns:
                    df_s["dsr"] = (df_s["annual_repayment"]
                                   / (df_s["annual_income"] + eps) * 100)

            if "asset_total" in df_s.columns:
                a_noise = rng.normal(1.0, asset_noise, size=len(df_s))
                df_s["asset_total"] = (df_s["asset_total"] * a_noise).clip(0)

            X_s    = df_s[HOUSEHOLD_FEATURES].values
            pred   = self.clf.predict(X_s)
            results.append(int((pred == 2).sum()))

        arr = np.array(results)
        summary = {
            "mean"   : float(arr.mean()),
            "std"    : float(arr.std()),
            "ci95_lo": float(np.percentile(arr, 2.5)),
            "ci95_hi": float(np.percentile(arr, 97.5)),
            "min"    : float(arr.min()),
            "max"    : float(arr.max()),
        }
        print(f"\n📊 Monte Carlo 결과 (금리 {delta:+.1f}%p, {n_iter}회 반복)")
        print(f"  평균 고위험 가구: {summary['mean']:,.0f}가구")
        print(f"  95% 신뢰구간:    [{summary['ci95_lo']:,.0f} ~ {summary['ci95_hi']:,.0f}]")
        return summary

    # ── 지역별 분석 ────────────────────────────────────────────────────────────
    def regional_analysis(self, df: pd.DataFrame, X: np.ndarray,
                          delta: float) -> pd.DataFrame:
        """
        금리 변동 시 지역별 고위험 가구 증가율 분석

        Returns
        -------
        pd.DataFrame  지역별 분석 결과
        """
        if "region" not in df.columns:
            print("region 컬럼 없음")
            return pd.DataFrame()

        df_s = self.apply_rate_shock(df, delta)
        X_s  = df_s[HOUSEHOLD_FEATURES].values

        pred_base = self.clf.predict(X)
        pred_new  = self.clf.predict(X_s)

        df_result = df[["region"]].copy()
        df_result["base_high"] = (pred_base == 2).astype(int)
        df_result["new_high"]  = (pred_new  == 2).astype(int)

        region_map = {1: "수도권", 2: "비수도권"}
        df_result["region_name"] = df_result["region"].map(region_map)

        agg = df_result.groupby("region_name").agg(
            base_high=("base_high", "sum"),
            new_high =("new_high",  "sum"),
            total    =("base_high", "count"),
        ).reset_index()
        agg["기준 고위험률(%)"]     = (agg["base_high"] / agg["total"] * 100).round(2)
        agg["시나리오 고위험률(%)"] = (agg["new_high"]  / agg["total"] * 100).round(2)
        agg["증감(%p)"]            = (agg["시나리오 고위험률(%)"] - agg["기준 고위험률(%)"]).round(2)

        print(f"\n📊 지역별 고위험 가구 변화 (금리 {delta:+.1f}%p)")
        print(agg[["region_name","기준 고위험률(%)","시나리오 고위험률(%)","증감(%p)"]].to_string(index=False))
        return agg


if __name__ == "__main__":
    import joblib
    from data.household_preprocessor import preprocess

    clf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "risk_classifier.pkl"
    )
    fsi_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "fsi_calculator.pkl"
    )

    clf = joblib.load(clf_path)
    fsi = joblib.load(fsi_path)

    df = preprocess()
    X  = df[HOUSEHOLD_FEATURES].values

    sim = ScenarioSimulator(clf, fsi)

    # 전체 시나리오 비교
    current_fsi = 0.42   # 예시 현재 FSI 값
    result = sim.run_all(df, X, current_fsi)

    # Monte Carlo: 금리 +0.5%p
    mc = sim.monte_carlo(df, X, delta=0.5, n_iter=500)

    # 지역별 분석
    regional = sim.regional_analysis(df, X, delta=0.5)
