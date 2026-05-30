"""
models/shap_analyzer.py
SHAP 기반 모델 설명 가능성 분석 모듈

[설치]
  pip install shap matplotlib seaborn
"""
"""
models/shap_analyzer.py
SHAP 기반 모델 설명 가능성 분석 모듈
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, HOUSEHOLD_FEATURES

matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rcParams["axes.unicode_minus"] = False


class SHAPAnalyzer:
    CLASS_NAMES = ["저위험", "중위험", "고위험"]

    def __init__(self, model, feature_names):
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(model)

    def _compute_shap(self, X):
        return self.explainer.shap_values(X)

    def _get_target_sv(self, shap_vals, class_idx):
        """멀티클래스/단일 클래스 모두 처리"""
        if isinstance(shap_vals, list):
            return shap_vals[class_idx]
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            return shap_vals[:, :, class_idx]
        else:
            return shap_vals

    # ── Summary Plot ────────────────────────────────────────────────────────
    def summary_plot(self, X, class_idx=2, save=True):
        shap_vals = self._compute_shap(X)
        target_sv = self._get_target_sv(shap_vals, class_idx)

        # X가 DataFrame이면 numpy로 변환
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X

        # feature_names 길이 맞추기
        n_feat = target_sv.shape[1] if target_sv.ndim == 2 else len(self.feature_names)
        feat_names = self.feature_names[:n_feat]

        shap.summary_plot(
            target_sv,
            X_np,
            feature_names=feat_names,
            show=False,
            max_display=12,
        )
        plt.title(f"SHAP Summary Plot ({self.CLASS_NAMES[class_idx]} 분류 기여도)")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, f"shap_summary_class{class_idx}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"저장: {path}")
        plt.show()
        plt.close()

    # ── 변수 중요도 테이블 ───────────────────────────────────────────────────
    def get_importance_df(self, X, class_idx=2):
        shap_vals = self._compute_shap(X)
        target_sv = self._get_target_sv(shap_vals, class_idx)

        # target_sv가 2D인지 확인
        if target_sv.ndim == 2:
            mean_abs = np.abs(target_sv).mean(axis=0)
        else:
            mean_abs = np.abs(target_sv).flatten()

        n_feat     = len(mean_abs)
        feat_names = self.feature_names[:n_feat]

        df = pd.DataFrame({
            "feature"    : feat_names,
            "importance" : mean_abs.tolist(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ── 막대 그래프 ──────────────────────────────────────────────────────────
    def bar_plot(self, X, class_idx=2, save=True):
        df = self.get_importance_df(X, class_idx).head(12)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(
            df["feature"].tolist()[::-1],
            df["importance"].tolist()[::-1],
            color="#3498DB", alpha=0.85
        )
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
        ax.set_xlabel("평균 |SHAP| 기여도")
        ax.set_title(f"피처 중요도 ({self.CLASS_NAMES[class_idx]} 분류 기준)")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, "shap_bar_importance.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"저장: {path}")
        plt.show()
        plt.close()

    # ── Waterfall Plot ───────────────────────────────────────────────────────
    def waterfall_plot(self, X, idx=0, class_idx=2, save=True):
        sv        = self.explainer(X)
        shap_vals = self._compute_shap(X)
        target_sv = self._get_target_sv(shap_vals, class_idx)

        n_feat     = target_sv.shape[1]
        feat_names = self.feature_names[:n_feat]

        sv_single = shap.Explanation(
            values        = target_sv[idx],
            base_values   = self.explainer.expected_value[class_idx]
                            if isinstance(self.explainer.expected_value, (list, np.ndarray))
                            else self.explainer.expected_value,
            data          = X[idx] if not hasattr(X, 'iloc') else X.iloc[idx].values,
            feature_names = feat_names,
        )

        shap.plots.waterfall(sv_single, show=False)
        plt.title(f"가구 #{idx} 위험도 판정 근거 ({self.CLASS_NAMES[class_idx]})")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, f"shap_waterfall_idx{idx}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"저장: {path}")
        plt.show()
        plt.close()

    # ── Dependence Plot ──────────────────────────────────────────────────────
    def dependence_plot(self, X, feature="dsr", class_idx=2, save=True):
        shap_vals = self._compute_shap(X)
        target_sv = self._get_target_sv(shap_vals, class_idx)

        n_feat     = target_sv.shape[1]
        feat_names = self.feature_names[:n_feat]

        if feature not in feat_names:
            print(f"'{feature}' 피처를 찾을 수 없습니다.")
            return

        feat_idx = feat_names.index(feature)

        shap.dependence_plot(
            feat_idx,
            target_sv,
            X if not hasattr(X, 'values') else X.values,
            feature_names=feat_names,
            show=False,
        )
        plt.title(f"{feature} 값 변화에 따른 SHAP 기여도")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, f"shap_dependence_{feature}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"저장: {path}")
        plt.show()
        plt.close()


if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from risk_classifier import RiskClassifier
    from data.household_preprocessor import preprocess

    clf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "risk_classifier.pkl"
    )

    if not os.path.exists(clf_path):
        print("risk_classifier.pkl 없음. risk_classifier.py 먼저 실행하세요.")
    else:
        clf = joblib.load(clf_path)
        df  = preprocess()
        X   = df[HOUSEHOLD_FEATURES].values
        y   = df["risk"].values

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        analyzer = SHAPAnalyzer(clf.model, HOUSEHOLD_FEATURES)

        print("Summary Plot 생성 중...")
        analyzer.summary_plot(X_test, class_idx=2)

        print("Bar Plot 생성 중...")
        analyzer.bar_plot(X_test, class_idx=2)

        print("Waterfall Plot 생성 중...")
        high_idx = np.where(y_test == 2)[0][0]
        analyzer.waterfall_plot(X_test, idx=high_idx, class_idx=2)

        print("Dependence Plot 생성 중...")
        analyzer.dependence_plot(X_test, feature="dsr", class_idx=2)

        print("\n📊 고위험 분류 주요 변수:")
        print(analyzer.get_importance_df(X_test, class_idx=2).to_string(index=False))