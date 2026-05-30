"""
models/risk_classifier.py
LightGBM 기반 가계부채 위험도 분류 모델

[분류 기준]
  0 = 저위험 (DSR < 20%)
  1 = 중위험 (DSR 20~40%)
  2 = 고위험 (DSR > 40%)

[설치]
  pip install lightgbm scikit-learn imbalanced-learn optuna joblib
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, average_precision_score
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, HOUSEHOLD_FEATURES, OUTPUT_DIR


class RiskClassifier:
    """
    가계부채 위험도 3분류 모델

    사용 순서:
        clf = RiskClassifier()
        clf.fit(X_train, y_train)       # Optuna 튜닝 포함
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        clf.save()
    """

    def __init__(self, n_trials: int = 50, cv_folds: int = 5,
                 random_state: int = 42):
        self.n_trials     = n_trials
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.model        = None
        self.scaler       = StandardScaler()
        self.best_params  = None

    # ── Optuna 목적 함수 ──────────────────────────────────────────────────────
    def _objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        params = {
            "num_leaves"       : trial.suggest_int("num_leaves", 20, 100),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.0, 1.0),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators"     : 300,
            "n_jobs"           : -1,
            "verbose"          : -1,
            "random_state"     : self.random_state,
            "objective"        : "multiclass",
            "num_class"        : 3,
        }

        skf    = StratifiedKFold(n_splits=self.cv_folds,
                                 shuffle=True,
                                 random_state=self.random_state)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # SMOTE 오버샘플링 (고위험 클래스 보강)
            X_tr_res, y_tr_res = SMOTE(
                random_state=self.random_state
            ).fit_resample(X_tr, y_tr)

            m = LGBMClassifier(**params)
            m.fit(X_tr_res, y_tr_res,
                  eval_set=[(X_val, y_val)],
                  callbacks=[])

            pred   = m.predict(X_val)
            scores.append(f1_score(y_val, pred, average="macro"))

        return np.mean(scores)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiskClassifier":
        """
        Optuna 하이퍼파라미터 튜닝 + 최종 모델 학습

        Parameters
        ----------
        X : np.ndarray  (n_samples, n_features)
        y : np.ndarray  (n_samples,)  0/1/2 레이블
        """
        print(f"\n🔍 Optuna 튜닝 시작 ({self.n_trials}회 trial)...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(
                                        seed=self.random_state))
        study.optimize(
            lambda t: self._objective(t, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        self.best_params = study.best_params
        print(f"✅ 최적 파라미터: {self.best_params}")
        print(f"   Macro F1 (CV): {study.best_value:.4f}")

        # SMOTE 최종 적용
        X_res, y_res = SMOTE(random_state=self.random_state).fit_resample(X, y)

        # 최종 모델 학습
        final_params = {**self.best_params,
                        "n_estimators": 500,
                        "n_jobs": -1,
                        "verbose": -1,
                        "random_state": self.random_state,
                        "objective": "multiclass",
                        "num_class": 3}
        self.model = LGBMClassifier(**final_params)
        self.model.fit(X_res, y_res)
        return self

    # ── 예측 ─────────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    # ── 평가 ─────────────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        테스트셋 성능 평가

        Returns
        -------
        dict  주요 지표 딕셔너리
        """
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        print("\n📊 가계부채 위험도 분류 성능")
        print("=" * 50)
        print(classification_report(
            y_test, y_pred,
            target_names=["저위험", "중위험", "고위험"]
        ))

        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        ap  = average_precision_score(
            pd.get_dummies(y_test).values, y_proba, average="macro"
        )
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"AUC-ROC (macro OvR): {auc:.4f}")
        print(f"Average Precision:   {ap:.4f}")
        print(f"Macro F1-Score:      {macro_f1:.4f}")

        return {
            "macro_f1": macro_f1,
            "auc_roc" : auc,
            "avg_prec": ap,
            "cm"      : confusion_matrix(y_test, y_pred),
        }

    def plot_confusion_matrix(self, X_test: np.ndarray,
                              y_test: np.ndarray, save: bool = True):
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        matplotlib.rc("font", family="Malgun Gothic")
        matplotlib.rcParams["axes.unicode_minus"] = False

        y_pred = self.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["저위험", "중위험", "고위험"],
                    yticklabels=["저위험", "중위험", "고위험"],
                    ax=ax)
        ax.set_xlabel("예측 등급"); ax.set_ylabel("실제 등급")
        ax.set_title("혼동행렬 (가계부채 위험도 분류)")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
            plt.savefig(path, dpi=150)
            print(f"저장: {path}")
        plt.show()

    # ── 저장 / 로드 ───────────────────────────────────────────────────────────
    def save(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "risk_classifier.pkl")
        joblib.dump(self, path)
        print(f"✅ 모델 저장: {path}")

    @staticmethod
    def load(path: str = None) -> "RiskClassifier":
        if path is None:
            path = os.path.join(MODEL_DIR, "risk_classifier.pkl")
        return joblib.load(path)


def run_training():
    """학습 전체 파이프라인 실행"""
    from data.household_preprocessor import preprocess

    # 1) 데이터 로드 & 전처리
    df = preprocess()
    X  = df[HOUSEHOLD_FEATURES].values
    y  = df["risk"].values

    # 2) 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"학습: {len(X_train):,}  |  테스트: {len(X_test):,}")

    # 3) 학습
    clf = RiskClassifier(n_trials=50)
    clf.fit(X_train, y_train)

    # 4) 평가
    clf.evaluate(X_test, y_test)
    clf.plot_confusion_matrix(X_test, y_test)

    # 5) 저장
    clf.save()
    return clf, X_test, y_test


if __name__ == "__main__":
    clf, X_test, y_test = run_training()
