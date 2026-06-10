"""
dashboard/app.py
금융안정 2층 모니터링 시스템 Streamlit 대시보드

[실행 방법]
  VSCode 터미널에서:
    streamlit run dashboard/app.py

  브라우저에서 http://localhost:8501 접속
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from risk_classifier import RiskClassifier
from fsi_calculator import FSICalculator

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    HOUSEHOLD_FEATURES, FSI_NORMAL, FSI_WARN
)

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="금융안정 2층 모니터링",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

# ── 사이드바 ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
st.sidebar.markdown("---")

delta = st.sidebar.slider(
    "기준금리 변동폭 (%p)",
    min_value=-1.0, max_value=1.0,
    value=0.0, step=0.1,
    help="양수=인상, 음수=인하"
)
st.sidebar.markdown(f"**현재 기준금리 가정:** {2.5 + delta:.1f}%")
st.sidebar.markdown("---")
show_mc = st.sidebar.checkbox("Monte Carlo 신뢰구간 표시", value=False)
n_mc    = st.sidebar.slider("MC 반복 횟수", 100, 1000, 300, 100,
                             disabled=not show_mc)

# ── 데이터 & 모델 로드 ───────────────────────────────────────────────────────
@st.cache_data
def load_macro():
    path = os.path.join(DATA_DIR, "macro_merged.csv")
    if not os.path.exists(path):
        # 샘플 데이터 생성 (실제 데이터 없을 때)
        dates = pd.date_range("2010-01-01", "2026-12-01", freq="MS")
        np.random.seed(42)
        base = np.cumsum(np.random.randn(len(dates)) * 0.02)
        fsi  = np.abs(base) + 0.1
        # 위기 구간 강조
        for s, e, boost in [("2020-03", "2020-06", 0.6),
                             ("2022-09", "2022-12", 0.4)]:
            mask = (dates >= s) & (dates <= e)
            fsi[mask] += boost
        df = pd.DataFrame({"date": dates, "fsi_value": fsi.clip(0, 1.5)})
        return df
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_resource
def load_models():
    clf_path = os.path.join(MODEL_DIR, "risk_classifier.pkl")
    fsi_path = os.path.join(MODEL_DIR, "fsi_calculator.pkl")
    clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
    fsi = joblib.load(fsi_path) if os.path.exists(fsi_path) else None
    return clf, fsi

@st.cache_data
def load_household():
    path = os.path.join(DATA_DIR, "household_features.csv")
    if not os.path.exists(path):
        from data.household_preprocessor import preprocess
        return preprocess()
    return pd.read_csv(path)


df_macro = load_macro()
clf, fsi_calc = load_models()
df_hh = load_household()

# FSI 계산 (모델 없을 때 샘플 사용)
if fsi_calc is not None and "macro_merged" in str(DATA_DIR):
    fsi_series = fsi_calc.transform(df_macro)
else:
    fsi_series = df_macro.get("fsi_value",
                               pd.Series(np.random.rand(len(df_macro)) * 0.8))
    fsi_series.index = pd.to_datetime(df_macro["date"])

current_fsi = float(fsi_series.iloc[-1])

# 가계 예측
X_hh = df_hh[HOUSEHOLD_FEATURES].values
if clf is not None:
    pred_base = clf.predict(X_hh)
else:
    # 모델 없을 때: DSR 기준 단순 분류
    pred_base = pd.cut(
        df_hh["dsr"], bins=[-np.inf, 20, 40, np.inf], labels=[0, 1, 2]
    ).astype(int).values

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.title("🏦 금융안정 2층 모니터링 시스템")
st.caption(
    "한국은행 2026 통화신용정책 운영방향(금융안정 도모) 기반 | "
    "AI소프트웨어학부 포트폴리오 | 백 유 진"
)
st.divider()

# ── KPI 카드 ─────────────────────────────────────────────────────────────────
fsi_label = "정상" if current_fsi < FSI_NORMAL else ("주의" if current_fsi < FSI_WARN else "경계")
fsi_delta = current_fsi - float(fsi_series.iloc[-2]) if len(fsi_series) > 1 else 0

n_high = int((pred_base == 2).sum())
n_total = len(pred_base)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재 FSI",       f"{current_fsi:.3f}",
          delta=f"{fsi_delta:+.3f} (전월비)")
c2.metric("경보 수준",       fsi_label,
          delta=None)
c3.metric("고위험 가구 수",  f"{n_high:,}가구",
          delta=f"{n_high/n_total*100:.1f}%")
c4.metric("분석 가구 수",    f"{n_total:,}가구")
c5.metric("분석 기간",       f"{len(fsi_series)}개월")

st.divider()

# ── 탭 구성 ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 FSI 실시간 모니터링",
    "💳 가계부채 위험도 분석",
    "🎯 금리 시나리오 시뮬레이터"
])

# ══════════════════════════════════════════════════════
# 탭 1: FSI 실시간 모니터링
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 금융 스트레스 지수(FSI) 시계열")

    # FSI 라인 차트
    fig_fsi = go.Figure()

    # 정상 구간 배경
    fig_fsi.add_hrect(
        y0=0, y1=FSI_NORMAL,
        fillcolor="rgba(46,204,113,0.08)", line_width=0,
        annotation_text="정상", annotation_position="top left"
    )
    # 주의 구간 배경
    fig_fsi.add_hrect(
        y0=FSI_NORMAL, y1=FSI_WARN,
        fillcolor="rgba(243,156,18,0.08)", line_width=0,
        annotation_text="주의", annotation_position="top left"
    )
    # 경계 구간 배경
    fig_fsi.add_hrect(
        y0=FSI_WARN, y1=2.0,
        fillcolor="rgba(231,76,60,0.08)", line_width=0,
        annotation_text="경계", annotation_position="top left"
    )

    # FSI 라인
    fig_fsi.add_trace(go.Scatter(
        x=fsi_series.index,
        y=fsi_series.values,
        name="FSI",
        line=dict(color="#2980B9", width=2),
        hovertemplate="%{x|%Y-%m}<br>FSI: %{y:.4f}<extra></extra>"
    ))

    # 임계선
    fig_fsi.add_hline(y=FSI_NORMAL, line_dash="dash",
                      line_color="orange", opacity=0.6,
                      annotation_text=f"주의선 ({FSI_NORMAL})")
    fig_fsi.add_hline(y=FSI_WARN, line_dash="dash",
                      line_color="red", opacity=0.6,
                      annotation_text=f"경계선 ({FSI_WARN})")

    # 과거 위기 구간 표시
    crises = [
        ("1997-11-01", "1998-12-01", "외환위기"),
        ("2008-09-01", "2009-06-01", "금융위기"),
        ("2020-03-01", "2020-06-01", "코로나 쇼크"),
        ("2022-09-01", "2022-12-01", "레고랜드"),
    ]
    for s, e, name in crises:
        fig_fsi.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(220,50,50,0.12)", line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation_font_size=9
        )

    fig_fsi.update_layout(
        height=420,
        xaxis=dict(title="날짜", tickangle=-45),  # x축 라벨을 -45도 회전
        yaxis_title="FSI",
        legend=dict(orientation="h"),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=40, b=80),   # 기울어진 글씨가 잘리지 않도록 하단 여백(b)을 80으로 확대
    )
    st.plotly_chart(fig_fsi, use_container_width=True)

    # FSI 구성 기여도
    if fsi_calc is not None:
        try:
            contrib = fsi_calc.get_contribution(df_macro)
            st.subheader("구성 지표별 FSI 기여도 (최근 24개월)")
            recent = contrib.tail(24)
            fig_contrib = go.Figure()
            colors = ["#3498DB","#E74C3C","#F39C12","#2ECC71","#9B59B6"]
            for i, col in enumerate(recent.columns):
                fig_contrib.add_trace(go.Bar(
                    name=col, x=recent.index, y=recent[col],
                    marker_color=colors[i % len(colors)]
                ))
            fig_contrib.update_layout(
                barmode="stack", height=550,
                xaxis=dict(title="날짜", tickangle=-20),  # x축 라벨을 -45도 회전
                yaxis_title="기여도",
                legend=dict(orientation="h"),
                margin=dict(b=0),                     # 하단 여백 추가
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        except Exception:
            pass

# ══════════════════════════════════════════════════════
# 탭 2: 가계부채 위험도 분석
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("💳 가계부채 위험도 분류 결과")

    df_vis = df_hh.copy()
    df_vis["위험등급"] = pd.Categorical(
        pred_base, categories=[0, 1, 2]
    ).rename_categories(["저위험", "중위험", "고위험"])

    col1, col2 = st.columns(2)

    # 파이 차트
    with col1:
        counts = df_vis["위험등급"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker_colors=["#2ECC71", "#F39C12", "#E74C3C"],
            hole=0.35,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title="위험 등급 분포", height=320,
                              showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # DSR 분포 히스토그램
    with col2:
        fig_dsr = px.histogram(
            df_vis, x="dsr", color="위험등급",
            color_discrete_map={
                "저위험": "#2ECC71",
                "중위험": "#F39C12",
                "고위험": "#E74C3C"
            },
            nbins=50,
            title="DSR 분포 (위험 등급별)",
            labels={"dsr": "DSR (%)"},
            barmode="overlay",
            opacity=0.7,
        )
        fig_dsr.update_layout(height=320)
        st.plotly_chart(fig_dsr, use_container_width=True)

    # 소득분위별 고위험 비율
    st.subheader("소득분위별 고위험 가구 비율")
    if "annual_income" in df_vis.columns:
        df_vis["소득분위"] = pd.qcut(
            df_vis["annual_income"], q=5,
            labels=["1분위\n(최저)", "2분위", "3분위", "4분위", "5분위\n(최고)"]
        )
        quintile = (
            df_vis.groupby("소득분위")
            .apply(lambda x: (x["위험등급"] == "고위험").mean() * 100)
            .reset_index()
        )
        quintile.columns = ["소득분위", "고위험 비율(%)"]

        fig_q = px.bar(
            quintile, x="소득분위", y="고위험 비율(%)",
            color="고위험 비율(%)",
            color_continuous_scale="Reds",
            text_auto=".1f",
            title="소득분위별 고위험 가구 비율 (%)",
        )
        fig_q.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_q, use_container_width=True)

    # 주요 통계 테이블
    st.subheader("위험 등급별 주요 지표 평균")
    summary_tbl = df_vis.groupby("위험등급")[
        ["dsr", "lti", "var_rate_pct", "annual_income"]
    ].mean().round(2)
    summary_tbl.columns = ["평균 DSR(%)", "LTI", "변동금리비중", "연소득(만원)"]
    st.dataframe(summary_tbl, use_container_width=True)

# ══════════════════════════════════════════════════════
# 탭 3: 금리 시나리오 시뮬레이터
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 기준금리 변동 시나리오 시뮬레이션")
    st.info(
        f"현재 사이드바에서 설정한 금리 변동폭: **{delta:+.1f}%p** "
        f"→ 기준금리 {2.5 + delta:.1f}%"
    )

    # 시나리오 적용
    from models.scenario_simulator import ScenarioSimulator, FSI_RATE_SENSITIVITY

    sim = ScenarioSimulator(clf, fsi_calc)

    # 금리 충격 적용
    df_stress = sim.apply_rate_shock(df_hh, delta)
    X_stress  = df_stress[HOUSEHOLD_FEATURES].values

    if clf is not None:
        pred_stress = clf.predict(X_stress)
    else:
        pred_stress = pd.cut(
            df_stress["dsr"], bins=[-np.inf, 20, 40, np.inf], labels=[0, 1, 2]
        ).astype(int).values

    n_high_base   = int((pred_base   == 2).sum())
    n_high_stress = int((pred_stress == 2).sum())
    diff_high     = n_high_stress - n_high_base
    fsi_new       = current_fsi + delta * FSI_RATE_SENSITIVITY

    # 결과 KPI
    st.markdown("### 시뮬레이션 결과")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("기준 고위험 가구",    f"{n_high_base:,}가구")
    r2.metric("시나리오 고위험 가구", f"{n_high_stress:,}가구",
              delta=f"{diff_high:+,}가구",
              delta_color="inverse")
    r3.metric("기준 FSI",     f"{current_fsi:.3f}")
    r4.metric("시나리오 FSI", f"{fsi_new:.3f}",
              delta=f"{delta * FSI_RATE_SENSITIVITY:+.3f}",
              delta_color="inverse")

    st.divider()

    # 고위험 가구 비교 막대 그래프
    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = go.Figure()
        fig_bar.add_bar(
            name="기준", x=["고위험 가구"],
            y=[n_high_base], marker_color="#3498DB",
            text=[f"{n_high_base:,}"], textposition="outside"
        )
        fig_bar.add_bar(
            name=f"금리 {delta:+.1f}%p", x=["고위험 가구"],
            y=[n_high_stress], marker_color="#E74C3C",
            text=[f"{n_high_stress:,}"], textposition="outside"
        )
        fig_bar.update_layout(
            barmode="group", height=380,
            title=f"금리 {delta:+.1f}%p 시 고위험 가구 변화",
            yaxis_title="가구 수",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        # DSR 분포 변화
        fig_dsr2 = go.Figure()
        fig_dsr2.add_trace(go.Histogram(
            x=df_hh["dsr"].clip(0, 120),
            name="기준", opacity=0.6,
            marker_color="#3498DB", nbinsx=50
        ))
        fig_dsr2.add_trace(go.Histogram(
            x=df_stress["dsr"].clip(0, 120),
            name=f"금리 {delta:+.1f}%p", opacity=0.6,
            marker_color="#E74C3C", nbinsx=50
        ))
        fig_dsr2.add_vline(x=40, line_dash="dash",
                           line_color="black", opacity=0.5,
                           annotation_text="고위험 기준(DSR=40%)")
        fig_dsr2.update_layout(
            barmode="overlay", height=380,
            title="DSR 분포 변화",
            xaxis_title="DSR (%)", yaxis_title="가구 수",
        )
        st.plotly_chart(fig_dsr2, use_container_width=True)

    # Monte Carlo
    if show_mc:
        st.subheader("📊 Monte Carlo 불확실성 분석")
        with st.spinner(f"Monte Carlo {n_mc}회 반복 중..."):
            mc = sim.monte_carlo(df_hh, X_hh, delta=delta, n_iter=n_mc)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("평균 고위험 가구", f"{mc['mean']:,.0f}가구")
        mc2.metric("95% 신뢰구간 하한", f"{mc['ci95_lo']:,.0f}가구")
        mc3.metric("95% 신뢰구간 상한", f"{mc['ci95_hi']:,.0f}가구")

    # 전체 시나리오 비교표
    st.subheader("📋 전체 시나리오 비교")
    from models.scenario_simulator import SCENARIOS

    rows = []
    for sc_name, sc_delta in SCENARIOS.items():
        df_s = sim.apply_rate_shock(df_hh, sc_delta)
        X_s  = df_s[HOUSEHOLD_FEATURES].values
        p    = clf.predict(X_s) if clf else (
            pd.cut(df_s["dsr"], bins=[-np.inf,20,40,np.inf], labels=[0,1,2]).astype(int).values
        )
        n_h  = int((p == 2).sum())
        rows.append({
            "시나리오"       : sc_name,
            "금리 변동(%p)"  : sc_delta,
            "고위험 가구(명)": n_h,
            "증감(명)"       : n_h - n_high_base,
            "예상 FSI"       : round(current_fsi + sc_delta * FSI_RATE_SENSITIVITY, 4),
        })

    df_sc = pd.DataFrame(rows)
    st.dataframe(
        df_sc.style
        .background_gradient(subset=["고위험 가구(명)"], cmap="Reds")
        .format({"증감(명)": "{:+,.0f}"}),
        use_container_width=True
    )

footer_html = """
<style>
/* 푸터 컨테이너 스타일 설정 */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa; /* 밝은 회색 배경 (다크모드 지원을 원하면 transparent 추천) */
    color: #6c757d;
    text-align: center;
    padding: 12px 0;
    font-size: 13px;
    border-top: 1px solid #dee2e6;
    z-index: 999; /* 다른 요소들보다 항상 위에 표시되도록 설정 */
}

/* 다크 모드일 때의 색상 처리 (Streamlit 테마 자동 대응) */
@media (prefers-color-scheme: dark) {
    .footer {
        background-color: #0e1117;
        color: #fafafa;
        border-top: 1px solid #333333;
    }
}

/* 본문 컨텐츠가 푸터에 가려지지 않도록 하단 여백 추가 */
.block-container {
    padding-bottom: 80px !important;
}
</style>

<div class="footer">
    <p style="margin: 0;">
        ⓒ 2026 <b>금융안정 2층 모니터링 시스템</b> | 
        Developed by AI소프트웨어학부 20233939 백유진 | 
        <i>Bank of Korea AX & CBDC Project</i>
    </p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
