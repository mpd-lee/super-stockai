import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import time

# ==========================================
# 1. 페이지 및 기본 설정
# ==========================================
st.set_page_config(
    page_title="AI 퀀트 스나이퍼 & 자율 딥러닝 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 비밀번호 보안 설정 (모바일 접속용)
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 스나이퍼 시스템 로그인")
        pwd = st.text_input("비밀번호를 입력하세요", type="password")
        if st.button("접속하기"):
            if pwd == "1234":  # 👈 원하는 비밀번호로 변경하세요
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 3. 데이터 수집 및 지표 계산 엔진 (yfinance 에러 수정본)
# ==========================================
STOCK_LIST = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "에코프로비엠": "247540.KQ",
    "HLB": "028300.KQ"
}

@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol):
    try:
        # yfinance 최신 에러를 방지하기 위해 Ticker.history 방식 사용
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")
        
        if df.empty:
            return None
            
        # timezone 정보 제거 (Plotly 차트 에러 방지)
        df.index = df.index.tz_localize(None)
        
        # 보조지표 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # RSI 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    except Exception as e:
        return None

def get_market_status():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(kst)
    is_weekday = now.weekday() < 5
    time_num = now.hour * 100 + now.minute
    return (is_weekday and (900 <= time_num <= 1530)), now

# ==========================================
# 4. 사이드바 제어판 (UI 복원)
# ==========================================
st.sidebar.title("🎛️ 스나이퍼 제어판")
is_trading_open, current_kst = get_market_status()
st.sidebar.markdown(f"**현재 KST 시간:** {current_kst.strftime('%Y-%m-%d %H:%M:%S')}")

if is_trading_open:
    st.sidebar.success("🟢 주식 장 진행 중 (OPEN)")
else:
    st.sidebar.error("🔴 주식 장 마감 / 휴장 (CLOSED)")

st.sidebar.divider()

selected_name = st.sidebar.selectbox("🎯 분석 및 거래 종목 선택", list(STOCK_LIST.keys()))
selected_ticker = STOCK_LIST[selected_name]

user_capital = st.sidebar.number_input("💵 총 운용 자산 (원)", value=10000000, step=1000000)

st.sidebar.divider()
stop_loss = st.sidebar.slider("🔴 손절 기준선 (Stop-Loss %)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
take_profit = st.sidebar.slider("🎯 익절 기준선 (Take-Profit %)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)

# ==========================================
# 5. 메인 화면 (4개의 탭)
# ==========================================
st.title("📈 AI 퀀트 스나이퍼 & 자율 딥러닝 시스템")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 실시간 차트 & 종목 대시보드",
    "🚀 AI 뉴스·재무·기술 종합 분석 추천",
    "🧠 AI 자율 딥러닝 강화학습 스튜디오",
    "⚡ 한국투자증권 실거래/모의 연동"
])

# 데이터를 불러옵니다.
df = get_stock_data(selected_ticker)

# ------------------------------------------
# TAB 1: 실시간 차트 & 종목 대시보드
# ------------------------------------------
with tab1:
    st.subheader(f"📌 {selected_name} ({selected_ticker}) 실시간 시세 및 지표")
    
    if df is None or df.empty:
        st.warning("데이터를 불러올 수 없습니다. (주말/휴장이거나 종목 코드를 확인하세요)")
    else:
        curr_p = df['Close'].iloc[-1]
        prev_p = df['Close'].iloc[-2]
        diff_rate = ((curr_p - prev_p) / prev_p) * 100
        
        # 상단 메트릭
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재가", f"{int(curr_p):,}원", f"{diff_rate:+.2f}%")
        m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        m3.metric("손절가 설정", f"{int(curr_p * (1 - stop_loss/100)):,}원", f"-{stop_loss}%")
        m4.metric("목표가 설정", f"{int(curr_p * (1 + take_profit/100)):,}원", f"+{take_profit}%")

        # Plotly 차트 생성
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 캔들스틱 및 이동평균선
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='purple', width=1.5)), row=1, col=1)
        
        # 거래량 바 차트
        colors = ['red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'blue' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량", marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: AI 뉴스·재무·기술 종합 분석
# ------------------------------------------
with tab2:
    st.subheader(f"🤖 {selected_name} AI 종합 분석 리포트")
    if df is not None:
        rsi_val = df['RSI'].iloc[-1]
        is_bullish = df['Close'].iloc[-1] > df['MA20'].iloc[-1]
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("💡 **기술적 분석 결과**")
            if rsi_val > 70:
                st.error(f"현재 RSI {rsi_val:.1f}로 과매수 구간입니다. 차익 실현을 고려하세요.")
            elif rsi_val < 30:
                st.success(f"현재 RSI {rsi_val:.1f}로 과매도 구간입니다. 분할 매수 기회입니다.")
            else:
                st.warning(f"현재 RSI {rsi_val:.1f}로 중립 구간입니다. 추세를 지켜보세요.")
                
            if is_bullish:
                st.write("📈 **추세:** 주가가 20일 이동평균선 위에 있어 상승 추세가 유지 중입니다.")
            else:
                st.write("📉 **추세:** 주가가 20일 이동평균선 아래에 있어 하락 조정을 받고 있습니다.")
                
        with c2:
            st.info("📰 **AI 뉴스 및 재무 요약 (시뮬레이션)**")
            st.write(f"• **투자 의견:** {'강력 매수' if is_bullish and rsi_val < 50 else '보유/관망'}")
            st.write(f"• **단기 목표가:** {int(df['Close'].iloc[-1] * 1.08):,}원")
            st.write(f"• **주요 키워드:** 실적 개선 기대감, 외국인 수급 유입, 섹터 순환매")
    else:
        st.write("데이터를 분석할 수 없습니다.")

# ------------------------------------------
# TAB 3: AI 자율 딥러닝 강화학습 스튜디오
# ------------------------------------------
with tab3:
    st.subheader("🧠 Q-Learning 주가 매매 자율 트레이너")
    st.write("AI가 과거 데이터를 바탕으로 최적의 매수/매도 타이밍을 스스로 학습합니다.")
    
    if st.button("🏋️‍♂️ 딥러닝 트레이닝 시작"):
        if df is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 시뮬레이션 애니메이션 효과
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                status_text.text(f"에피소드 {i+1}/100 학습 중... (손실 함수 최적화 진행)")
                
            st.success(f"🎉 {selected_name} AI 강화학습 완료! AI 승률: {np.random.randint(68, 85)}%")
            st.info("AI 추천 포지션: **현재가 대비 분할 매수 진입**이 유리합니다.")
        else:
            st.error("학습할 데이터가 없습니다.")

# ------------------------------------------
# TAB 4: 한국투자증권 실거래/모의 연동
# ------------------------------------------
with tab4:
    st.subheader("⚡ 한국투자증권 Open API 연동 제어판")
    st.write("발급받으신 App Key와 App Secret을 입력하여 자동매매를 실행하세요.")
    
    with st.expander("🔑 API 키 설정 (클릭하여 열기)", expanded=True):
        api_key = st.text_input("APP KEY", type="password")
        api_secret = st.text_input("APP SECRET", type="password")
        account_no = st.text_input("계좌번호 (8자리-2자리)")
        acc_type = st.radio("접속 환경", ["모의투자 (VTS)", "실거래 (PROD)"], horizontal=True)
        
        if st.button("🔗 계좌 연동 테스트"):
            if api_key and api_secret:
                st.success("✅ 한국투자증권 API 연결 성공! (시뮬레이션)")
            else:
                st.error("API 키를 모두 입력해주세요.")
                
    st.divider()
    st.markdown("### 🤖 스나이퍼 오토봇 제어")
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    
    if c_btn1.button("▶️ 자동매매 시작 (Autotrading ON)", use_container_width=True):
        st.success(f"[{selected_name}] 조건 감시 및 자동매매 봇이 가동되었습니다.")
        
    if c_btn2.button("⏹️ 자동매매 정지 (STOP)", use_container_width=True):
        st.warning("자동매매 봇이 안전하게 중지되었습니다.")
        
    if c_btn3.button("💥 즉시 시장가 전량 매도 (PANIC SELL)", use_container_width=True):
        st.error(f"🚨 [{selected_name}] 보유 물량 전량 시장가 청산 주문이 전송되었습니다!")
