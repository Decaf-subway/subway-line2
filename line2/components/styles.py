import streamlit as st

def inject_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
        .metric-card {
            background: #f8f9fa; border-radius: 12px; padding: 16px 20px;
            border: 1px solid #e9ecef; text-align: center;
        }
        .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
        .metric-value { font-size: 28px; font-weight: 700; color: #212529; }
        .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }
        .badge-쾌적   { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
        .badge-보통   { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
        .badge-혼잡   { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
        .badge-매우혼잡 { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
        .holiday-banner {
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border: 1px solid #667eea44; border-radius: 10px;
            padding: 12px 16px; margin-bottom: 16px;
            font-size: 14px; color: #4c3499;
        }
        .station-btn { cursor: pointer; transition: all 0.15s; }
        .rank-card {
            background: #fff; border: 1px solid #e9ecef; border-radius: 8px;
            padding: 10px 14px; margin-bottom: 6px;
            display: flex; justify-content: space-between; align-items: center;
        }
        /* 사이드바 여백 축소 및 헤더 제거로 로고를 최상단에 밀착 */
        [data-testid="stSidebarHeader"] {
            display: none !important;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 1.0rem !important;
        }
        /* 로고 이미지 가운데 정렬 */
        [data-testid="stSidebarUserContent"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)
