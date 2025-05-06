import streamlit as st
from streamlit_option_menu import option_menu
import page.home as hm
import page.analyzermov as al
import page.multi_model_dashboard as mmd
import page.applymodel as am
import page.motoring as mo
import page.setting as setg
import page.analyzermonitor as azm
import test.monitor_test as mot
import page.link as link
import os
import script.model_loader as ml

# ✅ 페이지 설정
st.set_page_config(page_title="배려대상자(노인, 아이, 거동불편자 등) 낙상 감지 및 알림 시스템", layout="wide")

# 초기 로딩 (한 번만 수행됨)
if "models" not in st.session_state:
    st.session_state.models = ml.load_models("ALL")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 220px;
        max-width: 220px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    menu_type = st.radio("", ["사용자", "관리자"], horizontal=True)

    if menu_type == "사용자":
        user_selected = option_menu(
            "사용자",
            ["실시간 감시", "데이터 분석"],
            icons=["activity", "graph-up"],
            default_index=0,
            key="user_menu"
        )
        admin_selected = None  # 동시에 값이 생기지 않도록 초기화

    else:  # menu_type == "관리자"
        admin_selected = option_menu(
            "관리자",
            ["설명", "영상 학습", "모델 생성", "모델 적용", "환경설정", "Github", "PPT"],
            icons=["house", "camera-video", "cpu", "box-fill", "gear", "github", "file-earmark-ppt"],
            default_index=0,
            key="admin_menu"
        )
        user_selected = None


# 사용자 메뉴
if user_selected == "실시간 감시":
    mo.show()
elif user_selected == "데이터 분석":
    azm.show()

# 관리자 메뉴
elif admin_selected == "설명":
    hm.show()
elif admin_selected == "영상 학습":
    al.show()
elif admin_selected == "모델 생성":
    mmd.show(True)
elif admin_selected == "모델 적용":
    am.show()
elif admin_selected == "환경설정":
    setg.show()
elif admin_selected == "Github":
    link.Github()
elif admin_selected == "PPT":
    link.PPT()
