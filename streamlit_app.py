import streamlit as st
from streamlit_option_menu import option_menu
import page.home as hm
import page.analyzermov as al
import page.createmodel as cm
import page.applymodel as am
import page.motoring as mo
import page.monitor_test as mot
import page.link as link
import os
import joblib
import numpy as np
import script.model_loader as ml


# ✅ 페이지 설정
st.set_page_config(page_title="배려대상자(노인, 아이, 거동불편자 등) 낙상 감지 및 알림 시스템", layout="wide")

model_dir = os.path.abspath(os.path.join("user", "model"))
model, scaler, _model_loaded_time = ml.load_models_cached() # ✅ 모델과 스케일러를 1번만 로딩

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

# ✅ 사이드바 메뉴 만들기
with st.sidebar:
    selected = option_menu(
        menu_title="메뉴",
        options=["홈", "영상 분석", "모델 생성", "모델 적용", "실시간 감시","Github", "PPT", "Test"],
        icons=["house", "camera-video", "cpu", "box-fill", "activity", "github","file-earmark-ppt", "activity"], # https://icons.getbootstrap.com/
        menu_icon="shield-lock",
        default_index=0,
    )

# ✅ 페이지별 내용 표시
if selected == "홈":
    hm.show(model)

elif selected == "영상 분석":  
    al.show()

elif selected == "모델 생성":    
    cm.show()

elif selected == "모델 적용":    
    am.show()

elif selected == "실시간 감시":
    mo.show()

elif selected == "Github":
    link.Github()

elif selected == "PPT":
    link.PPT()

elif selected == "PPT":
    link.PPT()

elif selected == "Test":
    mot.show()