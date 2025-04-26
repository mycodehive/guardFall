import streamlit as st
from streamlit_option_menu import option_menu
import page.home as hm
import page.analyzermov as al
import page.createmodel as cm
import page.motoring as mo
import page.link as link

# ✅ 페이지 설정
st.set_page_config(page_title="시스템", layout="wide")

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
        options=["홈", "영상 분석", "모델 생성", "실시간 감시","Github", "PPT"],
        icons=["house", "camera-video", "cpu", "activity", "github","file-earmark-ppt"], # https://icons.getbootstrap.com/
        menu_icon="shield-lock",
        default_index=0,
    )

# ✅ 페이지별 내용 표시
if selected == "홈":
    hm.show()

elif selected == "영상 분석":  
    al.show()

elif selected == "모델 생성":    
    cm.show()

elif selected == "실시간 감시":
    mo.show()

elif selected == "Github":
    link.Github()

elif selected == "PPT":
    link.PPT()