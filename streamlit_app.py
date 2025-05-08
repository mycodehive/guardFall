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
import base64
import script.model_loader as ml

# ✅ 사용자 계정 정보
users = {
    "admin": {"password": "1111", "role": "admin", "name": "김관리"},
    "user": {"password": "1111", "role": "user", "name": "김배려"}
}

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("user/img/indexbg.png")

# ✅ 페이지 설정
st.set_page_config(
    page_title="배려대상자(노인, 아이, 거동불편자 등) 낙상 감지 및 알림 시스템",
    layout="wide"
)

# ✅ 세션 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "name" not in st.session_state:
    st.session_state.name = None
if "role" not in st.session_state:
    st.session_state.role = None
if "models" not in st.session_state:
    st.session_state.models = ml.load_models("ALL")

# ✅ 로그인 페이지
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div style='text-align: center; padding: 30px 0;'>
                <img src="data:image/jpeg;base64,{image_base64}" width="100%">
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .shake {
            animation: shake 0.3s;
            animation-iteration-count: 2;
        }
        @keyframes shake {
            0% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            50% { transform: translateX(5px); }
            75% { transform: translateX(-5px); }
            100% { transform: translateX(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### 👤 사용자 정보 입력")
            username = st.text_input("아이디", placeholder="아이디를 입력하세요")
            password = st.text_input("비밀번호", placeholder="비밀번호를 입력하세요", type="password")

            submitted = st.form_submit_button("로그인", use_container_width=True)
            if submitted:
                user = users.get(username)
                if user and user["password"] == password:
                    with st.spinner("로그인 중입니다..."):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = user["role"]
                        st.session_state.name = user["name"]
                        st.experimental_rerun()
                else:
                    st.markdown("""
                        <div class='shake'>
                            <p style='color: red;'>❌ 아이디 또는 비밀번호가 잘못되었습니다.</p>
                        </div>
                    """, unsafe_allow_html=True)
    st.stop()

# ✅ 로그인 후 화면
st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 220px;
    max-width: 220px;
}
</style>
""", unsafe_allow_html=True)

# ✅ 사용자명 상단에 표시
st.markdown(f"### 👤 현재 로그인한 사용자: `{st.session_state.name}`")

# ✅ 사이드바 메뉴
with st.sidebar:
    st.write(f"👋 {st.session_state.name}님 환영합니다!")

    if st.button("🔓 로그아웃"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.experimental_rerun()

    if st.session_state.role == "user":
        user_selected = option_menu(
            "사용자",
            ["실시간 감지", "데이터 분석"],
            icons=["activity", "graph-up"],
            default_index=0,
            key="user_menu"
        )
        admin_selected = None

    elif st.session_state.role == "admin":
        admin_selected = option_menu(
            "관리자",
            ["설명", "영상 학습", "모델 생성", "모델 적용", "환경설정", "Github", "PPT"],
            icons=["house", "camera-video", "cpu", "box-fill", "gear", "github", "file-earmark-ppt"],
            default_index=0,
            key="admin_menu"
        )
        user_selected = None

# ✅ 사용자 메뉴 연결
if st.session_state.role == "user":
    if user_selected == "실시간 감지":
        mo.show()
    elif user_selected == "데이터 분석":
        azm.show()

# ✅ 관리자 메뉴 연결
elif st.session_state.role == "admin":
    if admin_selected == "설명":
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
