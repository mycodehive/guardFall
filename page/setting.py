import streamlit as st
import json
import os

csv_dir = os.path.abspath(os.path.join("user", "setting"))
DATA_FILE = os.path.join(csv_dir, 'config.json')

# 키 → 표시할 한글 이름
# 섹션 키 → 한글명
section_label_map = {
    "is_fallen": "🔧 사용자 낙상 기준",
    "is_fallen_model": "🔧 딥러닝 낙상 모델 기준",
    "is_fallen_Upperbody": "🔧 상체 낙상 기준(테스트)",
    "fallen_send_msg" : "🔧 텔레그램 전송 기준"
}

label_map = {
    "MIN_OK_PARTS": "유효한 OK 판정 최소 관절 수",
    "LS_Y": "왼쪽 어깨 Y값 기준",
    "RS_Y": "오른쪽 어깨 Y값 기준",
    "LK_Y": "왼쪽 무릎 Y값 기준",
    "RK_Y": "오른쪽 무릎 Y값 기준",
    "FALL_COUNT": "각 관절 낙상 판정 누적(Y값) 횟수",
    "FALL_PROBABILITY": "낙상 확률 임계값",
    "SEND_YN": "메세지 전송 여부"
}

# 🔄 데이터 불러오기
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# 💾 저장
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def editable_section(section, section_data):
    new_section = {}
    for key, value in section_data.items():
        label = label_map.get(key, key)
        widget_key = f"{section}_{key}"

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{label}**")  # 왼쪽: 텍스트 라벨
        with col2:
            if isinstance(value, int):
                new_section[key] = st.number_input(
                    label, value=value, step=1, key=widget_key, label_visibility="collapsed"
                )
            elif isinstance(value, float):
                new_section[key] = st.number_input(
                    label, value=value, format="%.2f", key=widget_key, label_visibility="collapsed"
                )
            elif isinstance(value, str) and key == "SEND_YN":
                new_section[key] = st.selectbox(
                    label,
                    options=["Y", "N"],
                    index=0 if value == "Y" else 1,
                    key=widget_key,
                    label_visibility="collapsed"
                )
            else:
                st.warning(f"{label}: 지원되지 않는 데이터 유형입니다.")
    return new_section


def show():
    st.title("📂 환경 설정")

    data = load_data()
    updated_data = {}

    with st.form("edit_form"):
        for section_key, section_values in data.items():
            section_label = section_label_map.get(section_key, section_key)
            st.markdown(f"### {section_label}")
            updated_data[section_key] = editable_section(section_key, section_values)
            st.markdown("---")

        submitted = st.form_submit_button("💾 저장")

    if submitted:
        save_data(updated_data)
        st.success("✅ 설정이 성공적으로 저장되었습니다.")       
