import streamlit as st
import os
import script.model_loader as ml
from datetime import datetime

model_path = os.path.join("user", "model", "fall_model.keras")

def show():
    st.title("🧠 모델 적용")
    st.write("기존 학습된 모델과 신규 학습된 모델 적용 여부를 선택하는 곳입니다. ")

    options = ("선택하세요", "예", "아니오")
    answer = st.radio(
        "현재 학습된 모델을 적용하시겠습니까?",
        options
    )

    # '선택하세요' 상태는 무시하고, 진짜 답변만 처리
    if answer == "예":
        file_modified_time =  datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
        model, scaler, _model_loaded_time = ml.reload_models() # ✅ 모델과 스케일러를 1번만 로딩
        loadtime = _model_loaded_time.strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        ---
        🆕 **현재 학습된 새로운 모델을 적용합니다!**
          - 파일 생성 시간 : {file_modified_time}
          - 메모리 로드 시간 : {loadtime}
        """)
    elif answer == "아니오":
        file_modified_time =  datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        ---
        📂 기존 메모리에 최초 학습된 모델을 계속 사용합니다!
          - 파일 생성시간 : {file_modified_time}"
        """)
    else:
        st.warning("⚡ 선택해 주세요!")