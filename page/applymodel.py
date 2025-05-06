import streamlit as st
import os, io
import script.model_loader as ml
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
from contextlib import redirect_stdout

model_path = os.path.join("user", "model")

def show():
    st.title("🧠 모델 적용")
    st.write("기존 학습된 모델과 신규 학습된 모델을 사용자에게 적용하기 위해 메모리에 업로드 합니다.")

    model_dir = os.path.abspath(os.path.join("user", "model"))

    # .keras 파일 목록 불러오기
    keras_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".keras") and f != "fall_model.keras"
    ]

    if not keras_files:
        st.warning("⚠️ .keras 파일이 존재하지 않습니다.")
        return
    
    st.markdown("---")

    st.subheader("📂 생성된 모델 파일 목록")
    for fname in keras_files:
        st.write(f"🔹 {fname}")

    @st.cache_resource
    def load_all_models(file_list):
        loaded = {}
        for fname in file_list:
            #model_name = os.path.splitext(fname)[0]  # 확장자 제거한 이름
            model_name = fname.replace("fall_model_", "").replace(".keras", "").replace("Model", "")
            model_path = os.path.join(model_dir, fname)
            try:
                loaded[model_name] = load_model(model_path)
            except Exception as e:
                st.error(f"❌ {fname} 로딩 실패: {e}")
        return loaded
    
    def get_model_summary(model):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        return buffer.getvalue()
    
    def generate_dummy_input(model_key):
        if model_key == "dense":
            return np.random.rand(1, 16)
        elif model_key == "lstm":
            return np.random.rand(1, 1, 16)
        elif model_key == "ensemble":
            return [np.random.rand(1, 16), np.random.rand(1, 1, 16)]


    # 모델 로딩 실행
    with st.spinner("모델을 메모리에 로딩 중입니다..."):
        st.session_state.models = load_all_models(keras_files)
        models = load_all_models(keras_files)

    # 로딩 완료 메시지
    st.success(f"✅ 총 {len(models)}개의 모델이 메모리에 성공적으로 로드되었습니다!")
    for model_name in models:
        st.write(f"✅ {model_name} 모델 로딩 완료 : 로드된 모델명 - {model_name}" )    

    st.markdown("---")

    st.subheader("🧠 모델 로드 정상 확인")
    selected = st.radio(
            "낙상판단 기준을 무슨 모델로로 할까요?",
            ("선택하세요", "dense", "lstm", "ensemble"),
            horizontal=True
        )

    # ✅ 모델 메모리 로딩
    if selected != "선택하세요":
        st.success(f"✅ '{selected}' 모델이 선택되었습니다.")
        selected_model = models[selected]
        st.subheader("input_shape : ")
        st.code(selected_model.input_shape)
        summary_text = get_model_summary(selected_model)
        st.subheader("📐 선택된 모델 구조 요약")
        st.code(summary_text, language='text')

    # ✅ 예측 실행
    try:
        prediction = models[selected].predict(generate_dummy_input(selected))
        st.write("🧾 테스트 예측 결과 - 모델은 정상로드되었습니다.", prediction)
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")