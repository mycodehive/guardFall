import streamlit as st
import script.model_loader as ml
import numpy as np
import os

def show():
    st.title("🛡️ 감시 모드")
    st.write("낙상 여부를 실시간으로 감지합니다.")

    model_dir = os.path.abspath(os.path.join("user", "model"))

    # ✅ streamlit_app.py랑 똑같이 호출하지만
    #    이미 캐시된 모델을 가져오게 됨
    model, scaler, _model_loaded_time = ml.load_models_cached()

    new_data = np.array([
        [0.44, 0.62, 1, 1, 0.52, 0.63, 1, 1, 0.46, 0.81, 0.63, 0, 0.49, 0.84, 0.71, 1],  # 샘플1
        [0.43, 0.62, 1, 1, 0.53, 0.62, 1, 1, 0.46, 0.82, 0.67, 0, 0.51, 0.75, 0.66, 0]   # 샘플2
    ])

    new_data_scaled = scaler.transform(new_data)

    predictions = model.predict(new_data_scaled)

    st.success(predictions)

    if predictions[0] > 0.55 :
        st.info("aaaaaa")
    
    if predictions[1] < 0.55 :
        st.info("bbbbb")