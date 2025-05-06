import streamlit as st
import os
from tensorflow.keras.models import load_model

# [20250506]
@st.cache_resource
def load_models(model_name):
    model_dir = os.path.abspath(os.path.join("user", "model"))

    if model_name == "ALL":
        keras_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".keras") and f != "fall_model.keras"
        ]

        if not keras_files:
            st.warning("⚠️ .keras 파일이 존재하지 않습니다.")
            return {}

        models = {}
        for fname in keras_files:
            # fall_model_denseModel.keras → dense
            key = fname.replace("fall_model_", "").replace("Model.keras", "")
            path = os.path.join(model_dir, fname)
            models[key] = load_model(path)

        return models
    else:
        file_name = f"fall_model_{model_name}Model.keras"
        path = os.path.join(model_dir, file_name)

        if not os.path.exists(path):
            st.warning(f"⚠️ {file_name} 파일이 존재하지 않습니다.")
            return None

        return {model_name: load_model(path)}