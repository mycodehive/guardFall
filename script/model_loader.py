import streamlit as st
import os, io
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# 모델과 스케일러를 저장할 변수
_model = None
_scaler = None
_model_loaded_time = None

def _load_models_real():
    """캐시 없이 진짜 모델 로딩"""
    global _model, _scaler, _model_loaded_time
    model_dir = os.path.abspath(os.path.join("user", "model"))
    fall_model = os.path.join(model_dir, "fall_model.keras")
    scaler_model = os.path.join(model_dir, "scaler.pkl")

    _model = load_model(fall_model)
    _scaler = joblib.load(scaler_model)
    _model_loaded_time = datetime.now()

@st.cache_resource
def load_models_cached():
    """Streamlit 캐시를 이용하는 버전"""
    _load_models_real()
    return _model, _scaler, _model_loaded_time

def reload_models():
    """강제 리로드 (캐시 무시)"""
    _load_models_real()
    return _model, _scaler, _model_loaded_time
