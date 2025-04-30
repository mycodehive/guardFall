import script.model_loader as ml
import numpy as np
import warnings
import streamlit as st

# ✅ 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 낙상 기준값
fall_cfg = st.secrets["is_fallen"]
model_cfg = st.secrets["is_fallen_model"]
upper_cfg = st.secrets["is_fallen_Upperbody"]

# 낙상 여부 판단
def is_fallen(ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok):
    if sum([ls_ok, rs_ok, lk_ok, rk_ok]) >= fall_cfg["MIN_OK_PARTS"]: # 정상범주 데이터여만 한다.
        count = 0
        if isinstance(ls_y, (int, float)) and ls_y >= fall_cfg["LS_Y"]:
            count += 1
        if isinstance(rs_y, (int, float)) and rs_y >= fall_cfg["RS_Y"]:
            count += 1
        if isinstance(lk_y, (int, float)) and lk_y >= fall_cfg["LK_Y"]:
            count += 1
        if isinstance(rk_y, (int, float)) and rk_y >= fall_cfg["RK_Y"]:
            count += 1
        return 1 if count >= fall_cfg["FALL_COUNT"] else 0 #1이 낙상, 0은 정상, 판별불가
    return 1

# 메모리에 올린 모델을 통해 예측한다.
def is_fallen_model(ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok):
    is_fallen = 0
    model, scaler, _model_loaded_time = ml.load_models_cached()
    new_data = np.array([
        [ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok]
    ])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled,verbose=0)
    fall_probability = prediction[0][0]

    if fall_probability >= model_cfg["FALL_PROBABILITY"]:
        is_fallen = 1
    else:
        is_fallen = 0

    return is_fallen

# 낙상 여부 판단
def is_fallen_Upperbody(ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok):
    if sum([ls_ok, rs_ok]) >= upper_cfg["MIN_OK_PARTS"]: # 정상범주 데이터여만 한다.
        count = 0
        if isinstance(ls_y, (int, float)) and ls_y >= upper_cfg["LS_Y"]:
            count += 1
        if isinstance(rs_y, (int, float)) and rs_y >= upper_cfg["RS_Y"]:
            count += 1
        return 1 if count >= upper_cfg["FALL_COUNT"] else 0 #1이 낙상, 0은 정상, 판별불가
    return 0