import script.model_loader as ml
import numpy as np
import warnings, os, json

# ✅ 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 경로 설정
config_path = os.path.abspath(os.path.join("user", "setting", "config.json"))

# JSON 파일 읽기
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 섹션별 변수 할당
is_fallen_json = config.get("is_fallen", {})
is_fallen_model_json = config.get("is_fallen_model", {})
is_fallen_upper_json = config.get("is_fallen_Upperbody", {})    

def extract_pose_data(row):
    return {
        "ls": [row["left_shoulder_x"], row["left_shoulder_y"], row["left_shoulder_v"], row["left_shoulder_vr"]],
        "rs": [row["right_shoulder_x"], row["right_shoulder_y"], row["right_shoulder_v"], row["right_shoulder_vr"]],
        "lk": [row["left_knee_x"], row["left_knee_y"], row["left_knee_v"], row["left_knee_vr"]],
        "rk": [row["right_knee_x"], row["right_knee_y"], row["right_knee_v"], row["right_knee_vr"]],
    }

# 낙상 여부 판단
def is_fallen(row) :
    pose = extract_pose_data(row)
    ls_x, ls_y, ls_v, ls_ok = pose["ls"]
    rs_x, rs_y, rs_v, rs_ok = pose["rs"]
    lk_x, lk_y, lk_v, lk_ok = pose["lk"]
    rk_x, rk_y, rk_v, rk_ok = pose["rk"]

    if sum([ls_ok, rs_ok, lk_ok, rk_ok]) >= is_fallen_json["MIN_OK_PARTS"]: # 정상범주 데이터여만 한다.
        count = 0      
        if isinstance(ls_y, (int, float)) and ls_y >= is_fallen_json["LS_Y"]:
            count += 1
        if isinstance(rs_y, (int, float)) and rs_y >= is_fallen_json["RS_Y"]:
            count += 1
        if isinstance(lk_y, (int, float)) and lk_y >= is_fallen_json["LK_Y"]:
            count += 1
        if isinstance(rk_y, (int, float)) and rk_y >= is_fallen_json["RK_Y"]:
            count += 1
        return 1 if count >= is_fallen_json["FALL_COUNT"] else 0 #1이 낙상, 0은 정상
    return -1 # 판별불가

# 메모리에 올린 모델을 통해 예측한다.
def is_fallen_model(row) :
    pose = extract_pose_data(row)
    ls_x, ls_y, ls_v, ls_ok = pose["ls"]
    rs_x, rs_y, rs_v, rs_ok = pose["rs"]
    lk_x, lk_y, lk_v, lk_ok = pose["lk"]
    rk_x, rk_y, rk_v, rk_ok = pose["rk"]
    
    is_fallen = 0
    model, scaler, _model_loaded_time = ml.load_models_cached()
    new_data = np.array([
        [ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok]
    ])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled,verbose=0)
    fall_probability = prediction[0][0]

    if fall_probability >= is_fallen_model_json["FALL_PROBABILITY"]:
        is_fallen = 1
    else:
        is_fallen = 0

    return is_fallen

# 낙상 여부 판단
def is_fallen_Upperbody(row):
    pose = extract_pose_data(row)
    ls_x, ls_y, ls_v, ls_ok = pose["ls"]
    rs_x, rs_y, rs_v, rs_ok = pose["rs"]
    lk_x, lk_y, lk_v, lk_ok = pose["lk"]
    rk_x, rk_y, rk_v, rk_ok = pose["rk"]

    if sum([ls_ok, rs_ok]) >= is_fallen_upper_json["MIN_OK_PARTS"]: # 정상범주 데이터여만 한다.
        count = 0
        if isinstance(ls_y, (int, float)) and ls_y >= is_fallen_upper_json["LS_Y"]:
            count += 1
        if isinstance(rs_y, (int, float)) and rs_y >= is_fallen_upper_json["RS_Y"]:
            count += 1
        return 1 if count >= is_fallen_upper_json["FALL_COUNT"] else 0 #1이 낙상, 0은 정상, 판별불가
    return 0