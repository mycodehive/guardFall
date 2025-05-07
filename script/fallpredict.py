import script.model_loader as ml
import numpy as np
import warnings, os, json
import joblib, math

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

def is_fallen_angle(row) :
    pose = extract_pose_data(row)
    ls_x, ls_y, ls_v, ls_ok = pose["ls"]
    rs_x, rs_y, rs_v, rs_ok = pose["rs"]
    lk_x, lk_y, lk_v, lk_ok = pose["lk"]
    rk_x, rk_y, rk_v, rk_ok = pose["rk"]

    # 유클리드 거리 계산 함수 (두 점 사이의 거리)
    def calculate_distance(x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    # 2. 각도 계산 함수 (벡터 간의 각도 계산)
    def calculate_angle(x1, y1, x2, y2, x3, y3):
        v1_x = x2 - x1
        v1_y = y2 - y1
        v2_x = x3 - x1
        v2_y = y3 - y1
        
        dot_product = v1_x * v2_x + v1_y * v2_y
        magnitude_v1 = math.sqrt(v1_x**2 + v1_y**2)
        magnitude_v2 = math.sqrt(v2_x**2 + v2_y**2)

        # 두 벡터 중 하나가 길이 0이면 계산 불가: 기본값으로 0 리턴턴
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        # 안전한 코사인 값(부동소수점 오차 방지)
        # cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle = math.acos(cos_angle)
        return math.degrees(angle)  # 라디안을 도로 변환
    
    # 어깨 간 거리 계산
    shoulder_distance = calculate_distance(ls_x, ls_y, rs_x, rs_y)

    # 무릎 간 거리 계산
    knee_distance = calculate_distance(lk_x, lk_y, rk_x, rk_y)

    # 어깨와 무릎의 각도 계산
    shoulder_angle = calculate_angle(ls_x, ls_y, rs_x, rs_y, (lk_x + rk_x) / 2, (lk_y + rk_y) / 2)

    # 어깨 비대칭성 (수평 차이 계산)
    delta_x = abs(ls_x - rs_x)
    delta_y = abs(ls_y - rs_y)

    # 무릎 비대칭성 (수평 차이 계산)
    knee_delta_x = abs(lk_x - rk_x)
    knee_delta_y = abs(lk_y - rk_y)

    # 임계값 설정
    threshold_distance = 0.2  # 어깨 및 무릎 간 거리 차이
    threshold_angle = 150  # 각도 임계값 (낙상 시 150도를 넘으면 위험)
    threshold_asymmetry = 0.1  # 비대칭성 임계값

    # 어깨 또는 무릎 간 거리 기준
    if shoulder_distance > threshold_distance or knee_distance > threshold_distance:
        return 11  # "낙상 위험 감지"

    # 어깨 각도 기준
    if shoulder_angle > threshold_angle:
        return 12  # "신체 균형 이상, 낙상 징후"

    # 어깨 비대칭성 감지
    if delta_x > threshold_asymmetry or delta_y > threshold_asymmetry:
        return 13  # "어깨 비대칭성 감지, 낙상 위험"

    # 무릎 비대칭성 감지
    if knee_delta_x > threshold_asymmetry or knee_delta_y > threshold_asymmetry:
        return 14  # "무릎 비대칭성 감지, 낙상 위험"

    return 0  # 정상 범주

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

def is_fallenlearned(modelname, selectedmodel, row) :

    # 1. 포즈 데이터 추출
    pose = extract_pose_data(row)
    ls_x, ls_y, ls_v, ls_ok = pose["ls"]
    rs_x, rs_y, rs_v, rs_ok = pose["rs"]
    lk_x, lk_y, lk_v, lk_ok = pose["lk"]
    rk_x, rk_y, rk_v, rk_ok = pose["rk"]

    # 2. 필요한 변수 선언
    is_fallen = 0

    # 3. 데이터 준비
    new_data = np.array([
        [ls_x, ls_y, ls_v, ls_ok, rs_x, rs_y, rs_v, rs_ok, lk_x, lk_y, lk_v, lk_ok, rk_x, rk_y, rk_v, rk_ok]
    ])

    scaler_path = os.path.join("user", "model", f"fall_model_{modelname}Model.pkl")
    scaler = joblib.load(scaler_path)
    new_data_scaled = scaler.transform(new_data)

    # 4. 모델 입력 형태에 따라 예측
    try:
        if modelname == "dense":
            prediction = selectedmodel.predict(new_data_scaled, verbose=0)
        elif modelname == "lstm":
            X_input = new_data_scaled.reshape((1, 1, 16))
            prediction = selectedmodel.predict(X_input, verbose=0)
        elif modelname == "ensemble":
            input_dense = new_data_scaled
            input_lstm = new_data_scaled.reshape((1, 1, 16))
            prediction = selectedmodel.predict([input_dense, input_lstm], verbose=0)
        else:
            return 0  # 알 수 없는 모델명 → 낙상 아님으로 처리
    except Exception as e:
        print(f"[에러] 모델 예측 중 오류 발생: {e}")
        return 0

    # 5. 낙상 판단
    fall_probability = float(prediction[0][0])

    if fall_probability >= is_fallen_model_json["FALL_PROBABILITY"]:
        is_fallen = 1
    else:
        is_fallen = 0

    return is_fallen