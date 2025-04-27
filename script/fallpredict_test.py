import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# ✅ 모델과 스케일러는 "1번만" 로딩
model_dir = os.path.abspath(os.path.join("user", "model"))
fall_model = os.path.join(model_dir, "fall_model.keras")
scaler_model = os.path.join(model_dir, "scaler.pkl")

model = load_model(fall_model)
scaler = joblib.load(scaler_model)

def predict_fall(single_data):
    """하나의 샘플(new_data) 예측하기"""
    single_data_scaled = scaler.transform(single_data.reshape(1, -1))
    prediction = model.predict(single_data_scaled, verbose=0)
    fall_probability = prediction[0][0]
    return fall_probability

def batch_predict_fall(batch_data):
    """여러 개 샘플(batch_data)을 한꺼번에 예측하기"""
    batch_data_scaled = scaler.transform(batch_data)  # batch_data는 (N, 16) 형태
    predictions = model.predict(batch_data_scaled, verbose=0)
    fall_probabilities = predictions.flatten()  # (N,) 형태로 변환
    return fall_probabilities

"""
if __name__ == "__main__":
    # ✅ 여기서는 new_data만 준비하고 함수 호출
    new_data = np.array([0.43, 0.62, 1, 1, 0.53, 0.62, 1, 1, 0.46, 0.82, 0.67, 0, 0.51, 0.75, 0.66, 0])
    
    fall_probability = predict_fall(new_data)

    print(f"낙상 확률: {fall_probability:.4f}")

    if fall_probability >= 0.5:
        print("🚨 낙상 감지!")
    else:
        print("✅ 정상 상태")
"""