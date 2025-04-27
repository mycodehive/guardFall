import script.fallpredict_test as ft
import numpy as np

new_data = np.array([
        [0.44, 0.62, 1, 1, 0.52, 0.63, 1, 1, 0.46, 0.81, 0.63, 0, 0.49, 0.84, 0.71, 1],  # 샘플1
        [0.43, 0.62, 1, 1, 0.53, 0.62, 1, 1, 0.46, 0.82, 0.67, 0, 0.51, 0.75, 0.66, 0]  # 샘플2
    ])

# fall_probability = ft.predict_fall(new_data)

print("✅ 여러 데이터에 대해 낙상 예측을 시작합니다...\n")

fall_probabilities = ft.batch_predict_fall(new_data)

for idx, prob in enumerate(fall_probabilities, start=1):
    print(f"샘플 {idx} 낙상 확률: {prob:.4f}")
    if prob >= 0.5:
        print("🚨 낙상 감지!\n")
    else:
        print("✅ 정상 상태\n")