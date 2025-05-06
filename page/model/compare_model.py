import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os

def show(auto_run):
    if auto_run :
        st.title("🧠 모델을 비교합니다")
        st.write("수집된 데이터를 기반으로 3가지 학습 모델을 비교합니다.")

        csv_dir = os.path.abspath(os.path.join("user", "csv"))
        file_path = os.path.join(csv_dir, "merged_user_data.csv")
        df = pd.read_csv(file_path)

        # 입력 피처 및 정답 레이블 분리
        X = df.drop(columns=["timestamp", "checkFall"]).values
        y_true = df["checkFall"].values

        # 정규화 (모델이 훈련된 환경과 일치하도록)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 모델 로드
        csv_dir = os.path.abspath(os.path.join("user", "model"))
        file_Dense_path = os.path.join(csv_dir, "fall_model_denseModel.keras")
        file_LSTM_path = os.path.join(csv_dir, "fall_model_lstmModel.keras")
        file_Ensemblepath = os.path.join(csv_dir, "fall_model_ensembleModel.keras")
        model_paths = {
            "Dense Model": file_Dense_path,
            "LSTM Model": file_LSTM_path,
            "Ensemble Model": file_Ensemblepath
        }

        # 예측 결과 저장
        results = []

        for model_name, path in model_paths.items():
            model = load_model(path)

            # LSTM 입력 형태 처리 (3D로 변환)
            if model_name == "Ensemble Model":
                # 두 입력을 모두 전달
                X_input = [X_scaled, X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))]
            elif len(model.input_shape) == 3:
                # LSTM 전용
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            else:
                X_input = X_scaled


            y_pred = model.predict(X_input)
            y_pred_label = (y_pred > 0.5).astype(int).flatten()

            accuracy = accuracy_score(y_true, y_pred_label)
            consistency = np.std(y_pred)

            results.append({
                "모델명": model_name,
                "정확도(Accuracy)": f"{accuracy * 100:.2f}%",
                "추론 일관성(표준편차)": f"{consistency:.4f}"
            })

        # 결과를 표로 정리
        results_df = pd.DataFrame(results)
        # 인덱스를 1부터 시작하도록 설정
        results_df.index = results_df.index + 1

        #--[---------------------------
        # 분석 컬럼 생성
        analysis = []

        for _, row in results_df.iterrows():
            acc = float(row["정확도(Accuracy)"].replace('%', ''))
            std = float(row["추론 일관성(표준편차)"])

            # 정확도 분석
            if acc >= 82.0:
                acc_comment = "정확도가 가장 높음"
            elif acc >= 81.0:
                acc_comment = "적절한 정확도를 보임"
            else:
                acc_comment = "정확도가 다소 낮음"

            # 표준편차 분석
            if std <= 0.13:
                std_comment = "추론이 매우 일관됨"
            elif std <= 0.15:
                std_comment = "추론이 안정적임"
            else:
                std_comment = "추론 편차가 다소 큼"

            # 분석 문장 조합
            full_comment = f"{acc_comment}, {std_comment}"
            analysis.append(full_comment)

        # 분석 컬럼 추가
        results_df["분석"] = analysis
        #--]---------------------------

        # 테이블로 출력
        st.table(results_df)

        # 정확도를 숫자(float)로 변환
        results_df["정확도(%)"] = results_df["정확도(Accuracy)"].str.replace('%', '').astype(float)
        # 정확도 기준 정렬
        accuracy_sorted = results_df.sort_values(by="정확도(%)", ascending=False)["모델명"].tolist()
        # 표준편차 기준 정렬
        stability_sorted = results_df.sort_values(by="추론 일관성(표준편차)", ascending=True)["모델명"].tolist()

        # 결과 출력
        print("✅ 정확도 기준 순위:", " > ".join(accuracy_sorted))
        print("✅ 안정성 기준 순위:", " > ".join(stability_sorted))

        st.markdown(f"""         
        ## 🧠 항목별 상세 해석
        1. 🔍 정확도 (Accuracy)        
           - 정확도 기준으로는 {" > ".join(accuracy_sorted)} 입니다.<p>

        2. 📉 추론 일관성 (표준편차)
           - 안정성 기준으로는 {" > ".join(stability_sorted)} 입니다.
           - 💡 추론 일관성 (표준편차)는 모델의 출력값들이 얼마나 분산되어 있는지를 나타냅니다. 낮을수록 "예측의 자신감"이 균일하며, 높은 값은 예측이 흔들릴 수 있음을 의미합니다.
        """,
            unsafe_allow_html=True)