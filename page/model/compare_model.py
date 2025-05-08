import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import os

def show(auto_run):
    if auto_run:
        st.title("모델을 비교합니다")
        st.write("수집된 데이터를 기반으로 3가지 학습 모델을 비교합니다.")

        csv_dir = os.path.abspath(os.path.join("user", "csv"))
        file_path = os.path.join(csv_dir, "merged_user_data.csv")
        df = pd.read_csv(file_path)

        # 입력 피처 및 정답 레이블 분리
        X = df.drop(columns=["timestamp", "checkFall"]).values
        y_true = df["checkFall"].values

        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 모델 로드
        model_dir = os.path.abspath(os.path.join("user", "model"))
        model_paths = {
            "Dense Model": os.path.join(model_dir, "fall_model_denseModel.keras"),
            "LSTM Model": os.path.join(model_dir, "fall_model_lstmModel.keras"),
            "Ensemble Model": os.path.join(model_dir, "fall_model_ensembleModel.keras"),
        }

        results = []

        for model_name, path in model_paths.items():
            model = load_model(path)

            # 입력 형태 처리
            if model_name == "Ensemble Model":
                X_input = [X_scaled, X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))]
            elif len(model.input_shape) == 3:
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            else:
                X_input = X_scaled

            y_pred = model.predict(X_input)
            y_pred_label = (y_pred > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_true, y_pred_label)
            recall = recall_score(y_true, y_pred_label)
            consistency = np.std(y_pred)

            results.append({
                "모델명": model_name,
                "정확도(Accuracy)": f"{accuracy * 100:.2f}%",
                "추론 일관성(표준편차)": f"{consistency:.4f}",
                "재현율(Recall)": f"{recall * 100:.2f}%"
            })

        # 결과를 표로 정리
        results_df = pd.DataFrame(results)
        results_df["정확도(%)"] = results_df["정확도(Accuracy)"].str.replace('%', '').astype(float)
        results_df["표준편차"] = results_df["추론 일관성(표준편차)"].astype(float)
        results_df["재현율(%)"] = results_df["재현율(Recall)"].str.replace('%', '').astype(float)

        # 테이블 출력
        st.table(results_df[["모델명", "정확도(Accuracy)", "추론 일관성(표준편차)","재현율(%)"]])

        # 정확도 및 안정성 기준 정렬
        accuracy_sorted = results_df.sort_values(by="정확도(%)", ascending=False)["모델명"].tolist()
        stability_sorted = results_df.sort_values(by="표준편차", ascending=True)["모델명"].tolist()
        recall_sorted = results_df.sort_values(by="재현율(%)", ascending=False)["모델명"].tolist()

        # 종합 분석 문장
        best_acc = results_df["정확도(%)"].max()
        acc_best_models = results_df[results_df["정확도(%)"] == best_acc]["모델명"].tolist()

        min_std = results_df["표준편차"].min()
        std_best_models = results_df[results_df["표준편차"] == min_std]["모델명"].tolist()

        # 최고 재현율 모델
        best_recall = results_df["재현율(%)"].max()
        recall_best_models = results_df[results_df["재현율(%)"] == best_recall]["모델명"].tolist()

        summary_md = f"""
        ## 📊 모델 종합 분석

        - **정확도 차지**: {' / '.join(acc_best_models)} 모델이 {best_acc:.2f}%로 가장 우수합니다.
        - **추론 일관성(표준편편차)**: {' / '.join(std_best_models)} 모델이 {min_std:.4f}로 가장 안정적입니다.
        - **재현율(Recall)**: {' / '.join(recall_best_models)} 모델이 {best_recall:.2f}%로 낙상을 가장 잘 감지합니다.

        🧠 결론적으로, 
        - **낙상 여부를 넓게 감지하고 싶다면**: `{', '.join(recall_best_models)}`
        - **높은 전체 정확도를 원한다면**: `{', '.join(acc_best_models)}`
        - **출력값의 일관성과 안정성을 중시한다면**: `{', '.join(std_best_models)}`
        """

        st.markdown(summary_md)

        # 추가 해석 출력
        st.markdown(f"""         
        ## 🧠 항목별 상세 해석
        1. 🔍 **정확도 (Accuracy)**        
          - 정확도 기준으로는 {' > '.join(accuracy_sorted)} 입니다.<p>

        2. 📈 **재현율 (Recall)**        
          - 재현율 기준으로는 {' > '.join(recall_sorted)} 입니다.
          - 💡 재현율은 실제 낙상 중에서 감지된 비율로, 놓치지 않는 민감도를 측정합니다.

        3. 📉 **추론 일관성 (표준편차)**        
          - 안정성 기준으로는 {' > '.join(stability_sorted)} 입니다.
          - 💡 추론 일관성(표준편차)는 예측 확률의 흔들림 정도를 나타냅니다. 낮을수록 안정된 예측을 의미합니다.
        """, unsafe_allow_html=True)
