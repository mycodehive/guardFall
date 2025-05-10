import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import os

def show(auto_run):
    if auto_run:
        st.title("모델을 비교합니다")
        st.write("수집된 데이터를 기반으로 3가지 학습 모델을 비교합니다.")

        csv_dir = os.path.abspath(os.path.join("user", "csv"))
        file_path = os.path.join(csv_dir, "merged_user_data.csv")
        df = pd.read_csv(file_path)

        X = df.drop(columns=["timestamp", "checkFall"]).values
        y_true = df["checkFall"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_dir = os.path.abspath(os.path.join("user", "model"))
        model_paths = {
            "Dense Model": os.path.join(model_dir, "fall_model_denseModel.keras"),
            "LSTM Model": os.path.join(model_dir, "fall_model_lstmModel.keras"),
            "Ensemble Model": os.path.join(model_dir, "fall_model_ensembleModel.keras"),
        }

        results = []

        for model_name, path in model_paths.items():
            model = load_model(path)

            if model_name == "Ensemble Model":
                X_input = [X_scaled, X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))]
            elif len(model.input_shape) == 3:
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            else:
                X_input = X_scaled

            y_pred = model.predict(X_input)
            y_pred_label = (y_pred > 0.5).astype(int).flatten()

            accuracy = accuracy_score(y_true, y_pred_label)
            recall = recall_score(y_true, y_pred_label, average='macro')
            f1 = f1_score(y_true, y_pred_label, average='macro')
            precision = precision_score(y_true, y_pred_label, average='macro')
            consistency = np.std(y_pred)

            results.append({
                "모델명": model_name,
                "정확도(Accuracy)": f"{accuracy * 100:.2f}%",
                "재현율(Recall)": f"{recall * 100:.2f}%",
                "정밀도(Precision)": f"{precision * 100:.2f}%",
                "F1-score": f"{f1 * 100:.2f}%",
                "추론 일관성(표준편차)": f"{consistency:.4f}"
            })

        results_df = pd.DataFrame(results)
        results_df["정확도(%)"] = results_df["정확도(Accuracy)"].str.replace('%', '').astype(float)
        results_df["재현율(%)"] = results_df["재현율(Recall)"].str.replace('%', '').astype(float)
        results_df["정밀도(%)"] = results_df["정밀도(Precision)"].str.replace('%', '').astype(float)
        results_df["F1(%)"] = results_df["F1-score"].str.replace('%', '').astype(float)
        results_df["표준편차"] = results_df["추론 일관성(표준편차)"].astype(float)

        st.table(results_df[["모델명", "정확도(Accuracy)", "재현율(Recall)", "정밀도(Precision)", "F1-score", "추론 일관성(표준편차)"]])

        # 종합 정렬
        accuracy_sorted = results_df.sort_values(by="정확도(%)", ascending=False)["모델명"].tolist()
        recall_sorted = results_df.sort_values(by="재현율(%)", ascending=False)["모델명"].tolist()
        precision_sorted = results_df.sort_values(by="정밀도(%)", ascending=False)["모델명"].tolist()
        f1_sorted = results_df.sort_values(by="F1(%)", ascending=False)["모델명"].tolist()
        stability_sorted = results_df.sort_values(by="표준편차", ascending=True)["모델명"].tolist()

        best_acc = results_df["정확도(%)"].max()
        acc_best_models = results_df[results_df["정확도(%)"] == best_acc]["모델명"].tolist()

        best_recall = results_df["재현율(%)"].max()
        recall_best_models = results_df[results_df["재현율(%)"] == best_recall]["모델명"].tolist()

        best_f1 = results_df["F1(%)"].max()
        f1_best_models = results_df[results_df["F1(%)"] == best_f1]["모델명"].tolist()

        min_std = results_df["표준편차"].min()
        std_best_models = results_df[results_df["표준편차"] == min_std]["모델명"].tolist()

        st.markdown(f"""
        ## 📊 모델 종합 분석

        - **정확도**: {' / '.join(acc_best_models)} 모델이 {best_acc:.2f}%로 가장 우수합니다.
        - **재현율(Recall)**: {' / '.join(recall_best_models)} 모델이 {best_recall:.2f}%로 낙상을 가장 잘 감지합니다.
        - **F1-score**: {' / '.join(f1_best_models)} 모델이 {best_f1:.2f}%로 정밀도와 재현율의 균형이 가장 좋습니다.
        - **추론 일관성(표준편차)**: {' / '.join(std_best_models)} 모델이 {min_std:.4f}로 가장 안정적입니다.
        """)

        st.markdown(f"""
        | 평가 항목       | 기준 순위                                | 설명 |
|----------------|-------------------------------------------|------|
| 🔍 정확도 (Accuracy)     | {' > '.join(accuracy_sorted)}           | 전체 예측 중 맞춘 비율 |
| 📈 재현율 (Recall)       | {' > '.join(recall_sorted)}           | 실제 낙상 중 감지한 비율 (민감도) |
| 🎯 정밀도 (Precision)   | {' > '.join(precision_sorted)}           | 감지된 낙상 중 실제 낙상일 확률 |
| ⚖️ F1-score            | {' > '.join(f1_sorted)}          | 정밀도와 재현율의 균형 |
| 📉 추론 일관성 (표준편차) | {' > '.join(stability_sorted)}          | 예측의 흔들림 없이 안정적인 정도 |
        """, unsafe_allow_html=True)