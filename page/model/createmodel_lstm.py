import streamlit as st
import numpy as np
import pandas as pd
import os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

def show(auto_run):
    if auto_run :
        st.title("🧠 LSTM 기반 모델 생성")
        st.write("수집된 데이터를 기반으로 LSTM 모델을 생성합니다.")

        filename = "fall_model_lstmModel.keras"
        pkl_filename = "fall_model_lstmModel.pkl"

        displaytime = 1.5
        progress_box = st.empty()
        csv_dir = os.path.abspath(os.path.join("user", "csv"))
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and f != "merged_user_data.csv"]

        time.sleep(displaytime)
        progress_box.success("✔️ 파일을 병합 중입니다...")

        df_list = [pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files]
        st.code("\n".join(csv_files))
        time.sleep(displaytime)
        merged_df = pd.concat(df_list, ignore_index=True)
        progress_box.success(f"✔️ 총 {len(csv_files)}개의 CSV 파일을 합쳐서 하나의 데이터로 만들었습니다.")

        time.sleep(displaytime)
        filtered_df = merged_df[merged_df["checkFall"] == 1]
        total_falls = filtered_df.shape[0]
        st.info(f"✔️ 전체 데이터 중 낙상으로 판별된 데이터({total_falls}건)만 보기 완료")
        st.dataframe(filtered_df)

        time.sleep(displaytime)
        csv_path = os.path.join(csv_dir, "merged_user_data.csv")
        merged_df.to_csv(csv_path, index=False)
        progress_box.success("✔️ CSV 병합 파일을 저장했습니다.")

        df = pd.read_csv(csv_path)

        time.sleep(displaytime)
        progress_box.success("✔️ [1] 입력(X)과 타겟(y)을 분리합니다.")
        X = df.drop(columns=['timestamp', 'checkFall'])
        y = df['checkFall']

        time.sleep(displaytime)
        progress_box.success("✔️ [2] 스케일링을 시작합니다.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # LSTM 용 입력

        time.sleep(displaytime)
        progress_box.success("✔️ [3] 학습/검증 데이터 분리를 시작합니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_3d, y, test_size=0.2, random_state=42
        )

        time.sleep(displaytime)
        progress_box.success("✔️ [4] LSTM 모델을 정의합니다.")
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        time.sleep(displaytime)
        progress_box.success("✔️ [5] 모델을 컴파일합니다.")
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        time.sleep(displaytime)
        progress_box.success("✔️ [6] EarlyStopping을 적용합니다.")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        time.sleep(displaytime)
        progress_box.warning("✔️ [7] 모델을 학습합니다. 시간이 조금 걸립니다. 조금만 기다려주세요.")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        model_dir = os.path.abspath(os.path.join("user", "model"))
        os.makedirs(model_dir, exist_ok=True)

        time.sleep(displaytime)
        model_save_path = os.path.join(model_dir, filename)
        model.save(model_save_path)
        progress_box.success("✔️ [8] 모델을 저장했습니다.")

        scaler_save_path = os.path.join(model_dir, pkl_filename)
        joblib.dump(scaler, scaler_save_path)
        progress_box.success("✔️ [9] 스케일러를 저장했습니다.")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        time.sleep(displaytime)
        progress_box.success("✔️ [10] 테스트 데이터로 정확도를 평가합니다.")

        col1, col2 = st.columns(2)
        with col1:
            fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
            ax_acc.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
            ax_acc.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
            ax_acc.set_title('Model Accuracy Curve')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.grid(True, linestyle='--', alpha=0.5)
            ax_acc.legend(loc='lower right')
            st.pyplot(fig_acc)

        with col2:
            fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
            ax_loss.plot(history.history['loss'], label='Train Loss', linewidth=2)
            ax_loss.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
            ax_loss.set_title('Loss Over Epochs')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.grid(True, linestyle='--', alpha=0.5)
            ax_loss.legend(loc='upper right')
            st.pyplot(fig_loss)

        acc_train_end = history.history['accuracy'][-1]
        acc_val_end = history.history['val_accuracy'][-1]
        acc_gap = abs(acc_train_end - acc_val_end)

        if acc_gap < 0.05:
            st.info("✅ 학습과 검증 정확도의 차이가 작아 일반화 성능이 우수한 모델입니다.")
        elif acc_train_end > acc_val_end:
            st.warning("⚠️ 학습 정확도는 높지만 검증 정확도가 낮아 과적합이 의심됩니다.")
        else:
            st.error("❌ 전체적으로 정확도가 낮아 모델 구조 또는 데이터 품질 개선이 필요합니다.")

        #progress_box.info(f"🎉 모델링이 완료되었습니다! '{model_dir}'에 저장되었습니다. 모델 정확도는 {accuracy * 100:.2f}% 입니다. ")
        progress_box.info(f"🎉 모델링이 완료되었습니다! '{model_dir}'에 저장되었습니다.")
