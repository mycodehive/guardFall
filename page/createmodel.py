import streamlit as st
import numpy as np
import pandas as pd
import os, time, io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def show():
    st.title("🧠 모델 생성")
    st.write("수집된 데이터를 기반으로 학습 모델을 생성합니다.")

    displaytime = 1.5
    progress_box = st.empty()
    question_box = st.empty()
    answer_box = st.empty()

    csv_dir = os.path.abspath(os.path.join("user", "csv"))
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and f != "merged_user_data.csv"]

    time.sleep(displaytime)
    progress_box.success("✔️ 파일을 병합 중입니다...")

    df_list = []
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    file_name = "\n".join(csv_files)
    st.code(file_name)

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

    time.sleep(displaytime)
    progress_box.success("✔️ CSV 병합 파일을 저장했습니다.")

    df = pd.read_csv(csv_path)
    #buffer = io.StringIO()
    #df.info(buf=buffer)
    #info_str = buffer.getvalue()
    #st.text(info_str)

    time.sleep(displaytime)
    progress_box.success("✔️ [1] 입력(X)과 타겟(y)을 분리합니다.")
    X = df.drop(columns=['timestamp', 'checkFall'])
    y = df['checkFall']

    time.sleep(displaytime)
    progress_box.success("✔️ [2] 스케일링을 시작합니다.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    time.sleep(displaytime)
    progress_box.success("✔️ [3] 학습/검증 데이터 분리를 시작합니다.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    time.sleep(displaytime)
    progress_box.success("✔️ [4] 모델을 정의합니다.")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    time.sleep(displaytime)
    progress_box.success("✔️ [5] 모델을 컴파일합니다.")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    time.sleep(displaytime)
    progress_box.success("✔️ [6] EarlyStopping을 적용합니다.")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

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
    progress_box.success("✔️ [8] 모델을 저장합니다.")
    model_save_path = os.path.join(model_dir, 'fall_model.keras')
    model.save(model_save_path)

    time.sleep(displaytime)
    progress_box.success("✔️ [9] 스케일러를 저장합니다.")
    scaler_save_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_save_path)

    time.sleep(displaytime)
    progress_box.info(f"🎉 모델링이 완료되었습니다! '{model_dir}'에 저장되었습니다.")