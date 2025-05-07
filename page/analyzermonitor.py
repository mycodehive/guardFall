import streamlit as st
import numpy as np
import pandas as pd
import os, time, io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def show():
    st.title("🧠 모니터링 데이터 분석")
    st.write("실시간 모니터링 한 데이터를 기반으로 GPT API로 분석합니다.")

    # 📁 1. 파일이 저장된 폴더 경로
    FOLDER_PATH = os.path.abspath(os.path.join("user", "monitor"))

    # 📦 2. 해당 폴더의 파일 리스트 가져오기
    try:
        files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]
    except FileNotFoundError:
        st.error(f"폴더가 존재하지 않습니다: {FOLDER_PATH}")
        files = []
    
    # 📋 3. 파일 리스트 보여주기
    selected_file = st.radio("파일을 선택하세요", files if files else ["(파일 없음)"])

    if selected_file != "(파일 없음)":
        csv_dir = os.path.abspath(os.path.join(f"user", "monitor", selected_file))

        # 파일 존재 여부 확인
        if os.path.exists(csv_dir):
            try:
                df = pd.read_csv(csv_dir)
                st.success(f"✅ 낙상 세그먼트 데이터({selected_file})가 성공적으로 로드되었습니다.")
                st.dataframe(df)  # Streamlit 테이블로 출력
            except Exception as e:
                st.error(f"❌ CSV 파일을 읽는 도중 오류가 발생했습니다: {e}")
        else:
            st.warning("⚠️ 'fall_segment.csv' 파일이 존재하지 않습니다.")
        
        #df.to_string(index=False)
        if selected_file == "fall_segment_상체모델(Test).csv":
            st.title("🧠 GPT 데이터 분석 결과(샘플)")
            st.markdown("""
    제공해주신 관절 좌표 데이터를 분석한 결과, **낙상(checkFall=1)이 감지된 시간 구간**은 아래와 같습니다:

    ```
    [2025-05-02 10:42:58.999] ~ [2025-05-02 10:42:59.874]
    ```

    이 시간 동안 관찰된 주요 특징을 토대로 보호자께 안내드릴 수 있는 **낙상 징후 해석**은 다음과 같습니다:

    ---

    ### 🧠 낙상 징후 분석 결과

    #### ✅ 1. 어깨 대비 무릎의 비정상적인 높이 차이

    * 일반적으로 서 있는 상태에서는 어깨의 Y좌표가 무릎보다 **크게 작아야** 합니다 (즉, 더 위에 위치).
    * 그러나 낙상이 감지된 시점에서는 **무릎의 Y좌표가 2.7에 가까운 높은 값**으로 측정되었고, 어깨는 1.0 이하의 값이었습니다.
    * 이는 **몸이 바닥에 거의 수평으로 누운 상태**임을 시사합니다.

    #### ✅ 2. 낙상 시기의 관절 인식률

    * 낙상 구간에서 무릎 관절(`left_knee_vr`, `right_knee_vr`)의 신뢰도가 모두 **0**으로 나타났습니다.
    * 이는 **낙상 시 무릎 부위가 카메라 시야에서 벗어나거나 인식이 어려운 자세로 전환되었음을 의미**합니다.

    #### ✅ 3. 시간에 따른 변화 흐름

    * 낙상이 발생하기 전 구간에서는 어깨와 무릎의 Y값이 **천천히 증가**하다가, 낙상 직전 프레임(10:42:58.886 → 10:42:58.999)에서 **갑자기 무릎 높이가 급격히 증가**합니다.
    * 이는 단순히 앉거나 숙인 동작이 아닌, **예상치 못한 자세 변화**를 반영합니다.

    ---

    ### 📢 보호자 안내 내용

    > 📌 **낙상 징후가 감지되었습니다.**
    >
    > 2025년 5월 2일 오전 10시 42분경, 어르신의 자세 변화에서 낙상으로 의심되는 행동이 관측되었습니다.
    > 감지된 시간 동안, 무릎 위치가 어깨보다 비정상적으로 높게 측정되었고, 무릎 관절이 카메라에서 사라진 것으로 보아 **갑작스러운 자세 붕괴 또는 바닥으로의 낙하** 가능성이 있습니다.
    > 이는 일상적인 움직임과는 다른, **위험한 자세 변화 패턴**으로, 즉각적인 확인이 필요합니다.

    ---
    """
        )