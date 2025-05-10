import streamlit as st
import pandas as pd
import os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import getgenai as gai
import threading

def show():
    st.title("🧠 모니터링 데이터 분석")
    st.write("실시간 모니터링 한 데이터를 기반으로 GPT API로 분석합니다. 낙상 발생 시점기준 이전 데이터까지 취합하여 분석자료로 활용됩니다.")

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

    col1, col2 = st.columns([9, 1], gap="small")
    with col1:
        msg_box = st.empty()
        msg_box.info("분석할 파일을 선택을 선택하시면 분석이 시작됩니다.")
    with col2:
        start = st.button("분석시작", use_container_width=True)

    progress_bar = st.progress(0)
    
    if selected_file != "(파일 없음)":
        csv_dir = os.path.abspath(os.path.join(f"user", "monitor", selected_file))

        # 파일 존재 여부 확인
        if os.path.exists(csv_dir):
            try:
                df = pd.read_csv(csv_dir)
                msg_box.success(f"✅ 낙상 세그먼트 데이터({selected_file})가 성공적으로 로드되었습니다.")
                #st.dataframe(df)  # Streamlit 테이블로 출력
            except Exception as e:
                st.error(f"❌ CSV 파일을 읽는 도중 오류가 발생했습니다: {e}")
        else:
            st.warning("⚠️ 'fall_segment.csv' 파일이 존재하지 않습니다.")
        
        #time.sleep(1)
        
        if start :
            # 결과를 저장할 변수 (mutable container)
            result_container = {"done": False, "result": None}

            def run_analysis():
                result_container["result"] = gai.show(df.to_csv(index=False))
                result_container["done"] = True

            # 별도 스레드에서 GPT 분석 실행
            thread = threading.Thread(target=run_analysis)
            thread.start()

            # 진행 바 표시 (실제 작업 완료까지 반복)
            percent_complete = 0
            while not result_container["done"]:
                percent_complete = min(99, percent_complete + 1)  # 최대 99까지만
                progress_bar.progress(percent_complete)
                time.sleep(0.1)  # 짧은 주기로 확인

            # 완료 시 100%로 설정
            progress_bar.progress(100)

            # 결과 출력
            st.title("🧠 GPT 데이터 분석 결과")
            st.markdown(result_container["result"])