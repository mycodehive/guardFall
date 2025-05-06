# ✅ multi_model_dashboard.py

import streamlit as st
import page.model.createmodel as cm
import page.model.createmodel_lstm as lstm
import page.model.createmodel_ensemble as ens
import page.model.compare_model as cmr

def show(auto_run=True):
    if not auto_run:
        if not st.button("▶ 모델 학습 시작"):
            return  # 버튼 누르지 않으면 함수 종료
        
    # ✅ 페이지 기본 설정
    st.title("🧠 모델 비교 대시보드")
    st.caption("Dense 모델 / LSTM 모델 / 앙상블 모델을 동시에 생성하고 비교할 수 있습니다.")

    # ✅ 탭 구성 또는 컬럼 구성 (중 택 1)

    # ✅ [1] 탭으로 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["모델 탭을 선택하세요", "🔹 Dense 모델", "🔸 LSTM 모델", "🔰 앙상블 모델","🧠 모델 비교"])

    with tab1:
        col1, col2, col3 = st.columns([3.3, 3.3, 3.3])
        with col1:
            st.markdown("""                            
            ### 1) Dense 모델
            - 정의
              - Dense 모델은 가장 기본적인 다층 퍼셉트론(MLP) 구조입니다.
              - 모든 입력 노드가 다음 층의 모든 노드와 연결됩니다.
            - 특징
              - 시계열 특성 없음: 데이터 간 시간 순서나 흐름이 없는 구조에 적합.
              - 빠른 학습: 구조가 단순해 학습 속도가 빠름.
              - 낙상 판단 기준: 관절 좌표 값만 보고 그 순간이 낙상인지 여부를 판단.
            """)
        with col2:
            st.markdown("""           
            ### 2) LSTM 모델
            - 정의
              - LSTM은 시계열 데이터를 다룰 수 있는 순환 신경망(RNN) 계열의 모델입니다.
              - 시간 흐름에 따른 이전 상태 정보를 기억하며 예측을 수행합니다.
            - 특징
              - 시간 흐름 고려: 과거의 움직임 흐름까지 반영해 예측.
              - 데이터 입력: (batch_size, time_steps, features) 형태로 입력됨 (예: (100, 1, 16))
              - 낙상 판단 기준: 이전 프레임의 좌표 움직임이 현재 낙상 여부에 영향을 준다고 가정.
            """)
        with col3:
            st.markdown("""              
            ### 3) 앙상블 모델
            - 정의
              - 서로 다른 두 모델(DNN과 LSTM)의 강점을 **병렬로 학습 후 통합(Concatenate)**하는 방식.
              - 각각의 모델이 독립적으로 데이터를 학습하고 마지막에 결과를 통합.
            - 특징
              - 결합형 구조: Dense는 현재 상태를, LSTM은 시간 흐름을 담당 → 상호 보완적 판단.
              - 높은 일반화 성능: 각기 다른 특성의 모델을 통합하므로 더 정확한 예측이 가능.
              - 낙상 판단 기준: 공간적 좌표(DNN) + 시간 흐름(LSTM)을 동시에 고려.  
            """)        

    with tab2:
        if st.button("Dense 모델 학습 시작"):
            cm.show(True)

    with tab3:
        if st.button("LSTM 모델 학습 시작"):
            lstm.show(True)

    with tab4:
        if st.button("앙상블 모델 학습 시작"):
            ens.show(True)

    with tab5:
        if st.button("모델비교 시작"):
            cmr.show(True)
        
