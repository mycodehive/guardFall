import streamlit as st
import cv2, time
import mediapipe as mp
import pandas as pd
import av
from script import util
from script import fallpredict

def show():
    st.title("🛡️ 감시 모드")
    st.write("낙상 여부를 실시간으로 감지합니다. 아래에서 낙상 모델을 선택하세요.")

    selected = st.radio(
        "",
        ("상체모델(Test)", "사용자모델", "딥러닝모델"),
        horizontal=True
    )

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    landmark_logs = []

    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 화면 상단 구성
    col1_top, col2_top, col3_top = st.columns([3.3, 3.4, 3.3])

    with col1_top:
        col1_box = st.empty()
        col1_box.info("카메라 OFF")

    with col2_top:
        col2_box = st.empty()
        col2_box.info("낙상여부 판단")

    with col3_top:
        col3_box = st.empty()
        col3_box.info("낙상모델")

    st.markdown("---")
    # 화면 하단 구성
    col1, col2 = st.columns([4, 6])

    with col2:
        st.markdown("### 📥 실시간 분석 데이터를 추출합니다.")
        landmarks_box = st.empty()

    with col1:
        st.markdown("### 🎥 웹캠을 이용한 실시간 분석")
        # 카메라 시작/종료 버튼을 가로로 나란히 배치
        button_col1, button_col2, button_col3 = st.columns([2, 2, 2], gap="small")
        with button_col1:
            start = st.button("카메라 시작")
        with button_col2:
            stop = st.button("카메라 종료")
        with button_col3:
            print("")
        frame_display = st.empty()  # 영상 표시용

        frame_placeholder = st.empty()

        landmark_data = []  # 누적 landmark 데이터 저장

        if start:
            st.session_state.camera = cv2.VideoCapture(0)
            pose = mp_pose.Pose()
            last_print_time = 0
            analyzing = True

            while analyzing:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.warning("카메라 프레임을 읽을 수 없습니다.")
                    break

                # BGR → RGB 변환
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe Pose 처리
                results = pose.process(image)

                # 랜드마크가 있으면 처리
                current_time = time.time()
                if results.pose_landmarks and current_time - last_print_time >= 3:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    lm = results.pose_landmarks.landmark
                    frame_landmarks = {
                        "timestamp": util.now_kst(),
                        "left_shoulder": (
                            round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 2),
                            round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y, 2),
                            round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility, 2),
                            1 if round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility, 2) >= 0.7 else 0
                        ),
                        "right_shoulder": (
                            round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 2),
                            round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 2),
                            round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility, 2),
                            1 if round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility, 2) >= 0.7 else 0
                        ),
                        "left_knee": (
                            round(lm[mp_pose.PoseLandmark.LEFT_KNEE].x, 2),
                            round(lm[mp_pose.PoseLandmark.LEFT_KNEE].y, 2),
                            round(lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility, 2),
                            1 if round(lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility, 2) >= 0.7 else 0
                        ),
                        "right_knee": (
                            round(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, 2),
                            round(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y, 2),
                            round(lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility, 2),
                            1 if round(lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility, 2) >= 0.7 else 0
                        ),
                    }
                    landmark_data.append(frame_landmarks)

                    # 누적 로그 3개 표시
                    log_text = f"""
                    ⏱ {frame_landmarks['timestamp']}  
        🦴 왼쪽 어깨: {frame_landmarks['left_shoulder']} 🦴 오른쪽 어깨: {frame_landmarks['right_shoulder']}  
        🦵 왼쪽 무릎: {frame_landmarks['left_knee']} 🦵 오른쪽 무릎: {frame_landmarks['right_knee']}  
                    """
                    landmark_logs.append(log_text)
                    landmarks_box.markdown("### 📝 누적 좌표 로그(x,y,신뢰도,적합여부)\n\n" + '\n---\n'.join(landmark_logs[-1:]), unsafe_allow_html=True)

                # 화면에 출력
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")

                # ✅ 카메라 오픈 여부 확인
                if st.session_state.camera.isOpened():
                    col1_box_msg ="🎥 카메라 ON"
                col1_box.success(col1_box_msg)

                # "상체모델(Test)", "사용자모델", "딥러닝모델"
                if selected == "상체모델(Test)":
                    fall_check_function  = fallpredict.is_fallen_Upperbody
                    fall_msg = "테스트를 위해 양쪽 어깨 좌표만 사용합니다."
                elif selected == "사용자모델":
                    fall_check_function  = fallpredict.is_fallen
                    fall_msg = "동영상 학습을 위한 기준 함수입니다."
                elif selected == "딥러닝모델":
                    fall_check_function  = fallpredict.is_fallen_model   
                    fall_msg = "keras 모델로 검증합니다."

                col2_box_msg = fall_check_function(
                                    frame_landmarks['left_shoulder'][0], frame_landmarks['left_shoulder'][1],
                                    frame_landmarks['left_shoulder'][2], frame_landmarks['left_shoulder'][3],
                                    frame_landmarks['right_shoulder'][0], frame_landmarks['right_shoulder'][1],
                                    frame_landmarks['right_shoulder'][2], frame_landmarks['right_shoulder'][3],
                                    frame_landmarks['left_knee'][0], frame_landmarks['left_knee'][1],
                                    frame_landmarks['left_knee'][2], frame_landmarks['left_knee'][3],
                                    frame_landmarks['right_knee'][0], frame_landmarks['right_knee'][1],
                                    frame_landmarks['right_knee'][2], frame_landmarks['right_knee'][3])
                if col2_box_msg == 1 :
                    col2_box.error("💥🧓💢 **낙상!!**  \n⚠️ 감지된 자세가 위험합니다. 즉시 확인하세요!", icon="🚨")
                else :
                    col2_box.success("정상")
                col3_box.success(f"낙상모델 : {selected}\n{fall_msg}")

                # 분석 중지 버튼이 눌리면
                if stop:
                    analyzing = False
                    break

                time.sleep(0.03)

            # 카메라 해제
            st.session_state.camera.release()
            st.session_state.camera = None