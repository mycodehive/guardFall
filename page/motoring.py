import streamlit as st
import cv2, time
import mediapipe as mp
import pandas as pd
import os, json
from script import util
from script import fallpredict
from script import sendmsg

def convert_landmarks_to_row(frame_landmarks):
    return {
        "left_shoulder_x": frame_landmarks["left_shoulder"][0],
        "left_shoulder_y": frame_landmarks["left_shoulder"][1],
        "left_shoulder_v": frame_landmarks["left_shoulder"][2],
        "left_shoulder_vr": frame_landmarks["left_shoulder"][3],

        "right_shoulder_x": frame_landmarks["right_shoulder"][0],
        "right_shoulder_y": frame_landmarks["right_shoulder"][1],
        "right_shoulder_v": frame_landmarks["right_shoulder"][2],
        "right_shoulder_vr": frame_landmarks["right_shoulder"][3],

        "left_knee_x": frame_landmarks["left_knee"][0],
        "left_knee_y": frame_landmarks["left_knee"][1],
        "left_knee_v": frame_landmarks["left_knee"][2],
        "left_knee_vr": frame_landmarks["left_knee"][3],

        "right_knee_x": frame_landmarks["right_knee"][0],
        "right_knee_y": frame_landmarks["right_knee"][1],
        "right_knee_v": frame_landmarks["right_knee"][2],
        "right_knee_vr": frame_landmarks["right_knee"][3],
    }

# 경로 설정
config_path = os.path.abspath(os.path.join("user", "setting", "config.json"))

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
    global fallen_count
    fallen_count = 0

    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 화면 상단 구성
    col1_top, col2_top, col3_top = st.columns([2, 3, 5])

    with col1_top:
        col1_box = st.empty()
        col1_box.info("카메라 OFF")

    with col2_top:
        col2_box = st.empty()
        col2_box.info("낙상여부 판단")

    with col3_top:
        col3_box = st.empty()
        col3_box.info("낙상모델")

    col1, col2 = st.columns([4, 6])
    with col1:
        msg_box = st.empty()
        msg_box.info("낙상 횟수를 보여줍니다.")
    with col2:
        msg_send = st.empty()
        msg_send.info("메세지 전송여부를 보여줍니다.")
        
    #col1_bottom, col2_bottom = st.columns([3, 7])
    #with col1_bottom :
    #    st.info("낙상 분석")
    #with col2_bottom :
    #    st.info("""
#2025년 5월 2일 오전 10시 42분경, 어르신의 자세 변화에서 낙상으로 의심되는 행동이 관측되었습니다.
#감지된 시간 동안, 무릎 위치가 어깨보다 비정상적으로 높게 측정되었고, 무릎 관절이 카메라에서 사라진 것으로 보아 갑작스러운 자세 붕괴 또는 바닥으로의 낙하 가능성이 있습니다.
#이는 일상적인 움직임과는 다른, 위험한 자세 변화 패턴으로, 즉각적인 확인이 필요합니다.
#"""
#)
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
        landmark_data_df = [] # df에 넣을 데이터 저장장

        if start:
            # ✅ JSON 파일 최신 상태로 다시 로드
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            fallen_send_msg_json = config.get("fallen_send_msg", {})
            st.session_state.camera = cv2.VideoCapture(0)
            pose = mp_pose.Pose(model_complexity=1)
            last_print_time = 0
            analyzing = True

            while analyzing:
                if "camera" not in st.session_state:
                    st.session_state.camera = cv2.VideoCapture(0)
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
                if results.pose_landmarks:                    
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
                    if current_time - last_print_time >= 1.5 :
                        last_print_time = current_time
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

                try:
                    col2_box_msg = fall_check_function(convert_landmarks_to_row(frame_landmarks))
                except UnboundLocalError:
                    col2_box_msg = "데이터가 아직 수신되지 않았습니다."


                # convert_landmarks_to_row에 timestamp 넣어서 df로 변환하고 발생시점 전 10개의 데이터 저장하기
                row = convert_landmarks_to_row(frame_landmarks)
                row = {"timestamp": frame_landmarks["timestamp"]}
                row.update(convert_landmarks_to_row(frame_landmarks))
                row["checkfall"] = col2_box_msg
                landmark_data_df.append(row)
                df = pd.DataFrame(landmark_data_df)
                util.save_fall_segments(df)

                if col2_box_msg == 1 :
                    col2_box.error("💥🧓💢 **낙상!!**  \n⚠️ 감지된 자세가 위험합니다.", icon="🚨")
                    fallen_count += 1
                    if (fallen_count % fallen_send_msg_json["FALL_COUNT"] == 0) :
                        msg_send.info(f"낙상 {fallen_count}회가 초과하였으므로 보호자에게 메세지를 발송합니다.")
                        if fallen_send_msg_json["SEND_YN"] == "Y" :
                            sendmsg.send_message(f"{frame_landmarks['timestamp']} 낙상 발생!\n빠른 시간안에 확인 바랍니다.")
                    else :
                        msg_box.info(f"낙상 {fallen_count}회 발생하였습니다.")
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