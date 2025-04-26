import streamlit as st
import cv2
import mediapipe as mp
from collections import defaultdict
import os
import pandas as pd
import ast
from script import util
from script import fallpredict

def show():
    st.title("🎥 영상 분석")
    st.write("영상의 좌표를 추출하고 추출한 좌표값으로 낙상여부를 체크합니다. 해당 데이터는 머신러닝 학습 데이터로 활용됩니다.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    landmark_logs = []
    csv_format = ()

    col1, col2 = st.columns([4,6])

    with col2:
        st.markdown("### 📥 분석 데이터를 추출합니다.")
        landmarks_box = st.empty()

    with col1:
        st.markdown("### 💾 파일을 업로드하세요.")
        video_file = st.file_uploader("🎞️ 분석할 영상 파일을 업로드하세요", type=["mp4", "avi"])
        progress_box = st.empty()
        if video_file:
            file_path = f"./user/mov/{video_file.name}"
            with open(file_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(file_path)
            frame_placeholder = st.empty()
            pose = mp_pose.Pose()
            stop_button = st.button("⏹ 영상 처리 중지")
            last_print_time = 0

            landmark_data = []  #추출한 데이터 저장용

            # [---여기서부터 영상 분석---------------------
            # 프레임 읽기
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR → RGB 변환
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe 처리
                results = pose.process(image)

                # 랜드마크가 존재하면
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # 예: 주요 관절 추출 (왼쪽 어깨, 오른쪽 어깨, 왼쪽 무릎, 오른쪽 무릎)
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

                    # 좌표 실시간 표시
                    #landmarks_box.markdown(f"""
                    #⏱ **시간**: {frame_landmarks['timestamp']}  
                    #🦴 **왼쪽 어깨**: {frame_landmarks['left_shoulder']}  
                    #🦴 **오른쪽 어깨**: {frame_landmarks['right_shoulder']}  
                    #🦵 **왼쪽 무릎**: {frame_landmarks['left_knee']}  
                    #🦵 **오른쪽 무릎**: {frame_landmarks['right_knee']}  
                    #""")

                    # 좌표 문자열 생성
                    log_text = f"""
                    ⏱ {frame_landmarks['timestamp']}  
        🦴 왼쪽 어깨: {frame_landmarks['left_shoulder']} 🦴 오른쪽 어깨: {frame_landmarks['right_shoulder']}  
        🦵 왼쪽 무릎: {frame_landmarks['left_knee']} 🦵 오른쪽 무릎: {frame_landmarks['right_knee']}  
                    """
                    landmark_logs.append(log_text)
                    
                    landmarks_box.markdown("### 📝 누적 좌표 로그(x,y,신뢰도,적합여부)\n\n" + '\n---\n'.join(landmark_logs[-5:]), unsafe_allow_html=True)

                # 화면에 출력
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")

                if stop_button:
                    break

            cap.release()

            # 👉 landmark 데이터 저장
            if landmark_data:
                df = pd.DataFrame(landmark_data)

                # 문자열 좌표를 튜플 형태로 변환 및 새로운 컬럼 생성
                for joint in ["left_shoulder", "right_shoulder", "left_knee", "right_knee"]:
                    df[joint] = df[joint].apply(util.safe_literal_eval)  # 안전한 변환 적용
                    df[[f"{joint}_x", f"{joint}_y", f"{joint}_v", f"{joint}_vr"]] = df[joint].apply(lambda x: pd.Series(x))

                # 기존 컬럼 제거
                df.drop(columns=["left_shoulder", "right_shoulder", "left_knee", "right_knee"], inplace=True)
                df["checkFall"] = df.apply(lambda row: fallpredict.is_fallen(
                                            row["left_shoulder_y"], row["right_shoulder_y"], 
                                            row["left_knee_y"], row["right_knee_y"], 
                                            row["left_shoulder_vr"], row["right_shoulder_vr"], 
                                            row["left_knee_vr"], row["right_knee_vr"]
                                        ), axis=1)

                csv_dir = os.path.abspath(os.path.join("user", "csv"))
                file_name = os.path.splitext(os.path.basename(file_path))[0]+"_landmarks.csv"
                csv_save_path = os.path.join(csv_dir, file_name)
                csv_display_path = os.path.join("user", "csv", file_name)
                df.to_csv(csv_save_path, index=False)
                progress_box.success(f"좌표 추출 완료! CSV 저장: {csv_display_path}")
            # ]---여기까지 영상 분석---------------------
