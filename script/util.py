from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import ast, os

# 현재 시간 가져오기
def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 초 단위까지 포함

# 변환을 시도하기 전에 데이터 타입 확인
def safe_literal_eval(x):
    if isinstance(x, str):  # 문자열일 때만 변환
        return ast.literal_eval(x)
    return x  # 이미 튜플이면 그대로 반환

# checkfall == 1이 포함된 경우를 찾아서 csv에 저장
def save_fall_segments(frame_landmarks: pd.DataFrame, selected):
    # 경로 설정
    output_dir = os.path.abspath(os.path.join("user", "monitor"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 낙상 발생 index 찾기
    fall_indices = frame_landmarks[frame_landmarks['checkfall'] == 1].index

    # 낙상 구간 저장용 리스트
    segments = []

    for fall_idx in fall_indices:
        start_idx = max(0, fall_idx - 10)
        segment_df = frame_landmarks.iloc[start_idx:]
        segments.append(segment_df)

    # 모든 낙상 구간 병합
    if segments:
        new_data = pd.concat(segments, ignore_index=True)
    else:
        return  # 낙상 데이터가 없으면 함수 종료

    # 파일 경로 지정
    filepath = os.path.join(output_dir, f"fall_segment_{selected}.csv")

    # 기존 파일이 있으면 이어붙이기
    if os.path.exists(filepath):
        try:
            existing_data = pd.read_csv(filepath)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        except Exception as e:
            print(f"기존 파일 읽기 오류: {e}")
            combined_data = new_data
    else:
        combined_data = new_data

    # 저장
    try:
        combined_data.to_csv(filepath, index=False)
    except Exception as e:
        print(f"저장 중 오류 발생: {e}")

#전체데이터 저장
def save_fall_all_segments(frame_landmarks: pd.DataFrame, selected):
    # 경로 설정
    output_dir = os.path.abspath(os.path.join("user", "monitor"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 파일명에 시간정보와 선택 값 포함
    filepath = os.path.join(output_dir, f"fall_segment_{selected}.csv")

    try:
        frame_landmarks.to_csv(filepath, index=False)  # ✅ 전체 저장
    except Exception as e:
        print(f"❌ 저장 중 오류 발생: {e}")