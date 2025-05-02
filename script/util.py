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
def save_fall_segments(frame_landmarks: pd.DataFrame):
    # 경로 설정
    output_dir = os.path.abspath(os.path.join("user", "monitor"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 낙상 발생 index 찾기
    fall_indices = frame_landmarks[frame_landmarks['checkfall'] == 1].index

    for i, fall_idx in enumerate(fall_indices):
        start_idx = max(0, fall_idx - 10)
        end_idx = fall_idx + 10  # 현재 행 포함

        segment_df = frame_landmarks.iloc[start_idx:end_idx]

    # 파일명에 시간정보와 순번 포함 (중복 방지)
    filepath = os.path.join(output_dir,"fall_segment.csv")

    try:
        segment_df.to_csv(filepath, index=False)
    except UnboundLocalError:
        print("")
