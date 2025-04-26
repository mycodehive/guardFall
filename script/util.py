from datetime import datetime
from zoneinfo import ZoneInfo
import ast

# 현재 시간 가져오기
def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 초 단위까지 포함

# 변환을 시도하기 전에 데이터 타입 확인
def safe_literal_eval(x):
    if isinstance(x, str):  # 문자열일 때만 변환
        return ast.literal_eval(x)
    return x  # 이미 튜플이면 그대로 반환