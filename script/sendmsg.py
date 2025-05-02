import asyncio
import nest_asyncio
import streamlit as st
from script import util as ut
from telegram import Bot

# ✅ 이벤트 루프 중첩 허용
nest_asyncio.apply()

# ✅ 시크릿에서 Telegram 토큰과 Chat ID 불러오기
TELEGRAM_TOKEN = st.secrets['telegram']['TELEGRAM_TOKEN']
CHAT_ID = st.secrets['telegram']['CHAT_ID']

# ✅ 봇 인스턴스 생성
bot = Bot(token=TELEGRAM_TOKEN)

# ✅ 비동기 메시지 전송 함수
async def send_telegram_message(text: str):
    await bot.send_message(chat_id=CHAT_ID, text=text)

# ✅ 외부에서 호출할 수 있는 동기 래퍼 함수
def send_message(text: str):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_telegram_message(text))
