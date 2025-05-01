import asyncio
import streamlit as st
from script import util as ut
from telegram import Bot
import toml
import os

TELEGRAM_TOKEN = st.secrets['telegram']['TELEGRAM_TOKEN']
CHAT_ID = st.secrets['telegram']['CHAT_ID']

bot = Bot(token=TELEGRAM_TOKEN)

async def send_telegram_message(text: str):
    await bot.send_message(chat_id=CHAT_ID, text=text)

# 필요 시 main에서 비동기 호출용 래퍼 함수
def send_message(text):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 이벤트 루프가 없으면 새로 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_telegram_message(text))
    else:
        # 이미 실행 중인 루프가 있으면 그 루프에 태스크로 등록
        asyncio.ensure_future(send_telegram_message(text))