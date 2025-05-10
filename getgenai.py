from openai import OpenAI
import httpx, re
import streamlit as st

def change_tags(text):
    cleaned_text = text.replace("~", "&#126;")
    return cleaned_text

def show(data_string):
    openai_api_key = st.secrets['openai']['API_KEY']
    # 프록시 없이 사용
    http_client = httpx.Client(
        timeout=20.0,
        transport=httpx.HTTPTransport(proxy=None)
    )

    # OpenAI 클라이언트 초기화
    client = OpenAI(
        api_key=openai_api_key,
        http_client=http_client
    )

    send_prompt_role_header = """너는 mediapipe로 추출한 관절좌표를 기반으로 낙상여부를 판별하는 전문가야.
아래 좌표값으로 데이터를 분석해줘. checkFall이 1이면 낙상을 의미해.
전체 데이터를 바탕으로 낙상의 징후와 어깨-무릎 관절 좌표 관계를 분석해줘.
마지막에는 [보호자님께]라고 보호자에게 안내하는 어조로 위 내용을 요약하고 조심해야 할 것들을 친절하게 설명해줘."""

    message_history_user = [{"role": "user", "content": data_string}]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"{send_prompt_role_header}"},
            *message_history_user
        ],
        temperature=0,
        max_tokens=2000
    )

    gptMessage = response.choices[0].message.content.strip()
    print(gptMessage)
    return change_tags(gptMessage)