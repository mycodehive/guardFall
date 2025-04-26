import streamlit as st

def Github():
    st.title("🐙 소스")
    st.write("Github 소스를 공유합니다.")
    # HTML을 이용해 외부 링크로 리다이렉트
    link("https://github.com/mycodehive/Study/tree/main/seoul-ict/project/guardFall", "🔗 Git 프로젝트 보기")

def PPT():
    st.title("📊 프리젠테이션")
    st.write("최초 개발계획안입니다.")
    # HTML을 이용해 외부 링크로 리다이렉트
    link("https://gamma.app/docs/GuardFall-AI-od3bunrb5l3o40z", "🔗 최초 발표자료 보기")

def link(url, text):
    st.markdown(f"""
    <a href="{url}" target="_blank">
        <button style="padding:10px 20px; font-size:16px;">{text}</button>
    </a>
    """, unsafe_allow_html=True)
