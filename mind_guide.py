import streamlit as st
import openai
import uuid
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPEN_API_KEY')

# OpenAI 챗 모델을 사용한 응답 생성 함수
def get_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message['content'].strip()

# Streamlit 애플리케이션 설정
st.set_page_config(layout="wide")

# 스타일 설정
st.markdown("""
    <style>
        .chat-container {
            height: 400px;
            overflow-y: auto;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('MindGuide ChatBot')

# 채팅 히스토리 저장소
if 'history' not in st.session_state:
    st.session_state.history = []

# 사용자 입력 저장소
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

# 채팅 히스토리 표시
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.text_area("You: ", value=chat["content"], height=100, max_chars=None, key=chat["id"], disabled=True)
        else:
            st.text_area("AI: ", value=chat["content"], height=100, max_chars=None, key=chat["id"], disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 사용자 입력
input_container = st.empty()
with input_container.container():
    user_input = st.text_input("You: ", key="user_input")

    if st.button("Send", key="send_button"):
        if user_input:
            # 현재 대화 기록에 사용자 입력 추가
            st.session_state.history.append({"role": "user", "content": user_input, "id": str(uuid.uuid4())})
            
            # OpenAI 챗 모델에 대한 메시지 형식 준비
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            messages += [{"role": chat["role"], "content": chat["content"]} for chat in st.session_state.history]
            
            # 챗 모델을 사용해 응답 생성
            response = get_response(messages)
            
            # 대화 기록에 챗봇 응답 추가
            st.session_state.history.append({"role": "assistant", "content": response, "id": str(uuid.uuid4())})
            
            # 입력 필드 비우기
            del st.session_state.user_input
            st.session_state.user_input = ''

            # 페이지 다시 로드하여 채팅 히스토리 업데이트
            st.experimental_rerun()
