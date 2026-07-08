import streamlit as st

from chatbot import HistoryChatBotRag
from config import MODEL_NAME, PROMPT_TEMPLATE,EMBEDDING_MODEL

### PAGE SETUP

st.set_page_config(
    page_title="My Chatbot",
    page_icon="J",
    layout="centered",
)


st.title("My Chatbot")
st.markdown("## ASK YOUR QUESTION RELATED TO HISTORY")
st.markdown("**DEVELOPED BY JAWAD AMIN**")

##INITIALIZE CHATBOT via SESSION.
if "chatbot" not in st.session_state:
    #st.session_state.chatbot = HistoryChatBotRag(MODEL_NAME,EMBEDDING_MODEL,PROMPT_TEMPLATE)
    with st.spinner("Building datastore... This may take a moment"):
        st.session_state.chatbot = HistoryChatBotRag(
            MODEL_NAME,
            EMBEDDING_MODEL,
            PROMPT_TEMPLATE
        )
    st.success("Datastore ready!")

##INITIALIZE MEMORY via SESSION.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input := st.chat_input("Type any of your history questions? Type exit for a new conversation."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("User"):
        st.markdown(user_input)

    if user_input.strip().lower() == "exit":
        st.session_state.messages = []
        st.session_state.chabot.clear_memory()
    else:
        with st.chat_message("AI AGENT"):
            with st.spinner("AI is processing. Please wait..."):
                response = st.session_state.chatbot.get_response(user_input)
            st.markdown(f"**Answer:**\n\n{response['answer']}")

            if response.get("sources"):
                st.markdown("**Sources:**")
                for src in response["sources"]:
                    st.markdown(f"- {src}")
        st.session_state.messages.append({"role": "AI Agent", "content": response})
