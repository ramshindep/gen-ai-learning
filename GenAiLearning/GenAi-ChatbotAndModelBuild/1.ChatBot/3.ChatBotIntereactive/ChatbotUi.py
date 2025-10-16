import streamlit as st
import requests

st.set_page_config(page_title="Chatbot", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "messages" not in st.session_state:
    st.session_state.messages = []       
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None



def new_chat():
    """Start a new chat and store the old one."""
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"][:30] + "..." if st.session_state.messages else "Untitled"
        st.session_state.chat_history.append({
            "title": title,
            "messages": st.session_state.messages.copy()
        })
    st.session_state.messages = []
    st.session_state.active_chat_index = None


def load_chat(index):
    """Load a previous chat from history."""
    st.session_state.messages = st.session_state.chat_history[index]["messages"].copy()
    st.session_state.active_chat_index = index



with st.sidebar:
    st.header("Chat Sidebar")

    # New chat button stays on top
    if st.button("New Chat", on_click=new_chat):
        st.rerun()  

    st.divider()
    st.subheader("📜 Previous Chats")

    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            index = len(st.session_state.chat_history) - 1 - i  
            if st.button(chat["title"], key=f"chat_{index}"):
                load_chat(index)
                st.rerun() 
    else:
        st.caption("No previous chats yet.")



chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")


input_container = st.empty()
with input_container.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...", key="chat_input")
    submit_button = st.form_submit_button(label="Send")


if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            "http://127.0.0.1:5400/api/chat",
            json={"message": user_input},
            timeout=30
        )
        if response.status_code == 200:
            ai_message = response.json().get("response", "No response from AI")
        else:
            ai_message = f"Error: {response.status_code}"
    except Exception as e:
        ai_message = f"Error: {e}"

    st.session_state.messages.append({"role": "bot", "content": ai_message})
    st.rerun()
