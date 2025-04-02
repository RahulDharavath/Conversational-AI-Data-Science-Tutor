import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Chat Model (✅ FIX: Added convert_system_message_to_human=True)
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=api_key,
    convert_system_message_to_human=True  # ✅ Fix for SystemMessage error
)

# SQLite Database Connection for Chat History
sqlite_db_path = "chat_history.db"
sqlite_connection = f"sqlite:///{sqlite_db_path}"

def get_msg_history_from_db(session_id):
    return SQLChatMessageHistory(connection_string=sqlite_connection, session_id=session_id)

# Define Prompts
prompts = {
    "Beginner": (
        "You are a friendly AI Data Science Tutor for beginners. Explain concepts in simple terms with real-world examples, "
        "like talking to a child. Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided. "
        "If the user's question is not related to Data Science, politely inform them: 'I'm trained to assist with Data Science topics only.'"
    ),
    "Intermediate": (
        "You are a helpful AI Data Science Tutor for intermediate learners. Provide in-depth explanations, suggest projects, and cover practical use cases. "
        "Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided. "
        "If the user's question is not related to Data Science, politely inform them: 'I'm trained to assist with Data Science topics only.'"
    ),
    "Advanced": (
        "You are a professional AI Data Science Tutor for advanced learners. Discuss complex topics, research papers, and optimization techniques. "
        "Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided. "
        "If the user's question is not related to Data Science, politely inform them: 'I'm trained to assist with Data Science topics only.'"
    )
}

def get_system_prompt(level):
    return prompts.get(level, "You are a general AI Data Science Tutor.")

# Streamlit App Configuration
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("🤖🧠 Conversational AI Data Science Tutor")

# Login Section
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    with st.form("login_form"):
        username_input = st.text_input("Enter your username:", "")
        login_button = st.form_submit_button("Login")
        
        if login_button and username_input:
            st.session_state.username = username_input
            st.success(f"Welcome, {username_input}!")
            st.rerun()  # ✅ Fixed: Now using st.rerun()

# If user is logged in, show chatbot
if st.session_state.username:
    st.subheader(f"Welcome, {st.session_state.username}! 👋")
    
    # User level selection
    user_level = st.selectbox("Select Your current knowledge Level:", ["Beginner", "Intermediate", "Advanced"])

    # User Input
    user_input = st.text_input("Ask a question about Data Science:", "")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ FIX: Removed SystemMessage
    chat_template = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(get_system_prompt(user_level)),  # ✅ Fix: Convert SystemMessage to HumanMessage
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    output_parser = StrOutputParser()

    # Conversation Chain with username as session ID
    chain = RunnableWithMessageHistory(
        chat_template | chat_model | output_parser,
        lambda session_id=st.session_state.username: get_msg_history_from_db(session_id),
        input_messages_key="human_input",
        history_messages_key="chat_history"
    )

    if st.button("Submit"):
        if user_input:
            query = {"human_input": user_input}
            response = chain.invoke(query, config={"configurable": {"session_id": st.session_state.username}})

            # Save messages to DB
            chat_history = get_msg_history_from_db(st.session_state.username)
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(response)

            # Update session history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "Bot", "content": response})

            # Display AI Response
            st.write("🤖 **Bot:**")
            st.write(response)
        else:
            st.warning("Please enter a question!")

# Logout Button
if st.session_state.username:
    if st.button("Logout"):
        st.session_state.username = None
        st.rerun()  # ✅ Fixed: Now using st.rerun()
