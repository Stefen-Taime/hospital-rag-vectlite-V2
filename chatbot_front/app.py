"""Streamlit frontend — entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from chatbot_front.components.chat_view import render_chat
from chatbot_front.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Montreal Hospital RAG",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()
render_chat()
