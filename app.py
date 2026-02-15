# app.py

import streamlit as st

from rag.pipeline import ask
import main


st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📘 Nutrition RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask a question...")

if user_query:
    with st.spinner("Thinking..."):
        answer, contexts = ask(
            query=user_query,
            embedding_model=main.embedding_model,
            index=main.index,
            pages_and_chunks=main.pages_and_chunks,
            reranker=main.reranker,
            tokenizer=main.tokenizer,
            llm_model=main.llm_model,
            top_k_retrieval=5,
            top_k_rerank=3,
            max_new_tokens=512,
            temperature=0.7
        )

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("bot", answer))


for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
