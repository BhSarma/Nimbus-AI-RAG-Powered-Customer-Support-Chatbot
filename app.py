"""Streamlit interface for NimbusCart AI."""
import streamlit as st
from main import answer_question, build_vectorstore

st.set_page_config(page_title="NimbusCart AI", page_icon="💬", layout="centered")
st.title("NimbusCart AI")
st.caption("RAG-powered customer support assistant")
st.write("Ask about orders, refunds, payments, shipping, accounts, and support policies.")

with st.sidebar:
    st.subheader("About this assistant")
    st.write("Answers are grounded in the provided NimbusCart Customer Support Policy Handbook.")
    st.caption("Local model: Ollama / llama3")

@st.cache_resource(show_spinner="Reading the handbook and building the search index…")
def load_vectorstore():
    return build_vectorstore()

try:
    vectorstore = load_vectorstore()
except Exception as exc:
    st.error("Could not prepare the RAG knowledge base.")
    st.exception(exc)
    st.info("Check that customer_support.pdf is in the project folder and that the dependencies are installed.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved handbook excerpts"):
                for i, source in enumerate(message["sources"], start=1):
                    page = source.metadata.get("page")
                    page_label = f"Page {page + 1}" if isinstance(page, int) else "Handbook excerpt"
                    st.markdown(f"**{i}. {page_label}**")
                    st.write(source.page_content)

question = st.chat_input("Ask a customer support question…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook and preparing an answer…"):
            try:
                answer, sources = answer_question(question, vectorstore)
                st.markdown(answer)
                if sources:
                    with st.expander("View retrieved handbook excerpts"):
                        for i, source in enumerate(sources, start=1):
                            page = source.metadata.get("page")
                            page_label = f"Page {page + 1}" if isinstance(page, int) else "Handbook excerpt"
                            st.markdown(f"**{i}. {page_label}**")
                            st.write(source.page_content)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })
            except Exception as exc:
                st.error("The answer could not be generated. Check that Ollama is running and the llama3 model is available.")
                st.exception(exc)
