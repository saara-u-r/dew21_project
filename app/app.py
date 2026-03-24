import streamlit as st
from src.rag import ask

st.title("⚡ DEW21 Assistant")

query = st.text_input("Ask a question")

if query:
    answer, docs = ask(query)

    st.write("### Answer")
    st.write(answer)

    with st.expander("Sources"):
        for d in docs:
            st.write(d.metadata["source"])