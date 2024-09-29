import streamlit as st
import uuid
from utils.rag import rag
from data_processing.qdrant_loader import upload_documents_qdrant, load_documents


def main():
    st.title("RAG on Anchor")
    # Session state initialization

    if "count" not in st.session_state:
        st.session_state.count = 0

    source = st.selectbox(
        "Select a source:", ["anchor_book.pdf", "GettingStartedwithAnchor.pdf"]
    )

    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            output = rag(user_input, source)
            st.success("Completed!")
            st.write(output)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            st.session_state.count += 1
    with col2:
        if st.button("-1"):
            st.session_state.count -= 1

    st.write(f"Current count: {st.session_state.count}")


if __name__ == "__main__":
    main()
