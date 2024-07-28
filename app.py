import streamlit as st

# from autofill import autofill_text

def main():
    """Doc AI"""

    # Title
    st.title("Doc AI")
    st.subheader("Intelligent Document Processing and Management")
    st.markdown("""
        #### Description
        + You can do all this here, for Free :
        Summarization, Q&A, Translation, Entity Extraction from PDF documents
        """)

    # PDF Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", text, height=200)

        # Summarization
        if st.checkbox("Show Text Summarization"):
            st.subheader("Summarize Your Text")
            summary_result = sumy_summarizer(text)
            st.success(summary_result)

        # Q&A
        if st.checkbox("Ask a Question"):
            st.subheader("Ask a Question about the Text")
            question = st.text_input("Enter your question")
            if st.button("Get Answer"):
                answer = answer_question(text, question)
                st.success(answer)

        # Translation
        if st.checkbox("Translate Text"):
            st.subheader("Translate Your Text")
            target_language = st.selectbox("Select Target Language", ['es', 'fr', 'de', 'zh'])
            if st.button("Translate"):
                translated_text = translate_text(text, target_language)
                st.success(translated_text)

        # Entity Extraction
        if st.checkbox("Show Named Entities"):
            st.subheader("Analyze Your Text")
            entities = entity_analyzer(text)
            st.json(entities)

    st.sidebar.subheader("Doc AI",)
    st.sidebar.text("Intelligent Document Processing and Management")
    st.sidebar.info("Kudos SnehaP and Team")

if __name__ == '__main__':
    main()
