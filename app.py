import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from PyPDF2 import PdfReader
import io
import openai
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from fpdf import FPDF
import pyttsx3
import matplotlib.pyplot as plt
from sumy.summarizers.lex_rank import LexRankSummarizer
from io import BytesIO
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import base64

# Initialize OpenAI API (replace 'your_openai_api_key' with your actual API key)
openai.api_key = 'sk-proj-bVv_AdsTqNdjJqtKrGLYu5xIMTgtnLaWlLQ5duREk6BJ71bI7wh3h26aBTT3BlbkFJ50PDxX_bN-QFY5zO5E07lywllo72jw8m8VW_se9WQhqDgwmGRgNq2cIoMA4'

def gpt_summarizer(docx, max_tokens=150, temperature=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{docx}\n\nSummary:"}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
    

# Function for Q & A
def answer_question(text, question):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(question, text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

    return answer

# Function for Q&A using BERT
def answer_question(text, question, model):
    if model == "bert":
        qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    elif model == "mistral-large":
        qa_model = pipeline("question-answering", model="mistral-large")
    elif model == "gpt3.5":
        qa_model = pipeline("question-answering", model="gpt-3.5-turbo")
    elif model == "gpt4":
        qa_model = pipeline("question-answering", model="gpt-4")
    elif model == "gpt4o":
        qa_model = pipeline("question-answering", model="gpt-4o")
    elif model == "gpt4-mini":
        qa_model = pipeline("question-answering", model="gpt-4-mini")
    elif model == "claude3.5-sonnet":
        qa_model = pipeline("question-answering", model="claude-3.5-sonnet")
    elif model == "gemmini1.5":
        qa_model = pipeline("question-answering", model="gemmini-1.5")
    else:
        raise ValueError("Model not supported.")
    
    result = qa_model(question=question, context=text)
    return result['answer']

# Function to Analyze Entities
@st.cache_data
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities

# Function to visualize entities
def visualize_entities(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    html = spacy.displacy.render(docx, style='ent', jupyter=False)
    return html

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Function for Voice Output
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to calculate document similarity
def document_similarity(doc1, doc2):
    documents = [doc1, doc2]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0, 1]

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def visualize_term_frequency(text):
    words = text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)
    df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=df, palette='viridis')
    plt.title('Top 20 Most Common Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    return plt

def main():
    """NLP Based App with Streamlit"""

    # Set page configuration
    st.set_page_config(page_title="Document AI", layout="wide")

    # Set background image
    img_path = "/images/DocAI1.jpg"
    img_base64 = get_base64_of_bin_file(img_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
# Function to adjust text size and contrast
def apply_accessibility_features():
    st.sidebar.subheader("Accessibility Settings")
    text_size = st.sidebar.slider("Adjust Text Size", 10, 40, 16)
    contrast = st.sidebar.selectbox("Adjust Color Contrast", ["Normal", "High Contrast"])

    # Apply styles dynamically
    style = f"""
    <style>
    body {{
        font-size: {text_size}px;
    }}
    """
    if contrast == "High Contrast":
        style += """
        .stApp {{
            background-color: black;
            color: white;
        }}
        """
    style += "</style>"
    st.markdown(style, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    """NLP Based App with Streamlit"""

    # Set page configuration
    st.set_page_config(page_title="Document AI", layout="wide")

    # Set background image
    img_path = "/images/DocAI1.jpg"
    img_base64 = get_base64_of_bin_file(img_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display logo and project description in the sidebar
    st.sidebar.image("images/Untitled_design_3_-removebg-preview.png", width=200)
    st.sidebar.markdown("""
        **DocAI** An intelligent solution for processing, summarizing, and analyzing documents.
    """)
    apply_accessibility_features()

    selected_option = option_menu(
        menu_title="",
        options=["HOME", "SUMMARIZATION", "TEXT TO SPEECH", "Q & A", "FEATURES", "INTERACTIVE VISUALIZATIONS"],
        icons=["house", "book", "volume-up", "question-circle", "layers", "chart-line"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "10px",
                "background-color": "#0E1117",
                "border": "1px solid #24272C",
                "border-radius": "8px"
            },
            "icon": {
                "color": "#d3d3d3",
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "20px",
                "text-align": "center",
                "margin": "10px",
                "--hover-color": "#252627",
                "color": "#d3d3d3",
                "font-family": "Arial, sans-serif",
            },
            "nav-link-selected": {
                "background-color": "#21545B"
            },
        },
    )

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        file_content = uploaded_file.read()
        text = extract_text_from_pdf(io.BytesIO(file_content))

    # Layout for selected options
    if selected_option == "HOME":
        st.title("Welcome to Document AI")
        st.write(""""Doc AI" is an advanced document analysis platform designed to facilitate seamless interaction with text-based documents through an array of powerful features, including summarization, translation, text-to-speech, and question answering. By integrating cutting-edge technologies such as GPT models, Hugging Face's Transformers, and various NLP tools, it offers users a comprehensive suite for enhancing productivity and understanding complex content. 
                 \n This platform serves as an essential tool for professionals and academics alike, providing intuitive navigation and user-friendly interfaces to efficiently manage, analyze, and derive insights from voluminous texts.
                 \n Github repo : https://github.com/Arjun-P-Dinesh/DocAI""")

    elif selected_option == "SUMMARIZATION":
        st.title("Summarization")
        if uploaded_file is not None and text:
            num_sentences = st.slider("Select Number of Sentences for Summary", min_value=1, max_value=10, value=3)
            if st.button("Summarize Text"):
                summary = gpt_summarizer(text, max_tokens=num_sentences*50)
                st.text_area("Summary", summary, height=300)
            
            if st.button("Document Summarization Settings"):
                st.write("Set preferences for document summarization, such as focus on specific sections or custom length.")
                max_length = st.slider("Max Length of Summary", min_value=50, max_value=500, value=150)
                temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7)
                if st.button("Apply Settings"):
                    summary = gpt_summarizer(text, max_tokens=max_length, temperature=temperature)
                    st.text_area("Custom Summary", summary, height=300)

    elif selected_option == "TEXT TO SPEECH":
        st.title("Text to Speech")
        if uploaded_file is not None and text:
            st.text_area("Extracted Text", text, height=300)
            if st.button("Speak Text"):
                speak_text(text)
                st.success("Text is being spoken.")
            st.button("Pause", on_click=lambda: st.stop())

    elif selected_option == "EXPORT RESULTS":
        st.title("Export Results")
        if uploaded_file is not None and text:
            file_format = st.selectbox("Choose File Format", ["PDF", "CSV"])
            if st.button("Export"):
                if file_format == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, text)
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    st.download_button("Download PDF", pdf_output.getvalue(), file_name="exported_text.pdf")
                elif file_format == "CSV":
                    df = pd.DataFrame({"Text": [text]})
                    csv_output = io.StringIO()
                    df.to_csv(csv_output, index=False)
                    st.download_button("Download CSV", csv_output.getvalue(), file_name="exported_text.csv")

    elif selected_option == "Q & A":
        st.title("Q & A")
        if uploaded_file is not None and text:
            st.write("Ask me anything about the document!")
            question = st.text_input("Enter Your Question")
            model_type = st.selectbox("Choose Model for Q&A", ["bert", "mistral-large", "gpt3.5", "gpt4", "gpt4o", "gpt4-mini", "claude3.5-sonnet", "gemmini1.5"])

            if st.button("Get Answer"):
                answer = answer_question(text, question, model_type)
                st.text_area("Answer", answer, height=100)

    elif selected_option == "FEATURES":
        st.title("Additional Features")
        st.write("Explore some additional NLP functionalities below.")
        feature_choice = st.selectbox("Choose Feature", ["Word Cloud", "Named Entity Recognition", "Document Similarity"])
        
        if feature_choice == "Named Entity Recognition":
            if uploaded_file is not None and text:
                entities = entity_analyzer(text)
                st.write("Named Entities", entities)

        elif feature_choice == "Word Cloud":
            if uploaded_file is not None and text:
                fig = generate_wordcloud(text)
                st.pyplot(fig)

        elif feature_choice == "Document Similarity":
            doc1 = st.text_area("Document 1")
            doc2 = st.text_area("Document 2")
            if st.button("Check Similarity"):
                similarity = document_similarity(doc1, doc2)
                st.write(f"Document Similarity Score: {similarity:.2f}")

            # Add Document Summarization Settings
            if st.button("Summarization Settings"):
                st.write("Set preferences for document summarization, such as focus on specific sections or custom length.")

            # Export options
            file_format = st.selectbox("Choose File Format", ["PDF", "CSV"])
            if st.button("Export Results"):
                if file_format == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, text)
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    st.download_button("Download PDF", pdf_output.getvalue(), file_name="exported_text.pdf")
                elif file_format == "CSV":
                    df = pd.DataFrame({"Text": [text]})
                    csv_output = io.StringIO()
                    df.to_csv(csv_output, index=False)
                    st.download_button("Download CSV", csv_output.getvalue(), file_name="exported_text.csv")

        else:
            st.write("Please upload a PDF file for summarization")

    # More options based on the user's selections and features described

    # INTERACTIVE VISUALIZATIONS
    if selected_option == "INTERACTIVE VISUALIZATIONS":
        st.title("Term Frequency Visualizations")
        if uploaded_file is not None and text:
            fig = visualize_term_frequency(text)
            st.pyplot(fig)
        else:
            st.write("Upload a PDF file to generate visualizations")

if __name__ == '__main__':
    main()
