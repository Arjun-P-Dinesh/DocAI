import streamlit as st
from streamlit_option_menu import option_menu
from textblob import TextBlob
import base64
import spacy
import subprocess
import sys
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import io
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import pyttsx3
import pandas as pd
from docx import Document
from fpdf import FPDF
import textstat
from transformers import BertForQuestionAnswering, BertTokenizer

nltk.download('punkt')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from sumy.parsers.plaintext import PlaintextParser
except ImportError:
    install("sumy")
    from sumy.parsers.plaintext import PlaintextParser

try:
    from PyPDF2 import PdfReader
except ImportError:
    install("PyPDF2")
    from PyPDF2 import PdfReader

try:
    from docx import Document
except ImportError:
    install("python-docx")
    from docx import Document

def sumy_summarizer(docx, num_sentences=3):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, num_sentences)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

login_img_base64 = get_img_as_base64("images/STBG.jpg")
background_img_base64 = get_img_as_base64("images/STBG.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpeg;base64,{background_img_base64}");
background-size: 106%;
background-repeat: no-repeat;
background-position: right;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpeg;base64,{login_img_base64}");
background-position: fill; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def answer_question_bert(text, question):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def translate_text(text, target_language):
    openai.api_key = 'groq_api_key'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following text to {target_language}:\n\n{text}",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.3,
    )
    translated_text = response.choices[0].text.strip()
    return translated_text

def visualize_entities(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    html = spacy.displacy.render(docx, style='ent', jupyter=False)
    return html

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def detect_language(text):
    return detect(text)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def export_results(text, filename, format):
    if format == 'PDF':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(200, 10, text)
        pdf.output(filename)
    elif format == 'DOCX':
        doc = Document()
        doc.add_paragraph(text)
        doc.save(filename)
    elif format == 'CSV':
        df = pd.DataFrame({"Text": [text]})
        df.to_csv(filename, index=False)

def text_analytics(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    readability = textstat.flesch_reading_ease(text)
    return sentiment, readability

def main():
    """Doc AI: Document Processing and Management System"""

    # Setting up the option menu for navigation
    selected_option = option_menu(
        menu_title="Main Menu",  # required
        options=["HOME", "SUMMARIZATION", "TRANSLATION", "TEXT-TO-SPEECH", "EXPORT-RESULTS", "QUESTION-ANSWERING", "ADDITIONAL-FUNCTIONS"],  # required
        icons=["house", "book", "translate", "volume-up", "cloud-upload", "question-circle", "layers"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#0E1117"},
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "grey"},
            "nav-link-selected": {"background-color": "blue"},
        },
    )

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)

    if selected_option == "HOME":
        st.title("Welcome to Doc AI")
        st.write("Welcome to Doc AI, where you get all Document solutions in just one place. Use the options in the menu to navigate through the functionalities.")
        
        if uploaded_file is not None:
            st.write("Do you want to see the extracted text?")
            if st.button("Show Extracted Text"):
                st.text_area("Extracted Text", text, height=200)

    elif selected_option == "SUMMARIZATION":
        st.title("Text Summarization")
        if uploaded_file is not None:
            num_sentences = st.slider("Number of Sentences (for Extractive Summarization)", 1, 10, 3)
            summary_result = sumy_summarizer(text, num_sentences)
            st.success(summary_result)
            
            if st.button("Show Keywords"):
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform([text])
                feature_names = tfidf_vectorizer.get_feature_names_out()
                dense = tfidf_matrix.todense()
                text_dense = dense.tolist()[0]
                keywords = [feature_names[i] for i in range(len(text_dense)) if text_dense[i] > 0.1]
                st.write("Keywords:", keywords)

    elif selected_option == "TRANSLATION":
        st.title("Text Translation")
        if uploaded_file is not None:
            target_language = st.selectbox("Select Target Language", ['es', 'fr', 'de', 'zh'])
            if st.button("Translate"):
                translated_text = translate_text(text, target_language)
                st.success(translated_text)
    
    elif selected_option == "TEXT TO SPEECH":
        st.title("Text to Speech")
        if uploaded_file is not None:
            summarized_text = sumy_summarizer(text)
            if st.button("Show Summarized Text"):
                st.write(summarized_text)
            if st.button("Play Summarized Text"):
                speak_text(summarized_text)

    elif selected_option == "EXPORT RESULTS":
        st.title("Export Results")
        if uploaded_file is not None:
            export_format = st.selectbox("Select export format", ["PDF", "DOCX", "CSV"])
            filename = st.text_input("Enter the filename (without extension)")
            if st.button("Export"):
                export_results(text, f"{filename}.{export_format.lower()}", export_format)
                st.success(f"Results exported as {filename}.{export_format.lower()}")

    elif selected_option == "QUESTION ANSWERING":
        st.title("Question Answering")
        if uploaded_file is not None:
            question = st.text_input("Ask a question:")
            if st.button("Get Answer"):
                answer = answer_question_bert(text, question)
                st.success(answer)

    elif selected_option == "ADDITIONAL FUNCTIONALITIES":
        additional_options = st.selectbox("Select Additional Functionality", ["Named Entity Recognition", "Word Cloud", "Text Simplification", "Multi-document Similarity"])

        if uploaded_file is not None:
            if additional_options == "Named Entity Recognition":
                st.title("Named Entity Recognition")
                entity_html = visualize_entities(text)
                st.write(entity_html, unsafe_allow_html=True)
            
            elif additional_options == "Word Cloud":
                st.title("Word Cloud")
                wordcloud_plot = generate_wordcloud(text)
                st.pyplot(wordcloud_plot)
            
            elif additional_options == "Text Simplification":
                st.title("Text Simplification")
                st.write("Text simplification functionality goes here.")
            
            elif additional_options == "Multi-document Similarity":
                st.title("Multi-document Similarity")
                st.write("Multi-document similarity functionality goes here.")

if __name__ == '__main__':
    main()
