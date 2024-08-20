import streamlit as st
from textblob import TextBlob
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from PyPDF2 import PdfReader
import io
import google.generativeai as genai
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
import base64
from fpdf import FPDF
import textstat
from transformers import BertForQuestionAnswering, BertTokenizer

nltk.download('punkt')

# Set up the Gemini API key
genai.configure(api_key="YOUR_API_KEY_HERE")

# Function for Sumy Summarization
def sumy_summarizer(docx, num_sentences=3):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, num_sentences)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# Function for GPT-4 Summarization (now using Gemini)
def abstractive_summarizer(docx):
    model = genai.GenerativeModel(model_name="gemini-pro")
    prompt = f"Summarize the following text:\n\n{docx}"
    response = model.generate_content(prompt)
    summary = response.text.strip()
    return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function for Q&A using BERT
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

# Function for Translation using Gemini
def translate_text(text, target_language):
    model = genai.GenerativeModel(model_name="gemini-pro")
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    response = model.generate_content(prompt)
    translated_text = response.text.strip()
    return translated_text

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

# Function for Language Detection
def detect_language(text):
    return detect(text)

# Function for Voice Output
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function for exporting results
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

# Function for advanced analytics
def text_analytics(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    readability = textstat.flesch_reading_ease(text)
    return sentiment, readability

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

def main():
    """Eda Mowne"""

    # Title
    st.title("Doc AI")
    st.subheader("Document Processing and Management System")
    st.markdown("""
        #### Description
        + Doc AI is a Document Processing and Management System designed to streamline the handling and organization of documents.
        """)

    # PDF Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", text, height=200)

        # Named Entity Recognition Visualization
        if st.checkbox("Show Named Entity Visualization"):
            st.subheader("Named Entity Recognition")
            entity_html = visualize_entities(text)
            st.write(entity_html, unsafe_allow_html=True)

        # Word Cloud Visualization
        if st.checkbox("Show Word Cloud"):
            st.subheader("Word Cloud")
            wordcloud_plot = generate_wordcloud(text)
            st.pyplot(wordcloud_plot)

        # Summarization
        if st.checkbox("Show Text Summarization"):
            st.subheader("Summarize Your Text")
            summary_type = st.selectbox("Choose Summarization Type", ["Extractive (Sumy)", "Abstractive (Gemini)"])
            num_sentences = st.slider("Number of Sentences (for Extractive Summarization)", 1, 10, 3)

            if summary_type == "Extractive (Sumy)":
                summary_result = sumy_summarizer(text, num_sentences)
            else:
                summary_result = abstractive_summarizer(text)
                
            st.success(summary_result)

        # Translation
        if st.checkbox("Translate Text"):
            st.subheader("Translate Your Text")
            target_language = st.selectbox("Select Target Language", ['es', 'fr', 'de', 'zh'])
            if st.button("Translate"):
                translated_text = translate_text(text, target_language)
                st.success(translated_text)

        # Language Detection
        if st.checkbox("Detect Language"):
            st.subheader("Detected Language")
            detected_lang = detect_language(text)
            st.success(f"The detected language is: {detected_lang}")

        # Voice Output for Summarized Text
        if st.checkbox("Voice Output for Summarized Text"):
            st.subheader("Listen to the Summary")
            summary_result = sumy_summarizer(text)
            if st.button("Play Summary"):
                speak_text(summary_result)

        # Export Results
        if st.checkbox("Export Results"):
            st.subheader("Export the results")
            export_format = st.selectbox("Select export format", ["PDF", "DOCX", "CSV"])
            filename = st.text_input("Enter the filename (without extension)")
            if st.button("Export"):
                export_results(text, f"{filename}.{export_format.lower()}", export_format)
                st.success(f"Results exported as {filename}.{export_format.lower()}")

if __name__ == '__main__':
    main()
