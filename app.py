import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLP resources
nltk.download("stopwords")
nltk.download("punkt")

# Streamlit Page Config
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# ✅ **Custom CSS Styling**
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f4f4f4;
        }
        .main-container {
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .ranking-container {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .highlight {
            font-weight: bold;
            color: #1976d2;
        }
        .stButton>button {
            background-color: #1976d2 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 8px 20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🚀 **App Layout**
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🚀 AI Resume Screening & Ranking System")
st.write("Upload multiple resumes and a job description to get rankings.")

# 📄 **Job Description Input (Inside a Card)**
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Upload Job Description")
job_desc_text = st.text_area("Paste Job Description Here")
job_desc_file = st.file_uploader("Or Upload JD (PDF)", type=["pdf"], key="jd")
st.markdown('</div>', unsafe_allow_html=True)

if job_desc_file is not None:
    with pdfplumber.open(job_desc_file) as pdf:
        job_desc_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# 📂 **Resume Upload (Inside a Card)**
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload Resumes")
resume_files = st.file_uploader("Upload Multiple Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="resumes")
st.markdown('</div>', unsafe_allow_html=True)

# ✅ **Function to Process Text**
@st.cache_data
def process_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stopwords.words("english")])

# ✅ **Use Function**
@st.cache_data
def use(job_desc_text, resume_files):
    processed_jd = process_text(job_desc_text)
    resume_texts = []
    resume_names = []

    for resume in resume_files:
        with pdfplumber.open(resume) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            if text:
                resume_texts.append(process_text(text))
                resume_names.append(resume.name)

    vectorizer = TfidfVectorizer()
    all_texts = [processed_jd] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1:]

    return sorted(zip(resume_names, similarities), key=lambda x: x[1], reverse=True)

# 🔍 **Process Resumes & Rank**
if job_desc_text and resume_files:
    with st.spinner("🔍 Processing resumes and ranking..."):
        ranked_resumes = use(job_desc_text, resume_files)

        # 🔹 **Progress Bar Simulation**
        progress_text = st.empty()
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_text.text(f"Processing... {percent_complete}%")
            progress_bar.progress(percent_complete + 1)

        # 🏆 **Display Rankings (Inside a Card)**
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Resume Rankings")
        for rank, (name, score) in enumerate(ranked_resumes, start=1):
            st.markdown(
                f'<div class="ranking-container">'
                f'<p class="highlight">{rank}. {name} - Match Score: {score:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
