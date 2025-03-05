import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

# Download NLP resources
nltk.download("stopwords")
nltk.download("punkt")

# ✅ Streamlit Page Config
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# ✅ Custom Dark Theme & Styling
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #121212;
        }
        .main-container {
            background-color: #1e1e1e;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.1);
        }
        .card {
            background-color: #232323;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        .ranking-container {
            background-color: #2b2b2b;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            color: #ffffff;
        }
        .highlight {
            font-weight: bold;
            color: #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 8px 20px !important;
            border: none !important;
        }
        .stTextInput>div>div>input, .stTextArea>div>textarea {
            background-color: #2b2b2b !important;
            color: white !important;
            border-radius: 5px !important;
            border: 1px solid #4CAF50 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🚀 App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🚀 AI Resume Screening & Ranking System")
st.write("Upload multiple resumes and a job description to get ranked results.")

# 📄 Job Description Input
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Upload Job Description")
job_desc_text = st.text_area("Paste Job Description Here")
job_desc_file = st.file_uploader("Or Upload JD (PDF)", type=["pdf"], key="jd")
st.markdown('</div>', unsafe_allow_html=True)

if job_desc_file is not None:
    with pdfplumber.open(job_desc_file) as pdf:
        job_desc_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# 📂 Resume Upload
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload Resumes")
resume_files = st.file_uploader("Upload Multiple Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="resumes")
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Function to Process Text
@st.cache_data
def process_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stopwords.words("english")])

# ✅ Function to Rank Resumes (Fixes ranking issue)
@st.cache_data
def rank_resumes(job_desc_text, resumes):
    try:
        if not job_desc_text:
            st.error("Please provide a job description.")
            return []

        processed_jd = process_text(job_desc_text)
        ranked_results = []

        vectorizer = TfidfVectorizer()
        jd_vector = vectorizer.fit_transform([processed_jd])  # ✅ Job Description ko fix rakhenge

        for resume in resumes:
            with pdfplumber.open(resume) as pdf:
                text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                if text:
                    resume_text = process_text(text)
                    resume_vector = vectorizer.transform([resume_text])  # ✅ Har resume ka alag score calculate hoga
                    similarity = (jd_vector * resume_vector.T).toarray()[0][0]
                    ranked_results.append((resume.name, similarity))

        return sorted(ranked_results, key=lambda x: x[1], reverse=True)  # ✅ Fix rankings (stable sort)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# 🔍 Process & Rank Resumes
if job_desc_text and resume_files:
    with st.spinner("🔍 Processing resumes and ranking..."):
        ranked_resumes = rank_resumes(job_desc_text, resume_files)

        # 🔹 Progress Bar Simulation
        progress_text = st.empty()
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)  # Simulate processing time
            progress_text.text(f"Processing... {percent_complete}%")
            progress_bar.progress(percent_complete + 1)

        # 🏆 Display Rankings
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
