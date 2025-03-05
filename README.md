# AI Resume Screening & Ranking System 🚀

An AI-powered tool to screen and rank resumes based on their relevance to a given job description. Built with **Python, Streamlit, and NLP techniques**.

---

## 🔥 Features  

✅ **Resume Ranking** – Upload multiple resumes and a job description to get ranked results.  
✅ **PDF Support** – Supports PDF files for both resumes and job descriptions.  
✅ **NLP Processing** – Uses **NLTK** for text preprocessing and **TF-IDF** for similarity scoring.  
✅ **User-Friendly Interface** – Built with **Streamlit** for an intuitive and interactive experience.  

---

## 🛠️ Technologies Used  

- **Python** 🐍 – Core programming language  
- **Streamlit** 🌐 – For building the web application interface  
- **PDFPlumber** 📄 – For extracting text from PDF files  
- **NLTK** 🤖 – For natural language processing (tokenization, stopwords removal)  
- **Scikit-learn** 🔢 – For **TF-IDF vectorization** and **similarity scoring**  

---

## 📦 Installation  

### 1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/Vaibhatekale/AI-Resume-Screening-Ranking-System.git
cd AI-Resume-Screening-Ranking-System

 Create Virtual Environment & Activate
 python -m venv .venv  
source .venv/bin/activate  # For Linux/macOS  
.venv\Scripts\activate     # For Windows  

Install Dependencies
pip install -r requirements.txt

Run the Application
streamlit run app.py

📂 Project Structure
AI-Resume-Screening-Ranking-System/
├── app.py                # Main Streamlit application
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── .gitignore            # Files to ignore in Git

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

