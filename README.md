
AI Resume Screening & Ranking System
This project is an AI-powered resume screening and ranking system built using scikit-learn and PyPDF2. The application allows users to upload resumes in PDF format, compares them to a given job description, and ranks them based on cosine similarity using TF-IDF.

🚀 Features
📂 Upload multiple PDF resumes

✍️ Enter a job description to compare

🔍 Uses TF-IDF & Cosine Similarity for ranking

📊 Displays ranked resumes with similarity scores

⚡ Easy to use and deploy with Flask
📂 Folder Structure
AI Resume Screening & Ranking System/
│
├── app.py                  # Streamlit app ka main file
├── requirements.txt        # Dependencies ka list
├── resume_ranking/         # Resume processing aur ranking logic ka module
│   ├── __init__.py         # Module initialization
│   ├── resume_processor.py # PDF se text extract aur clean karne ka code
│   └── similarity.py       # TF-IDF aur cosine similarity logic
│
├── data/                   # Uploaded resumes ka folder
│   └── sample_resume.pdf   # Sample resume testing ke liye
│
├── README.md               # Project ka documentation
└── .gitignore              # Git ignore file

