<<<<<<< HEAD
# AI Resume Screening & Ranking System  

 An AI-powered system to screen and rank resumes based on job descriptions.  

##  Features  
✅ Upload multiple resumes in **PDF format**  
✅ Enter a **job description** for comparison  
✅ Uses **TF-IDF & Cosine Similarity** for ranking  
✅ Displays **ranked resumes with similarity scores**  
✅ Built with **Python, Flask, and NLP techniques**  

## 📂 Project Structure  
AI Resume Screening & Ranking System/
│── app.py # Main application file
│── requirements.txt # Dependencies list
│── resume_ranking/ # Resume processing logic
│ ├── init.py # Module initialization
│ ├── resume_processor.py # Extract and clean text from PDFs
│ ├── similarity.py # TF-IDF & Cosine Similarity logic
│── data/ # Folder for uploaded resumes
│── sample_resume.pdf # Sample resume for testing
│── README.md # Project documentation
│── .gitignore # Git ignore file


##  Installation  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd AI-powered-Resume-Screening-and-Ranking-System

python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
.venv\Scripts\activate     # For Windows

pip install -r requirements.txt

python app.py

How It Works
Upload multiple resumes in PDF format
Enter the job description
The system extracts text from PDFs and preprocesses it
It calculates TF-IDF scores and applies cosine similarity
Resumes are ranked based on similarity scores

Technologies Used
Python 🐍
Flask 🌐
Streamlit (For UI)
Scikit-Learn 🤖
pdfplumber 📄 (For PDF text extraction)
NLTK (For text preprocessing)
🤝 Contributing
Feel free to contribute! Fork the repo, make changes, and submit a pull request.

📜 License
This project is open-source and available under the MIT License.

---

Ab **is README.md ko paste karke commit kar**:  
```bash
git add README.md
git commit -m "Added proper README file"
git push origin main
=======
# AI-Resume-Screening-Ranking-System
