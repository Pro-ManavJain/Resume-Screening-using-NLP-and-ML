Resume Screening using NLP & Machine Learning
Overview
Recruiters often receive hundreds of resumes for a single job opening, making the initial screening process time-consuming and repetitive. This project aims to simplify that process by automatically analyzing resumes, comparing them with a job description, and ranking candidates based on their relevance.
The application uses Natural Language Processing (NLP) techniques along with Machine Learning to extract meaningful information from resumes, predict the candidate's most suitable job role, and calculate how closely the resume matches the given job description.
The project is built using Python, Scikit-Learn, and Streamlit, providing an easy-to-use interface for uploading and evaluating multiple resumes at once.

Features
Upload multiple resumes simultaneously.
Supports PDF, DOCX, and TXT resume formats.
Automatic resume text extraction.
NLP-based text preprocessing and cleaning.
Resume categorization using a Machine Learning model.
Job description matching using cosine similarity.
Candidate ranking based on match score.
Confidence score for predicted role.
Interactive web interface built with Streamlit.

How It Works
1. Resume Parsing
The system extracts text from uploaded resumes using:
PyPDF2 for PDF files
python-docx for DOCX files
Standard file reading for TXT files

2. Text Preprocessing
Before any analysis, the resume text is cleaned through several NLP steps:
Conversion to lowercase
Removal of URLs
Removal of email addresses
Removal of numbers
Removal of punctuation
Stopword removal using NLTK
Word lemmatization using WordNet Lemmatizer
This preprocessing helps reduce noise and improves the quality of extracted features.

3. Feature Extraction
The cleaned text is converted into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).
The vectorizer uses:
Maximum 5000 features
Unigrams and bigrams (ngram_range=(1,2))
This allows the model to capture both individual keywords and meaningful word combinations.

4. Role Prediction
A Logistic Regression classifier is trained on labeled resume data.
The model predicts:
Candidate's most suitable job category
Prediction confidence score
The training process includes:
Train-test split (80:20)
Stratified sampling
Model evaluation using accuracy score

5. Resume Matching
To determine how well a resume fits a particular job opening, the project computes the Cosine Similarity between:
Resume text vector
Job description vector
The resulting score indicates how closely the candidate's skills and experience align with the requirements of the job.

6. Candidate Ranking
All uploaded resumes are ranked according to their match score.
The final results display:
Candidate Name
Match Score
Predicted Role
Confidence Score
The system also highlights the highest-ranked candidate as the best match for the provided job description.

Tech Stack
| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Streamlit    | Web application interface |
| Scikit-Learn | Machine Learning models   |
| NLTK         | Text preprocessing        |
| Pandas       | Data handling             |
| PyPDF2       | PDF text extraction       |
| python-docx  | DOCX text extraction      |


Dataset
The model is trained using a resume dataset containing:
Resume text
Corresponding job categories
Required columns:
Resume_str, Category
