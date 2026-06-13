import re, string, io
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import docx
import pandas as pd

print(pd.read_csv('resumes.csv').columns.tolist())

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

#file text extraction
def extract_text(file):
    name = file.name.lower()
    if name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith('.docx'):
        document = docx.Document(file)
        return " ".join(p.text for p in document.paragraphs)
    elif name.endswith('.txt'):
        return file.read().decode('utf-8', errors='ignore')
    else:
        return ""

#model training
@st.cache_resource
def train_model(csv_path='resumes.csv'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Resume_str', 'Category'])
    df['cleaned'] = df['Resume_str'].apply(clean_text)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, tfidf, le, acc

def predict_category(model, tfidf, le, resume_text):
    cleaned = clean_text(resume_text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return le.inverse_transform([pred])[0], round(proba, 3)

def match_score(tfidf, resume_text, jd_text):
    r = clean_text(resume_text)
    j = clean_text(jd_text)
    vecs = tfidf.transform([r, j])
    return round(cosine_similarity(vecs[0], vecs[1])[0][0], 3)

#using streamlit UI
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screening using NLP & ML")

with st.spinner("Loading model..."):
    model, tfidf, le, acc = train_model()
st.success(f"Model ready (test accuracy: {acc:.2%})")

jd_text = st.text_area("Paste Job Description here:", height=150)

uploaded_files = st.file_uploader(
    "Upload resumes (PDF, DOCX, or TXT) — multiple allowed",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if st.button("Screen Resumes"):
    if not jd_text.strip():
        st.warning("Please paste a job description first.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        results = []
        for file in uploaded_files:
            text = extract_text(file)
            if not text.strip():
                st.error(f"Could not extract text from {file.name}")
                continue
            score = match_score(tfidf, text, jd_text)
            category, conf = predict_category(model, tfidf, le, text)
            results.append({
                "Candidate": file.name,
                "Match Score": score,
                "Predicted Role": category,
                "Confidence": conf
            })

        if results:
            results_df = pd.DataFrame(results).sort_values("Match Score", ascending=False)
            st.subheader("Ranked Results")
            st.dataframe(results_df, use_container_width=True)

            top = results_df.iloc[0]
            st.info(f"Best match: **{top['Candidate']}** "
                    f"(score: {top['Match Score']}, role: {top['Predicted Role']})")