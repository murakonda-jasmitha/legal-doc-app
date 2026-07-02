# =========================================================
# LEGAL DOCUMENT SUMMARIZER USING NLP
# =========================================================

import streamlit as st
import sqlite3
import re
import docx
import PyPDF2
import nltk
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

# =========================================================
# NLTK DOWNLOAD
# =========================================================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

download_nltk()

# =========================================================
# DATABASE
# =========================================================
def get_db():
    conn = sqlite3.connect("summaries.db", check_same_thread=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        gender TEXT,
        email TEXT,
        phone TEXT)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filename TEXT,
        summary TEXT,
        similarity REAL,
        compression REAL,
        created_at TEXT)
    """)

    return conn

db = get_db()

# =========================================================
# SESSION STATE
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None

# =========================================================
# AUTH FUNCTIONS
# =========================================================
def register_user(username,password,gender,email,phone):
    try:
        db.execute(
            "INSERT INTO users VALUES(NULL,?,?,?,?,?)",
            (username,password,gender,email,phone)
        )
        db.commit()
        return True
    except:
        return False


def login_user(username,password):
    result = db.execute(
        "SELECT 1 FROM users WHERE username=? AND password=?",
        (username,password)
    ).fetchone()
    return result is not None


# =========================================================
# # =========================================================
# LOGIN / REGISTER
# =========================================================
if not st.session_state.user:

    st.title("🔐 Legal Document Summarizer")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.user = username
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

    with tab2:

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        gender = st.selectbox("Gender", ["Male", "Female"])
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")

        if st.button("Register"):
            if register_user(new_user, new_pass, gender, email, phone):
                st.success("Registered Successfully")
            else:
                st.error("Username already exists")

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Dashboard")
st.sidebar.write("👤",st.session_state.user)

menu = st.sidebar.radio("Navigation",["Upload","History","Logout"])


# =========================================================
# TEXT EXTRACTION
# =========================================================
def extract_text(file):

    text = []
    name = file.name.lower()

    if name.endswith(".pdf"):

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

    elif name.endswith(".docx"):

        doc = docx.Document(file)

        for p in doc.paragraphs:
            if p.text.strip():
                text.append(p.text)

    return "\n".join(text)


# =========================================================
# NLP PREPROCESS
# =========================================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]","",text)

    words = word_tokenize(text)

    return " ".join(
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words
    )


# =========================================================
# SUMMARY
# =========================================================
def generate_summary(sentences,clean_sentences):

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(clean_sentences)

    similarity = cosine_similarity(tfidf)

    np.fill_diagonal(similarity,0)

    scores = similarity.sum(axis=1)

    top_n = min(5,len(sentences))

    top_idx = scores.argsort()[-top_n:]

    top_idx = sorted(top_idx)

    summary = " ".join([sentences[i] for i in top_idx])

    return summary


# =========================================================
# UPLOAD PAGE
# =========================================================
if menu=="Upload":

    st.title("📄 Upload Legal Document")

    file = st.file_uploader("Upload PDF or DOCX",["pdf","docx"])

    if file:

        text = extract_text(file)

        if not text:
            st.error("No readable text found")
            st.stop()

        sentences = sent_tokenize(text)

        clean_sentences = [preprocess(s) for s in sentences]

        if st.button("Generate Summary"):

            start_time = time.time()

            summary = generate_summary(sentences,clean_sentences)

            end_time = time.time()

            processing_time = end_time - start_time

            st.session_state.sentences = sentences
            st.session_state.clean_sentences = clean_sentences
            st.session_state.summary = summary

            # Evaluation
            vect = TfidfVectorizer()

            tfidf = vect.fit_transform([text,summary])

            similarity_score = cosine_similarity(
                tfidf[0:1],tfidf[1:2]
            )[0][0]*100

            compression_ratio = (
                1-(len(summary.split())/len(text.split()))
            )*100


            # Save to DB
            db.execute("""
                INSERT INTO summaries
                (username,filename,summary,similarity,compression,created_at)
                VALUES(?,?,?,?,?,?)
            """,(
                st.session_state.user,
                file.name,
                summary,
                similarity_score,
                compression_ratio,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

            db.commit()

            # Show Summary
            st.subheader("📌 Summary")

            st.write(summary)

            # Metrics
            st.subheader("📊 Evaluation")

            st.write("Similarity:",round(similarity_score,2),"%")

            st.write("Reduction:",round(compression_ratio,2),"%")


            # Graph
            fig = plt.figure()

            plt.bar(
                ["Similarity","Reduction"],
                [similarity_score,compression_ratio]
            )

            plt.ylim(0,100)

            st.pyplot(fig)

            # Processing Time
            st.subheader("⏱ Processing Time")

            st.write(round(processing_time,2),"seconds")


            # Download
            st.subheader("📥 Download Summary")

            st.download_button(
                "Download Summary",
                summary,
                "summary.txt"
            )


    # =========================================================
    # QUESTION ANSWERING
    # =========================================================
    if "sentences" in st.session_state:

        st.subheader("❓ Ask Questions About Document")

        question = st.text_input("Enter your question")

        if st.button("Get Answer"):

            clean_q = preprocess(question)

            sentences = st.session_state.sentences

            clean_sentences = st.session_state.clean_sentences

            vect = TfidfVectorizer()

            tfidf = vect.fit_transform(clean_sentences + [clean_q])

            scores = cosine_similarity(
                tfidf[-1],
                tfidf[:-1]
            )[0]

            idx = scores.argmax()

            st.success("Answer")

            st.write(sentences[idx])


# =========================================================
# HISTORY
# =========================================================
elif menu=="History":

    st.title("📚 Summary History")

    rows = db.execute("""
        SELECT filename,summary,similarity,compression,created_at
        FROM summaries
        WHERE username=?
        ORDER BY id DESC
    """,(st.session_state.user,)).fetchall()

    if rows:

        for f,s,sim,comp,d in rows:

            with st.expander(f"{f} ({d})"):

                st.write(s)

    else:

        st.info("No summaries found")


# =========================================================
# LOGOUT
# =========================================================
elif menu=="Logout":

    st.session_state.clear()

    st.success("Logged out successfully")

    st.rerun()