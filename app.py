# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import tempfile
# import PyPDF2
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# CORS(app)  

# MENTOR_PSYCHOMETRIC_TESTS_FOLDER = 'mentor_data/mentor_psychometric_tests'
# MENTOR_PDFS_FOLDER = 'mentor_data/mentor_pdfs'
# MENTEE_PSYCHOMETRIC_TEST_PATH = 'mentee_data/mentee_psychometric_test/mentee_psy.txt'

# def read_scores_from_file(file_path):
#     scores = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             score = line.strip().split(": ")[1]
#             scores.append(score)
#     return scores

# def psychometric_to_binary(scores):
#     binary_scores = []
#     for score in scores:
#         if score == "high":
#             binary_scores.append(1)
#         elif score == "low":
#             binary_scores.append(0)
#         else:
#             print("Invalid input in file. Scores should be either 'high' or 'low'.")
#             return None
#     return binary_scores

# def extract_antimatched_mentor_resumes(mentor_pdf_paths, mentor_binary_scores, mentee_binary_scores):
#     antimatched_mentor_resumes = []
#     for i, pdf_path in enumerate(mentor_pdf_paths):
#         if mentor_binary_scores[i] != mentee_binary_scores:
#             antimatched_mentor_resumes.append(pdf_path)
#     return antimatched_mentor_resumes

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# def preprocess_text(text):
#     if text:
#         return text.lower()  
#     return ""  

# @app.route('/recommend_mentor', methods=['POST'])
# def recommend_mentor():

#     mentee_resume_pdf_file = request.files['mentee_resume_pdf']
#     # mentee_preferences = request.form['mentee_preferences']
#     mentee_preferences = "data visualization"

#     with tempfile.TemporaryDirectory() as tempdir:
        
#         mentee_resume_pdf_path = os.path.join(tempdir, mentee_resume_pdf_file.filename)
#         mentee_resume_pdf_file.save(mentee_resume_pdf_path)

#         mentor_psychometric_test_paths = [
#             os.path.join(MENTOR_PSYCHOMETRIC_TESTS_FOLDER, fname)
#             for fname in os.listdir(MENTOR_PSYCHOMETRIC_TESTS_FOLDER)
#             if fname.endswith('.txt')
#         ]

#         mentor_binary_scores = []
#         for mentor_file_path in mentor_psychometric_test_paths:
#             mentor_scores = read_scores_from_file(mentor_file_path)
#             if mentor_scores:
#                 mentor_binary_scores.append(psychometric_to_binary(mentor_scores))

#         mentee_scores = read_scores_from_file(MENTEE_PSYCHOMETRIC_TEST_PATH)
#         if mentee_scores:
#             mentee_binary_scores = psychometric_to_binary(mentee_scores)

#         mentor_pdf_paths = [
#             os.path.join(MENTOR_PDFS_FOLDER, fname)
#             for fname in os.listdir(MENTOR_PDFS_FOLDER)
#             if fname.endswith('.pdf')
#         ]

#         antimatched_mentor_resumes = extract_antimatched_mentor_resumes(mentor_pdf_paths, mentor_binary_scores, mentee_binary_scores)

#         mentee_resume_text = extract_text_from_pdf(mentee_resume_pdf_path)
#         mentee_resume_text = preprocess_text(mentee_resume_text)
#         mentee_preferences = preprocess_text(mentee_preferences)
#         mentee_combined_text = mentee_resume_text + ' ' + mentee_preferences

#         mentor_texts = [extract_text_from_pdf(pdf_path) for pdf_path in antimatched_mentor_resumes]
#         mentor_texts = [preprocess_text(text) for text in mentor_texts]

#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform(mentor_texts + [mentee_combined_text])
#         similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
#         sorted_mentors = sorted(zip(antimatched_mentor_resumes, similarity_scores[0]), key=lambda x: x[1], reverse=True)
#         top_mentor = sorted_mentors[0]

#     return jsonify({
#         'recommended_mentor': top_mentor[0],
#         'similarity_score': top_mentor[1]
#     })

# @app.route('/mentor_pdfs/<path:filename>')
# def mentor_pdfs(filename):
#     return send_from_directory(MENTOR_PDFS_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

MENTOR_PSYCHOMETRIC_TESTS_FOLDER = 'mentor_data/mentor_psychometric_tests'
MENTOR_PDFS_FOLDER = 'mentor_data/mentor_pdfs'
MENTEE_PSYCHOMETRIC_TEST_PATH = 'mentee_data/mentee_psychometric_test/mentee_psy.txt'


def read_scores_from_file(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            score = line.strip().split(": ")[1]
            scores.append(score)
    return scores


def psychometric_to_binary(scores):
    binary_scores = []
    for score in scores:
        if score == "high":
            binary_scores.append(1)
        elif score == "low":
            binary_scores.append(0)
        else:
            print("Invalid input in file. Scores should be either 'high' or 'low'.")
            return None
    return binary_scores


def extract_antimatched_mentor_resumes(mentor_pdf_paths, mentor_binary_scores, mentee_binary_scores):
    antimatched_mentor_resumes = []
    for i, pdf_path in enumerate(mentor_pdf_paths):
        if mentor_binary_scores[i] != mentee_binary_scores:
            antimatched_mentor_resumes.append(pdf_path)
    return antimatched_mentor_resumes


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def preprocess_text(text):
    if text:
        return text.lower()
    return ""


def extract_name(resume_text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    ]

    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc = nlp(resume_text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text

    return None


def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number


def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email


def extract_skills_from_resume(resume_text):
    resume_text = resume_text.lower()
    tokens = word_tokenize(resume_text)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    token_str = " ".join(tokens)
    vectorizer = TfidfVectorizer()
    token_vectors = vectorizer.fit_transform([token_str])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = token_vectors.toarray()[0]
    top_indices = tfidf_scores.argsort()[-20:][::-1]
    skills = [feature_names[idx] for idx in top_indices]

    return skills


def extract_experience_from_resume(resume_text):
    resume_text = resume_text.replace('\n', ' ').replace('\r', ' ')

    experience_pattern = re.compile(
        r'(experience|employment history|work history|professional experience|career summary|work experience):?\s*(.*?)\s*(education|skills|projects|certifications|awards|$)',
        re.IGNORECASE | re.DOTALL
    )

    experience_section = experience_pattern.search(resume_text)

    if not experience_section:
        return []

    experience_text = experience_section.group(2)

    job_pattern = re.compile(
        r'(?P<title>.+?)\s+at\s+(?P<company>.+?)\s*(?P<dates>\([\d\s\-]+\)|\d{4}-\d{4}|\d{4}–\d{4}|\d{4}–Present|Present)?:?\s*(?P<description>.*?)(?=\s+.+?\s+at\s+.+?|\s*$)',
        re.IGNORECASE | re.DOTALL
    )

    job_matches = job_pattern.findall(experience_text)

    experience = []
    for match in job_matches:
        title, company, dates, description = match
        title = title.strip()
        company = company.strip()
        dates = dates.strip() if dates else ''
        description = " ".join(description.strip().split())

        experience.append(f"{title} at {company} {dates}: {description}")

    return '\n'.join(experience)

@app.route('/recommend_mentor', methods=['POST'])
def recommend_mentor():
    mentee_resume_pdf_file = request.files['mentee_resume_pdf']
    mentee_preferences = "data visualization"

    with tempfile.TemporaryDirectory() as tempdir:
        mentee_resume_pdf_path = os.path.join(tempdir, mentee_resume_pdf_file.filename)
        mentee_resume_pdf_file.save(mentee_resume_pdf_path)

        mentor_psychometric_test_paths = [
            os.path.join(MENTOR_PSYCHOMETRIC_TESTS_FOLDER, fname)
            for fname in os.listdir(MENTOR_PSYCHOMETRIC_TESTS_FOLDER)
            if fname.endswith('.txt')
        ]

        mentor_binary_scores = []
        for mentor_file_path in mentor_psychometric_test_paths:
            mentor_scores = read_scores_from_file(mentor_file_path)
            if mentor_scores:
                mentor_binary_scores.append(psychometric_to_binary(mentor_scores))

        mentee_scores = read_scores_from_file(MENTEE_PSYCHOMETRIC_TEST_PATH)
        if mentee_scores:
            mentee_binary_scores = psychometric_to_binary(mentee_scores)

        mentor_pdf_paths = [
            os.path.join(MENTOR_PDFS_FOLDER, fname)
            for fname in os.listdir(MENTOR_PDFS_FOLDER)
            if fname.endswith('.pdf')
        ]

        antimatched_mentor_resumes = extract_antimatched_mentor_resumes(mentor_pdf_paths, mentor_binary_scores, mentee_binary_scores)

        mentee_resume_text = extract_text_from_pdf(mentee_resume_pdf_path)
        mentee_resume_text = preprocess_text(mentee_resume_text)
        mentee_preferences = preprocess_text(mentee_preferences)
        mentee_combined_text = mentee_resume_text + ' ' + mentee_preferences

        mentor_texts = [extract_text_from_pdf(pdf_path) for pdf_path in antimatched_mentor_resumes]
        mentor_texts = [preprocess_text(text) for text in mentor_texts]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(mentor_texts + [mentee_combined_text])
        similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
        sorted_mentors = sorted(zip(antimatched_mentor_resumes, similarity_scores[0]), key=lambda x: x[1], reverse=True)
        top_mentor_path, top_similarity_score = sorted_mentors[0]

        top_mentor_text = extract_text_from_pdf(top_mentor_path)
        print(top_mentor_path)
        name = extract_name(top_mentor_text)
        number = extract_contact_number_from_resume(top_mentor_text)
        email = extract_email_from_resume(top_mentor_text)
        skills = extract_skills_from_resume(top_mentor_text)
        experience = extract_experience_from_resume(top_mentor_text)

    return jsonify({
        'name': name,
        'contact_number': number,
        'email': email,
        'skills': skills,
        'experience': experience,
        'similarity_score': top_similarity_score
    })

# @app.route('/mentor_pdfs/<path:filename>')
# def mentor_pdfs(filename):
#     return send_from_directory(MENTOR_PDFS_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
