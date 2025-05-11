import streamlit as st
import spacy
from transformers import pipeline
import PyPDF2
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker

nlp = spacy.load("en_core_web_sm")
semantic = pipeline('feature-extraction')

engine = db.create_engine('sqlite:///candidates.db')
metadata = db.MetaData()

candidates = db.Table('candidates', metadata,
                      db.Column('id', db.Integer(), primary_key=True),
                      db.Column('name', db.String(255)),
                      db.Column('skills', db.Text()),
                      db.Column('score', db.Float()))
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def analyze_resume(text, job_desc):
    resume_vec = semantic(text)[0][0]
    job_vec = semantic(job_desc)[0][0]
    similarity = sum(r*j for r, j in zip(resume_vec, job_vec))
    doc = nlp(text)
    skills = [chunk.text for chunk in doc.noun_chunks]
    return skills, similarity

st.title('Automated Resume Screening')
job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)

if st.button('Analyze'):
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        skills, score = analyze_resume(text, job_desc)
        insert_stmt = candidates.insert().values(name=uploaded_file.name, skills=", ".join(skills), score=score)
        session.execute(insert_stmt)
        session.commit()
    st.success("Resumes analyzed and stored!")

results = session.query(candidates).order_by(candidates.c.score.desc()).all()
st.write("## Ranked Candidates")
for res in results:
    st.write(f"### {res.name}")
    st.write(f"Score: {res.score}")
    st.write(f"Skills: {res.skills}")
