import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
import PyPDF2
from docx import Document

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Email Generator")
    uploaded_file = st.file_uploader("Upload a job description file (.pdf or .docx):", type=["pdf", "docx"])
    url_input = st.text_input("Or Enter a Job Description URL:")
    tone = st.selectbox("Select Email Tone:", options=["Formal", "Casual"], index=0)
    submit_button = st.button("Generate Email")

    if submit_button:
        try:
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    job_description = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    job_description = extract_text_from_docx(uploaded_file)
                else:
                    st.error("Unsupported file format.")
                    return

                data = clean_text(job_description)

            elif url_input:
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)

            else:
                st.error("Please upload a file or provide a URL.")
                return
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            for job in jobs:
                skills = job.get('skills', [])
                match_percentage, matched_skills = portfolio.calculate_matching_percentage(skills)
                st.write(f"Matching Skills: {len(matched_skills)} / {len(skills)}")
                st.write(f"Matching Percentage: {match_percentage:.2f}%")
                st.write(f"Matched Skills: {', '.join(matched_skills)}")

                if match_percentage < 50:
                    st.warning("The match percentage is low, you might want to reconsider sending the email.")
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links, tone=tone.lower())
                st.subheader(f"Generated Email for: {job.get('role', 'N/A')}")
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Email Generator")
    create_streamlit_app(chain, portfolio, clean_text)
