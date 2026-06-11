import re
import pdfplumber
import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined skills list
SKILLS = [
    "Python", "Java", "C++", "C", "JavaScript", "SQL",
    "Machine Learning", "Deep Learning", "Data Science",
    "TensorFlow", "PyTorch", "Power BI", "Tableau",
    "Excel", "AWS", "Azure", "Docker", "Kubernetes",
    "HTML", "CSS", "React", "Node.js", "Git"
]

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract email and phone number
def extract_contact_info(text):
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    phone_pattern = r'(\+?\d[\d\s\-]{8,}\d)'

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "Email": emails[0] if emails else None,
        "Phone": phones[0] if phones else None
    }

# Extract candidate name
def extract_name(text):
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    lines = text.split("\n")
    if lines:
        return lines[0].strip()

    return None

# Extract skills
def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    return list(set(found_skills))

# Extract education keywords
def extract_education(text):
    education_keywords = [
        "B.Tech", "B.E", "M.Tech", "M.E",
        "B.Sc", "M.Sc", "MBA", "PhD",
        "Bachelor", "Master"
    ]

    education = []

    for edu in education_keywords:
        if edu.lower() in text.lower():
            education.append(edu)

    return list(set(education))

# Main parser
def parse_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    contact = extract_contact_info(text)

    data = {
        "Name": extract_name(text),
        "Email": contact["Email"],
        "Phone": contact["Phone"],
        "Skills": ", ".join(extract_skills(text)),
        "Education": ", ".join(extract_education(text))
    }

    return data

# Example Usage
if __name__ == "__main__":
    resume_path = "resume.pdf"   # Replace with your resume file

    parsed_data = parse_resume(resume_path)

    print("\n===== Resume Parsing Result =====")
    for key, value in parsed_data.items():
        print(f"{key}: {value}")

    # Save to CSV
    df = pd.DataFrame([parsed_data])
    df.to_csv("parsed_resume.csv", index=False)

    print("\nData saved to parsed_resume.csv")  