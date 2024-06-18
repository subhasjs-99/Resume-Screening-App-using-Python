import streamlit as st
import pickle
import re
import nltk
import PyPDF2
from io import BytesIO
from langdetect import detect
from googletrans import Translator

# In this context provides a convenient
# and efficient way to process PDF files with PyPDF2 without the need for temporary files on disk.
# It's a common approach in scenarios where file I/O can be minimized or avoided.


nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# # web app

st.title("Resume Screening App")

uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])


def is_pdf(file):
    """
    Checking if the uploaded file is a PDF.
    """
    if file.type == 'application/pdf':
        return True
    else:
        return False


def extract_text_from_pdf(uploaded_file):
    """
        Extract text from a PDF file.
    """
    text = ""
    with BytesIO(uploaded_file.read()) as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text




count=0

from langdetect import detect

def realvsFake(resume_text):
    try:
        # Detect the language of the text
        language = detect(resume_text)

        # Define language codes for the specified languages
        language_codes = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            # Add more language codes as needed
        }

        # Check if the detected language is supported
        if language in language_codes:
            # Define keywords for each supported language
            keywords = {
                'en': ["skills", "education", "profile", "curriculum vitae"],
                'hi': ["कौशल", "शिक्षा", "प्रोफ़ाइल", "जीवन परिचय"],
                'es': ["habilidades", "educación", "perfil", "currículum vitae"],
                'la': ["scientia", "educatio", "profilium", "curriculum vitae"],
                'ru': ["навыки", "образование", "профиль", "резюме"],
                # Add more keywords for other languages as needed
            }

            # Get keywords for the detected language
            lang_keywords = keywords.get(language, [])

            # Check if any of the language-specific keywords are present in the resume text
            for keyword in lang_keywords:
                if keyword.lower() in resume_text.lower():
                    return True  # Resume is considered real
            return False  # None of the keywords found, resume might be fake
        else:
            return False  # Unsupported language
    except Exception as e:
        print("Error:", e)
        return False  # Error occurred, unable to determine

def main():


    if uploaded_file is not None:

        # If it's a PDF then Calls the above fn to convert pdf to text

        if is_pdf(uploaded_file):
            resume_text = extract_text_from_pdf(uploaded_file)

        # Otherwise treat it as text
        else:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

        if realvsFake(resume_text):
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            st.write(prediction_id)


            # Map category ID to category name

            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            st.write("Predicted Category:", category_name)
        else:
            st.write("Not a CV")

# python main
if __name__ == "__main__":
    main()



