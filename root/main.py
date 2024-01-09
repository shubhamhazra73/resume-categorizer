from flask import Flask, redirect, session, url_for, flash, render_template, request
import os
import spacy, re
from pdfminer.high_level import extract_text
import spacy
from spacy.matcher import Matcher
from werkzeug.utils import secure_filename
from gensim.models import Word2Vec
import numpy as np
import pickle
###############################################
w2v = Word2Vec.load("word2vec_model.bin", "rb")
################################################
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
UPLOAD_FOLDER = 'all_uploads'
app.secret_key = 'Nothing'
def cleanResume(resumeText):

    resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText)
    resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
    return resumeText.lower()
def allowed_file(filename) :
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}
################################################################
@app.route('/')
def index() :
    return render_template('index.html', message = 'NO FILE')
###################################################################
def extract_name(resume_text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    # Define name patterns
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name, Middle name, and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]  # First name, Middle name, Middle name, and Last name
        # Add more patterns as needed
    ]

    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc = nlp(resume_text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text

    return ''
def extract_education(text):
    education = []
    education_keywords = [
    'Igdtuw','DCE','DTU','Bsc', 'Bachelor of Science', 'B. Pharmacy', 'Bachelor of Pharmacy',
    'B Pharmacy', 'Master of Science', 'M. Pharmacy', 'Master of Pharmacy', 'Ph.D', 'Doctor of Philosophy',
    'Bachelor', 'Bachelor Degree', 'Master', 'Master Degree', 'B.Tech', 'Bachelor of Technology', 'MBA',
    'Master of Business Administration', 'MCA', 'Master of Computer Applications', 'B.A.', 'Bachelor of Arts',
    'M.A.', 'Master of Arts', 'B.Com', 'Bachelor of Commerce', 'M.Com', 'Master of Commerce', 'B.E.',
    'Bachelor of Engineering','M.E.', 'Master of Engineering','BBA', 'Bachelor of Business Administration',
    'BCA', 'Bachelor of Computer Applications',  'Diploma', 'High School', 'High School Diploma',
    'Secondary Education', 'Secondary School Certificate', 'Higher Secondary', 'Higher Secondary Certificate',
    'Undergraduate', 'Undergraduate Degree', 'Postgraduate', 'Postgraduate Degree', 'Doctorate', 'Doctorate Degree',
    'Associate Degree', 'Associate of Arts/Science', 'Certificate', 'Specialization', 'Specialization',
    'Training', 'Training Certification','SSC', 'Secondary School Certificate', 'HSC', 'Higher Secondary Certificate',
    'CBSE', 'Central Board of Secondary Education', 'ICSE', 'Indian Certificate of Secondary Education', 'A-levels',
    'Advanced Level Certificate', 'O-levels', 'Ordinary Level Certificate', 'GED', 'General Educational Development',
    'Delhi University', 'Indian Institute of Technology Delhi', 'Jamia Millia Islamia', 'Jawaharlal Nehru University',
    'Amity University', 'Indraprastha Institute of Information Technology Delhi', 'Delhi Technological University',
    'Netaji Subhas University of Technology', 'Guru Gobind Singh Indraprastha University',   'Shri Ram College of Commerce',
    'National Institute of Fashion Technology Delhi', 'Faculty of Management Studies', 'All India Institute of Medical Sciences Delhi',
    'Indian Statistical Institute Delhi', 'Delhi College of Engineering', 'Indira Gandhi Delhi Technical University for Women']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())


    return education
def extract_skills(text):
    skills = []
    skills_list = [
    'Python', 'Data Analysis', 'Machine Learning', 'Communication','Project Management', 'Deep Learning', 'SQL', 'Tableau',
    'JavaScript','HTML', 'CSS', 'React', 'Angular', 'Node.js', 'Git', 'Docker', 'AWS','Azure', 'Statistics', 'Natural Language Processing',
    'Big Data', 'Cloud Computing', 'Cybersecurity', 'UX/UI Design','Java','C++','C#','PHP','Ruby','Swift','Objective-C','Kotlin',
    'Flutter','Vue.js','TypeScript','GraphQL','RESTful API','Microservices','DevOps','CI/CD','Blockchain','IoT (Internet of Things)','AR/VR Development','Game Development','Mobile App Development',
    'Web Development','Backend Development','Frontend Development','Database Management','Network Administration','Linux Administration','Windows Administration','Digital Marketing','SEO' ,'Search Engine Optimization',
    'SEM (Search Engine Marketing)','Frontend', 'Backend', 'Bootstrap', 'Figma','Web Developer', 'API', 'Nodejs', 'Node', 'Flask','ML', 'DL', 'DSA', 'Data Structures and Algorithms', 'OOPS', 'DBMS', 'Algorithms', 'Dynamic Programming', 'Programmer', 'Rust',
    'Content Writing','Graphic Design','Motion Graphics','Video Editing','Photography','Animation','3D Modeling','User Research',
    'Product Design','Product Management','Business Analysis','Salesforce','ERP Systems','CRM Systems','E-commerce Platforms','Supply Chain Management',
    'Financial Analysis','Investment Banking','Risk Management','Actuarial Intern','Quantitative Analysis','Forex Trading',
    'Cryptocurrency','Machine Vision','Robotics','Control Systems','Systems Integration','IoT Security','Penetration Testing',
    'Ethical Hacking','Malware Analysis','Threat Intelligence','Cloud Security','Data Privacy','Regulatory Compliance','User Experience Testing',
    'Accessibility Testing','Quality Assurance (QA)','Automated Testing','Performance Testing','API Testing','Mobile Testing',
    'Web Security','Network Security','Endpoint Security']

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)
def listToString(s):

    str1 = ""

    for ele in s:
        str1 += ele

    return str1
def extract_contact_number(text):

    matcher = Matcher(nlp.vocab)
    indian_phone_number_pattern = [{"TEXT": {"REGEX": "^[6-9][0-9]{9}$"}}]
    matcher.add("INDIAN_PHONE_NUMBER", [indian_phone_number_pattern])
    doc = nlp(text)
    matches = matcher(doc)

    ans = ""
    for match_id, start, end in matches:
        span = doc[start:end]
        ans += span.text
    return ans

    return contact_number
def extract_email(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email
##########################################################################
@app.route('/upload', methods = ['GET', 'POST'])
def upload() :
    if(request.method == 'POST') :
        file = request.files['file']


        if file.filename == '' :
            return redirect(url_for('index'))

        if file and allowed_file(file.filename) :
            # temporary storage
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            # Extract text from the PDF file
            resume_text = extract_text_from_pdf(file_path)

            name = str(extract_name(resume_text))
            education = (extract_education(resume_text))
            skills = (extract_skills(resume_text))
            contact_number = str(extract_contact_number(resume_text))
            email = str(extract_email(resume_text))
            def vectorize_text(text):
                tokens = [token.text for token in nlp(text)]
                vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
                if vectors:
                    return np.mean(vectors, axis=0)
                else:
                    return np.zeros(100)

            resume_vector = vectorize_text( cleanResume(listToString(skills) + listToString(education) ) )


            info = " "

            with open('resume_model.pkl', 'rb') as f :
                model = pickle.load(f, encoding = 'latin1', errors = 'ignore')
                info = model.predict(resume_vector.reshape(1, 100))[0]

            if(email == 'None' ) :
                email = ''

            return render_template('res.html',
                                   name = name,
                                   education = education,
                                   skills = skills,
                                   contact_number = contact_number,
                                   email = email,
                                   info = info)

        else :
            return render_template('invalid.html')

    return redirect(url_for('index'))


if __name__ == '__main__' :
    app.run(debug = False, host = '0.0.0.0')

