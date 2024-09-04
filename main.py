
from sqlalchemy import create_engine, Column, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize database
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Check if the users table exists
inspector = inspect(engine)
if not inspector.has_table('users'):
    users_table = Table(
        "users", metadata,
        Column("username", String, primary_key=True),
        Column("password", String)
    )
    metadata.create_all(engine)

# Bind users_table to metadata after checking for existence
users_table = Table("users", metadata, autoload_with=engine)

Session = sessionmaker(bind=engine)
session = Session()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Set up Streamlit page
st.set_page_config(page_title="JEE & NEET Prep Assistant", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7b6; /* light yellow */
        color: #333333;
    }
    .title {
        color: #0FEAC7;
        text-align: center;
        margin-bottom: 20px;
    }
    .navbar {
        display: flex;
        justify-content: space-around;
        background-color: #0feac7;
        padding: 10px;
    }
    .navbar a {
        color: #ffcc00;
        text-decoration: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .navbar a:hover {
        background-color: #ffcc00;
        color: #003366;
    }
    .container {
        margin: 20px;
    }
    .stButton>button {
        background-color: #003366; /* blue */
        color: white;
    }
    .stButton>button:hover {
        background-color: #002244; /* darker blue */
    }
    .stDownloadButton>button {
        background-color: #003366; /* blue */
        color: white;
    }
    .stDownloadButton>button:hover {
        background-color: #002244; /* darker blue */
    }
    </style>
    """, unsafe_allow_html=True
)


st.markdown('<h1 class="title">JEE & NEET Prep Assistant 🧠📚</h1>', unsafe_allow_html=True)

# Configure Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.0
)

# Set up vector database
persist_directory = "docs"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load or create vector database
if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting your bot. This might take a while'):
        try:
            pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
            pdf_documents = pdf_loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
            pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
            pdfs = splitter.split_text(pdf_context)

            vectordb = Chroma.from_texts(pdfs, embeddings, persist_directory=persist_directory)
            vectordb.persist()
            st.success("Vector database created successfully!")
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            st.stop()
else:
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# Functions for the application
def retrieve_and_generate(query):
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.invoke(full_prompt)
    return response.content


def get_response(prompt, subject):
    formatted_prompt = f"As a JEE and NEET preparation assistant, answer the following {subject} question: {prompt}"
    try:
        response = retrieve_and_generate(formatted_prompt)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def suggest_related_topics(subject, prompt):
    suggestion_prompt = f"Based on the {subject} question '{prompt}', suggest 3-5 related topics for further study."
    try:
        suggestions = retrieve_and_generate(suggestion_prompt)
        return suggestions
    except Exception as e:
        return f"An error occurred while generating suggestions: {str(e)}"


def generate_mcqs(subject):
    mcq_prompt = f"""Generate 5 multiple-choice questions suitable for JEE or NEET preparation on the subject of {subject}. 
    Follow these guidelines strictly:
    1. Questions should be at the difficulty level of actual JEE and NEET exams.
    2. Cover different topics within {subject} that are relevant to JEE/NEET syllabus.
    3. Include calculation-based questions, conceptual questions, and application-based questions.
    4. Each question should have 4 options (A, B, C, D) with only one correct answer.
    5. Ensure that wrong options are plausible and relate to common misconceptions.

    Format each question EXACTLY as follows, with each component on a new line:
    Q1. [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Correct option letter]
    Explanation: [Brief explanation of the correct answer]

    [Leave two blank lines between questions]
    """
    try:
        mcqs = retrieve_and_generate(mcq_prompt)
        # Ensure proper line breaks
        formatted_mcqs = mcqs.replace(". ", ".\n").replace(") ", ")\n").replace("Correct Answer:",
                                                                                "\nCorrect Answer:").replace(
            "Explanation:", "\nExplanation:")
        return formatted_mcqs
    except Exception as e:
        return f"An error occurred while generating MCQs: {str(e)}"


def generate_flashcards(subject):
    flashcard_prompt = f"Generate 10 flashcards for revision on the subject of {subject}. Each flashcard should have a question and an answer."
    try:
        flashcards = retrieve_and_generate(flashcard_prompt)
        # Remove extra '' after flashcard numbers and adjust formatting
        formatted_flashcards = flashcards.replace("**Flashcard", "Flashcard").replace("**Question:**",
                                                                                     "Question:").replace(
            "**Answer:**", "Answer:")
        # Clean up any remaining '' and ensure correct formatting
        formatted_flashcards = formatted_flashcards.replace("**", "").strip()
        return formatted_flashcards
    except Exception as e:
        return f"An error occurred while generating flashcards: {str(e)}"


def get_concept_explanation(concept, subject):
    explanation_prompt = f"Provide a detailed explanation of the concept '{concept}' in {subject}."
    try:
        explanation = retrieve_and_generate(explanation_prompt)
        return explanation
    except Exception as e:
        return f"An error occurred while generating explanation: {str(e)}"


def generate_practice_problems(subject):
    practice_prompt = f"""Generate 5 practice problems suitable for JEE or NEET preparation on the subject of {subject}. 
        Follow these guidelines:
        1. Provide a clear problem statement.
        2. Include detailed solutions with step-by-step explanations.

        Example format:

        Problem 1 (Easy)

        Question: [Problem statement]

        Solution:

    [Solution steps]


        Problem 2 (Medium)

        Question: [Problem statement]

        Solution:

    [Solution steps]


        And so on...
        """

    try:
        problems = retrieve_and_generate(practice_prompt)
        # Format the output to remove stars around problems, questions, and solutions
        formatted_problems = problems.replace("**Problem", "Problem").replace("**Question:**", "Question:").replace(
            "**Solution:**", "Solution:")
        # Clean up any remaining '' and ensure correct formatting
        formatted_problems = formatted_problems.replace("**", "").strip()
        return formatted_problems
    except Exception as e:
        return f"An error occurred while generating practice problems: {str(e)}"


def generate_study_plan(subject, duration):
    study_plan_prompt = f"Create a study plan for {subject} to cover in {duration} weeks, including daily tasks and weekly goals."
    try:
        study_plan = retrieve_and_generate(study_plan_prompt)
        return study_plan
    except Exception as e:
        return f"An error occurred while generating study plan: {str(e)}"


def get_exam_tips(subject):
    tips_prompt = f"Provide tips and strategies for preparing for exams in {subject}."
    try:
        tips = retrieve_and_generate(tips_prompt)
        return tips
    except Exception as e:
        return f"An error occurred while generating tips: {str(e)}"


# Functions for authentication
def login(username, password):
    user = session.query(users_table).filter_by(username=username, password=password).first()
    if user:
        st.session_state.authenticated = True
        st.session_state.page = "Home"
    else:
        st.error("Invalid username or password")


def logout():
    st.session_state.authenticated = False
    st.session_state.page = "Login"


def signup(new_username, new_password):
    if session.query(users_table).filter_by(username=new_username).first():
        st.error("Username already exists. Please choose a different username.")
    else:
        new_user = users_table.insert().values(username=new_username, password=new_password)
        session.execute(new_user)
        session.commit()
        st.success("Account created successfully! Please log in.")


# Render main content based on authentication status
if st.session_state.authenticated:

    pages = ["Home", "Chat", "Generate MCQs", "Generate Flashcards", "Concept Explanations", "Practice Problems", "Study Plans",
                 "Exam Tips"]
    page = st.radio("", pages, index=pages.index(st.session_state.page),  horizontal=True)
    st.session_state.page = page
    # Main content
    st.markdown('<div class="container">', unsafe_allow_html=True)




    if st.session_state.page == "Home":
        # Load the image
        image = Image.open("Designer.jpeg")

        # Resize the image
        new_width = 1100
        new_height = 500
        resized_image = image.resize((new_width, new_height))

        # Display the resized image in Streamlit
        st.image(resized_image)


        st.write("""
            Welcome to JEE & NEET Prep Assistant! This platform is meticulously designed to support JEE and NEET aspirants throughout their preparation journey. With our interactive chatbot, you can ask any questions related to the JEE or NEET syllabus and receive precise, subject-specific answers. Whether you need clarification on complex concepts in Physics, Chemistry, Biology, or Mathematics, our assistant is here to help.

        Additionally, you can generate customized Multiple Choice Questions (MCQs) for practice by simply inputting a topic or subject name. The chatbot will provide MCQs with four options, along with instant feedback and correct answers. You can even select the difficulty level of the questions to match your preparation stage. For quick revision, the platform offers the creation of topic-wise flashcards, which you can interactively flip through with questions on one side and answers on the other. These flashcards are fully customizable to suit your study preferences.

        Moreover, the chatbot can generate in-depth concept explanations to help you understand and master various topics. Beyond practice questions and explanations, the platform can also generate practice problems with solutions, aiding in effective problem-solving practice. For a more structured approach, you can develop a personalized study plan tailored to your specific needs and timelines. Additionally, the platform offers valuable exam tips and strategies to enhance your preparation and boost your confidence.

        Use the navigation menu on the left to explore these features and more. Happy studying!
        """)
        st.button("Logout", on_click=logout)

    elif st.session_state.page == "Chat":


        st.markdown('<h2 id="chat">Chat with your Assistant</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        for msg in st.session_state.history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        prompt = st.chat_input("Ask your question:")
        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(prompt, subject)
                    st.markdown(response)
                    st.session_state.history.append({"role": "assistant", "content": response})


    elif st.session_state.page == "Generate MCQs":
        st.markdown('<h2 id="generate-mcqs">Generate MCQs</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        if st.button("Generate MCQs"):
            with st.spinner('🧠 Generating MCQs...'):
                mcqs = generate_mcqs(subject)
                st.markdown("### Generated MCQs")
                st.code(mcqs)

                st.download_button(
                    label="Download MCQs",
                    data=mcqs,
                    file_name=f"{subject}_MCQs.txt",
                    mime="text/plain"
                )

    elif st.session_state.page == "Generate Flashcards":
        st.markdown('<h2 id="flashcards">Flashcards</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        if st.button("Generate Flashcards"):
            with st.spinner('🧠 Generating flashcards...'):
                flashcards = generate_flashcards(subject)
                st.markdown("### Generated Flashcards")
                st.code(flashcards)

                st.download_button(
                    label="Download Flashcards",
                    data=flashcards,
                    file_name=f"{subject}_Flashcards.txt",
                    mime="text/plain"
                )
    elif st.session_state.page == "Concept Explanations":
        st.markdown('<h2 id="concept-explanations">Concept Explanations</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        concept = st.text_input("Enter the concept you want explained:")
        if st.button("Get Explanation"):
            with st.spinner('💡 Thinking...'):
                explanation = get_concept_explanation(concept, subject)
                st.markdown("### Concept Explanation")
                st.write(explanation)



    elif st.session_state.page == "Practice Problems":
        st.markdown('<h2 id="practice-problems">Practice Problems</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        if st.button("Generate Practice Problems"):
            with st.spinner('🧠 Generating practice problems...'):
                problems = generate_practice_problems(subject)
                st.markdown("### Generated Practice Problems")
                st.code(problems)

                st.download_button(
                    label="Download Practice Problems",
                    data=problems,
                    file_name=f"{subject}_Practice_Problems.txt",
                    mime="text/plain"
                )


    elif st.session_state.page == "Study Plans":
        st.markdown('<h2 id="study-plans">Study Plans</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        duration = st.number_input("Enter the duration in weeks:", min_value=1, max_value=52, step=1)
        if st.button("Generate Study Plan"):
            with st.spinner('📅 Generating study plan...'):
                study_plan = generate_study_plan(subject, duration)
                st.markdown("### Generated Study Plan")
                st.write(study_plan)

                st.download_button(
                    label="Download Study Plan",
                    data=study_plan,
                    file_name=f"{subject}_Study_Plan.txt",
                    mime="text/plain"
                )

    elif st.session_state.page == "Exam Tips":
        st.markdown('<h2 id="exam-tips">Exam Tips</h2>', unsafe_allow_html=True)
        subject = st.selectbox("Choose subject:", ["Physics", "Chemistry", "Biology", "Mathematics"])
        if st.button("Get Tips"):
            with st.spinner('💡 Thinking...'):
                tips = get_exam_tips(subject)
                st.markdown("### Exam Tips and Strategies")
                st.write(tips)

                st.download_button(
                    label="Download Tips",
                    data=tips,
                    file_name=f"{subject}_Exam_Tips.txt",
                    mime="text/plain"
                )



else:
    st.sidebar.header("Account")
    auth_mode = st.sidebar.selectbox("Select Mode", ["Login", "Sign Up"])

    if auth_mode == "Login":
        st.header("Login")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        st.button("Login", on_click=login, args=(username, password))
    else:
        st.header("Sign Up")
        new_username = st.text_input("New Username:")
        new_password = st.text_input("New Password:", type="password")
        st.button("Sign Up", on_click=signup, args=(new_username, new_password))
