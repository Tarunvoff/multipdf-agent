import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ PDF Reading ------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ------------------ Text Chunking ------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ------------------ Vector Store ------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ------------------ Conversational Chain ------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "answer is not available in the context." 
    Don't provide incorrect answers.

    Context: {context}
    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# ------------------ User Question Handling ------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    with st.chat_message("assistant"):
        st.markdown(f"**📖 AI Reply:** \n{response['output_text']}")

# ------------------ Main App ------------------
def main():
    st.set_page_config("⭐ Multi PDF Chatbot", page_icon="📜", layout="wide")

    # Custom CSS for background and fonts
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f7fa;
            color: #333333;
        }
        .css-18e3th9 {
            padding: 2rem;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            padding: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 Multi-PDF AI Chat Agent")
    st.subheader("Interact with your documents like never before! 🎓")

    with st.sidebar:
        st.image("img/Robot.jpg", use_column_width=True)
        st.write("---")
        st.header("📎 Upload your PDFs")
        pdf_docs = st.file_uploader("Upload multiple PDF files:", accept_multiple_files=True)

        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                st.success("🎉 PDFs processed successfully! Start asking questions.")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    # Main Input Area
    user_question = st.text_input("🔎 Ask your question about the PDFs:")

    if user_question:
        with st.spinner("Thinking..."):
            user_input(user_question)

    st.markdown("""
        <hr style='margin-top: 3rem; margin-bottom: 1rem;'>
        <p style='text-align: center;'>Made  by Tarun  | Powered by Google Gemini + Langchain</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
