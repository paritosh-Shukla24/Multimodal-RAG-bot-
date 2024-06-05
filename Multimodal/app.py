import streamlit as st
import os
import requests
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to load the model
def load_model(model_name):
    if model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    return llm

# Function to get image from URL
def get_image(url, filename, extension):
    content = requests.get(url).content
    with open(f'/tmp/{filename}.{extension}', 'wb') as f:
        f.write(content)
    image = Image.open(f"/tmp/{filename}.{extension}")
    return image

# Function to split text into chunks
def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            text += page_text
        else:
            st.warning(f"No text extracted from page {page_num}. Attempting OCR.")
            # Perform OCR on the page image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            text += ocr_text
    return text

# Load models
model_text = load_model("gemini-pro")
model_vision = load_model("gemini-pro-vision")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit UI
st.title("Multimodal RAG Application")

# Input: Text, Image, and PDF
text_input = st.text_area("Enter text:")
uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

template = """
{query}

Provide a concise answer limited to 5-6 lines based on the documents {context}.
"""
prompt = ChatPromptTemplate.from_template(template)

# Process uploaded PDFs
if uploaded_pdfs:
    pdf_text = ""
    for pdf_file in uploaded_pdfs:
        st.write(f"## PDF File: {pdf_file.name}")
        pdf_text += extract_text_from_pdf(pdf_file) + "\n"
    
    # Process PDF text and create retriever
    pdf_docs = get_text_chunks_langchain(pdf_text)

    # Debugging: Check the content of pdf_docs
    if not pdf_docs:
        st.error("No text chunks created from the extracted PDF text.")
    else:
        st.write(f"Number of document chunks: {len(pdf_docs)}")

        # Generate embeddings and create vector store
        try:
            pdf_vectorstore = FAISS.from_documents(pdf_docs, embedding=embeddings)
            pdf_retriever = pdf_vectorstore.as_retriever()

            # Define the RAG chain for PDF
            pdf_rag_chain = (
                {"context": pdf_retriever, "query": str}
                | prompt
                | model_text
                | StrOutputParser()
            )
            pdf_query = st.text_input("Enter your query for the PDFs:", key="pdf_query")
            if pdf_query:
                pdf_result = pdf_rag_chain.invoke(pdf_query)
                st.markdown(pdf_result)
        except IndexError as e:
            st.error(f"Error in processing embeddings: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Process uploaded image
if uploaded_image:
    # Display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")

    # Process image and get summary
    prompt_text = "Give me a detailed summary of this image in 4-5 lines and extract useful information accordingly."
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": image,
            },
        ]
    )
    image_summary = model_vision.invoke([message]).content
    st.write("Image Summary:", image_summary)

    # Allow querying based on the image summary
    image_query = st.text_input("Enter your query for the image:", key="image_query")
    if image_query:
        query = f"{image_query}\n{image_summary}"
        result = model_text.invoke(query)
        st.markdown(result)

# Process text input
if text_input:
    docs = get_text_chunks_langchain(text_input)
    if not docs:
        st.error("No text chunks created from the input text.")
    else:
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Define the RAG chain
        rag_chain = (
            {"context": retriever, "query": str}
            | prompt
            | model_text
            | StrOutputParser()
        )
        query = st.text_input("Enter your query for the text input:", key="text_query")
        if query:
            result = rag_chain.invoke(query)
            st.markdown(result)
