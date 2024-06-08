import streamlit as st
import os
import requests
from PIL import Image
import fitz  
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
from langchain_community.document_loaders import WebBaseLoader

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_model(model_name):
    if model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    return llm

def get_image(url, filename, extension):
    content = requests.get(url).content
    with open(f'/tmp/{filename}.{extension}', 'wb') as f:
        f.write(content)
    image = Image.open(f"/tmp/{filename}.{extension}")
    return image

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            text += page_text
        else:
            st.warning(f"No text extracted from page {page_num}. Attempting OCR.")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            text += ocr_text
    return text

def extract_text_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

model_text = load_model("gemini-pro")
model_vision = load_model("gemini-pro-vision")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

st.title("Multimodal RAG Application")

text_input = st.text_area("Enter text:")
uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
url_input = st.text_input("Enter URL:")

template = """
{query}

Provide a concise answer limited to 5-6 lines based on the documents {context}.
"""
prompt = ChatPromptTemplate.from_template(template)

combined_docs = []

if uploaded_pdfs:
    pdf_text = ""
    for pdf_file in uploaded_pdfs:
        st.write(f"## PDF File: {pdf_file.name}")
        pdf_text += extract_text_from_pdf(pdf_file) + "\n"
    pdf_docs = get_text_chunks_langchain(pdf_text)
    combined_docs.extend(pdf_docs)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")
    prompt_text = "Give me a detailed summary of this image in 4-5 lines and extract useful information accordingly."
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": uploaded_image,
            },
        ]
    )
    image_summary = model_vision.invoke([message]).content
    st.write("Image Summary:", image_summary)
    combined_docs.append(Document(page_content=image_summary))

if text_input:
    text_docs = get_text_chunks_langchain(text_input)
    combined_docs.extend(text_docs)

if url_input:
    try:
        url_docs = extract_text_from_url(url_input)
        combined_docs.extend(url_docs)
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")

if combined_docs:
    try:
        vectorstore = FAISS.from_documents(combined_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"context": retriever, "query": str}
            | prompt
            | model_text
            | StrOutputParser()
        )
        
        query = st.text_input("Enter your query for the combined documents:", key="combined_query")
        if query:
            result = rag_chain.invoke(query)
            st.markdown(result)
    except IndexError as e:
        st.error(f"Error in processing embeddings: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")



