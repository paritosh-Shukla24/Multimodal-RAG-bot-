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


model_text = load_model("gemini-pro")
model_vision = load_model("gemini-pro-vision")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


st.title("Multimodal RAG Application")

text_input = st.text_area("Enter text:")
uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

template = """
{query}

Provide a concise answer limited to 5-6 lines based on the documents {context}.
"""
prompt = ChatPromptTemplate.from_template(template)


if uploaded_pdfs:
    pdf_text = ""
    for pdf_file in uploaded_pdfs:
        st.write(f"## PDF File: {pdf_file.name}")
        pdf_text += extract_text_from_pdf(pdf_file) + "\n"
    
    
    pdf_docs = get_text_chunks_langchain(pdf_text)

    
    if not pdf_docs:
        st.error("No text chunks created from the extracted PDF text.")
    else:
        st.write(f"Number of document chunks: {len(pdf_docs)}")

        
        try:
            pdf_vectorstore = FAISS.from_documents(pdf_docs, embedding=embeddings)
            pdf_retriever = pdf_vectorstore.as_retriever()

            
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
                "image_url": image,
            },
        ]
    )
    image_summary = model_vision.invoke([message]).content
    st.write("Image Summary:", image_summary)


    image_query = st.text_input("Enter your query for the image:", key="image_query")
    if image_query:
        query = f"{image_query}\n{image_summary}"
        result = model_text.invoke(query)
        st.markdown(result)

 
if text_input:
    docs = get_text_chunks_langchain(text_input)
    if not docs:
        st.error("No text chunks created from the input text.")
    else:
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        
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

