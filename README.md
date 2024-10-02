
# Multimodal RAG Application for Document Search and Chat

This repository contains the code for a **Streamlit application** that leverages a **Multimodal Retrieval-Augmented Generation (RAG)** system for document search and chat functionalities. Users can upload documents, and the application processes them using RAG to provide insightful responses to user queries.

## Features

- **Document Upload**: Supports user-uploaded text or PDF documents for processing.
- **Multimodal Retrieval**: Handles both text-based and image-based data using Optical Character Recognition (OCR).
- **Efficient Search**: Utilizes a dense passage retrieval model and vector database for efficient semantic search.
- **Response Generation**: Employs a large language model (LLM) to generate responses based on retrieved document passages.
- **Mitigation of Hallucinations**: Several strategies are employed to ensure the accuracy of generated responses, including data cleaning and confidence scoring.
- **Human-in-the-loop**: Feedback mechanisms allow users to evaluate responses for further refinement.

## Tech Stack

- **Streamlit**: User-friendly web interface.
- **Chroma**: Vector database for embedding storage and retrieval.
- **LLM (Large Language Model)**: Used for generating responses.
- **OCR (Pytesseract)**: For extracting text from images in PDFs.

## Key Components

### 1. Text Input RAG
A text-based search interface for querying user-uploaded documents.

![Text Input Interface](https://github.com/paritosh-Shukla24/Multimodal-RAG-bot-/text_input_image.png)

### 2. Image-to-Text RAG
Processes images in PDFs using OCR and extracts meaningful text for search and response generation.

![Image to Text RAG](https://github.com/paritosh-Shukla24/Multimodal-RAG-bot-/blob/main/pdf/Screenshot%202024-06-05%20231627.png?raw=true)


### 3. Multiple PDF RAG Application
Extracts data from multiple PDFs. If text is not found, OCR is used to extract it from images, followed by embedding creation.

![Multiple PDF RAG](https://github.com/paritosh-Shukla24/Multimodal-RAG-bot-/multiple_pdf_image.png)

### 4. Web URL RAG
Supports URL-based searches for document retrieval and response generation.

![Web URL RAG](https://github.com/paritosh-Shukla24/Multimodal-RAG-bot-/web_url_image.png)

## Installation

To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/paritosh-Shukla24/Multimodal-RAG-bot-.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Multimodal-RAG-bot-
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Future Work

- Integration with a **Knowledge Graph** database for better reasoning.
- Exploration of advanced retrieval techniques like dense retrieval with transformers.
- Enhanced feedback mechanisms for user input on response quality.

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes!

## License

This project is licensed under the MIT License.
# Multimodal RAG Chat bot for Vision and text to text
