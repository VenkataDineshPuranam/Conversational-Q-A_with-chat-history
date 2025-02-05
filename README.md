# Conversational Q&A Chatbot with Chat history

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using Groq's LLM API and OpenAI's embeddings. The application maintains chat history and provides conversational context for follow-up questions.

## Features

- PDF document upload and processing
- Conversational Q&A with context memory
- Document chunking and vector storage using FAISS
- Response time tracking
- Source document reference
- Chat history display

## Prerequisites

Before running the application, make sure you have:
- Python 3.7+
- Groq API key
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install streamlit langchain-groq langchain-core langchain faiss-cpu PyPDF2 openai uuid
```

## Environment Variables

The application requires two API keys:
- `GROQ_API_KEY`: Your Groq API key
- `OPENAI_API_KEY`: Your OpenAI API key

These can be entered directly in the application's interface.

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open the provided URL in your web browser

3. Enter your Groq and OpenAI API keys

4. Upload a PDF document

5. Ask questions about the content of your PDF

## Features in Detail

### PDF Processing
- Splits documents into manageable chunks
- Creates embeddings using OpenAI's embedding model
- Stores vectors in a FAISS index for efficient retrieval

### Question Answering
- Uses Groq's gemma2-9b-it model for generating responses
- Maintains conversation history for context
- Provides source documents for verification
- Displays response generation time

### User Interface
- Clean and intuitive Streamlit interface
- Chat history display
- Option to view source documents
- Real-time response time tracking

## Contributing

Feel free to submit issues and enhancement requests!


## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [Langchain](https://python.langchain.com/) for LLM integration
- Powered by [Groq](https://groq.com/) and [OpenAI](https://openai.com/)
