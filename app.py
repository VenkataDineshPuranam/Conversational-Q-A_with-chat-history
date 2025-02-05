import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import uuid
import time

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  
    )
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# App title
st.title("Conversational Q&A with Chat History")

# API key input
groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if groq_api_key and openai_api_key:
    # Set API keys
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file:
        # Process PDF and create vector store
        def process_pdf():
            try:
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load and process the PDF
                loader = PyPDFDirectoryLoader(".")
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_documents(splits, embeddings)
                
                return vector_store
            
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return None
            finally:
                # Clean up
                if os.path.exists("temp.pdf"):
                    os.remove("temp.pdf")
        
        # Create or get vector store
        if st.session_state.vector_store is None:
            with st.spinner('Processing PDF...'):
                st.session_state.vector_store = process_pdf()
        
        if st.session_state.vector_store:
            # Create chat model
            llm = ChatGroq(
                temperature=0.7,
                model="gemma2-9b-it",
                groq_api_key=groq_api_key
            )
            
            # Create conversation chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(),
                memory=st.session_state.memory,
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={'prompt': ChatPromptTemplate.from_template("""
                    Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    
                    {context}
                    
                    Question: {question}
                    Helpful Answer:""")}
            )
            
            # Question input
            question = st.text_input("Ask a question about your PDF:")
            
            if question:
                with st.spinner('Generating response...'):
                    start_time = time.time()
                    
                    try:
                        # Get response
                        response = qa_chain({"question": question})
                        
                        # Display response
                        st.write("Answer:", response['answer'])
                        
                        # Optionally display source documents
                        if st.checkbox("Show source documents"):
                            st.write("Sources:")
                            for doc in response['source_documents']:
                                st.write(doc.page_content)
                        
                        # Display response time
                        end_time = time.time()
                        st.write(f"Response time: {end_time - start_time:.2f} seconds")
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                
                # Display chat history
                st.write("\nChat History:")
                for message in st.session_state.memory.chat_memory.messages:
                    role = "User" if message.type == 'human' else "Assistant"
                    st.write(f"{role}: {message.content}")
