import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings model
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("PDF Chatbot with Conversation History")
st.write("Upload PDFs and chat with their content")

# Get API key from user
api_key = st.text_input("Enter your Groq API key:", type="password")

# Initialize session management
session_id = st.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}

# Get session history helper function
def get_session_history(session_id):
    """Return chat history for the given session ID"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# File upload handling
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if api_key and uploaded_files:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    
    # Process uploaded PDFs
    documents = []
    for uploaded_file in uploaded_files:
        # Save temporary file
        temp_pdf = f"./temp_{uploaded_file.name}"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        
        # Load and process PDF
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)
        
        # Clean up temp file
        os.remove(temp_pdf)
    
    # Create vector store from documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Context-aware question reformulation prompt
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the user's question to be standalone, considering the chat history context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    
    # Answer generation prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question concisely using the context provided. Maximum three sentences. {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Create conversational chain with history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Process user input
    user_input = st.text_input("Your question:")
    if user_input:
        # Get response with history
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Display response and history
        st.write("Assistant:", response['answer'])
        
        # Show chat history in collapsible section
        with st.expander("View Chat History"):
            history = get_session_history(session_id)
            for i, msg in enumerate(history.messages):
                role = "User" if i % 2 == 0 else "Assistant"
                st.write(f"{role}: {msg.content}")
                
elif not api_key:
    st.warning("Please enter your Groq API key")
elif not uploaded_files:
    st.info("Please upload at least one PDF file")
