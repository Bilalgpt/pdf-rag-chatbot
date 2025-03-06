# PDF RAG Chatbot with Conversation History

A Streamlit application that allows users to upload PDFs and have interactive conversations about their content using Retrieval Augmented Generation (RAG).

## Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Conversation Memory**: Maintains chat history for contextual follow-up questions
- **RAG Implementation**: Uses vector embeddings and semantic search for relevant context retrieval
- **Session Management**: Supports multiple chat sessions with unique identifiers

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Groq API key
- HuggingFace API token (for embeddings)

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install streamlit langchain langchain-chroma langchain-groq langchain-huggingface python-dotenv
   ```

3. **Create a `.env` file with your API keys**
   ```
   HF_TOKEN=your_huggingface_token
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter your Groq API key in the password field
2. Provide a session ID (or use the default)
3. Upload one or more PDF files
4. Type your questions in the input box
5. View conversation history by expanding the chat history section

## How It Works

1. **Document Processing Pipeline**:
   - PDFs are uploaded and temporarily stored
   - Documents are split into chunks and embedded using HuggingFace embeddings
   - Embeddings are stored in a Chroma vector database

2. **Question Answering Chain**:
   - User questions are reformulated considering conversation history
   - Relevant document chunks are retrieved from the vector store
   - LLM generates concise answers based on the retrieved context

3. **Conversation Management**:
   - Chat history is maintained for each unique session ID
   - Questions are contextualized based on previous interactions
   - History can be viewed and referenced during the conversation

## Customization

- Change the embedding model by modifying the `model_name` parameter
- Adjust chunk size and overlap in the `RecursiveCharacterTextSplitter`
- Modify the LLM model by changing the `model_name` in the `ChatGroq` initialization
- Edit system prompts to change the tone or verbosity of responses

## Limitations

- Requires internet connection for API access
- Limited by the context window of the underlying LLM
- Processing large PDFs may take some time
- Temporary files are created during processing