import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configuration moved to .env file for better security and flexibility
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "/vectorstore")
PDF_PATH = os.getenv("PDF_PATH", "/data/EMPLOYEE_AGREEMENT.pdf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")


@st.cache_resource
def get_vectorstore(vectorstore_path: str):
    """
    Load the vector store from disk. Cached to avoid reloading on every run.
    """
    if Path(vectorstore_path).exists():
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def init(pdf_path: str = PDF_PATH, vectorstore_path: str = VECTORSTORE_PATH) -> None:
    """
    Initialize the vector store by loading PDF, splitting, embedding, and saving.
    
    Args:
        pdf_path: Path to the PDF file
        vectorstore_path: Path where the vector store will be saved
    """
    pdf_file = Path(pdf_path)

    # If the configured PDF path doesn't exist, allow the user to upload one.
    if not pdf_file.is_file():
        st.warning(f"PDF not found at: {pdf_path}")
        uploaded = st.file_uploader("Upload a PDF to initialize the vector store:", type=["pdf"])
        if uploaded is None:
            st.info(f"Place your PDF at {pdf_path} or upload it here to proceed.")
            return
        else:
            try:
                pdf_file.parent.mkdir(parents=True, exist_ok=True)
                with open(pdf_file, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.success(f"Saved uploaded PDF to: {pdf_path}")
            except Exception as e:
                st.error(f"Failed to save uploaded PDF: {e}")
                return

    try:
        with st.spinner("Loading and processing PDF..."):
            # Load PDF document
            pdf_loader = PyPDFLoader(str(pdf_file))
            pdf_pages = pdf_loader.load_and_split()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            doc_chunks = text_splitter.create_documents(
                [page.page_content for page in pdf_pages]
            )

            # Create embeddings
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

            # Create and save vector store
            vector_store = FAISS.from_documents(doc_chunks, embeddings)
            vector_store.save_local(vectorstore_path)

            # Clear cache to ensure the new vector store is loaded next time
            st.cache_resource.clear()

        st.success("✅ Vector store initialized successfully!")

    except Exception as e:
        st.error(f"❌ Error initializing vector store: {str(e)}")


def chat(query: str, vectorstore) -> str:
    """
    Chat with the user and retrieve response from LLM using RAG.
    
    Args:
        query: User's question
        vectorstore: The loaded vector store object
        
    Returns:
        Response from the LLM
    """
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        llm = ChatOpenAI(model=CHAT_MODEL)
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below:

{context}

Question: {question}
""")
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create and invoke chain
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return response
        
    except Exception as e:
        return f"Error processing query: {str(e)}"


def main() -> None:
    """
    Main application logic. Handles initialization and chat functionality.
    """
    st.title("RAG Demo - Using LangChain, FAISS, and OpenAI")
    
    # Try to load vector store (cached)
    vectorstore = get_vectorstore(VECTORSTORE_PATH)
    
    # Sidebar for initialization
    st.sidebar.header("Setup")
    
    if not vectorstore:
        st.sidebar.warning("⚠️ Vector store not found. Please initialize first.")
        if st.sidebar.button("Initialize Vector Store"):
            init()
            st.rerun()
    else:
        st.sidebar.success("✅ Vector store ready")
        if st.sidebar.button("Reinitialize Vector Store"):
            init()
            st.rerun()
    
    # Chat interface
    st.header("Chat Interface")

    if not vectorstore:
        st.info("Chat is disabled until the vector store is initialized. Use the sidebar to initialize.")
        # Try to render a disabled input when supported by Streamlit; fall back to no input.
        try:
            query = st.text_input("Ask your question", disabled=True)
        except TypeError:
            query = ""
    else:
        query = st.text_input("Ask your question")

        if query:
            with st.spinner("Generating response..."):
                response = chat(query, vectorstore)
                st.write(response)


if __name__ == "__main__":
    main()
