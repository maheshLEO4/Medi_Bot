"""
data_processing.py - ONLY loads pre-built Chroma DB
Database is built by build_db.py during deployment
"""
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/"
CHROMA_DB_PATH = "chroma_db"

def log_message(message):
    """Log message"""
    logger.info(message)
    print(message)

def get_embedding_model():
    """Get the embedding model"""
    try:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return model
    except Exception as e:
        log_message(f"❌ Error loading embedding model: {str(e)}")
        return None

def get_vectorstore():
    """
    Load pre-built Chroma DB
    This should NEVER build - only load existing DB
    """
    log_message("📚 Loading MediBot Vector Store...")
    
    # Check if DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        log_message("❌ ERROR: Chroma DB not found!")
        log_message("   The database should have been built during deployment.")
        log_message("   Run: python build_db.py")
        return None
    
    if not os.listdir(CHROMA_DB_PATH):
        log_message("❌ ERROR: Chroma DB folder is empty!")
        log_message("   The database build may have failed.")
        return None
    
    try:
        # Load embedding model
        embedding_model = get_embedding_model()
        if not embedding_model:
            return None
        
        # Load existing database
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
        
        # Verify it loaded
        collection_count = db._collection.count()
        log_message(f"✅ Loaded database with {collection_count} chunks")
        
        return db
        
    except Exception as e:
        log_message(f"❌ ERROR loading Chroma DB: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Set custom prompt template"""
    try:
        prompt = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["context", "question"]
        )
        return prompt
    except Exception as e:
        log_message(f"⚠️ Error setting custom prompt: {str(e)}")
        # Fallback to simple prompt
        return PromptTemplate(
            template="""Use the following context to answer the question.

Context: {context}
Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

def check_pdf_files():
    """Check available PDF files"""
    if not os.path.exists(DATA_PATH):
        return []
    return [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]

def get_vectorstore_status():
    """Get status of vector store"""
    return {
        "data_folder_exists": os.path.exists(DATA_PATH),
        "chroma_db_exists": os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0,
        "pdf_files": len(check_pdf_files()) > 0
    }

# NEW: Function for build_db.py to use
def process_documents():
    """Process documents and create vector store - used by build_db.py"""
    try:
        # Check data directory
        if not os.path.exists(DATA_PATH):
            log_message(f"❌ Data directory '{DATA_PATH}' not found")
            return None
        
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            log_message("❌ No PDF files found in data directory")
            return None
        
        log_message(f"📁 Found {len(pdf_files)} PDF files")
        
        # Load documents
        log_message("📚 Loading PDF files...")
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        
        if not documents:
            log_message("❌ No documents loaded")
            return None
        
        log_message(f"📄 Loaded {len(documents)} pages")
        
        # Split documents
        log_message("✂️ Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        log_message(f"📦 Created {len(chunks)} chunks")
        
        # Create embeddings and vector store
        log_message("🔧 Creating vector store...")
        embeddings = get_embedding_model()
        if not embeddings:
            return None
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        log_message("✅ Vector store created successfully")
        return vectorstore
        
    except Exception as e:
        log_message(f"❌ Error processing documents: {str(e)}")
        return None