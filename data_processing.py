"""
data_processing.py - ONLY loads pre-built Pinecone Vector Store
Database is built by build_db.py during deployment
"""
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/"
PINECONE_INDEX_NAME = "medi-bot-medical"

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
    Load pre-built Pinecone Vector Store
    """
    log_message("📚 Loading MediBot Pinecone Vector Store...")
    
    # Check if API key is available
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        log_message("❌ ERROR: PINECONE_API_KEY environment variable not found!")
        log_message("   Please set your Pinecone API key as an environment variable")
        return None
    
    try:
        # Load embedding model
        embedding_model = get_embedding_model()
        if not embedding_model:
            return None
        
        # Initialize Pinecone client to check index status
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            log_message(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found!")
            log_message("   The database should have been built during deployment.")
            log_message("   Run: python build_db.py")
            return None
        
        # Load existing vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedding_model
        )
        
        log_message(f"✅ Loaded Pinecone index: {PINECONE_INDEX_NAME}")
        
        return vectorstore
        
    except Exception as e:
        log_message(f"❌ ERROR loading Pinecone: {str(e)}")
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
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    try:
        if pinecone_api_key:
            pc = Pinecone(api_key=pinecone_api_key)
            existing_indexes = [index.name for index in pc.list_indexes()]
            pinecone_ready = PINECONE_INDEX_NAME in existing_indexes
        else:
            pinecone_ready = False
    except:
        pinecone_ready = False
    
    return {
        "data_folder_exists": os.path.exists(DATA_PATH),
        "pinecone_ready": pinecone_ready,
        "api_key_available": bool(pinecone_api_key),
        "pdf_files": len(check_pdf_files()) > 0
    }