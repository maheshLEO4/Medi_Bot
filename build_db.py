"""
build_db.py - Builds Chroma DB ONCE during deployment
This script runs automatically when the app is deployed
"""
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_PATH = "data/"
CHROMA_DB_PATH = "chroma_db"

def build_database():
    """Build the Chroma database from PDFs"""
    print("=" * 60)
    print("BUILDING MEDIBOT DATABASE (ONE-TIME SETUP)")
    print("=" * 60)
    
    # Check if DB already exists
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("✅ Chroma DB already exists. Skipping build.")
        return True
    
    # Validate data folder
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} folder not found!")
        print(f"   Please create a 'data' folder with PDF files")
        return False
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {DATA_PATH}")
        print(f"   Please add PDF files to the 'data' folder")
        return False
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    # Load all PDFs using directory loader (more efficient)
    print("\n📚 Loading PDF files...")
    try:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        all_documents = loader.load()
        
        # Enhance metadata
        for doc in all_documents:
            if 'source' in doc.metadata:
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])
            if 'page' in doc.metadata:
                doc.metadata['page'] = doc.metadata['page'] + 1
        
        print(f"   Loaded {len(all_documents)} pages total")
        
    except Exception as e:
        print(f"❌ Error loading PDFs: {str(e)}")
        return False
    
    if not all_documents:
        print("❌ No documents loaded!")
        return False
    
    # Split into chunks
    print("\n✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"   Created {len(chunks)} chunks")
    
    # Load embedding model
    print("\n🔧 Loading embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")
        return False
    
    # Build Chroma DB
    print("\n🏗️ Building Chroma database...")
    try:
        # Remove existing DB if empty/corrupted
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Verify build
        collection_count = db._collection.count()
        print(f"✅ Database built successfully!")
        print(f"📊 Total chunks in database: {collection_count}")
        
        print("\n" + "=" * 60)
        print("🎉 MEDIBOT DATABASE READY FOR USE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error building database: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = build_database()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)