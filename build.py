"""
build_db.py - Builds Chroma DB ONCE during deployment
This script runs automatically when the app is deployed
"""
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import subprocess

if not os.path.exists("chroma_db"):  # or your DB folder
    print("📦 Building database...")
    subprocess.run(["python", "build_db.py"])


DATA_PATH = "data/"
CHROMA_DB_PATH = "chroma_db"

def build_database():
    """Build the Chroma database from PDFs"""
    print("=" * 60)
    print("BUILDING MEDIBOT DATABASE (ONE-TIME SETUP)")
    print("=" * 60)
    
    # Check if DB already exists
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("Chroma DB already exists. Skipping build.")
        return True
    
    # Validate data folder
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} folder not found!")
        return False
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in {DATA_PATH}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Load all PDFs
    print("\nLoading PDF files...")
    all_documents = []
    
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(DATA_PATH, pdf_file)
            print(f"   Loading: {pdf_file}...", end=" ")
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['source'] = pdf_file
                if 'page' in doc.metadata:
                    doc.metadata['page'] = doc.metadata['page'] + 1
            
            all_documents.extend(docs)
            print(f"Done ({len(docs)} pages)")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    if not all_documents:
        print("No documents loaded!")
        return False
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    
    # Split into chunks
    print("\nSplitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"   Created {len(chunks)} chunks")
    
    # Load embedding model
    print("\nLoading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   Model loaded")
    
    # Build Chroma DB
    print("\nBuilding Chroma database...")
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Verify
    collection_count = db._collection.count()
    print(f"Database built successfully!")
    print(f"Total chunks in database: {collection_count}")
    
    print("\n" + "=" * 60)
    print("MEDIBOT DATABASE READY FOR USE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = build_database()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)