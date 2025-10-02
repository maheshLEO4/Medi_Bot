"""
build_db.py - Builds Pinecone Vector Database ONCE during deployment
This script runs automatically when the app is deployed
"""
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Configuration
DATA_PATH = "data/"
PINECONE_INDEX_NAME = "medi-bot-medical"  # Will be created automatically

def build_database():
    """Build the Pinecone vector database from PDFs"""
    print("=" * 60)
    print("BUILDING MEDIBOT PINEcone DATABASE (ONE-TIME SETUP)")
    print("=" * 60)
    
    # Validate environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("❌ Error: PINECONE_API_KEY environment variable not found!")
        print("   Please set your Pinecone API key as an environment variable")
        return False
    else:
        print("✅ Pinecone API key found")
    
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
    
    # Load all PDFs
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
    
    # Initialize Pinecone and build vector store
    print("\n🏗️ Building Pinecone database...")
    try:
        # Initialize Pinecone client
        print("   Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Delete existing index if it exists (reset on redeployment)
        existing_indexes = [index.name for index in pc.list_indexes()]
        if PINECONE_INDEX_NAME in existing_indexes:
            print(f"🗑️ Deleting existing index: {PINECONE_INDEX_NAME}")
            pc.delete_index(PINECONE_INDEX_NAME)
            print("   Waiting for deletion to complete...")
            time.sleep(10)  # Wait for deletion to complete
        
        # Create new index
        print(f"📦 Creating new index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        time.sleep(30)
        
        # Build vector store using langchain_community
        print("📤 Uploading documents to Pinecone...")
        vectorstore = Pinecone.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=PINECONE_INDEX_NAME
        )
        
        # Verify build
        index_stats = pc.describe_index(PINECONE_INDEX_NAME)
        print(f"✅ Database built successfully!")
        print(f"📊 Index: {PINECONE_INDEX_NAME}")
        print(f"📈 Dimension: {index_stats.dimension}")
        print(f"📚 Total chunks uploaded: {len(chunks)}")
        
        print("\n" + "=" * 60)
        print("🎉 MEDIBOT PINEcone DATABASE READY FOR USE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error building database: {str(e)}")
        import traceback
        traceback.print_exc()
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