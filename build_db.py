import os
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# 🔐 API KEYS
# -------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "medical-documents"

# --- Step 2: Test Connection First ---
print("🔗 Testing Qdrant connection...")
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    
    # Test the connection
    collections = client.get_collections()
    print(f"✅ Successfully connected to Qdrant Cloud!")
    print(f"📊 Available collections: {[col.name for col in collections.collections]}")
    
except Exception as e:
    print(f"❌ Failed to connect to Qdrant: {e}")
    exit()

# --- Step 3: Load PDFs ---
DATA_PATH = "/content/data"

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

print("📚 Loading PDF documents...")
documents = load_pdf_files(DATA_PATH)
print(f"✅ Loaded {len(documents)} PDF pages from {DATA_PATH}")

# --- Step 4: Split into chunks ---
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

print("✂️ Splitting documents into chunks...")
text_chunks = create_chunks(documents)
print(f"✅ Created {len(text_chunks)} text chunks")

# --- Step 5: Load Embedding model ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Step 6: Upload to Qdrant Cloud ---
print("⏳ Uploading to Qdrant Cloud...")

try:
    # Upload in smaller batches to avoid timeouts
    batch_size = 100
    
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[i:i + batch_size]
        
        if i == 0:
            # First batch - create collection
            vectorstore = QdrantVectorStore.from_documents(
                documents=batch,
                embedding=embedding_model,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
                force_recreate=True
            )
        else:
            # Subsequent batches - add to existing collection
            vectorstore.add_documents(batch)
    
    print("✅ All embeddings successfully stored in Qdrant Cloud!")
    print(f"🔗 Access your dashboard at: https://cloud.qdrant.io/")
    
    
    # Show collection info
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"📈 Collection status: {collection_info.status}")
    print(f"📊 Vectors count: {collection_info.vectors_count}")
    
except Exception as e:
    print(f"❌ Upload failed: {e}")