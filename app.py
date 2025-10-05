import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# 🔑 API KEYS
# -------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# ⚙️ CONFIG
# -------------------------
COLLECTION_NAME = "medical-documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="MediBot AI",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# 🔧 Initialize Components
# -------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def init_qdrant():
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if COLLECTION_NAME not in collection_names:
            st.error("Medical database not found. Please check your setup.")
            return None
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=load_embedding_model()
        )
        return vectorstore
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

# Initialize components
try:
    vectorstore = init_qdrant()
    llm = load_llm()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""You are a helpful medical assistant. Answer the question naturally and conversationally using the medical context provided.

Medical Context:
{context}

Question: {question}

Provide a clear, helpful answer in a friendly tone:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# -------------------------
# 💬 Chat Interface (DocuBot Style)
# -------------------------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize source documents storage
if "source_docs" not in st.session_state:
    st.session_state.source_docs = {}

# --- Sidebar ---
with st.sidebar:
    st.title("MediBot Controls")
    st.markdown("AI-powered medical assistant for your health questions.")
    
    st.markdown("---")
    st.success("Using Qdrant Cloud Storage")
    
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_docs = {}
        st.toast("Chat history cleared!", icon="🧹")
        st.rerun()
    
    st.markdown("---")
    st.subheader("Knowledge Base Info")
    st.write(f"**Collection:** {COLLECTION_NAME}")
    st.write(f"**Model:** llama-3.1-8b-instant")

# --- Main Chat ---
st.title("MediBot AI: Your Medical Assistant")
st.markdown("Ask questions about medical topics and get informed answers.")

# Display welcome message
if not st.session_state.messages:
    st.success("Ready! Ask me any medical questions.")
    st.info("Using Qdrant Cloud for fast, scalable medical knowledge retrieval")

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Show sources for assistant messages
        if message['role'] == 'assistant' and idx in st.session_state.source_docs:
            source_documents = st.session_state.source_docs[idx]
            if source_documents:
                with st.expander("Source References"):
                    st.caption("Sources from medical knowledge base")
                    
                    for i, doc in enumerate(source_documents, 1):
                        source_name = doc.metadata.get('source', 'Unknown Document')
                        page_info = doc.metadata.get('page', 'N/A')
                        
                        # Display source name
                        display_name = os.path.basename(source_name)
                        if len(display_name) > 50:
                            display_name = display_name[:47] + "..."
                        
                        st.markdown(f"**📄 Source {i}:** `{display_name}`")
                        
                        if page_info != 'N/A':
                            st.caption(f"**Page:** {page_info}")
                        
                        excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        st.caption(f'**Excerpt:** "{excerpt}"')
                        st.markdown("---")

# Handle user input
if prompt := st.chat_input("Ask a medical question..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        with st.spinner("Thinking..."):
            import time
            start_time = time.time()
            
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            sources = response.get("source_documents", [])
            
            processing_time = time.time() - start_time
            
            with st.chat_message('assistant'):
                st.markdown(answer)
                
                if sources:
                    with st.expander("Source References"):
                        st.caption("Sources from medical knowledge base")
                        
                        for i, doc in enumerate(sources, 1):
                            source_name = doc.metadata.get('source', 'Unknown Document')
                            page_info = doc.metadata.get('page', 'N/A')
                            
                            display_name = os.path.basename(source_name)
                            if len(display_name) > 50:
                                display_name = display_name[:47] + "..."
                            
                            st.markdown(f"**📄 Source {i}:** `{display_name}`")
                            
                            if page_info != 'N/A':
                                st.caption(f"**Page:** {page_info}")
                            
                            excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            st.caption(f'**Excerpt:** "{excerpt}"')
                            st.markdown("---")
            
            # Store message and sources
            message_index = len(st.session_state.messages)
            st.session_state.messages.append({'role': 'assistant', 'content': answer})
            st.session_state.source_docs[message_index] = sources
            
    except Exception as e:
        error_msg = f"An error occurred while processing your question: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({'role': 'assistant', 'content': error_msg})