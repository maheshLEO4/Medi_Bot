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
# 🔐 API KEYS
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
    page_title="MediBot",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------
# 🎨 Clean CSS
# -------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .assistant-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 15%;
        background: white;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Style the chat input better */
    .stChatInput input {
        border-radius: 25px;
        padding: 12px 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 🏥 Simple Header
# -------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🏥 MediBot")
    st.markdown("**AI Medical Assistant**")
    st.markdown('</div>', unsafe_allow_html=True)

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
            st.error(f"Collection '{COLLECTION_NAME}' not found.")
            return None
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=load_embedding_model()
        )
        return vectorstore
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
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
    embedding_model = load_embedding_model()
    vectorstore = init_qdrant()
    llm = load_llm()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""You are a medical assistant. Answer the question naturally using the context.

Context:
{context}

Question: {question}

Answer conversationally:""",
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
# 💬 Clean Chat Interface
# -------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediBot. How can I assist you with medical questions today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-message"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)

    # Get response
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            
            # Display assistant response
            st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{answer}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = "I apologize, but I encountered an error. Please try again."
            st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{error_msg}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# -------------------------
# 🎯 Simple Footer
# -------------------------
st.markdown("---")
st.caption("MediBot - AI Medical Assistant")