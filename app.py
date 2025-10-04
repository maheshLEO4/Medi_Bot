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
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #e8f4fd;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
    }
    .chat-container {
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .clear-btn {
        margin-bottom: 1rem;
    }
    .sources-dropdown {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Chat input styling */
    .stChatInput {
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 🏥 Simple Header
# -------------------------
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
    template="""You are MediBot, a helpful and knowledgeable medical assistant. Use the following medical context to answer the user's question in a natural, conversational way. Provide accurate, helpful information without using phrases like "based on the context" or "according to the documents".

Medical Context:
{context}

Question: {question}

Answer in a friendly, professional tone as if you're having a conversation:""",
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
        {"role": "assistant", "content": "Hello! I'm MediBot, your AI medical assistant. How can I help you with medical questions today?"}
    ]

# Store source documents
if "source_docs" not in st.session_state:
    st.session_state.source_docs = {}

# Clear chat function
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediBot, your AI medical assistant. How can I help you with medical questions today?"}
    ]
    st.session_state.source_docs = {}

# Clear chat button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
        clear_chat()
        st.rerun()

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        
        # Show sources dropdown for assistant messages that have sources
        if i in st.session_state.source_docs and st.session_state.source_docs[i]:
            with st.expander("📚 View Sources", expanded=False):
                sources = st.session_state.source_docs[i]
                for j, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get('source', 'Unknown Document')
                    page_info = doc.metadata.get('page', '')
                    
                    st.markdown(f"**Source {j}:**")
                    st.markdown(f"📄 {os.path.basename(source_name)}")
                    if page_info:
                        st.markdown(f"📖 Page: {page_info}")
                    st.markdown(f"**Content:** {doc.page_content[:200]}...")
                    st.markdown("---")

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display response
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            sources = response.get("source_documents", [])
            
            # Store sources for this message
            message_index = len(st.session_state.messages)
            st.session_state.source_docs[message_index] = sources
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = "I apologize, but I encountered an error processing your question. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

# -------------------------
# 🎯 Simple Footer
# -------------------------
st.markdown("---")
st.caption("MediBot - Your AI Medical Assistant")