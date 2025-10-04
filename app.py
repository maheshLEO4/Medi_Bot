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
    page_title="🏥 MediBot", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------
# 🎨 Custom CSS for Better UI
# -------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .assistant-message {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .source-docs {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 80%;
        left: 10%;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 🏥 Header
# -------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏥 MediBot")
st.markdown("**Your AI Medical Assistant**")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# 🧠 Load embedding model
# -------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

try:
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"❌ Failed to load embedding model: {e}")
    st.stop()

# -------------------------
# 📚 Initialize Qdrant Cloud Vector Store
# -------------------------
@st.cache_resource
def init_qdrant():
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        
        # Check if our collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if COLLECTION_NAME not in collection_names:
            st.error(f"❌ Collection '{COLLECTION_NAME}' not found. Please run the data ingestion script first.")
            return None
        
        # Create vector store
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant Cloud: {e}")
        return None

vectorstore = init_qdrant()
if vectorstore is None:
    st.stop()

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# -------------------------
# ⚡ Initialize Groq LLM
# -------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"❌ Failed to initialize Groq LLM: {e}")
    st.stop()

# -------------------------
# 🔗 Create RetrievalQA Chain with Custom Prompt
# -------------------------
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
# 💬 Chat Interface
# -------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediBot, your AI medical assistant. How can I help you with your medical questions today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask your medical question..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-message"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)

    # Get response
    with st.spinner("🧠 Thinking..."):
        try:
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            sources = response.get("source_documents", [])
            
            # Display assistant response
            st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{answer}</div>', unsafe_allow_html=True)
            
            # Display sources in a subtle way
            if sources:
                unique_sources = set()
                for doc in sources:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in unique_sources:
                        unique_sources.add(source)
                
                if unique_sources:
                    sources_text = "References: " + ", ".join([os.path.basename(src) for src in unique_sources])
                    st.markdown(f'<div class="source-docs">{sources_text}</div>', unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your question. Please try again."
            st.markdown(f'<div class="assistant-message"><strong>MediBot:</strong><br>{error_msg}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# 🔧 Footer with Info
# -------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**⚡ Powered by Groq**")
with col2:
    st.markdown("**🗂️ Qdrant Cloud**")
with col3:
    st.markdown("**🔒 Secure & Private**")

# Clear chat button
if st.button("🗑️ Clear Conversation"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediBot, your AI medical assistant. How can I help you with your medical questions today?"}
    ]
    st.rerun()