import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from data_processing import get_vectorstore, set_custom_prompt, get_vectorstore_status, check_pdf_files
import subprocess
import sys
import traceback

# Auto-build database on startup if missing
def ensure_database():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("PINECONE_API_KEY environment variable not set! Please add it to your Streamlit secrets.")
        return False
    
    status = get_vectorstore_status()
    if not status["pinecone_ready"]:
        st.warning("First-time setup: Building knowledge base... This may take a few minutes.")
        try:
            status_text = st.empty()
            status_text.text("Starting database build...")
            
            result = subprocess.run([sys.executable, "build_db.py"], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                st.success("Database built successfully!")
                return True
            else:
                st.error(f"Build failed: {result.stderr}")
                if result.stdout:
                    with st.expander("Build Output"):
                        st.code(result.stdout)
                return False
        except subprocess.TimeoutExpired:
            st.error("Build timed out. Please check your Pinecone account and try again.")
            return False
        except Exception as e:
            st.error(f"Build error: {str(e)}")
            return False
    return True

st.set_page_config(
    page_title="MediBot - Medical Assistant",
    page_icon="🏥",
    layout="wide"
)

# Enhanced CSS for much better visibility
st.markdown("""
    <style>
        /* Force dark text on light background */
        .stApp {
            background-color: #f0f2f6;
        }
        
        /* Main content area */
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1200px;
        }
        
        /* All text elements */
        p, span, div, h1, h2, h3, label {
            color: #0e1117 !important;
        }
        
        /* Chat messages */
        .stChatMessage {
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* User messages - blue tint */
        .stChatMessage[data-testid*="user"] {
            background: linear-gradient(135deg, #e3f2fd 0%, #f5f9ff 100%) !important;
            border-left: 4px solid #2196f3 !important;
        }
        
        /* Assistant messages - green tint */
        .stChatMessage[data-testid*="assistant"] {
            background: linear-gradient(135deg, #e8f5e9 0%, #f5fdf5 100%) !important;
            border-left: 4px solid #4caf50 !important;
        }
        
        /* Chat message text */
        .stChatMessage p {
            color: #1a1a1a !important;
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: white !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        div[data-testid="metric-container"] > div {
            color: #0e1117 !important;
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
        }
        
        /* Chat input */
        .stChatInput {
            border: 2px solid #667eea !important;
            border-radius: 10px !important;
            background-color: white !important;
        }
        
        .stChatInput textarea {
            color: #1a1a1a !important;
            font-size: 16px !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: white !important;
            color: #0e1117 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #fafafa !important;
            border: 1px solid #e0e0e0 !important;
            color: #0e1117 !important;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 8px !important;
            border-left-width: 4px !important;
        }
        
        /* Success message */
        .stSuccess {
            background-color: #e8f5e9 !important;
            color: #2e7d32 !important;
        }
        
        /* Error message */
        .stError {
            background-color: #ffebee !important;
            color: #c62828 !important;
        }
        
        /* Warning message */
        .stWarning {
            background-color: #fff3e0 !important;
            color: #ef6c00 !important;
        }
        
        /* Info message */
        .stInfo {
            background-color: #e3f2fd !important;
            color: #1565c0 !important;
        }
        
        /* Title */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            padding: 1rem 0 !important;
        }
        
        /* Subheader */
        h2, h3 {
            color: #2c3e50 !important;
            font-weight: 700 !important;
            margin-top: 1.5rem !important;
        }
        
        /* Caption/Footer */
        .stCaption {
            color: #666 !important;
            text-align: center !important;
            padding: 1rem !important;
            border-top: 2px solid #e0e0e0 !important;
            margin-top: 2rem !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #667eea transparent transparent transparent !important;
        }
        
        /* Code blocks */
        code {
            background-color: #f5f5f5 !important;
            color: #d32f2f !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 4px !important;
        }
        
        pre {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🏥 MediBot</h1>
        <p style='font-size: 1.2rem; color: #666;'>Your AI-Powered Medical Document Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False
if 'database_building' not in st.session_state:
    st.session_state.database_building = False

# Status Dashboard
st.markdown("### System Status")
status = get_vectorstore_status()
pdf_files = check_pdf_files()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Medical Documents", len(pdf_files))
with col2:
    if st.session_state.vectorstore_loaded:
        st.metric("Vector Database", "Pinecone")
    else:
        st.metric("Vector Database", "Loading...")
with col3:
    status_icon = "Ready" if status["api_key_available"] else "Missing"
    st.metric("API Status", status_icon)
with col4:
    if st.session_state.database_building:
        st.metric("Status", "Building DB")
    else:
        status_text = "Ready" if st.session_state.vectorstore_loaded else "Initializing"
        st.metric("Status", status_text)

st.markdown("---")

# Debug info in expander
with st.expander("Technical Details"):
    st.write(f"**Pinecone API Key:** {'Configured' if os.getenv('PINECONE_API_KEY') else 'Missing'}")
    st.write(f"**GROQ API Key:** {'Configured' if os.getenv('GROQ_API_KEY') else 'Missing'}")
    st.write(f"**Vectorstore Status:** {'Loaded' if st.session_state.vectorstore_loaded else 'Initializing'}")
    st.write(f"**Pinecone Index:** {'Ready' if status['pinecone_ready'] else 'Not Found'}")
    st.write(f"**PDF Files:** {len(pdf_files)} documents")
    if pdf_files:
        with st.expander("View PDF Files"):
            for i, pdf in enumerate(pdf_files, 1):
                st.write(f"{i}. {pdf}")

# Load vectorstore ONCE when app starts
if not st.session_state.vectorstore_loaded:
    if not status["pinecone_ready"]:
        st.info("**Database Setup Required**\n\nClick the button below to build the medical knowledge base from your PDF documents.")
        
        if st.button("Build Knowledge Base", type="primary", use_container_width=True):
            st.session_state.database_building = True
            st.rerun()
    
    if st.session_state.database_building:
        with st.spinner("Building Pinecone database... This may take 2-5 minutes."):
            if ensure_database():
                st.session_state.database_building = False
                st.success("Database built successfully! Loading now...")
                st.rerun()
            else:
                st.session_state.database_building = False
                st.error("Database build failed. Please check the errors above.")
                st.stop()
    
    if status["pinecone_ready"] and not st.session_state.database_building:
        with st.spinner("Loading MediBot knowledge base from Pinecone..."):
            vectorstore = get_vectorstore()
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_loaded = True
                st.success("MediBot is ready to assist you!")
                st.rerun()
            else:
                st.error("""
                **Failed to load knowledge base**
                
                The Pinecone vector database exists but failed to load. Possible causes:
                - API compatibility issues
                - Network connectivity problems
                - Index corruption
                
                **Solution:** Try rebuilding the database
                """)
                if st.button("Rebuild Database", type="primary"):
                    st.session_state.database_building = True
                    st.rerun()

# Chat Interface
if st.session_state.vectorstore_loaded:
    st.markdown("---")
    
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Medical Consultation")
    with col3:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Enhanced medical prompt template
    MEDICAL_PROMPT_TEMPLATE = """You are MediBot, an expert AI medical assistant providing accurate medical information.

MEDICAL CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided medical context
2. If information is not in the context, clearly state this
3. Provide clear, structured responses
4. Always emphasize consulting healthcare professionals
5. Be precise and evidence-based

Response:"""

    # Chat input
    prompt = st.chat_input("Ask about medical conditions, treatments, medications...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Searching medical knowledge base..."):
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise Exception("GROQ_API_KEY not found")
                
                if not st.session_state.vectorstore:
                    raise Exception("Vectorstore not initialized")
                
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={'k': 3}
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="llama-3.1-8b-instant",
                        temperature=0.1,
                        groq_api_key=groq_api_key,
                    ),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(MEDICAL_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                st.session_state.messages.append({"role": "assistant", "content": result})
                
                with st.chat_message("assistant"):
                    st.markdown(result)
                
                if source_documents:
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(source_documents, 1):
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.markdown(f"**Source {i}:** `{source_file}` (Page {page_num})")
                            st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            if i < len(source_documents):
                                st.divider()
                
            except Exception as e:
                error_detail = f"An error occurred: {type(e).__name__}"
                st.session_state.messages.append({"role": "assistant", "content": error_detail})
                
                with st.chat_message("assistant"):
                    st.markdown(error_detail)
                    st.error("Please check technical details below")
                
                with st.expander("Error Details"):
                    st.error(f"**Error Type:** {type(e).__name__}")
                    st.error(f"**Message:** {str(e)}")
                    
                    if "GROQ" in str(e) or "API" in str(e):
                        st.warning("**GROQ API Issue** - Check your API key and credits")
                    
                    if "pinecone" in str(e).lower() or "vector" in str(e).lower():
                        st.warning("**Vector Database Issue** - Pinecone connection problem")
                    
                    st.code(f"Traceback:\n{traceback.format_exc()}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0; color: #666;'>
        <p style='font-size: 0.9rem; margin: 0;'>
            <strong>Medical Disclaimer:</strong> This assistant provides informational support only.
        </p>
        <p style='font-size: 0.9rem; margin: 0.5rem 0;'>
            Always consult qualified healthcare professionals for medical decisions and treatment.
        </p>
        <p style='font-size: 0.8rem; color: #999; margin-top: 1rem;'>
            Powered by Pinecone Vector DB | LangChain | GROQ AI
        </p>
    </div>
""", unsafe_allow_html=True)