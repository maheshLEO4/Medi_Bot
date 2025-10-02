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
        st.error("❌ PINECONE_API_KEY environment variable not set! Please add it to your Streamlit secrets.")
        return False
    
    status = get_vectorstore_status()
    if not status["pinecone_ready"]:
        st.warning("🚀 First-time setup: Building knowledge base... This may take a few minutes.")
        try:
            # Show a simple status instead of progress bar
            status_text = st.empty()
            status_text.text("Starting database build...")
            
            result = subprocess.run([sys.executable, "build_db.py"], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                st.success("✅ Database built successfully!")
                return True
            else:
                st.error(f"❌ Build failed: {result.stderr}")
                if result.stdout:
                    with st.expander("Build Output"):
                        st.code(result.stdout)
                return False
        except subprocess.TimeoutExpired:
            st.error("❌ Build timed out. Please check your Pinecone account and try again.")
            return False
        except Exception as e:
            st.error(f"❌ Build error: {str(e)}")
            return False
    return True

st.set_page_config(
    page_title="medi_bot - Medical Document Assistant",
    page_icon="🏥",
    layout="centered"
)

# Fix CSS for better visibility
st.markdown("""
    <style>
        /* Hide sidebar */
        .css-1d391kg {display: none}
        .css-1lcbmhc {display: none}
        
        /* Make chat messages more visible */
        .stMarkdown {
            color: black !important;
        }
        div[data-testid="stMarkdownContainer"] {
            color: black !important;
        }
        /* Ensure text is visible in chat containers */
        .stContainer {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 medi_bot - Medical Document Assistant")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False
if 'database_building' not in st.session_state:
    st.session_state.database_building = False

# Status indicator at top
status_container = st.container()

with status_container:
    status = get_vectorstore_status()
    pdf_files = check_pdf_files()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Medical Documents", len(pdf_files))
    with col2:
        if st.session_state.vectorstore_loaded:
            st.metric("Vector DB", "Pinecone")
        else:
            st.metric("Vector DB", "Loading...")
    with col3:
        status_icon = "✅" if status["api_key_available"] else "❌"
        st.metric("API Key", f"{status_icon} {'Set' if status['api_key_available'] else 'Missing'}")
    with col4:
        if st.session_state.database_building:
            st.metric("Status", "🔄 Building DB")
        else:
            status_icon = "✅" if st.session_state.vectorstore_loaded else "🔄"
            st.metric("Status", f"{status_icon} {'Online' if st.session_state.vectorstore_loaded else 'Initializing'}")

# Debug info in expander
with st.expander("🔧 System Status Details"):
    st.write(f"Pinecone API Key: {'✅ Set' if os.getenv('PINECONE_API_KEY') else '❌ Missing'}")
    st.write(f"GROQ API Key: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    st.write(f"Vectorstore loaded: {st.session_state.vectorstore_loaded}")
    st.write(f"Pinecone index ready: {status['pinecone_ready']}")
    st.write(f"PDF files found: {pdf_files}")
    st.write(f"Data folder exists: {status['data_folder_exists']}")

# Load vectorstore ONCE when app starts
if not st.session_state.vectorstore_loaded:
    # First check if we need to build the database
    if not status["pinecone_ready"]:
        st.info("📦 Database setup required. Click the button below to build the knowledge base.")
        
        if st.button("🚀 Build Knowledge Base", type="primary"):
            st.session_state.database_building = True
            st.rerun()
    
    # If database is building or we need to build it
    if st.session_state.database_building:
        with st.spinner("🔧 Building Pinecone database... This may take 2-5 minutes."):
            if ensure_database():
                st.session_state.database_building = False
                st.success("✅ Database built successfully! Loading now...")
                st.rerun()
            else:
                st.session_state.database_building = False
                st.error("❌ Database build failed. Please check the errors above.")
                st.stop()
    
    # If database is ready, load the vectorstore
    if status["pinecone_ready"] and not st.session_state.database_building:
        with st.spinner("📚 Loading MediBot knowledge base from Pinecone..."):
            vectorstore = get_vectorstore()
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_loaded = True
                st.success("✅ MediBot Ready!")
                st.rerun()
            else:
                st.error("""
                ❌ **Failed to load knowledge base**
                
                The Pinecone vector database exists but failed to load.
                This might be due to:
                - Index corruption
                - Network issues
                - API rate limits
                
                **Try rebuilding the database:**
                """)
                if st.button("🔄 Rebuild Database"):
                    st.session_state.database_building = True
                    st.rerun()

# Chat Interface
if st.session_state.vectorstore_loaded:
    # Clear chat button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    st.subheader("💬 Medical Consultation")
    
    # Chat container - SIMPLIFIED VERSION
    for message in st.session_state.messages:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])

    # Enhanced medical prompt template
    MEDICAL_PROMPT_TEMPLATE = """You are medi_bot, an expert AI medical assistant designed to provide accurate, helpful medical information.

MEDICAL CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Answer based SOLELY on the medical context provided above
2. If the context doesn't contain relevant information, clearly state this limitation
3. Provide clear, structured medical information when available in context
4. Always emphasize consulting healthcare professionals for medical decisions
5. Be precise and avoid speculation
6. Format your response for easy reading with clear sections if appropriate

Please provide your medical response:"""

    # Chat input
    prompt = st.chat_input("Ask about medical conditions, treatments, medications...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.spinner("🔍 Searching medical knowledge..."):
            try:
                # Test if GROQ API key is working
                groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise Exception("GROQ_API_KEY not found in secrets or environment variables")
                
                # Test if vectorstore is working
                if not st.session_state.vectorstore:
                    raise Exception("Vectorstore not initialized")
                
                # Create retriever from the vectorstore
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={'k': 3}
                )
                
                # Create QA chain
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

                # Execute query
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": result})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(result)
                
                # Show sources in expander
                if source_documents:
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(source_documents, 1):
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.write(f"**Source {i}:** {source_file} (Page {page_num})")
                            st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            st.divider()
                
            except Exception as e:
                # Better error logging
                error_detail = f"Sorry, I encountered an error: {type(e).__name__}"
                st.session_state.messages.append({"role": "assistant", "content": error_detail})
                
                with st.chat_message("assistant"):
                    st.write(error_detail)
                    st.error("Please check the technical details below for more information.")
                
                # Show detailed error in expander for debugging
                with st.expander("🔧 Technical Error Details"):
                    st.error(f"Error type: {type(e).__name__}")
                    st.error(f"Error message: {str(e)}")
                    
                    # Check specific common issues
                    if "GROQ" in str(e) or "API" in str(e):
                        st.warning("🔑 **GROQ API Issue Detected**")
                        st.info("Please check that your GROQ_API_KEY is valid and has sufficient credits.")
                    
                    if "pinecone" in str(e).lower() or "vector" in str(e).lower() or "retriever" in str(e).lower():
                        st.warning("🗄️ **Vector Database Issue Detected**")
                        st.info("There might be an issue with the Pinecone connection or the vectorstore configuration.")
                    
                    # Show full traceback
                    st.code(f"Full traceback:\n{traceback.format_exc()}")

# Simple footer
st.markdown("---")
st.caption("💡 Always consult healthcare professionals for medical decisions. This assistant provides informational support only.")