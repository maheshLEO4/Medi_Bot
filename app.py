import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from data_processing import get_vectorstore, set_custom_prompt, get_vectorstore_status, check_pdf_files
import subprocess
import sys

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
                                  capture_output=True, text=True, timeout=600)  # Increased timeout to 10 minutes
            
            if result.returncode == 0:
                st.success("✅ Database built successfully!")
                st.rerun()
            else:
                st.error(f"❌ Build failed: {result.stderr}")
                if result.stdout:
                    with st.expander("Build Output"):
                        st.code(result.stdout)
        except subprocess.TimeoutExpired:
            st.error("❌ Build timed out. Please check your Pinecone account and try again.")
        except Exception as e:
            st.error(f"❌ Build error: {str(e)}")
        return False
    return True

st.set_page_config(
    page_title="medi_bot - Medical Document Assistant",
    page_icon="🏥",
    layout="centered"
)

# Hide sidebar completely
st.markdown("""
    <style>
        .css-1d391kg {display: none}
        .css-1lcbmhc {display: none}
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
        status_icon = "✅" if st.session_state.vectorstore_loaded else "🔄"
        st.metric("Status", f"{status_icon} {'Online' if st.session_state.vectorstore_loaded else 'Initializing'}")

# Load vectorstore ONCE when app starts
if not st.session_state.vectorstore_loaded:
    # First check if we need to build the database
    if not get_vectorstore_status()["pinecone_ready"]:
        with st.spinner("🔧 Setting up Pinecone database..."):
            if ensure_database():
                st.rerun()
            else:
                st.stop()
    
    # Now load the vectorstore
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
            
            The Pinecone vector database was not found or failed to load.
            This usually means:
            - The database wasn't built during deployment
            - PINECONE_API_KEY is not set correctly
            - The Pinecone index doesn't exist
            
            **For developers:**
            1. Set PINECONE_API_KEY environment variable
            2. Run `python build_db.py` to build the database
            """)
            st.stop()

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
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 12px; border-radius: 10px; margin: 5px 0;'>
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #e6f3ff; padding: 12px; border-radius: 10px; margin: 5px 0;'>
                    <strong>MediBot:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)

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
        
        # Generate response
        with st.spinner("🔍 Searching medical knowledge..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="llama-3.1-8b-instant",
                        temperature=0.1,
                        groq_api_key=st.secrets["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(
                        search_kwargs={'k': 3}
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(MEDICAL_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": result})
                
                # Show sources in expander
                if source_documents:
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(source_documents, 1):
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.write(f"**Source {i}:** {source_file} (Page {page_num})")
                            st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            st.divider()
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while processing your query. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(f"Error: {str(e)}")
                st.rerun()

# Simple footer
st.markdown("---")
st.caption("💡 Always consult healthcare professionals for medical decisions. This assistant provides informational support only.")