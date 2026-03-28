"""
Streamlit UI for Post-Silicon Validation RAG Platform
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from typing import List, Dict
import time

from src.rag.rag_pipeline import get_rag_pipeline
from src.rag.ingestion import get_ingestion_pipeline
from src.rag.conversation import get_conversation_manager
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.rag_pipeline = None
        st.session_state.ingestion_pipeline = None
        st.session_state.conversation_manager = None


def initialize_components():
    """Initialize RAG components (lazy loading)."""
    try:
        if st.session_state.rag_pipeline is None:
            with st.spinner("🔧 Initializing RAG pipeline..."):
                st.session_state.rag_pipeline = get_rag_pipeline()
                st.session_state.ingestion_pipeline = get_ingestion_pipeline()
                st.session_state.conversation_manager = get_conversation_manager()
            st.success("✅ RAG pipeline initialized!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG pipeline: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return False
    return True


def render_header():
    """Render application header."""
    st.title("🚀 Post-Silicon Validation RAG Platform")
    st.markdown("""
    <style>
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    <p class="subtitle">Intelligent Q&A for Validation Logs, Test Reports, and Documentation</p>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        config = get_config()
        model_provider = config.llm.provider
        st.info(f"**Model:** {model_provider.upper()}")
        
        if model_provider == "ollama":
            st.info(f"**LLM:** {config.llm.ollama.model}")
        else:
            st.info(f"**LLM:** {config.llm.openai.model}")
        
        st.info(f"**Embeddings:** instructor-large")
        
        st.divider()
        
        # Statistics
        st.header("📊 Statistics")
        if st.session_state.ingestion_pipeline:
            stats = st.session_state.ingestion_pipeline.get_stats()
            st.metric("Documents", stats.get("document_count", 0))
            st.metric("Chunks", stats.get("chunk_count", 0))
        
        st.divider()
        
        # Clear conversation
        st.header("🔄 Actions")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.conversation_manager:
                st.session_state.conversation_manager.clear_messages()
            st.rerun()
        
        st.divider()
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        **Version:** 1.0.0
        
        **Tech Stack:**
        - LangChain
        - ChromaDB
        - instructor-large
        - Ollama/OpenAI
        
        **Documentation:**
        - [README](README.md)
        - [Setup Guide](MANUAL_TASKS.md)
        """)


def render_file_upload_page():
    """Render file upload page."""
    st.header("📁 Document Management")
    
    # Upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['txt', 'log', 'report', 'csv', 'xlsx', 'pdf', 'docx'],
        help="Supported formats: .txt, .log, .report, .csv, .xlsx, .pdf, .docx"
    )
    
    if uploaded_files:
        if st.button("📤 Upload and Process", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save file temporarily
                temp_path = Path("data/raw") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest file
                result = st.session_state.ingestion_pipeline.ingest_file(str(temp_path))
                results.append(result)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Show results
            status_text.empty()
            progress_bar.empty()
            
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]
            
            if successful:
                st.success(f"✅ Successfully processed {len(successful)} file(s)")
                for result in successful:
                    st.info(f"📄 **{result['file_name']}**: {result['num_chunks']} chunks created")
            
            if failed:
                st.error(f"❌ Failed to process {len(failed)} file(s)")
                for result in failed:
                    st.error(f"📄 **{result['file_path']}**: {result['error']}")
    
    st.divider()
    
    # Paste content section
    st.subheader("📝 Paste Content")
    text_content = st.text_area(
        "Paste text content directly",
        height=200,
        max_chars=10000,
        help="Paste validation logs, test reports, or documentation"
    )
    
    content_name = st.text_input(
        "Content name",
        value="pasted_content",
        help="Name for this content"
    )
    
    if st.button("📥 Add Content", disabled=not text_content):
        with st.spinner("Processing..."):
            result = st.session_state.ingestion_pipeline.ingest_text_content(
                text_content,
                content_name
            )
            
            if result["status"] == "success":
                st.success(f"✅ Content added: {result['num_chunks']} chunks created")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    st.divider()
    
    # List documents
    st.subheader("📚 Ingested Documents")
    if st.session_state.ingestion_pipeline:
        documents = st.session_state.ingestion_pipeline.list_documents()
        
        if documents:
            st.write(f"Total documents: {len(documents)}")
            
            for doc in documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}"):
                        result = st.session_state.ingestion_pipeline.delete_document(doc)
                        if result["status"] == "success":
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(f"Error deleting {doc}")
        else:
            st.info("No documents ingested yet. Upload files above to get started.")


def render_chat_page():
    """Render chat interface."""
    st.header("💬 Chat Interface")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Source Documents"):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            **Source {idx}:** `{source['source']}`  
                            **Chunk:** {source['chunk_id']}  
                            **Type:** {source['file_type']}  
                            > {source['content']}
                            """)
                
                # Show confidence
                if message["role"] == "assistant" and "confidence" in message:
                    confidence = message["confidence"]
                    if confidence == "high":
                        st.success(f"🎯 Confidence: {confidence.upper()}")
                    elif confidence == "medium":
                        st.warning(f"⚠️ Confidence: {confidence.upper()}")
                    else:
                        st.info(f"ℹ️ Confidence: {confidence.upper()}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.rag_pipeline.query(prompt)
                    
                    # Display answer
                    st.markdown(response.answer)
                    
                    # Display sources
                    with st.expander("📚 Source Documents"):
                        for idx, source in enumerate(response.source_documents, 1):
                            st.markdown(f"""
                            **Source {idx}:** `{source['source']}`  
                            **Chunk:** {source['chunk_id']}  
                            **Type:** {source['file_type']}  
                            > {source['content']}
                            """)
                    
                    # Display confidence
                    if response.confidence == "high":
                        st.success(f"🎯 Confidence: {response.confidence.upper()}")
                    elif response.confidence == "medium":
                        st.warning(f"⚠️ Confidence: {response.confidence.upper()}")
                    else:
                        st.info(f"ℹ️ Confidence: {response.confidence.upper()}")
                    
                    # Add to conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.source_documents,
                        "confidence": response.confidence
                    })
                    
                    # Save to conversation manager
                    if st.session_state.conversation_manager:
                        st.session_state.conversation_manager.add_message(
                            "user", prompt
                        )
                        st.session_state.conversation_manager.add_message(
                            "assistant",
                            response.answer,
                            sources=response.source_documents,
                            confidence=response.confidence
                        )
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Query error: {e}", exc_info=True)


def render_history_page():
    """Render conversation history page."""
    st.header("💾 Conversation History")
    
    # Current conversation
    st.subheader("Current Conversation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Conversation", use_container_width=True):
            if st.session_state.conversation_manager:
                filepath = st.session_state.conversation_manager.save_conversation()
                st.success(f"✅ Saved: {Path(filepath).name}")
    
    with col2:
        if st.button("📄 Export to Markdown", use_container_width=True):
            if st.session_state.conversation_manager:
                filepath = st.session_state.conversation_manager.export_to_markdown()
                st.success(f"✅ Exported: {Path(filepath).name}")
    
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.conversation_manager:
                st.session_state.conversation_manager.clear_messages()
            st.rerun()
    
    # Show current messages
    if st.session_state.messages:
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        for idx, msg in enumerate(st.session_state.messages, 1):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            st.text(f"{idx}. {role_emoji} {msg['role'].title()}: {msg['content'][:80]}...")
    else:
        st.info("No messages in current conversation")
    
    st.divider()
    
    # Saved conversations
    st.subheader("📂 Saved Conversations")
    
    if st.session_state.conversation_manager:
        saved_convos = st.session_state.conversation_manager.list_saved_conversations()
        
        if saved_convos:
            st.write(f"**Total saved:** {len(saved_convos)}")
            
            for convo in saved_convos:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"💬 {convo['filename']}")
                    st.caption(f"Created: {convo['created_at'][:10]} | Messages: {convo['num_messages']}")
                with col2:
                    if st.button("📂", key=f"load_{convo['filename']}"):
                        st.session_state.conversation_manager.load_conversation(convo['filename'])
                        st.session_state.messages = [
                            {
                                "role": msg.role,
                                "content": msg.content,
                                "sources": msg.sources,
                                "confidence": msg.confidence
                            }
                            for msg in st.session_state.conversation_manager.get_messages()
                        ]
                        st.success(f"Loaded {convo['filename']}")
                        st.rerun()
        else:
            st.info("No saved conversations yet")


def main():
    """Main application."""
    st.set_page_config(
        page_title="Post-Silicon Validation RAG",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    render_header()
    render_sidebar()
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs([
        "📁 Document Management",
        "💬 Chat",
        "💾 History"
    ])
    
    with tab1:
        render_file_upload_page()
    
    with tab2:
        render_chat_page()
    
    with tab3:
        render_history_page()
    
    # Footer
    st.divider()
    st.caption("🚀 Post-Silicon Validation RAG Platform v1.0.0 | Built with LangChain, ChromaDB, and Streamlit")


if __name__ == "__main__":
    main()
