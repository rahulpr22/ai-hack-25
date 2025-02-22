import streamlit as st
import asyncio
from src.data_ingestion.data_ingestion_orchestrator import DataIngestionOrchestrator
from src.vector_store.vector_store import VectorStore
from openai import OpenAI
import tempfile
import os
from typing import List, Dict
from dotenv import load_dotenv
from src.config.config import get_settings

# Load environment variables
load_dotenv()
settings = get_settings()

# Set page configuration
st.set_page_config(
    page_title="Car Sales Assistant",
    page_icon="🚗",
    layout="wide"
)

# Initialize components
orchestrator = DataIngestionOrchestrator()
vector_store = VectorStore()

# Initialize OpenAI client with API key from settings
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_chat_response(messages: List[Dict], car_data: Dict = None) -> str:
    """Generate a response using OpenAI with context from vector store"""
    try:
        # Get relevant context from vector store if available
        context = ""
        if car_data:
            context = "Here is the relevant car information:\n"
            for section, items in car_data.items():
                if items:
                    context += f"\n{section.upper()}:\n"
                    for item in items:
                        context += f"• {item}\n"

        # Prepare the messages for the API
        system_message = {
            "role": "system",
            "content": """You are an AI car sales assistant. Your goal is to help customers understand car features and make informed decisions.
            Be professional but conversational. Use the provided car information when available.
            If you don't have specific information about something, be honest about it.
            Format your responses in a clear, easy-to-read way using bullet points where appropriate."""
        }
        
        context_message = {
            "role": "system",
            "content": context
        } if context else None
        
        api_messages = [system_message]
        if context_message:
            api_messages.append(context_message)
        api_messages.extend(messages)

        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed to a standard model
            messages=[msg for msg in api_messages if msg is not None],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

async def process_uploaded_file(file, file_type: str):
    """Process uploaded file using the orchestrator"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file.flush()
        
        try:
            result = await orchestrator.process_car_data(tmp_file.name)
            return result
        finally:
            os.unlink(tmp_file.name)

async def render_chat_tab():
    """Render the chat interface"""
    st.header("💬 Chat with AI Sales Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about car features, specifications, or comparisons..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant car data from vector store
        car_data = None
        try:
            results = await vector_store.search_car_data(query=prompt, top_k=5)
            if results:
                car_data = {}
                for result in results:
                    section = result.get('metadata', {}).get('section', 'general')
                    if section not in car_data:
                        car_data[section] = []
                    car_data[section].append(result['text'])
        except Exception as e:
            st.warning(f"Could not retrieve car data: {str(e)}")

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_chat_response(st.session_state.messages, car_data)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

async def render_ingestion_tab():
    """Render the data ingestion tab"""
    st.header("📚 Knowledge Base Management")
    st.write("Upload car brochures to enrich the knowledge base")
    
    # File upload tabs
    upload_tabs = st.tabs(["📄 PDF Upload", "📝 Markdown"])
    
    # PDF Tab
    with upload_tabs[0]:
        uploaded_pdf = st.file_uploader(
            "Upload PDF Brochure",
            type=['pdf'],
            help="Upload a car brochure in PDF format",
            key="pdf_uploader"
        )
        
        if uploaded_pdf:
            try:
                progress_text = "Processing brochure..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Process PDF
                progress_bar.progress(25, text="Analyzing document...")
                result = await process_uploaded_file(uploaded_pdf, 'pdf')
                
                progress_bar.progress(100, text="Complete!")
                st.success("✅ Successfully processed brochure")
                
                # Show extracted information
                car_model = result.pop('car_model', 'Unknown Model')
                st.subheader(f"📍 Extracted Information for {car_model}")
                
                for section, items in result.items():
                    if items:
                        with st.expander(f"🔹 {section.title()}", expanded=True):
                            for item in items:
                                st.write(f"• {item}")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                if 'progress_bar' in locals():
                    progress_bar.empty()
    
    # Markdown Tab
    with upload_tabs[1]:
        uploaded_markdown = st.file_uploader(
            "Upload Markdown File",
            type=['md', 'markdown'],
            help="Upload a markdown file containing car specifications",
            key="markdown_uploader"
        )
        
        if uploaded_markdown:
            try:
                progress_text = "Processing markdown..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Process markdown
                progress_bar.progress(25, text="Analyzing document...")
                result = await process_uploaded_file(uploaded_markdown, 'markdown')
                
                progress_bar.progress(100, text="Complete!")
                st.success("✅ Successfully processed markdown")
                
                # Show extracted information
                car_model = result.pop('car_model', 'Unknown Model')
                st.subheader(f"📍 Extracted Information for {car_model}")
                
                for section, items in result.items():
                    if items:
                        with st.expander(f"🔹 {section.title()}", expanded=True):
                            for item in items:
                                st.write(f"• {item}")
                
            except Exception as e:
                st.error(f"Error processing markdown: {str(e)}")
            finally:
                if 'progress_bar' in locals():
                    progress_bar.empty()

def main():
    st.title("🚗 AI Car Sales Assistant")
    
    # Create tabs
    tabs = st.tabs(["💬 Chat", "📚 Knowledge Base"])
    
    # Chat Tab
    with tabs[0]:
        asyncio.run(render_chat_tab())
    
    # Knowledge Base Tab
    with tabs[1]:
        asyncio.run(render_ingestion_tab())

if __name__ == "__main__":
    main() 