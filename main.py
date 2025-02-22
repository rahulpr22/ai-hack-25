import streamlit as st
import asyncio
from src.data_ingestion.brochure_processor import BrochureProcessor
from src.vector_store import VectorStore
import base64

# Set page configuration
st.set_page_config(
    page_title="Car Sales Assistant",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f2f6;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1fae5;
        border: 1px solid #059669;
    }
    /* File upload styling */
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 1rem 0;
        background-color: #f8f9fa;
        text-align: center;
    }
    
    .upload-header {
        color: #2E7D32;
        font-size: 1.2em;
        margin: 0;
        padding: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .upload-header emoji {
        font-size: 1.4em;
        line-height: 1;
    }
    
    /* Image preview section */
    .image-preview {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
    }
    
    .image-metadata {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    /* Guidelines section */
    .guidelines {
        background-color: #f0f7ff;
        border-left: 4px solid #1976D2;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Hide default file uploader label */
    .stFileUploader > label {
        display: none;
    }
    
    /* Remove extra padding from file uploader */
    .stFileUploader > div[data-testid="stFileUploader"] {
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

async def render_ingestion_tab(processor: BrochureProcessor, vector_store: VectorStore):
    """Render the data ingestion tab"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📚 Knowledge Base Management")
        st.write("Upload car brochures with optional images")
        
        # File upload tabs
        upload_tabs = st.tabs(["📄 PDF Upload", "📝 Markdown"])
        
        # PDF Tab
        with upload_tabs[0]:
            uploaded_pdf = st.file_uploader(
                "Upload PDF Brochure",
                type=['pdf'],
                help="Upload a car brochure in PDF format",
                key="pdf_uploader",
                label_visibility="collapsed"
            )
            
            uploaded_images = st.file_uploader(
                "Upload Additional Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload additional car images",
                key="pdf_images",
                label_visibility="collapsed"
            )
            
            if uploaded_pdf:
                try:
                    with st.spinner("Analyzing PDF brochure..."):
                        try:
                            structured_data = processor.process_brochure(uploaded_pdf, 'pdf')
                        except Exception as e:
                            st.error(f"🚫 {str(e)}")
                            st.info("Please check your OpenAI API key in the .env file")
                            return
                        
                        car_model = uploaded_pdf.name.replace('.pdf', '')
                        
                        # Process optional images if provided
                        images_data = []
                        if uploaded_images:
                            cols = st.columns(3)
                            for idx, image in enumerate(uploaded_images):
                                with cols[idx % 3]:
                                    st.image(image, caption=image.name, use_container_width=True)
                                    with st.container():
                                        st.markdown('<div class="image-metadata">', unsafe_allow_html=True)
                                        image_type = st.selectbox(
                                            "Image Type",
                                            ["Exterior", "Interior", "Feature"],
                                            key=f"pdf_img_type_{idx}"
                                        )
                                        description = st.text_area(
                                            "Description",
                                            key=f"pdf_img_desc_{idx}",
                                            height=100,
                                            placeholder="Enter image description..."
                                        )
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    try:
                                        metadata = {
                                            "type": image_type,
                                            "description": description,
                                            "car_model": car_model
                                        }
                                        image_data = processor.process_image(image, metadata)
                                        images_data.append(image_data)
                                    except Exception as e:
                                        st.error(f"Error processing image: {str(e)}")
                        
                        await vector_store.upsert_car_data(
                            car_data=structured_data,
                            car_model=car_model,
                            images=images_data if images_data else None
                        )
                    
                    st.success("✅ Successfully processed PDF brochure" + 
                             (f" and {len(images_data)} images" if images_data else ""))
                    
                    # Show extracted information
                    for section, items in structured_data.items():
                        if items:
                            with st.expander(f"📍 {section.title()}", expanded=True):
                                for item in items:
                                    st.write(f"• {item}")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        # Markdown Tab
        with upload_tabs[1]:
            uploaded_markdown = st.file_uploader(
                "Upload Markdown File",
                type=['md', 'markdown'],
                help="Upload a markdown file containing car specifications and details",
                key="markdown_uploader",
                label_visibility="collapsed"
            )
            
            uploaded_md_images = st.file_uploader(
                "Upload Additional Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload additional car images",
                key="md_images",
                label_visibility="collapsed"
            )
            
            if uploaded_markdown:
                try:
                    with st.spinner("Analyzing markdown document..."):
                        # Parse markdown content
                        content = uploaded_markdown.getvalue().decode('utf-8')
                        
                        # Extract structured data
                        structured_data = processor.process_markdown(content)
                        car_model = uploaded_markdown.name.replace('.md', '')
                        
                        # Process optional images if provided
                        images_data = []
                        if uploaded_md_images:
                            cols = st.columns(3)
                            for idx, image in enumerate(uploaded_md_images):
                                with cols[idx % 3]:
                                    st.image(image, caption=image.name, use_container_width=True)
                                    with st.container():
                                        st.markdown('<div class="image-metadata">', unsafe_allow_html=True)
                                        image_type = st.selectbox(
                                            "Image Type",
                                            ["Exterior", "Interior", "Feature"],
                                            key=f"md_img_type_{idx}"
                                        )
                                        description = st.text_area(
                                            "Description",
                                            key=f"md_img_desc_{idx}",
                                            height=100,
                                            placeholder="Enter image description..."
                                        )
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    try:
                                        metadata = {
                                            "type": image_type,
                                            "description": description,
                                            "car_model": car_model
                                        }
                                        image_data = processor.process_image(image, metadata)
                                        images_data.append(image_data)
                                    except Exception as e:
                                        st.error(f"Error processing image: {str(e)}")
                        
                        # Store in database
                        await vector_store.upsert_car_data(
                            car_data=structured_data,
                            car_model=car_model,
                            images=images_data if images_data else None
                        )
                    
                    st.success("✅ Successfully processed markdown" + 
                             (f" and {len(images_data)} images" if images_data else ""))
                    
                    # Show extracted information
                    for section, items in structured_data.items():
                        if items:
                            with st.expander(f"📍 {section.title()}", expanded=True):
                                for item in items:
                                    st.write(f"• {item}")
                
                except Exception as e:
                    st.error(f"Error processing markdown: {str(e)}")
    
    with col2:
        st.markdown('<div class="guidelines">', unsafe_allow_html=True)
        st.markdown("""
        ### 📋 Guidelines
        
        **For Brochures:**
        - Use PDF or markdown format
        - Include car specifications
        - Add pricing details
        - List all features
        - Mention safety aspects
        
        **For Images (Optional):**
        - High quality photos
        - Clear visibility
        - Multiple angles
        - Interior & exterior shots
        - Feature highlights
        """)
        st.markdown('</div>', unsafe_allow_html=True)

async def render_chat_tab(vector_store: VectorStore):
    """Render the chat interface tab"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🚗 Car Sales Assistant")
        st.write("Your AI-powered automotive consultant")
        
        # Initialize or get chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hello! I'm your automotive consultant. How can I help you find the perfect car today?"
                }
            ]
    
    with col2:
        st.markdown("""
        ### 💡 Try asking about:
        - Car specifications
        - Price comparisons
        - Safety features
        - Performance details
        - Available colors
        """)
    
    # Chat display area
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about our cars..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            with st.spinner("Searching for information..."):
                results = await vector_store.search_car_data(prompt)
            
            response = "I'd be happy to help you with that!\n\n"
            for result in results:
                response += f"• {result['text']}\n\n"
            response += "\nIs there anything specific you'd like to know more about?"
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Add image search toggle
            show_images = st.toggle("Include images in search", value=False)

            if show_images:
                image_results = await vector_store.search_images(prompt)
                if image_results:
                    st.write("Relevant images:")
                    cols = st.columns(3)
                    for idx, result in enumerate(image_results):
                        with cols[idx % 3]:
                            st.image(
                                base64.b64decode(result["image_data"]),
                                caption=f"{result['car_model']} - {result['type']}",
                                use_container_width=True  # Updated parameter
                            )
            
        except Exception as e:
            st.error("I apologize, but I'm having trouble accessing that information right now.")

async def main():
    st.title("🚗 Automotive Sales Assistant")
    
    try:
        processor = BrochureProcessor()
        vector_store = VectorStore()
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        st.info("Please make sure your API keys are properly set in the .env file")
        return
    
    tab1, tab2 = st.tabs(["💬 Sales Assistant", "⚙️ Knowledge Management"])
    
    with tab1:
        await render_chat_tab(vector_store)
    
    with tab2:
        await render_ingestion_tab(processor, vector_store)

if __name__ == "__main__":
    asyncio.run(main()) 