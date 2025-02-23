import asyncio
import base64
import json
import os

import streamlit as st

from src.config import get_settings
from src.data_ingestion.brochure_processor import BrochureProcessor
from src.data_ingestion.video_processor import VideoProcessor
from src.eleven_labs.conversational_ai import get_agent_id
from src.vector_store import VectorStore
import threading


settings = get_settings()
# Set page configuration
st.set_page_config(
    page_title="Sales Assistant",
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

    video {
        playbackRate: 0.5 !important;
    }
</style>
""", unsafe_allow_html=True)


async def render_ingestion_tab(processor: BrochureProcessor, vector_store: VectorStore, video_processor: VideoProcessor):
    """Render the data ingestion tab"""
    st.header("📚 Knowledge Base Management")
    st.write("Upload car brochures to add product information")
    
    # File upload tabs
    upload_tabs = st.tabs(["📄 PDF Upload"])

    # PDF Tab
    with upload_tabs[0]:
        with st.form("car_details_pdf"):
            product_name = st.text_input(
                "Product Name",
                placeholder="e.g., Toyota Camry XSE V6",
                help="Enter the full name of the product"
            )
            
            product_description = st.text_input(
                "Product Description",
                placeholder="e.g., Premium mid-size sedan with advanced features",
                help="Provide a brief description of the product's key features"
            )
            
            # PDF upload
            uploaded_pdf = st.file_uploader("Upload Brochure as PDF", type="pdf")
            
            submit_button = st.form_submit_button("Process")
            
            if submit_button:
                if not uploaded_pdf:
                    st.error("Please upload a PDF file")
                elif not product_name:
                    st.error("Please fill in the Product Name")
                else:
                    try:
                        # Process PDF
                        with st.spinner("Processing PDF..."):
                            car_data = processor.process_pdf(uploaded_pdf, product_name)

                        # Upsert car data
                        with st.spinner("Storing car data in database..."):
                            await vector_store.upsert_car_data(
                                car_data=car_data,
                                car_model=product_name,
                            )

                        st.success("Successfully processed and stored car information!")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")

import cv2
import os
import time

# Function to get all video files from a folder
def get_video_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Function to play videos continuously
def play_videos(video_files):
    if not video_files:
        st.error("No video files found in the folder!")
        return
    
    video_index = 0
    video_placeholder = st.empty()
    while True:
        video_path = video_files[video_index]
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB")
            time.sleep(1/30)  # Adjust frame rate to match video
        
        cap.release()
        video_index = (video_index + 1) % len(video_files)

async def render_chat_tab():
    """Render the chat interface with video and chat sections"""
    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col2:
        st.markdown("<h3 style='text-align: center;'>🤖 AI Sales Assistant</h3>", unsafe_allow_html=True)
        if "user_name" not in st.session_state:
            st.session_state.user_name = "Karthik"

        # eleven labs conversational ai integration
        agent_id = get_agent_id("New agent")
        dynamic_variables = {
            "user_name": st.session_state.user_name
        }
        widget = f"""
            <elevenlabs-convai 
                agent-id={agent_id}
                dynamic-variables='{json.dumps(dynamic_variables)}'
            </elevenlabs-convai>
            <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
        """
        st.components.v1.html(widget)

    
        # Video Column
    with col1:
        st.subheader("🎥 Product Showcase")
        folder = "videos"  # Set the default folder containing videos
        video_files = get_video_files(folder)
        if video_files:
            play_videos(video_files)
        else:
            st.error("No video files found in the specified folder.")


async def render_video_tab(video_processor: VideoProcessor):
    """Render the video creation tab"""
    st.header("🎥 Create Product Videos")
    st.write("Upload images to create transition videos")

    # Initialize categories in session state if not exists
    if "categories" not in st.session_state:
        # Initialize with one empty category by default
        st.session_state.categories = [{}]

    # Category management section (above form)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Categories")

    # Main form for video creation
    with st.form("video_creation_form", clear_on_submit=False):
        # Product name
        product_name = st.text_input(
            "Product Name",
            placeholder="e.g., Toyota Camry XSE V6",
            help="Enter the product name for the video"
        )

        # Display existing categories
        for idx, _ in enumerate(st.session_state.categories):
            st.subheader(f"Category {idx + 1}")
            category_name = st.text_input(
                "Category Name",
                placeholder="e.g., Exterior Views, Performance Features",
                key=f"category_{idx}_name",
                help="Enter a name for this category of images"
            )
            
            # Add max files info
            st.markdown("_Upload up to 4 images for this category_")
            uploaded_images = st.file_uploader(
                "Upload images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"category_{idx}_images"
            )

            # Initialize image descriptions list
            image_descriptions = []
            
            # Image handling without descriptions
            if uploaded_images:
                if len(uploaded_images) > 4:
                    st.error("⚠️ Please select only 4 images for this category")
                    uploaded_images = uploaded_images[:4]
                
                cols = st.columns(min(len(uploaded_images), 4))
                for img_idx, image in enumerate(uploaded_images):
                    with cols[img_idx]:
                        st.image(image, use_container_width=True)
                        # Add a default description
                        image_descriptions.append({
                            "type": category_name if category_name else f"category_{idx}",
                            "description": f"Image {img_idx + 1} of {category_name if category_name else f'Category {idx + 1}'}"
                        })
            
            # Store the data in session state
            st.session_state.categories[idx] = {
                "name": category_name,
                "images": uploaded_images if uploaded_images else [],
                "descriptions": image_descriptions
            }

        # Buttons row at the bottom
        col1, col2 = st.columns([6, 1])  # Wider first column, narrow second column
        with col1:
            if len(st.session_state.categories) < 4:  # Limit to 4 categories
                if st.form_submit_button("➕ Add New Category", type="secondary"):
                    st.session_state.categories.append({})
                    st.rerun()
        with col2:
            submit = st.form_submit_button("Submit", type="primary", use_container_width=True)

        # Process form submission
        if submit:
            if not product_name:
                st.error("Please enter a product name")
            elif not st.session_state.categories:
                st.error("Please add at least one category")
            else:
                try:
                    # Process each category
                    for idx, category in enumerate(st.session_state.categories):
                        if not category.get("name"):
                            st.error(f"Please provide a name for category {idx + 1}")
                            continue
                        
                        if not category.get("images"):
                            st.warning(f"No images provided for {category['name']}")
                            continue

                        with st.spinner(f"Creating {category['name']} video..."):
                            processed_images = []
                            for img, desc in zip(category["images"], category["descriptions"]):
                                img_bytes = img.read()
                                img_base64 = base64.b64encode(img_bytes).decode()
                                processed_images.append({
                                    "image_data": img_base64,
                                    "metadata": desc
                                })

                            try:
                                await video_processor.create_video_from_images(
                                    images=processed_images,
                                    product_name=f"{product_name}_{category['name'].lower().replace(' ', '_')}",
                                    duration_per_image=3
                                )
                                st.success(f"{category['name']} video created successfully!")
                            except Exception as e:
                                st.error(f"Error creating {category['name']} video: {str(e)}")

                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

async def main():
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    st.title("Sales Assistant")

    try:
        processor = BrochureProcessor()
        vector_store = VectorStore()
        video_processor = VideoProcessor()  # Initialize video processor
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        st.info("Please make sure your API keys are properly set in the .env file")
        return

    tabs = st.tabs(["💬 Chat", "📚 Add a New Product", "🎥 Create Video"])

    with tabs[0]:
        await render_chat_tab()

    with tabs[1]:
        await render_ingestion_tab(processor, vector_store, video_processor)
        
    with tabs[2]:
        await render_video_tab(video_processor)


if __name__ == "__main__":
    asyncio.run(main())