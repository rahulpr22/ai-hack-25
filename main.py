import streamlit as st
import asyncio
import os

from src.data_ingestion import BrochureProcessor
from src.data_ingestion.data_ingestion_orchestrator import DataIngestionOrchestrator
from src.vector_store.vector_store import VectorStore
from openai import OpenAI
import tempfile
from typing import List, Dict
from dotenv import load_dotenv
from src.config.config import get_settings
from src.eleven_labs.conversational_ai import create_conversational_ai, get_agent_id
import json
from src.config import get_settings
from src.data_ingestion.brochure_processor import BrochureProcessor
from src.vector_store import VectorStore
from src.data_ingestion.video_processor import VideoProcessor
import base64

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
</style>
""", unsafe_allow_html=True)


async def render_ingestion_tab(processor: BrochureProcessor, vector_store: VectorStore, video_processor: VideoProcessor):
    """Render the data ingestion tab"""
    st.header("📚 Knowledge Base Management")
    st.write("Upload car brochures with optional images")

    # File upload tabs
    # upload_tabs = st.tabs(["📄 PDF Upload", "📝 Markdown/Text"])
    upload_tabs = st.tabs(["📄 PDF Upload"])

    # PDF Tab
    with upload_tabs[0]:
        # st.subheader("Upload PDF and Images")

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

            # Image upload section with descriptions
            st.write("Upload Images (Maximum 4)")
            uploaded_images = st.file_uploader("Choose images",
                                             type=["jpg", "jpeg", "png"],
                                             accept_multiple_files=True)

            # Image descriptions
            image_descriptions = []
            if uploaded_images:
                if len(uploaded_images) > 4:
                    st.error("Please upload a maximum of 4 images")
                else:
                    # Display uploaded images in a grid with description inputs
                    for idx, image in enumerate(uploaded_images):
                        st.write(f"Image {idx + 1}")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(image, use_container_width=True)
                        with col2:
                            view_type = st.selectbox(
                                "View Type",
                                ["exterior", "interior", "feature", "other"],
                                key=f"view_type_{idx}"
                            )
                            description = st.text_area(
                                "Description",
                                placeholder="Describe what this image shows...",
                                key=f"desc_{idx}",
                                height=100
                            )
                            image_descriptions.append({
                                "type": view_type,
                                "description": description
                            })

            # PDF upload
            uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

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

                        # Upsert car data without videos
                        with st.spinner("Storing car data in database..."):
                            await vector_store.upsert_car_data(
                                car_data=car_data,
                                car_model=product_name,
                            )

                        # Process images if provided
                        processed_images = []
                        image_paths = []  # Track image paths for video creation
                        if uploaded_images and image_descriptions:
                            with st.spinner("Processing images..."):
                                for img, desc in zip(uploaded_images, image_descriptions):
                                    img_bytes = img.read()
                                    img_base64 = base64.b64encode(img_bytes).decode()
                                    processed_images.append({
                                        "image_data": img_base64,
                                        "metadata": {
                                            "type": desc["type"],
                                            "description": desc["description"]
                                        }
                                    })
                                    # Save image temporarily for video creation
                                    temp_path = f"/tmp/{img.name}"
                                    with open(temp_path, "wb") as f:
                                        f.write(base64.b64decode(img_base64))
                                    image_paths.append(temp_path)

                        # Create video from images if available
                        if image_paths:
                            with st.spinner("Creating video from images..."):
                                try:
                                    # Convert image data to expected format
                                    image_data_list = [
                                        {
                                            "image_data": img["image_data"],
                                            "metadata": img["metadata"]
                                        } for img in processed_images
                                    ]

                                    # Create new event loop for video processing
                                    # loop = asyncio.new_event_loop()
                                    # asyncio.set_event_loop(loop)

                                    # Process video in the new loop
                                    await video_processor.create_video_from_images(
                                        images=image_data_list,
                                        duration_per_image=3,
                                        product_name=product_name,
                                    )

                                    # loop.run_until_complete(
                                    #     video_processor.create_video_from_images(
                                    #         images=image_data_list,
                                    #         duration_per_image=3,
                                    #         product_name=product_name,
                                    #     )
                                    # )
                                    # loop.close()
                                except Exception as e:
                                    st.error(f"Error creating video: {str(e)}")
                        
                        st.success("Successfully processed and stored car information!")

                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")


async def render_chat_tab(vector_store: VectorStore):
    """Render the chat interface tab"""
    st.header("🚗 Car Sales Assistant")
    st.write("Your AI-powered automotive consultant")

async def render_chat_tab():
    """Render the chat interface with video and chat sections"""
    # Create two columns
    col1, col2 = st.columns([1, 1])

    # Video Column
    with col1:
        st.subheader("🎥 Product Showcase")

        # Ensure assets directory exists
        assets_dir = "assets"
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)

        # Get available video files
        video_files = [f for f in os.listdir(assets_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            # No videos found
            st.warning("No video files found in assets folder")
            st.info("""
                Please add video files to the 'assets' folder.
                Supported formats: .mp4, .avi, .mov
            """)
            # Display placeholder image or message
            st.image("https://via.placeholder.com/400x300?text=No+Video+Available", use_container_width=True)
        else:
            # Initialize video path in session state if not exists or if current path is invalid
            if ("video_path" not in st.session_state or
                not os.path.exists(st.session_state.video_path)):
                st.session_state.video_path = os.path.join(assets_dir, video_files[0])

            # Add video selector
            selected_video = st.selectbox(
                "Select Video",
                video_files,
                index=video_files.index(os.path.basename(st.session_state.video_path)),
                key="video_selector",
                label_visibility="collapsed"
            )

            # Update video path in session state
            if selected_video:
                st.session_state.video_path = os.path.join(assets_dir, selected_video)

            # Display video from local file
            try:
                with open(st.session_state.video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
                st.info("Please make sure the video file exists and is not corrupted")

    # Chat Widget Column
    with col2:
        st.markdown("<h3 style='text-align: center;'>🤖 AI Sales Assistant</h3>", unsafe_allow_html=True)
        if "user_name" not in st.session_state:
            st.session_state.user_name = "John Doe"

        # eleven labs conversational ai integration
        agent_id = get_agent_id("New agent")
        dynamic_variables = {
            "user_name": "John Doe"
        }
        widget = f"""
            <elevenlabs-convai 
                agent-id={agent_id}
                dynamic-variables='{json.dumps(dynamic_variables)}'
                style="display: flex; justify-content: center; align-items: center;"
            </elevenlabs-convai>
            <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
        """
        columns = st.columns([1, 2, 1.3])
        with columns[0]:
            st.empty()
        with columns[1]:
            st.components.v1.html(widget)
        with columns[2]:
            st.empty()

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
                                use_container_width=True
                            )

            # Add video search toggle
            show_videos = st.toggle("Include videos in search", value=False)

            if show_videos:
                with st.spinner("Fetching and processing videos..."):
                    video_results = await vector_store.search_videos(prompt)
                    if video_results:
                        st.write("Related videos:")
                        for result in video_results:
                            # Create a unique key for each video
                            video_key = f"video_{result['car_model']}_{result['description']}"

                            # Display video using HTML video player
                            video_html = f"""
                            <video width="100%" controls>
                                <source src="data:video/mp4;base64,{result['video_data']}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            """
                            st.markdown(video_html, unsafe_allow_html=True)
                            st.caption(f"{result['car_model']} - {result['description']}")

        except Exception as e:
            st.error("I apologize, but I'm having trouble accessing that information right now.")


async def main():
    st.title("Sales Assistant")

    try:
        processor = BrochureProcessor()
        vector_store = VectorStore()
        video_processor = VideoProcessor()  # Initialize video processor
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        st.info("Please make sure your API keys are properly set in the .env file")
        return

    tabs = st.tabs(["💬 Chat", "📚 Add a New Product"])

    with tabs[0]:
        asyncio.run(render_chat_tab())

    with tabs[1]:
        asyncio.run(render_ingestion_tab())


if __name__ == "__main__":
    main()