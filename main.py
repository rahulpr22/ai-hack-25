import streamlit as st
from src.ui.components import BrochureUI
from src.pdf_processor import PDFProcessor

def main():
    # Initialize components
    ui = BrochureUI()
    
    # Render header
    ui.render_header()
    
    # File upload
    uploaded_file = ui.render_file_uploader()
    
    if uploaded_file is not None:
        try:
            # Process and store document
            status = ui.pdf_processor.process_and_store(uploaded_file)
            
            # Show processing status
            ui.render_processing_status(status)
            
            # Show content and download button
            text_content = ui.pdf_processor.read_pdf_content(uploaded_file)
            ui.render_content_display(text_content)
            ui.render_download_button(text_content, uploaded_file.name)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 