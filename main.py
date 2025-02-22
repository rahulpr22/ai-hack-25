import streamlit as st
from src.ui.components import BrochureUI
from src.pdf_processor import PDFProcessor

def main():
    # Initialize components
    ui = BrochureUI()
    pdf_processor = PDFProcessor()
    
    # Render header
    ui.render_header()
    
    # File upload
    uploaded_file = ui.render_file_uploader()
    
    if uploaded_file is not None:
        try:
            # Show file details
            ui.render_file_details(uploaded_file)
            
            # Extract and display text
            text_content = pdf_processor.read_pdf_content(uploaded_file)
            ui.render_content_display(text_content)
            
            # Add download button
            ui.render_download_button(text_content, uploaded_file.name)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 