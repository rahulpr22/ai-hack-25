import streamlit as st
from src.pdf_processor import PDFProcessor
from typing import Dict
import json

class BrochureUI:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
    
    def render_header(self):
        """Render the application header"""
        st.title("Brochure PDF Processor")
        st.write("Upload your PDF brochure to extract its content")
    
    def render_file_uploader(self):
        """Render the file upload component"""
        return st.file_uploader("Choose a PDF file", type=['pdf'])
    
    def render_file_details(self, file):
        """Render file details"""
        details = self.pdf_processor.get_file_details(file)
        st.write("File details:")
        for key, value in details.items():
            st.write(f"- {key.capitalize()}: {value}")
    
    def render_content_display(self, content: str):
        """Render the extracted content display"""
        st.subheader("Extracted Content")
        st.text_area("Content", content, height=300)
    
    def render_download_button(self, content: str, filename: str):
        """Render the download button"""
        st.download_button(
            label="Download extracted text",
            data=content,
            file_name=f"{filename}_extracted.txt",
            mime="text/plain"
        )
    
    def render_processing_status(self, status: Dict):
        """Render processing status with analysis"""
        st.success(f"✅ Successfully processed document:")
        st.write(f"- Name: {status['name']}")
        st.write(f"- Size: {status['size']}")
        st.write(f"- Chunks processed: {status['chunks_processed']}")
        
        # Show analysis results
        st.subheader("Document Analysis")
        st.write(f"Document Type: {status['analysis']['document_type'].title()}")
        
        st.subheader("Extracted Information")
        structured_info = json.loads(status['analysis']['structured_info'])
        for key, value in structured_info.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}") 