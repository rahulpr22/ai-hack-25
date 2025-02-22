import streamlit as st
from src.pdf_processor import PDFProcessor
from typing import Dict
import json

class BrochureUI:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
    
    def render_header(self):
        """Render the application header"""
        st.title("Car Brochure Analyzer")
        st.write("Upload a car brochure PDF to extract structured information")
    
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
        st.subheader("Raw Content")
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
        st.write(f"- Sections processed: {status['sections_processed']}")
        
        # Show analysis results
        st.subheader("Document Analysis")
        st.write(f"Document Type: {status['analysis']['document_type'].title()}")
        
        st.subheader("Extracted Information")
        structured_info = status['analysis']['structured_info']
        # Don't parse if already a dict
        if isinstance(structured_info, str):
            structured_info = json.loads(structured_info)
        
        # Display each section with proper formatting
        for section, data in structured_info.items():
            st.write(f"**{section.replace('_', ' ').title()}**")
            if isinstance(data, dict):
                for key, value in data.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            elif isinstance(data, list):
                for item in data:
                    st.write(f"- {item}")
            else:
                st.write(f"- {data}") 