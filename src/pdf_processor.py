import PyPDF2
import io
from typing import Dict

class PDFProcessor:
    @staticmethod
    def read_pdf_content(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def get_file_details(file) -> Dict[str, str]:
        """Get file details"""
        return {
            "name": file.name,
            "size": f"{file.size} bytes"
        } 