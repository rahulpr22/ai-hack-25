import PyPDF2
import io
from typing import Dict, List
from .vector_store import VectorStore
from .document_analyzer import DocumentAnalyzer

class PDFProcessor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.analyzer = DocumentAnalyzer()
    
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def process_and_store(self, file) -> Dict:
        """Process PDF and store in vector DB with structured information"""
        # Extract text
        text = self.read_pdf_content(file)
        
        # Analyze document and extract structured information
        analysis = self.analyzer.process_document(text)
        
        # Store in vector DB with enhanced metadata
        metadata = {
            "source": file.name,
            "chunk_id": 0,
            "document_type": analysis["document_type"],
            "structured_info": analysis["structured_info"]
        }
        
        # self.vector_store.store_document(text, metadata)
        
        return {
            "chunks_processed": 1,
            "name": file.name,
            "size": f"{file.size} bytes",
            "analysis": analysis
        }

    @staticmethod
    def get_file_details(file) -> Dict[str, str]:
        """Get file details"""
        return {
            "name": file.name,
            "size": f"{file.size} bytes"
        } 