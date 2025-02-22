from typing import Dict, List, Optional
import markdown
from bs4 import BeautifulSoup
import re
from pathlib import Path
import io
from PIL import Image
import base64
from openai import OpenAI
import logging
import PyPDF2
from ..config.config import get_settings

class BrochureProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None  # Initialize later when needed
        self.settings = get_settings()  # Get settings in constructor        
        self.sections = {
            "specifications": [],
            "interior": [],
            "technology": [],
            "exterior": [],
            "safety": [],
            "performance": [],
            "pricing": [],
            "colors": []
        }
        
        self.extraction_prompt = """Extract detailed car information from this text into these categories:
        1. Specifications (engine, dimensions, capacity, etc.)
        2. Interior & Comfort features
        3. Technology features
        4. Exterior features
        5. Safety features
        6. Performance details
        7. Pricing information
        8. Available colors

        Format the information in clear, structured bullet points. If certain information is not found, skip that category.
        Be very specific and accurate with technical details, numbers, and features."""

    def _ensure_client(self):
        """Ensure OpenAI client is initialized"""
        if not self.client:
            try:
                # Check if API key exists and has correct format
                if not self.settings.OPENAI_API_KEY or not self.settings.OPENAI_API_KEY.startswith('sk-'):
                    raise Exception("Invalid OpenAI API key format")
                    
                self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                # Test the connection
                self.client.models.list()
            except Exception as e:
                raise Exception("Failed to initialize OpenAI client. Please check your API key.")

    def process_brochure(self, file, file_type: str) -> Dict[str, List[str]]:
        """Process brochure file (PDF or Markdown)"""
        if file_type == 'pdf':
            return self._process_pdf(file)
        elif file_type == 'markdown':
            content = file.getvalue().decode('utf-8')
            return self.process_markdown(content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _process_pdf(self, file) -> Dict[str, List[str]]:
        """Process PDF using PyPDF2 and Vision API"""
        try:
            # Initialize OpenAI client if needed
            self._ensure_client()
            
            # First try regular text extraction with PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            if not full_text.strip():
                raise Exception("No text could be extracted from the PDF")
            
            try:
                # Use AI to extract structured information
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": self.extraction_prompt + "\n\nText:\n" + full_text
                    }],
                    temperature=0.5
                )
            except Exception as e:
                raise Exception(f"OpenAI API error: {str(e)}")
            
            # Parse AI response into sections
            extracted_info = {key: [] for key in self.sections.keys()}
            current_section = None
            
            for line in response.choices[0].message.content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Determine section
                section = self._determine_section(line.lower())
                if section:
                    current_section = section
                elif current_section and line.startswith(('•', '-', '*')):
                    # Clean and add the bullet point
                    clean_line = line.lstrip('•-* ').strip()
                    if clean_line and clean_line not in extracted_info[current_section]:
                        extracted_info[current_section].append(clean_line)
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def process_markdown(self, markdown_content: str) -> Dict[str, List[str]]:
        """Process markdown content and extract structured information"""
        # Reset sections for new processing
        self.sections = {key: [] for key in self.sections.keys()}
        
        # Convert markdown to HTML for easier parsing
        html_content = markdown.markdown(markdown_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract sections based on headers
        current_section = None
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                current_section = self._determine_section(element.text.lower())
            elif current_section and element.text.strip():
                if element.name == 'ul':
                    items = [li.text.strip() for li in element.find_all('li')]
                    self.sections[current_section].extend(items)
                else:
                    self.sections[current_section].append(element.text.strip())
        
        return self.sections

    def _determine_section(self, text: str) -> Optional[str]:
        """Determine which section a text belongs to"""
        text = text.lower()
        
        if any(word in text for word in ['spec', 'dimension', 'engine', 'technical']):
            return 'specifications'
        elif any(word in text for word in ['interior', 'comfort', 'cabin']):
            return 'interior'
        elif any(word in text for word in ['technology', 'digital', 'connectivity']):
            return 'technology'
        elif any(word in text for word in ['exterior', 'design', 'style']):
            return 'exterior'
        elif any(word in text for word in ['safety', 'security', 'assist']):
            return 'safety'
        elif any(word in text for word in ['performance', 'handling', 'efficiency']):
            return 'performance'
        elif any(word in text for word in ['price', 'cost', 'msrp']):
            return 'pricing'
        elif any(word in text for word in ['color', 'paint', 'shade']):
            return 'colors'
        return None

    def process_image(self, image_file, metadata: Dict) -> Dict:
        """Process image and prepare for storage"""
        try:
            # Read and preprocess image
            image = Image.open(image_file).convert('RGB')
            
            # Resize if image is too large while maintaining aspect ratio
            image.thumbnail((800, 800))
            
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Convert dimensions tuple to string
            width, height = image.size
            dimensions = f"{width}x{height}"
            
            return {
                "image_data": img_str,
                "metadata": {
                    **metadata,
                    "dimensions": dimensions,
                    "type": "image"
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}") 