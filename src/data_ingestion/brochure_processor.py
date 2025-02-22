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

settings = get_settings()  # Get settings in constructor


class BrochureProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None  # Initialize later when needed
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
                logging.info("api keys", settings.OPENAI_GPT_KEY, settings.PINECONE_API_KEY, settings.OPENAI_GPT_KEY)
                if not settings.OPENAI_GPT_KEY or not settings.OPENAI_GPT_KEY.startswith('sk-'):
                    raise Exception("Invalid OpenAI API key format")
                    
                self.client = OpenAI(api_key=settings.OPENAI_GPT_KEY)
                # Test the connection
                self.client.models.list()
            except Exception as e:
                raise Exception("Failed to initialize OpenAI client. Please check your API key.")

    def process_brochure(self, file, file_type: str, product_name: str) -> Dict[str, List[str]]:
        """Process brochure file (PDF or Markdown)"""
        if file_type == 'pdf':
            return self.process_pdf(file, product_name)
        elif file_type == 'markdown':
            content = file.getvalue().decode('utf-8')
            return self.process_markdown(content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def process_pdf(self, file, product_name) -> Dict[str, List[str]]:
        """Process PDF using PyPDF2 and Vision API"""
        try:
            # Initialize OpenAI client if needed
            self._ensure_client()
            
            # First try regular text extraction with PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = f"Product Name: {product_name} "
            
            # Extract text from all pages and do initial preprocessing
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    # Clean and preprocess the text
                    text = self._preprocess_text(text)
                    full_text += text + "\n\n"
            
            if not full_text.strip():
                raise Exception("No text could be extracted from the PDF")

            # Extract car model first
            car_model = self.extract_car_model(full_text)
            if not car_model:
                raise ValueError("Could not extract car model from content")
            
            # Reset sections
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

            # Extract key information first using regex patterns
            extracted_info = self._extract_key_info(full_text)
            
            # Create a summarized version for OpenAI
            summarized_text = self._create_summary(full_text, extracted_info)
            
            try:
                # Use AI to extract structured information from the summary
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": self.extraction_prompt + "\n\nText:\n" + summarized_text
                    }],
                    temperature=0.5
                )
            except Exception as e:
                raise Exception(f"OpenAI API error: {str(e)}")
            
            # Parse AI response into sections
            ai_extracted = self._parse_ai_response(response.choices[0].message.content)
            
            # Merge AI-extracted information with sections
            self._merge_extracted_info(ai_extracted)
            return self.sections
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep necessary punctuation
        text = re.sub(r'[^\w\s.,;:()\-]', '', text)
        return text.strip()

    def _extract_key_info(self, text: str) -> Dict[str, List[str]]:
        """Extract key information using regex patterns"""
        extracted = {
            "specifications": [],
            "pricing": [],
            "colors": []
        }
        
        # Example patterns (expand these based on your needs)
        spec_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:L|cc|hp|kW)',  # Engine specs
            r'(\d+(?:\.\d+)?)\s*(?:mph|km/h)',    # Speed
            r'(\d+(?:\.\d+)?)\s*(?:in|mm|cm)'     # Dimensions
        ]
        
        price_patterns = [
            # Dollar
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'USD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Euro
            r'€\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'EUR\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Rupee
            r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'INR\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Pound
            r'£\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'GBP\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Generic price patterns
            r'MSRP.*?([₹\$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Price.*?([₹\$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Cost.*?([₹\$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        color_patterns = [
            r'(?:available|exterior)\s+colors?[:\s]+([^.]+)',
            r'(?:comes|available)\s+in\s+([^.]+?(?:black|white|silver|blue|red|gray|grey)[^.]+)'
        ]
        
        # Extract matches for each pattern
        for pattern in spec_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted["specifications"].extend(matches)
        
        # Extract price matches
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Clean up the matches and add currency symbol if not present
            for match in matches:
                if match:
                    extracted["pricing"].append(match.strip())
        
        # Extract color matches
        for pattern in color_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_color = match.strip().rstrip('.,')
                if clean_color:
                    extracted["colors"].append(clean_color)
        
        return extracted

    def _create_summary(self, full_text: str, extracted_info: Dict[str, List[str]]) -> str:
        """Create a condensed summary of the text for OpenAI"""
        # Start with extracted information
        summary_parts = []
        
        # Add extracted specs
        if extracted_info["specifications"]:
            summary_parts.append("Specifications: " + ", ".join(extracted_info["specifications"]))
        
        # Add first few paragraphs (likely containing important information)
        paragraphs = full_text.split('\n\n')[:3]
        summary_parts.extend(paragraphs)
        
        # Add any sections with key terms
        key_terms = ['safety', 'technology', 'interior', 'exterior', 'performance']
        for para in paragraphs[3:]:
            if any(term in para.lower() for term in key_terms):
                summary_parts.append(para)
        
        return "\n\n".join(summary_parts)

    def extract_car_model(self, content: str) -> str:
        """
        Extract car model information from the content
        """
        # Try to find car model in markdown headers
        headers = re.findall(r'^#\s+(.+?)(?:\s+-|\n|$)', content, re.MULTILINE)
        if headers:
            # Look for car model patterns in the first header
            model_match = re.search(r'(\d{4}\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*)', headers[0])
            if model_match:
                return model_match.group(2).strip()
        
        # If not found in headers, try to extract from content
        model_patterns = [
            r'Model:\s*([A-Za-z0-9\s]+)',
            r'Car Model:\s*([A-Za-z0-9\s]+)',
            r'Vehicle:\s*([A-Za-z0-9\s]+)',
            r'Product Name:\s*([A-Za-z0-9\s]+)',
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, use AI to extract model name
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract only the car model name from the following text. Respond with just the model name, nothing else."},
                    {"role": "user", "content": content[:1000]}  # Use first 1000 chars
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Failed to extract car model using AI: {str(e)}")
            return None

    def process_markdown(self, content: str) -> Dict[str, List[str]]:
        """
        Process markdown content and extract car information
        """
        # Extract car model first
        car_model = self.extract_car_model(content)
        if not car_model:
            raise ValueError("Could not extract car model from content")
            
        # Reset sections
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
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Process each section
        current_section = None
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                current_section = self._determine_section(element.text.strip().lower())
            elif current_section and element.text.strip():
                if element.name == 'ul':
                    items = [li.text.strip() for li in element.find_all('li')]
                    self.sections[current_section].extend(items)
                else:
                    self.sections[current_section].append(element.text.strip())
        
        # Use AI to enhance and categorize the information
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.extraction_prompt},
                    {"role": "user", "content": content}
                ]
            )
            
            # Parse and merge AI-extracted information
            ai_extracted = self._parse_ai_response(response.choices[0].message.content)
            self._merge_extracted_info(ai_extracted)
            
        except Exception as e:
            self.logger.error(f"Failed to enhance content with AI: {str(e)}")
        
        return {"car_model": car_model, **self.sections}

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

    def _parse_ai_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse the AI response into structured sections"""
        sections = {
            "specifications": [],
            "interior": [],
            "technology": [],
            "exterior": [],
            "safety": [],
            "performance": [],
            "pricing": [],
            "colors": []
        }
        
        current_section = None
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            section = self._determine_section(line)
            if section:
                current_section = section
            elif current_section and line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # Clean and add the item
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line and clean_line not in sections[current_section]:
                    sections[current_section].append(clean_line)
        
        return sections

    def _merge_extracted_info(self, ai_extracted: Dict[str, List[str]]):
        """Merge AI-extracted information with existing sections"""
        for section, items in ai_extracted.items():
            if section in self.sections:
                # Create a set of existing items (lowercase for case-insensitive comparison)
                existing_items = {item.lower() for item in self.sections[section]}
                
                # Add new items if they don't exist
                for item in items:
                    if item.lower() not in existing_items:
                        self.sections[section].append(item)
                        existing_items.add(item.lower())

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