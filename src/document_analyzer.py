from typing import Dict, List
import openai
import re

class DocumentAnalyzer:
    def __init__(self):
        self.categories = {
            "car": ["model", "price", "specifications", "features", "engine", "mileage", "safety"],
            "property": ["location", "price", "area", "amenities", "contact"],
            "electronics": ["model", "price", "specifications", "features", "warranty", "technical_details"],
            # Add more categories as needed
        }
    
    def detect_document_type(self, text: str) -> str:
        """Detect the type of document using OpenAI"""
        prompt = f"""Analyze this text and determine if it's a brochure for a car, property, or electronics.
        Text: {text[:1000]}...
        Return only one word: car, property, or electronics."""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower()

    def extract_structured_info(self, text: str, doc_type: str) -> Dict:
        """Extract structured information based on document type"""
        categories = self.categories.get(doc_type, [])
        if not categories:
            return {"error": "Unsupported document type"}

        prompt = f"""Extract the following information from this brochure text:
        Categories: {', '.join(categories)}
        
        Text: {text}
        
        Return a JSON object with the found information. If information is not found, mark it as "Not specified".
        Be concise but include all relevant details."""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        return response.choices[0].message.content

    def process_document(self, text: str) -> Dict:
        """Process document and extract structured information"""
        # Detect document type
        doc_type = self.detect_document_type(text)
        
        # Extract structured information
        structured_info = self.extract_structured_info(text, doc_type)
        
        return {
            "document_type": doc_type,
            "structured_info": structured_info
        } 