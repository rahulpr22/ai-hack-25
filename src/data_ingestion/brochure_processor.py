from typing import Dict, List, Optional
import markdown
from bs4 import BeautifulSoup
import re
from pathlib import Path

class BrochureProcessor:
    def __init__(self):
        self.sections = {
            "specifications": [],
            "features": [],
            "pricing": [],
            "performance": [],
            "safety": []
        }
    
    def process_markdown(self, markdown_content: str) -> Dict[str, List[str]]:
        """
        Process markdown content and extract structured information
        """
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
    
    def _determine_section(self, header_text: str) -> Optional[str]:
        """
        Determine which section a header belongs to
        """
        header_text = header_text.lower()
        
        if any(word in header_text for word in ['spec', 'dimension', 'engine', 'technical']):
            return 'specifications'
        elif any(word in header_text for word in ['feature', 'equipment', 'interior', 'exterior']):
            return 'features'
        elif any(word in header_text for word in ['price', 'cost', 'msrp']):
            return 'pricing'
        elif any(word in header_text for word in ['performance', 'handling', 'efficiency']):
            return 'performance'
        elif any(word in header_text for word in ['safety', 'security', 'assist']):
            return 'safety'
        return None
    
    def extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from text using common patterns
        """
        pairs = {}
        # Pattern for "key: value" or "key - value"
        patterns = [
            r'([^:]+):\s*(.+)',
            r'([^-]+)-\s*(.+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key = match.group(1).strip()
                value = match.group(2).strip()
                pairs[key] = value
        
        return pairs
    
    def save_structured_data(self, output_path: Path):
        """
        Save the structured data to a file
        """
        with open(output_path, 'w') as f:
            for section, items in self.sections.items():
                f.write(f"## {section.upper()}\n")
                for item in items:
                    f.write(f"- {item}\n")
                f.write("\n") 