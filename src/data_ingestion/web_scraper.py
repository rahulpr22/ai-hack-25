import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set, Tuple
import logging
from urllib.parse import urljoin, urlparse, quote
import re
from ..config.config import get_settings
import random
import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = get_settings()

class CarWebScraper:
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.perplexity_headers = {
            'Authorization': f'Bearer {settings.PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.openai_client = OpenAI(api_key=settings.OPENAI_GPT_KEY)
        self.logger.debug("Initialized CarWebScraper with Perplexity and OpenAI clients")
        self.search_urls = {
            "cardekho": {
                "base": "https://www.cardekho.com/cars/{model}",
                "specs": "https://www.cardekho.com/cars/{model}/specifications",
                "features": "https://www.cardekho.com/cars/{model}/features",
                "colors": "https://www.cardekho.com/cars/{model}/colors",
                "search": "https://www.cardekho.com/search/results?q={query}"
            },
            "carwale": {
                "base": "https://www.carwale.com/{model}",
                "specs": "https://www.carwale.com/{model}/specifications",
                "features": "https://www.carwale.com/{model}/features",
                "search": "https://www.carwale.com/search/?q={query}"
            },
            "autocarindia": {
                "base": "https://www.autocarindia.com/cars/{model}",
                "search": "https://www.autocarindia.com/search?q={query}"
            }
        }

    async def __aenter__(self):
        if self.session is None:
            self.logger.debug("Creating new aiohttp ClientSession")
            self.session = aiohttp.ClientSession(headers=self.perplexity_headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            self.logger.debug("Closing aiohttp ClientSession")
            await self.session.close()
            self.session = None

    def _rotate_user_agent(self):
        """Rotate user agent for each request"""
        self.headers['User-Agent'] = random.choice(settings.USER_AGENTS)
        if self.session:
            self.session._default_headers.update(self.headers)

    async def fetch_page(self, url: str, retries: int = settings.MAX_RETRIES) -> Optional[str]:
        """
        Fetch a page with retry logic and user agent rotation
        """
        for attempt in range(retries):
            try:
                self._rotate_user_agent()  # Rotate user agent before each attempt
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Too Many Requests
                        wait_time = (attempt + 1) * settings.SCRAPING_DELAY
                        self.logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(f"Failed to fetch {url}. Status: {response.status}")
                        return None
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(settings.SCRAPING_DELAY)
                else:
                    return None
        return None

    def is_allowed_domain(self, url: str) -> bool:
        """Check if the domain is in the allowed list"""
        domain = urlparse(url).netloc.replace('www.', '')
        return any(allowed in domain for allowed in settings.ALLOWED_DOMAINS)

    async def scrape_car_data(self, car_model: str, brochure_data: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
        """
        Use RAG with Perplexity.ai to gather detailed car information
        """
        self.logger.info(f"Starting car data scraping for model: {car_model}")
        
        car_data = {
            "specifications": [],
            "interior": [],
            "technology": [],
            "exterior": [],
            "safety": [],
            "performance": [],
            "pricing": [],
            "colors": []
        }

        try:
            async with self as scraper:
                # Process brochure data using RAG
                if brochure_data:
                    # Create chunks and embeddings from brochure data
                    chunks = self._create_chunks(brochure_data)
                    chunk_embeddings = self._get_embeddings([chunk["text"] for chunk in chunks])
                    
                    # Generate section-specific queries and retrieve relevant context
                    for section in car_data.keys():
                        query = self._generate_section_query(car_model, section)
                        query_embedding = self._get_embeddings([query])[0]
                        
                        # Get most relevant chunks for this section
                        relevant_chunks = self._get_relevant_chunks(
                            query_embedding, 
                            chunk_embeddings, 
                            chunks, 
                            top_k=3
                        )
                        
                        # Create augmented query with relevant context
                        augmented_query = self._create_augmented_query(
                            car_model,
                            section,
                            relevant_chunks
                        )
                        
                        # Get information from Perplexity with augmented query
                        self.logger.info(f"Sending augmented request for section: {section}")
                        result = await self._query_perplexity(augmented_query)
                        
                        if result:
                            self._parse_section_response(result, car_data, section)
                else:
                    # If no brochure data, use basic query
                    query = self._generate_comprehensive_query(car_model)
                    result = await self._query_perplexity(query)
                    if result:
                        self._parse_and_categorize_response(result, car_data)
        
        except Exception as e:
            self.logger.error(f"Error in scrape_car_data: {str(e)}", exc_info=True)
        
        return car_data

    def _create_chunks(self, brochure_data: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Create meaningful chunks from brochure data
        """
        chunks = []
        for section, items in brochure_data.items():
            # Create section-level chunk
            section_text = f"{section.upper()}:\n" + "\n".join(items)
            chunks.append({
                "text": section_text,
                "section": section
            })
            
            # Create smaller chunks for longer sections
            if len(items) > 5:
                for i in range(0, len(items), 5):
                    chunk_items = items[i:i+5]
                    chunk_text = f"{section.upper()} (Part {i//5 + 1}):\n" + "\n".join(chunk_items)
                    chunks.append({
                        "text": chunk_text,
                        "section": section
                    })
        
        return chunks

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings using OpenAI's embedding model
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [np.array(embedding.embedding) for embedding in response.data]
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            return [np.zeros(1536) for _ in texts]  # Return zero embeddings as fallback

    def _get_relevant_chunks(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray],
        chunks: List[Dict[str, str]],
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Get most relevant chunks using cosine similarity
        """
        similarities = [
            cosine_similarity(query_embedding.reshape(1, -1), 
                            chunk_embedding.reshape(1, -1))[0][0]
            for chunk_embedding in chunk_embeddings
        ]
        
        # Get indices of top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [chunks[i] for i in top_indices]

    def _generate_section_query(self, car_model: str, section: str) -> str:
        """
        Generate a focused query for a specific section
        """
        section_prompts = {
            "specifications": f"What are the technical specifications and dimensions of the {car_model}?",
            "interior": f"What are the interior features and comfort options in the {car_model}?",
            "technology": f"What technology and infotainment features does the {car_model} have?",
            "exterior": f"What are the exterior design features of the {car_model}?",
            "safety": f"What safety features and systems does the {car_model} have?",
            "performance": f"What are the performance capabilities of the {car_model}?",
            "pricing": f"What is the pricing structure for the {car_model}?",
            "colors": f"What colors and finishes are available for the {car_model}?"
        }
        return section_prompts.get(section, f"Tell me about the {section} of the {car_model}")

    def _create_augmented_query(
        self,
        car_model: str,
        section: str,
        relevant_chunks: List[Dict[str, str]]
    ) -> str:
        """
        Create an augmented query using retrieved context
        """
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        return f"""Based on the following information about the {car_model}:

CONTEXT:
{context}

Please provide detailed and accurate information about the {section.upper()} of the {car_model}.
Format the response as bullet points using the • symbol.
Include both the information from the context and any additional relevant details you know about the car.
Focus specifically on {section}-related features and specifications.
"""

    def _parse_section_response(self, content: str, car_data: Dict[str, List[str]], section: str):
        """
        Parse response for a specific section
        """
        for line in content.split('\n'):
            line = line.strip()
            if line and line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line and clean_line not in car_data[section]:
                    car_data[section].append(clean_line)

    def _generate_comprehensive_query(self, car_model: str) -> str:
        """
        Generate a single comprehensive query that covers all aspects
        """
        # Base query structure
        query = f"""Please provide detailed information about the {car_model} car model. Structure your response using these exact headers and bullet points for each feature:

SPECIFICATIONS:
• Engine details
• Dimensions
• Technical specifications

INTERIOR:
• Cabin features
• Comfort options
• Seating

TECHNOLOGY:
• Infotainment
• Connectivity
• Driver assistance

EXTERIOR:
• Design elements
• Styling features
• Body characteristics

SAFETY:
• Safety features
• Ratings
• Protection systems

PERFORMANCE:
• Engine performance
• Driving capabilities
• Efficiency

PRICING:
• Price range
• Trim levels
• Available packages

COLORS:
• Exterior colors
• Interior colors
• Special finishes
"""
        
        return query

    async def _query_perplexity(self, query: str) -> Optional[str]:
        """
        Make a single query to Perplexity.ai API
        """
        try:
            self.logger.debug("Preparing Perplexity API request")
            
            request_data = {
                "model": "sonar", 
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a car information specialist. Provide accurate, detailed information about cars. Use bullet points (•) for listing features and maintain the exact section headers as provided in the query. Keep responses factual and concise."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            self.logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
            
            async with self.session.post(
                'https://api.perplexity.ai/chat/completions',
                json=request_data
            ) as response:
                response_text = await response.text()
                self.logger.debug(f"Response status: {response.status}")
                self.logger.debug(f"Response headers: {response.headers}")
                self.logger.debug(f"Response text: {response_text}")
                
                if response.status == 200:
                    result = json.loads(response_text)
                    return result['choices'][0]['message']['content']
                else:
                    self.logger.error(f"Perplexity API error {response.status}: {response_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error querying Perplexity: {str(e)}", exc_info=True)
            return None

    def _parse_and_categorize_response(self, content: str, car_data: Dict[str, List[str]]):
        """
        Parse the comprehensive response into sections
        """
        current_section = None
        
        # Map section headers to our categories
        section_mapping = {
            "SPECIFICATIONS:": "specifications",
            "INTERIOR:": "interior",
            "TECHNOLOGY:": "technology",
            "EXTERIOR:": "exterior",
            "SAFETY:": "safety",
            "PERFORMANCE:": "performance",
            "PRICING:": "pricing",
            "COLORS:": "colors"
        }
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            for header, category in section_mapping.items():
                if line.upper().startswith(header):
                    current_section = category
                    break
            
            # If not a header and we have a current section, process the line
            if current_section and (line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or not any(header in line.upper() for header in section_mapping.keys())):
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line and clean_line not in car_data[current_section]:
                    car_data[current_section].append(clean_line)

    def _generate_search_urls(self, car_model: str) -> List[str]:
        """Generate search URLs for all sources"""
        model_slug = car_model.lower().replace(' ', '-')
        urls = []
        for source in self.search_urls.values():
            if 'base' in source:
                urls.append(source['base'].format(model=model_slug))
        return urls

    def _extract_specifications(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract car specifications from the page
        """
        specs = []
        spec_patterns = [
            "engine",
            "transmission",
            "horsepower",
            "torque",
            "fuel economy",
            "dimensions",
            "weight"
        ]
        
        for pattern in spec_patterns:
            elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                parent = element.parent
                if parent and parent.text.strip():
                    specs.append(parent.text.strip())
        
        return list(set(specs))  # Remove duplicates

    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract car features from the page
        """
        features = []
        feature_patterns = [
            "features",
            "equipment",
            "technology",
            "comfort",
            "convenience"
        ]
        
        for pattern in feature_patterns:
            elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                parent = element.parent
                if parent:
                    feature_list = parent.find_next('ul')
                    if feature_list:
                        features.extend([li.text.strip() for li in feature_list.find_all('li')])
        
        return list(set(features))

    def _extract_pricing(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract pricing information from the page
        """
        pricing = []
        price_patterns = [
            "msrp",
            "price",
            "cost",
            "invoice"
        ]
        
        for pattern in price_patterns:
            elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                parent = element.parent
                if parent and parent.text.strip():
                    pricing.append(parent.text.strip())
        
        return list(set(pricing)) 