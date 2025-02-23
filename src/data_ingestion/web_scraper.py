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
        # Add competitor analysis sections to car_data structure
        self.default_sections = {
            "specifications": [],
            "interior": [],
            "technology": [],
            "exterior": [],
            "safety": [],
            "performance": [],
            "pricing": [],
            "colors": [],
            "competitors_same_segment": [],
            "competitors_upper_segment": [],
            "competitors_lower_segment": [],
            "comparative_analysis": []
        }
        self.search_urls = {
            "cardekho": {
                "base": "https://www.cardekho.com/cars/{model}",
                "specs": "https://www.cardekho.com/cars/{model}/specifications",
                "features": "https://www.cardekho.com/cars/{model}/features",
                "colors": "https://www.cardekho.com/cars/{model}/colors",
                "search": "https://www.cardekho.com/search/results?q={query}",
                "compare": "https://www.cardekho.com/compare-cars/{models}"
            },
            "carwale": {
                "base": "https://www.carwale.com/{model}",
                "specs": "https://www.carwale.com/{model}/specifications",
                "features": "https://www.carwale.com/{model}/features",
                "search": "https://www.carwale.com/search/?q={query}",
                "compare": "https://www.carwale.com/compare-cars/{models}"
            },
            "autocarindia": {
                "base": "https://www.autocarindia.com/cars/{model}",
                "search": "https://www.autocarindia.com/search?q={query}",
                "compare": "https://www.autocarindia.com/car-comparison/{models}"
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
        Use RAG with Perplexity.ai to gather detailed car information including competitor analysis
        """
        self.logger.info(f"Starting car data scraping for model: {car_model}")
        
        car_data = self.default_sections.copy()

        try:
            async with self as scraper:
                # Get competitor information first
                competitor_info = await self._get_competitor_info(car_model)
                if competitor_info:
                    car_data["competitors_same_segment"] = competitor_info.get("same_segment", [])
                    car_data["competitors_upper_segment"] = competitor_info.get("upper_segment", [])
                    car_data["competitors_lower_segment"] = competitor_info.get("lower_segment", [])

                    # Generate comparative analysis
                    comparative_query = f"""Based on the competitor analysis, provide a detailed comparative analysis of {car_model} including:
                    • Overall market positioning
                    • Value proposition
                    • Competitive advantages and disadvantages
                    • Best use cases and target audience
                    • Price-to-feature ratio analysis
                    • Long-term ownership comparison
                    • Resale value comparison
                    • Customer satisfaction comparison
                    Format as detailed bullet points."""

                    comparative_analysis = await self._query_perplexity(comparative_query)
                    if comparative_analysis:
                        car_data["comparative_analysis"] = [line.lstrip('• ').strip() 
                            for line in comparative_analysis.split('\n') 
                            if line.strip().startswith('•')]

                # Process brochure data using RAG
                if brochure_data:
                    chunks = self._create_chunks(brochure_data)
                    chunk_embeddings = self._get_embeddings([chunk["text"] for chunk in chunks])
                    
                    for section in car_data.keys():
                        if section not in ["competitors_same_segment", "competitors_upper_segment", 
                                         "competitors_lower_segment", "comparative_analysis"]:
                            query = self._generate_section_query(car_model, section)
                            query_embedding = self._get_embeddings([query])[0]
                            
                            relevant_chunks = self._get_relevant_chunks(
                                query_embedding, 
                                chunk_embeddings, 
                                chunks, 
                                top_k=3
                            )
                            
                            augmented_query = self._create_augmented_query(
                                car_model,
                                section,
                                relevant_chunks
                            )
                            
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
        Generate a focused query for a specific section with detailed sub-questions
        """
        section_prompts = {
            "specifications": f"""Provide extremely detailed technical specifications for the {car_model}, including:
• Complete engine specifications (displacement, cylinders, valvetrain, compression ratio)
• Detailed dimensions (length, width, height, wheelbase, track width, ground clearance)
• Transmission details (type, gear ratios, shift mechanism)
• Chassis and suspension specifications
• Brake system specifications
• Wheel and tire specifications
• Weight details (kerb weight, gross weight, payload)
• Fuel system specifications
• Performance figures (power, torque, acceleration, top speed)""",

            "interior": f"""Describe in detail all interior features of the {car_model}, including:
• Seating configuration and materials
• Interior dimensions (headroom, legroom, shoulder room)
• Dashboard and instrument panel layout
• Storage solutions and cargo space
• Climate control system
• Interior lighting
• Sound insulation and NVH levels
• Premium features and materials
• Ergonomics and comfort features""",

            "technology": f"""List all technology features in the {car_model}, including:
• Infotainment system specifications
• Display screens and their capabilities
• Connectivity options (Bluetooth, USB, wireless)
• Sound system details
• Driver assistance technologies
• Navigation system
• Voice control capabilities
• Smartphone integration
• Digital services and connected features""",

            "exterior": f"""Detail all exterior features of the {car_model}, including:
• Design language and styling elements
• Lighting system (headlamps, DRLs, tail lamps)
• Aerodynamic features
• Body construction and materials
• Paint technology and protection
• Glass specifications
• Door and mirror features
• Wheel designs and sizes
• Special exterior packages""",

            "safety": f"""Provide comprehensive safety information for the {car_model}, including:
• Active safety systems
• Passive safety features
• Airbag configuration and coverage
• Crash test ratings and results
• Child safety features
• Security systems
• Emergency response features
• Driver assistance safety features
• Structural safety elements""",

            "performance": f"""Detail the complete performance capabilities of the {car_model}, including:
• Engine performance characteristics
• Acceleration and speed figures
• Handling and dynamics
• Braking performance
• Fuel efficiency
• Driving modes
• Performance upgrades
• Real-world performance data
• Track/off-road capabilities""",

            "pricing": f"""Provide detailed pricing information for the {car_model}, including:
• Trim level pricing
• Optional package costs
• Warranty and service package pricing
• Insurance cost estimates
• Financing options
• Maintenance cost estimates
• Resale value projections
• Regional price variations
• Current offers and incentives""",

            "colors": f"""Detail all color options for the {car_model}, including:
• Available exterior colors
• Interior color schemes
• Special edition colors
• Color pricing differences
• Popular color choices
• Color-specific features
• Paint protection options
• Special finish options
• Color availability by trim"""
        }
        return section_prompts.get(section, f"Tell me everything about the {section} of the {car_model}")

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
        Generate a comprehensive query that covers all possible customer questions
        """
        return f"""Please provide extremely detailed information about the {car_model} car model, answering every possible question a customer might ask a car salesman. Structure your response using these exact headers and bullet points:

SPECIFICATIONS:
• What are the exact engine specifications (displacement, cylinders, configuration)?
• What are the detailed dimensions (length, width, height, wheelbase, ground clearance)?
• What is the fuel tank capacity and fuel efficiency (city/highway)?
• What are the transmission options and their gear ratios?
• What is the boot space/cargo capacity?
• What is the kerb weight and gross vehicle weight?
• What are the brake specifications (front/rear)?
• What are the tire and wheel specifications?

INTERIOR:
• What materials are used for the seats and dashboard?
• How many passengers can it accommodate?
• What is the legroom and headroom in front and rear seats?
• What are the seat adjustment options?
• What storage compartments are available?
• What is the quality of interior fit and finish?
• What interior color options are available?
• What are the comfort features for different weather conditions?

TECHNOLOGY:
• What is the infotainment system specification?
• What smartphone connectivity options are available?
• What is the sound system configuration?
• What driver assistance features are included?
• What are the display screens and their sizes?
• What are the charging/USB port locations?
• What smart/connected car features are available?
• What are the instrument cluster features?

EXTERIOR:
• What are the signature design elements?
• What are the lighting specifications (headlamps, taillamps, DRLs)?
• What aerodynamic features are incorporated?
• What are the special exterior styling elements?
• What are the window and sunroof specifications?
• What are the door handle and mirror features?
• What are the wheel design options?
• What are the paint quality and protection features?

SAFETY:
• What active safety features are included?
• What passive safety features are present?
• What are the airbag configurations?
• What are the crash test ratings?
• What child safety features are available?
• What security features are included?
• What are the brake assist and stability features?
• What are the parking assistance features?

PERFORMANCE:
• What is the engine power and torque output?
• What is the acceleration (0-60/100) time?
• What is the top speed?
• What are the different driving modes?
• How does it handle in different conditions?
• What is the braking performance?
• What is the suspension setup?
• What is the fuel efficiency in real-world conditions?

PRICING:
• What are the different trim levels and their prices?
• What warranty packages are available?
• What service packages are offered?
• What are the insurance costs?
• What are the available financing options?
• What are the maintenance costs?
• What are the optional package costs?
• What are the dealership offers and discounts?

COLORS:
• What exterior colors are available?
• What interior color combinations are offered?
• What are the premium/special edition colors?
• What are the most popular color choices?
• What are the color-specific maintenance tips?
• What are the color pricing differences?
• What are the color availability by trim level?
• What are the special finish options?

OWNERSHIP EXPERIENCE:
• What is the expected resale value?
• What are the maintenance intervals?
• What are common owner reviews and feedback?
• What are the pros and cons compared to competitors?
• What are the known issues or recalls?
• What are the customization options?
• What are the extended warranty options?
• What is the dealer service experience?

COMPETITIVE COMPARISON:
• How does it compare to direct competitors in the same price range?
• What are the key advantages over competitors?
• What are the disadvantages compared to competitors?
• How does the price-to-feature ratio compare with competitors?
• What features are class-exclusive or best-in-segment?
• What features are missing compared to competitors?
• How does the warranty compare to competitors?
• How does the service network compare to competitors?
• What is the resale value compared to competitors?
• How does the build quality compare?
• What are the reliability ratings compared to competitors?
• How do maintenance costs compare?
• What are expert opinions on this model vs competitors?
• How does it rank in independent comparison tests?
• What are customer satisfaction ratings compared to competitors?
• How does the brand reputation compare?

SEGMENT ANALYSIS:
• What is the exact segment positioning?
• What are all the alternatives in the same price range?
• What premium alternatives are available in the segment above?
• What budget alternatives are available in the segment below?
• How does it compare to segment leaders?
• What is the target audience compared to competitors?
• What are the segment-specific features?
• How has it evolved compared to previous models?
• What are the expected segment trends?
• How does it align with segment expectations?

VALUE PROPOSITION:
• What is the overall value for money?
• What are the unique selling points?
• Who is the ideal buyer for this model?
• What are the long-term ownership benefits?
• What are the lifestyle and status implications?
• How does it meet different user needs?
• What are the practical advantages for daily use?
• What are the special use case benefits?
• What makes it stand out in its segment?
• What type of buyers should consider alternatives?

Format each point as a detailed bullet point with the • symbol. Include both the information from official sources and real-world experiences. Be specific with comparisons and include actual model names and features when comparing."""

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
                        "content": """You are an expert automotive information specialist with deep knowledge of cars, their features, and the automotive industry. Your role is to:

1. Provide extremely detailed, accurate, and comprehensive information about cars
2. Include both official specifications and real-world experiences/reviews
3. Be specific with numbers, measurements, and technical details
4. Compare features with competitor vehicles when relevant
5. Include both positive aspects and potential drawbacks
6. Provide practical insights that would be valuable to potential buyers
7. Structure information clearly using bullet points with the • symbol
8. Maintain consistent formatting and section organization
9. Focus on factual information rather than marketing language
10. Include regional variations when applicable (features, pricing, availability)

Always verify information from multiple sources and indicate if certain details might vary by market or model year."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.5,
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

    async def _get_competitor_info(self, car_model: str) -> Dict[str, List[Dict]]:
        """
        Get information about competitors in same, upper, and lower segments
        """
        try:
            # First, get the segment and price information for our car
            segment_query = f"""What is the exact segment and price range of {car_model}? 
            Also list all direct competitors in the same price segment, 
            competitors in one segment above (premium alternatives), 
            and competitors in one segment below (budget alternatives). 
            Format as bullet points with exact price ranges."""
            
            segment_info = await self._query_perplexity(segment_query)
            if not segment_info:
                return {}

            # Now get detailed comparison for each competitor
            competitors = {
                "same_segment": [],
                "upper_segment": [],
                "lower_segment": []
            }

            for segment_type in competitors.keys():
                comparison_query = f"""Compare {car_model} with its {segment_type.replace('_', ' ')} competitors.
                For each competitor, provide:
                • Model name and price range
                • Key advantages over {car_model}
                • Key disadvantages compared to {car_model}
                • Unique features and selling points
                • Performance comparison
                • Feature-by-feature comparison
                • Value for money analysis
                • Customer satisfaction and reliability comparison
                Format as detailed bullet points."""

                comparison_data = await self._query_perplexity(comparison_query)
                if comparison_data:
                    competitors[segment_type] = self._parse_competitor_data(comparison_data)

            return competitors

        except Exception as e:
            self.logger.error(f"Error getting competitor info: {str(e)}")
            return {}

    def _parse_competitor_data(self, comparison_data: str) -> List[Dict]:
        """
        Parse competitor comparison data into structured format
        """
        competitors = []
        current_competitor = {}
        current_section = None

        for line in comparison_data.split('\n'):
            line = line.strip()
            if not line:
                if current_competitor:
                    competitors.append(current_competitor)
                    current_competitor = {}
                continue

            if line.startswith('•'):
                item = line.lstrip('• ').strip()
                if ':' in item:
                    key, value = item.split(':', 1)
                    current_competitor[key.strip().lower().replace(' ', '_')] = value.strip()
                else:
                    if current_section:
                        if current_section not in current_competitor:
                            current_competitor[current_section] = []
                        current_competitor[current_section].append(item)
            else:
                current_section = line.lower().replace(' ', '_')

        if current_competitor:
            competitors.append(current_competitor)

        return competitors 