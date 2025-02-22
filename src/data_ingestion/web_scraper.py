import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging
from urllib.parse import urljoin, urlparse
import re
from ..config.config import get_settings

settings = get_settings()

class CarWebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str, retries: int = settings.MAX_RETRIES) -> Optional[str]:
        """
        Fetch a page with retry logic
        """
        for attempt in range(retries):
            try:
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
        """
        Check if the URL belongs to allowed domains
        """
        domain = urlparse(url).netloc
        return any(allowed_domain in domain for allowed_domain in settings.ALLOWED_DOMAINS)

    async def scrape_car_data(self, car_model: str) -> Dict[str, List[str]]:
        """
        Scrape car data from multiple sources
        """
        car_data = {
            "specifications": [],
            "features": [],
            "reviews": [],
            "pricing": [],
            "safety_ratings": []
        }

        search_urls = self._generate_search_urls(car_model)
        
        async with self:  # Use context manager for session handling
            tasks = [self.process_source(url, car_data) for url in search_urls]
            await asyncio.gather(*tasks)

        return car_data

    def _generate_search_urls(self, car_model: str) -> List[str]:
        """
        Generate search URLs for different sources
        """
        car_model_query = car_model.replace(" ", "+")
        urls = [
            f"https://www.cardekho.com/carmodels/Tata/Tata_Safari"
        ]
        return urls

    async def process_source(self, url: str, car_data: Dict[str, List[str]]):
        """
        Process each source and extract relevant information
        """
        if not self.is_allowed_domain(url):
            return

        html_content = await self.fetch_page(url)
        if not html_content:
            return

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract specifications
        specs = self._extract_specifications(soup)
        car_data["specifications"].extend(specs)

        # Extract features
        features = self._extract_features(soup)
        car_data["features"].extend(features)

        # Extract pricing
        pricing = self._extract_pricing(soup)
        car_data["pricing"].extend(pricing)

        # Extract safety ratings
        safety = self._extract_safety_ratings(soup)
        car_data["safety_ratings"].extend(safety)

        await asyncio.sleep(settings.SCRAPING_DELAY)  # Respect rate limiting

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

    def _extract_safety_ratings(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract safety ratings from the page
        """
        safety = []
        safety_patterns = [
            "safety",
            "crash test",
            "rating",
            "iihs",
            "nhtsa"
        ]
        
        for pattern in safety_patterns:
            elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                parent = element.parent
                if parent and parent.text.strip():
                    safety.append(parent.text.strip())
        
        return list(set(safety)) 