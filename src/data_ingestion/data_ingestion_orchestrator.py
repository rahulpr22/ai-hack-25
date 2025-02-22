import asyncio
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from .brochure_processor import BrochureProcessor
from .web_scraper import CarWebScraper
from ..vector_store.vector_store import VectorStore

class DataIngestionOrchestrator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.brochure_processor = BrochureProcessor()
        self.web_scraper = CarWebScraper()
        self.vector_store = VectorStore()

    async def process_car_data(self, brochure_path: str) -> Dict[str, List[str]]:
        """
        Process both brochure and web data for a car and store in vector database
        """
        try:
            # Process brochure
            with open(brochure_path, 'r') as f:
                brochure_content = f.read()
            
            # Process brochure and extract car model
            brochure_data = self.brochure_processor.process_markdown(brochure_content)
            car_model = brochure_data.pop('car_model')  # Extract car model from processed data
            
            if not car_model:
                raise ValueError("Could not determine car model from brochure")
                
            self.logger.info(f"Successfully processed brochure for {car_model}")

            # Initialize web scraper and scrape data using brochure data for context
            web_data = {}
            try:
                async with self.web_scraper as scraper:
                    web_data = await scraper.scrape_car_data(car_model, brochure_data)
                self.logger.info(f"Successfully scraped web data for {car_model}")
            except Exception as e:
                self.logger.warning(f"Web scraping failed for {car_model}: {str(e)}")
                web_data = {k: [] for k in brochure_data.keys()}

            # Merge and deduplicate data
            combined_data = self._merge_data(brochure_data, web_data)
            
            # Validate the data
            if not self._validate_data(combined_data):
                raise ValueError(f"Invalid data format for {car_model}")
            
            # Save the combined data
            self._save_data(combined_data, car_model)
            
            # Store in vector database
            await self.vector_store.upsert_car_data(combined_data, car_model)
            self.logger.info(f"Successfully stored vector data for {car_model}")
            
            return {"car_model": car_model, **combined_data}
            
        except Exception as e:
            self.logger.error(f"Error processing car data: {str(e)}")
            raise

    async def search_car_info(
        self,
        query: str,
        car_model: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for car information in the vector store
        """
        try:
            results = await self.vector_store.search_car_data(
                query=query,
                car_model=car_model,
                section=section,
                top_k=top_k
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching car info: {str(e)}")
            raise

    async def delete_car_data(self, car_model: str):
        """
        Delete all data for a specific car model
        """
        try:
            # Delete from vector store
            await self.vector_store.delete_car_data(car_model)
            
            # Delete JSON files
            for file in self.output_dir.glob(f"{car_model.replace(' ', '_')}*.json"):
                file.unlink()
            
            self.logger.info(f"Successfully deleted all data for {car_model}")
        except Exception as e:
            self.logger.error(f"Error deleting car data: {str(e)}")
            raise

    def _merge_data(self, brochure_data: Dict[str, List[str]], web_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Merge and deduplicate data from brochure and web sources
        """
        merged_data = {}
        for key in brochure_data.keys():
            # Combine lists and remove duplicates while preserving order
            combined = []
            seen = set()
            
            # Add brochure data first (primary source)
            for item in brochure_data[key]:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    combined.append(item)
            
            # Add web data (secondary source)
            for item in web_data.get(key, []):
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    combined.append(item)
            
            merged_data[key] = combined
        
        return merged_data

    def _save_data(self, data: Dict[str, List[str]], car_model: str):
        """
        Save the processed data to a JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{car_model}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "car_model": car_model,
                "timestamp": timestamp,
                "data": data
            }, f, indent=2)
            
        self.logger.info(f"Saved processed data to {filename}")

    def _validate_data(self, data: Dict[str, List[str]]) -> bool:
        """
        Validate the data structure
        """
        required_sections = {
            "specifications",
            "interior",
            "technology",
            "exterior",
            "safety",
            "performance",
            "pricing",
            "colors"
        }
        
        # Check if all required sections exist
        if not all(section in data for section in required_sections):
            return False
            
        # Check if all values are lists
        if not all(isinstance(v, list) for v in data.values()):
            return False
            
        # Check if all list items are strings
        if not all(all(isinstance(item, str) for item in items) for items in data.values()):
            return False
            
        return True

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    orchestrator = DataIngestionOrchestrator()
    
    # Example car model and brochure path
    car_model = "Tata Safari 2024"
    brochure_path = "brochures/tata_safari_2024.md"
    
    try:
        # Process and store car data
        combined_data = await orchestrator.process_car_data(brochure_path)
        print(f"Successfully processed and stored data for {car_model}")
        
        # Example search
        search_results = await orchestrator.search_car_info(
            query="What are the safety features?",
            car_model=car_model,
            section="safety"
        )
        print("\nSearch Results:")
        for result in search_results:
            print(f"\nScore: {result['score']}")
            print(f"Text: {result['text']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 