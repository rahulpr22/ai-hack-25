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

    async def process_car_data(self, brochure_path: str, car_model: str) -> Dict[str, List[str]]:
        """
        Process both brochure and web data for a car and store in vector database
        """
        try:
            # Process brochure
            with open(brochure_path, 'r') as f:
                brochure_content = f.read()
            
            brochure_data = self.brochure_processor.process_markdown(brochure_content)
            self.logger.info(f"Successfully processed brochure for {car_model}")

            # Scrape web data
            web_data = await self.web_scraper.scrape_car_data(car_model)
            self.logger.info(f"Successfully scraped web data for {car_model}")

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
            
            return combined_data

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
        Merge and deduplicate data from both sources
        """
        merged_data = {}
        all_keys = set(brochure_data.keys()) | set(web_data.keys())
        
        for key in all_keys:
            brochure_items = set(brochure_data.get(key, []))
            web_items = set(web_data.get(key, []))
            merged_data[key] = list(brochure_items | web_items)  # Union of both sets
            
        return merged_data

    def _save_data(self, data: Dict[str, List[str]], car_model: str):
        """
        Save the processed data to a JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{car_model.replace(' ', '_')}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved processed data to {filename}")

    def _validate_data(self, data: Dict[str, List[str]]) -> bool:
        """
        Validate the processed data
        """
        required_sections = {"specifications", "features", "pricing"}
        
        # Check if all required sections are present
        if not all(section in data for section in required_sections):
            return False
        
        # Check if sections have content
        if not all(len(data[section]) > 0 for section in required_sections):
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
        combined_data = await orchestrator.process_car_data(brochure_path, car_model)
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