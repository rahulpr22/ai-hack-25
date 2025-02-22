"""
Data ingestion package for processing car brochures and web data
"""

from .brochure_processor import BrochureProcessor
from .web_scraper import CarWebScraper
from .data_ingestion_orchestrator import DataIngestionOrchestrator

__all__ = ['BrochureProcessor', 'CarWebScraper', 'DataIngestionOrchestrator'] 