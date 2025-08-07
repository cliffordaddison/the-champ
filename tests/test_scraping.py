"""Simple test script for web scraping functionality"""
import sys, os, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.scraper import LotteryScraper
logging.basicConfig(level=logging.INFO)
def test_scraping():
    scraper = LotteryScraper()
    try:
        results = scraper.scrape_all_sites()
