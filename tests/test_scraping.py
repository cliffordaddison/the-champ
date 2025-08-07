"""
Simple test script for web scraping functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.utils.scraper import LotteryScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraping():
    """Simple test for scraping functionality"""
    logger.info("Starting simple scraping test...")
    
    scraper = LotteryScraper()
    
    try:
        # Test scraping
        results = scraper.scrape_all_sites()
        
        if results:
            logger.info(f"Scraping successful! Found {len(results)} results")
            logger.info(f"Sample result: {results[0] if results else 'None'}")
            return True
        else:
            logger.warning("No results found")
            return False
            
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return False
    finally:
        scraper.cleanup()

def main():
    """Main test function"""
    logger.info("Running simple scraping test...")
    
    success = test_scraping()
    
    if success:
        logger.info("✅ Test passed!")
    else:
        logger.error("❌ Test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 