"""
Test script for web scraping functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.utils.scraper import LotteryScraper
from src.ml.config import scraping_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraping_functionality():
    """Test the scraping functionality"""
    logger.info("Starting scraping test...")
    
    scraper = LotteryScraper()
    
    try:
        # Test primary site
        logger.info("Testing primary site...")
        primary_results = scraper.scrape_primary_site()
        
        if primary_results:
            logger.info(f"Primary site test successful! Found {len(primary_results)} results")
            logger.info(f"Sample result: {primary_results[0] if primary_results else 'None'}")
        else:
            logger.warning("Primary site test failed - no results found")
        
        # Test backup site
        logger.info("Testing backup site...")
        backup_results = scraper.scrape_backup_site()
        
        if backup_results:
            logger.info(f"Backup site test successful! Found {len(backup_results)} results")
            logger.info(f"Sample result: {backup_results[0] if backup_results else 'None'}")
        else:
            logger.warning("Backup site test failed - no results found")
        
        # Test combined scraping
        logger.info("Testing combined scraping...")
        all_results = scraper.scrape_all_sites()
        
        if all_results:
            logger.info(f"Combined scraping successful! Found {len(all_results)} total results")
        else:
            logger.error("Combined scraping failed - no results from any site")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error during scraping test: {e}")
        return None
    finally:
        scraper.cleanup()

def analyze_website_structure():
    """Analyze the structure of lottery websites"""
    logger.info("Analyzing website structure...")
    
    scraper = LotteryScraper()
    
    try:
        # Test primary site structure
        logger.info("Analyzing primary site structure...")
        if scraper.setup_selenium():
            scraper.driver.get(scraping_config.primary_url)
            
            # Wait for page to load
            import time
            time.sleep(5)
            
            # Get page source
            page_source = scraper.driver.page_source
            
            # Save for analysis
            with open('test_primary_site.html', 'w', encoding='utf-8') as f:
                f.write(page_source)
            
            logger.info("Primary site HTML saved to test_primary_site.html")
            
            # Analyze structure
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Look for tables
            tables = soup.find_all('table')
            logger.info(f"Found {len(tables)} tables on primary site")
            
            for i, table in enumerate(tables):
                logger.info(f"Table {i+1}:")
                rows = table.find_all('tr')
                logger.info(f"  - {len(rows)} rows")
                if rows:
                    cells = rows[0].find_all(['td', 'th'])
                    logger.info(f"  - {len(cells)} columns in first row")
                    if cells:
                        logger.info(f"  - Sample content: {cells[0].get_text(strip=True)[:50]}...")
            
            # Look for lottery-related elements
            lottery_elements = soup.find_all(text=lambda text: text and any(word in text.lower() for word in ['lottery', 'draw', 'number', 'result']))
            logger.info(f"Found {len(lottery_elements)} lottery-related text elements")
            
            # Test backup site structure
            logger.info("Analyzing backup site structure...")
            scraper.driver.get(scraping_config.backup_url)
            time.sleep(5)
            
            backup_source = scraper.driver.page_source
            with open('test_backup_site.html', 'w', encoding='utf-8') as f:
                f.write(backup_source)
            
            logger.info("Backup site HTML saved to test_backup_site.html")
            
            # Analyze backup site
            backup_soup = BeautifulSoup(backup_source, 'html.parser')
            backup_tables = backup_soup.find_all('table')
            logger.info(f"Found {len(backup_tables)} tables on backup site")
            
            for i, table in enumerate(backup_tables):
                logger.info(f"Backup Table {i+1}:")
                rows = table.find_all('tr')
                logger.info(f"  - {len(rows)} rows")
                if rows:
                    cells = rows[0].find_all(['td', 'th'])
                    logger.info(f"  - {len(cells)} columns in first row")
                    if cells:
                        logger.info(f"  - Sample content: {cells[0].get_text(strip=True)[:50]}...")
        
    except Exception as e:
        logger.error(f"Error analyzing website structure: {e}")
    finally:
        scraper.cleanup()

def suggest_css_selectors():
    """Suggest CSS selectors based on website analysis"""
    logger.info("Suggesting CSS selectors...")
    
    try:
        # Read saved HTML files
        primary_html = None
        backup_html = None
        
        if os.path.exists('test_primary_site.html'):
            with open('test_primary_site.html', 'r', encoding='utf-8') as f:
                primary_html = f.read()
        
        if os.path.exists('test_backup_site.html'):
            with open('test_backup_site.html', 'r', encoding='utf-8') as f:
                backup_html = f.read()
        
        if primary_html:
            logger.info("Analyzing primary site for selectors...")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(primary_html, 'html.parser')
            
            # Look for date patterns
            date_elements = soup.find_all(text=lambda text: text and any(char.isdigit() for char in text))
            logger.info(f"Found {len(date_elements)} potential date elements")
            
            # Look for number patterns
            number_elements = soup.find_all(text=lambda text: text and any(char.isdigit() for char in text))
            logger.info(f"Found {len(number_elements)} potential number elements")
            
            # Suggest selectors
            logger.info("Suggested CSS selectors for primary site:")
            logger.info("  - Date selector: Look for elements containing date patterns")
            logger.info("  - Numbers selector: Look for elements containing 6 numbers")
            logger.info("  - Bonus ball selector: Look for elements containing single numbers")
        
        if backup_html:
            logger.info("Analyzing backup site for selectors...")
            backup_soup = BeautifulSoup(backup_html, 'html.parser')
            
            # Similar analysis for backup site
            logger.info("Suggested CSS selectors for backup site:")
            logger.info("  - Date selector: Look for elements containing date patterns")
            logger.info("  - Numbers selector: Look for elements containing 6 numbers")
            logger.info("  - Bonus ball selector: Look for elements containing single numbers")
    
    except Exception as e:
        logger.error(f"Error suggesting CSS selectors: {e}")

def main():
    """Main test function"""
    logger.info("Starting comprehensive scraping test...")
    
    # Test basic scraping
    results = test_scraping_functionality()
    
    # Analyze website structure
    analyze_website_structure()
    
    # Suggest CSS selectors
    suggest_css_selectors()
    
    logger.info("Scraping test completed!")
    logger.info("Check the generated HTML files for detailed analysis:")
    logger.info("  - test_primary_site.html")
    logger.info("  - test_backup_site.html")
    
    if results:
        logger.info(f"Successfully scraped {len(results)} results")
        return True
    else:
        logger.warning("No results were scraped - check website availability and selectors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 