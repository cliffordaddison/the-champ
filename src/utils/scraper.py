"""
Web scraping utility for lottery data collection
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.ml.config import scraping_config, system_config

logger = logging.getLogger(__name__)

class LotteryScraper:
    """Scraper for lottery websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.driver = None
        
    def setup_selenium(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            return True
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            return False
    
    def scrape_primary_site(self) -> Optional[List[Dict]]:
        """Scrape data from primary lottery website"""
        try:
            logger.info("Attempting to scrape primary site...")
            
            if not self.driver:
                if not self.setup_selenium():
                    return None
            
            self.driver.get(scraping_config.primary_url)
            time.sleep(3)  # Wait for page to load
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract data (selectors will be updated after testing)
            results = []
            
            # Look for table with lottery results
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:  # Expecting date, numbers, bonus ball
                        try:
                            # Extract date
                            date_text = cells[0].get_text(strip=True)
                            date = self.parse_date(date_text)
                            
                            # Extract numbers
                            numbers_text = cells[1].get_text(strip=True)
                            numbers = self.parse_numbers(numbers_text)
                            
                            # Extract bonus ball
                            bonus_text = cells[2].get_text(strip=True)
                            bonus_ball = self.parse_bonus_ball(bonus_text)
                            
                            if date and numbers and bonus_ball is not None:
                                results.append({
                                    'date': date,
                                    'numbers': numbers,
                                    'bonus_ball': bonus_ball
                                })
                        except Exception as e:
                            logger.debug(f"Failed to parse row: {e}")
                            continue
            
            logger.info(f"Successfully scraped {len(results)} results from primary site")
            return results
            
        except Exception as e:
            logger.error(f"Error scraping primary site: {e}")
            return None
    
    def scrape_backup_site(self) -> Optional[List[Dict]]:
        """Scrape data from backup lottery website"""
        try:
            logger.info("Attempting to scrape backup site...")
            
            if not self.driver:
                if not self.setup_selenium():
                    return None
            
            self.driver.get(scraping_config.backup_url)
            time.sleep(3)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            results = []
            
            # Look for lottery results (will be updated after testing)
            # This is a placeholder implementation
            lottery_elements = soup.find_all(['div', 'span'], class_=lambda x: x and 'lottery' in x.lower())
            
            for element in lottery_elements:
                try:
                    # Extract data from element
                    # This will be implemented after testing the actual site structure
                    pass
                except Exception as e:
                    logger.debug(f"Failed to parse element: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(results)} results from backup site")
            return results
            
        except Exception as e:
            logger.error(f"Error scraping backup site: {e}")
            return None
    
    def parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from text"""
        try:
            # Try different date formats
            date_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y/%m/%d',
                '%b %d, %Y',
                '%B %d, %Y'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_text, fmt)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_text}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return None
    
    def parse_numbers(self, numbers_text: str) -> Optional[List[int]]:
        """Parse winning numbers from text"""
        try:
            # Remove common separators and split
            numbers_text = numbers_text.replace(',', ' ').replace('-', ' ')
            numbers = []
            
            for num_str in numbers_text.split():
                try:
                    num = int(num_str.strip())
                    if 1 <= num <= system_config.max_number:
                        numbers.append(num)
                except ValueError:
                    continue
            
            # Check if we have the right number of results
            if len(numbers) == system_config.numbers_to_predict:
                return sorted(numbers)
            else:
                logger.warning(f"Expected {system_config.numbers_to_predict} numbers, got {len(numbers)}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing numbers: {e}")
            return None
    
    def parse_bonus_ball(self, bonus_text: str) -> Optional[int]:
        """Parse bonus ball from text"""
        try:
            bonus_text = bonus_text.strip()
            bonus = int(bonus_text)
            
            if 1 <= bonus <= system_config.bonus_ball_range:
                return bonus
            else:
                logger.warning(f"Bonus ball {bonus} out of range")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing bonus ball: {e}")
            return None
    
    def scrape_all_sites(self) -> List[Dict]:
        """Scrape from all available sites"""
        results = []
        
        # Try primary site first
        primary_results = self.scrape_primary_site()
        if primary_results:
            results.extend(primary_results)
            logger.info("Primary site scraping successful")
        else:
            logger.warning("Primary site failed, trying backup site")
            
            # Try backup site
            backup_results = self.scrape_backup_site()
            if backup_results:
                results.extend(backup_results)
                logger.info("Backup site scraping successful")
            else:
                logger.error("All scraping attempts failed")
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def test_scraping():
    """Test function to check website structure"""
    scraper = LotteryScraper()
    
    try:
        logger.info("Testing primary site structure...")
        if scraper.setup_selenium():
            scraper.driver.get(scraping_config.primary_url)
            time.sleep(5)
            
            soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')
            
            # Save page source for analysis
            with open('test_primary_site.html', 'w', encoding='utf-8') as f:
                f.write(scraper.driver.page_source)
            
            logger.info("Primary site test completed. Check test_primary_site.html for structure.")
            
            # Test backup site
            logger.info("Testing backup site structure...")
            scraper.driver.get(scraping_config.backup_url)
            time.sleep(5)
            
            with open('test_backup_site.html', 'w', encoding='utf-8') as f:
                f.write(scraper.driver.page_source)
            
            logger.info("Backup site test completed. Check test_backup_site.html for structure.")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test scraping
    test_scraping() 