#!/usr/bin/env python3
"""
Champion Winner - Startup Script
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app, initialize_trainer
from src.ml.config import system_config

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Champion Winner Lottery Prediction System')
    parser.add_argument('--host', default=system_config.host, help='Host to bind to')
    parser.add_argument('--port', type=int, default=system_config.port, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test-scraping', action='store_true', help='Test web scraping functionality')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test_scraping:
        print("Testing web scraping functionality...")
        from tests.test_scraping import main as test_scraping
        test_scraping()
        return
    
    # Initialize trainer
    print("Initializing Champion Winner system...")
    initialize_trainer()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"Starting Champion Winner on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the application
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == '__main__':
    main() 