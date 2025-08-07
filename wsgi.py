#!/usr/bin/env python3
"""
WSGI entry point for Champion Winner Flask app
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import app

if __name__ == "__main__":
    app.run() 