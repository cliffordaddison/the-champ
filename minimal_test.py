#!/usr/bin/env python3
"""
Minimal Flask test
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.main import app
    print("✅ Flask app imported successfully")
    
    # Check if routes are registered
    routes = list(app.url_map.iter_rules())
    print(f"✅ Found {len(routes)} routes:")
    for route in routes:
        print(f"  {route.rule} -> {route.endpoint}")
        
    # Test the app with a test client
    from flask.testing import FlaskClient
    client = FlaskClient(app)
    
    # Test health endpoint
    response = client.get('/api/health')
    print(f"✅ Health endpoint test: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.get_json()}")
    else:
        print(f"Error: {response.get_data(as_text=True)}")
        
    # Test root endpoint
    response = client.get('/')
    print(f"✅ Root endpoint test: {response.status_code}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 