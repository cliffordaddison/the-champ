#!/usr/bin/env python3
"""
Simple Flask test script
"""

import requests
import time

def test_local_flask():
    """Test the local Flask app"""
    print("Testing local Flask app...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Local Flask app not running")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_remote_flask():
    """Test the remote Flask app"""
    print("\nTesting remote Flask app...")
    
    urls = [
        'https://champion-winner-api.onrender.com',
        'https://champion-winner.onrender.com'
    ]
    
    for url in urls:
        print(f"\nTesting {url}...")
        try:
            # Test basic connectivity
            response = requests.get(url, timeout=10)
            print(f"Basic connectivity: {response.status_code}")
            
            # Test health endpoint
            response = requests.get(f'{url}/api/health', timeout=10)
            print(f"Health endpoint: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_local_flask()
    test_remote_flask() 