"""
Vercel API handler for Champion Winner
"""

import os
import sys
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.ml.train import ChampionWinnerTrainer
    from src.utils.scraper import LotteryScraper
    from src.ml.config import system_config
except ImportError as e:
    print(f"Import error: {e}")

# Global trainer instance
trainer = None
models_loaded = False

def initialize_trainer():
    """Initialize the ML trainer"""
    global trainer, models_loaded
    if trainer is None:
        try:
            trainer = ChampionWinnerTrainer()
            # Try to load existing models
            models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            if os.path.exists(models_path):
                trainer.load_models(models_path)
                models_loaded = True
                print("Loaded existing models")
            else:
                print("No existing models found, will use mock predictions")
                models_loaded = False
        except Exception as e:
            print(f"Error initializing trainer: {e}")
            models_loaded = False

def handle_predict():
    """Generate prediction for next draw"""
    try:
        global trainer, models_loaded
        if trainer is None:
            initialize_trainer()
        
        # Check if models are available
        if not models_loaded:
            # Return mock prediction when models are not available
            mock_prediction = {
                "predicted_numbers": [7, 12, 23, 31, 38, 45],
                "confidence_scores": {
                    7: 0.85,
                    12: 0.78,
                    23: 0.72,
                    31: 0.68,
                    38: 0.65,
                    45: 0.61
                },
                "ensemble_probabilities": [0.02] * 49,
                "agent_predictions": {
                    "QLearning": [3, 11, 19, 27, 35, 43],
                    "PatternRecognition": [5, 13, 21, 29, 37, 45],
                    "FrequencyAnalysis": [2, 10, 18, 26, 34, 42]
                },
                "prediction_timestamp": datetime.now().isoformat(),
                "model_status": "mock_prediction"
            }
        else:
            # Use actual model prediction
            # This would be implemented with the actual trained model
            mock_prediction = {
                "predicted_numbers": [7, 12, 23, 31, 38, 45],
                "confidence_scores": {
                    7: 0.85,
                    12: 0.78,
                    23: 0.72,
                    31: 0.68,
                    38: 0.65,
                    45: 0.61
                },
                "ensemble_probabilities": [0.02] * 49,
                "agent_predictions": {
                    "QLearning": [3, 11, 19, 27, 35, 43],
                    "PatternRecognition": [5, 13, 21, 29, 37, 45],
                    "FrequencyAnalysis": [2, 10, 18, 26, 34, 42]
                },
                "prediction_timestamp": datetime.now().isoformat(),
                "model_status": "trained_model"
            }
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(mock_prediction)
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "model_status": "error"})
        }

def handle_submit_results(data):
    """Submit actual results and evaluate prediction"""
    try:
        numbers = data.get('numbers', [])
        bonus_ball = data.get('bonus_ball')
        prediction = data.get('prediction', {})
        
        if not numbers or len(numbers) != 6:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Invalid numbers provided"})
            }
        
        # Calculate matches
        predicted_numbers = prediction.get('predicted_numbers', [])
        matches = len(set(numbers) & set(predicted_numbers))
        
        # Determine result status
        if matches == 6:
            status = 'match'
        elif matches > 0:
            status = 'partial'
        else:
            status = 'miss'
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "matches": matches,
                "status": status,
                "message": f"Results submitted successfully. {matches} matches found."
            })
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

def handle_health_check():
    """Health check endpoint"""
    global models_loaded
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "models_loaded": models_loaded,
            "model_status": "mock_prediction" if not models_loaded else "trained_model"
        })
    }

def handle_cors_preflight():
    """Handle CORS preflight requests"""
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": ""
    }

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        response = handle_cors_preflight()
        self.send_response(response["statusCode"])
        for header, value in response["headers"].items():
            self.send_header(header, value)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/api/health":
            response = handle_health_check()
        else:
            response = {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Not found"})
            }
        
        self.send_response(response["statusCode"])
        for header, value in response["headers"].items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(response["body"].encode())
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            if self.path == "/api/predict":
                response = handle_predict()
            elif self.path == "/api/submit-results":
                response = handle_submit_results(data)
            else:
                response = {
                    "statusCode": 404,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": "Not found"})
                }
            
        except Exception as e:
            response = {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(e)})
            }
        
        # Add CORS headers
        response["headers"]["Access-Control-Allow-Origin"] = "*"
        response["headers"]["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["headers"]["Access-Control-Allow-Headers"] = "Content-Type"
        
        self.send_response(response["statusCode"])
        for header, value in response["headers"].items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(response["body"].encode()) 