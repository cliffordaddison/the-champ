"""
Champion Winner - Main Flask Application
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time

from src.ml.train import ChampionWinnerTrainer
from src.utils.scraper import LotteryScraper
from src.ml.config import system_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='web',
            template_folder='web')
CORS(app)

# Global variables
trainer = None
training_thread = None
training_status = {
    'is_training': False,
    'progress': 0,
    'status': 'Ready',
    'completed': False
}
models_loaded = False

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

def initialize_trainer():
    """Initialize the ML trainer"""
    global trainer, models_loaded
    if trainer is None:
        trainer = ChampionWinnerTrainer()
        models_loaded = False
        
        # Check if model files exist (using actual file names)
        model_files = [
            'models/q_learning_model.pkl',
            'models/pattern_recognition_model.pkl', 
            'models/frequency_analysis_model.pkl',
            'models/ensemble_model.pkl',
            'models/feature_scaler.pkl'
        ]
        
        # Check if all required model files exist
        all_models_exist = all(os.path.exists(f) for f in model_files)
        
        if all_models_exist:
            try:
                trainer.load_models('models')
                models_loaded = True
                logger.info("Successfully loaded all existing models")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                models_loaded = False
        else:
            missing_models = [f for f in model_files if not os.path.exists(f)]
            logger.info(f"Missing model files: {missing_models}")
            models_loaded = False

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Champion Winner API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/health",
            "/api/predict",
            "/api/submit-results",
            "/api/performance-metrics",
            "/api/recent-results",
            "/api/refresh-data"
        ],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('web', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    # Only serve specific file types
    allowed_extensions = ['.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory('web', filename)

# Add a simple test endpoint
@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({"message": "API is working!", "timestamp": datetime.now().isoformat()})

# Add a debug endpoint
@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    return jsonify({
        "app_name": "Champion Winner",
        "environment": os.environ.get('FLASK_ENV', 'development'),
        "debug_mode": os.environ.get('FLASK_DEBUG', 'False'),
        "port": os.environ.get('PORT', '8000'),
        "timestamp": datetime.now().isoformat(),
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate prediction for next draw"""
    try:
        global trainer, models_loaded
        
        if trainer is None:
            initialize_trainer()
        
        # Use real model prediction if available
        if models_loaded and trainer:
            try:
                # Load historical data for prediction
                data_files = ['data/fixed_ont49.csv', 'data/fixed_ont49_new.csv', 'data/cleaned_ont49.csv', 'data/Ont49.csv']
                history = []
                dates = []
                
                for file_path in data_files:
                    if os.path.exists(file_path):
                        try:
                            import pandas as pd
                            df = pd.read_csv(file_path)
                            
                            if not df.empty:
                                # Extract numbers and dates from the data
                                for _, row in df.iterrows():
                                    numbers = []
                                    
                                    # Check if we have a 'numbers' column with comma-separated values
                                    if 'numbers' in df.columns:
                                        numbers_str = str(row['numbers'])
                                        # Remove quotes and split by comma
                                        numbers_str = numbers_str.strip('"').strip("'")
                                        try:
                                            numbers = [int(x.strip()) for x in numbers_str.split(',')]
                                        except:
                                            logger.warning(f"Could not parse numbers: {numbers_str}")
                                            continue
                                    else:
                                        # Try to find individual number columns (like 1,2,3,4,5,6)
                                        for col in df.columns:
                                            if col.isdigit() or (col.startswith('Unnamed:') and col.replace('Unnamed:', '').isdigit()):
                                                try:
                                                    value = row[col]
                                                    if pd.notna(value) and str(value).isdigit():
                                                        numbers.append(int(value))
                                                except:
                                                    continue
                                        
                                        # If we didn't find individual columns, try the old method
                                        if not numbers:
                                            for col in df.columns:
                                                if 'number' in col.lower() or 'ball' in col.lower():
                                                    try:
                                                        value = row[col]
                                                        if pd.notna(value) and str(value).isdigit():
                                                            numbers.append(int(value))
                                                    except:
                                                        continue
                                    
                                    if len(numbers) >= 6:
                                        history.append(numbers[:6])
                                        
                                        # Try to get date
                                        if 'date' in df.columns:
                                            try:
                                                date = pd.to_datetime(row['date'])
                                                dates.append(date)
                                            except:
                                                dates.append(datetime.now())
                                        else:
                                            dates.append(datetime.now())
                                
                                logger.info(f"Loaded {len(history)} historical draws from {file_path}")
                                break  # Use first available file
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")
                            continue
                
                if not history:
                    return jsonify({
                        "error": "No historical data available for prediction",
                        "message": "Please refresh data or upload training data first",
                        "model_status": "no_data"
                    }), 503
                
                # Get real prediction from the trained model
                prediction = trainer.predict_next_draw(history, dates)
                
                # Check if prediction has error
                if 'error' in prediction:
                    return jsonify({
                        "error": prediction['error'],
                        "model_status": "error"
                    }), 500
                
                # Format the prediction for the API response
                real_prediction = {
                    "predicted_numbers": prediction.get('predicted_numbers', []),
                    "confidence_scores": prediction.get('confidence_scores', {}),
                    "ensemble_probabilities": prediction.get('ensemble_probabilities', []),
                    "agent_predictions": prediction.get('agent_predictions', {}),
                    "prediction_timestamp": datetime.now().isoformat(),
                    "model_status": "trained_model"
                }
                
                return jsonify(real_prediction)
                
            except Exception as e:
                logger.error(f"Error getting real prediction: {e}")
                return jsonify({
                    "error": "Failed to generate prediction from trained models",
                    "details": str(e),
                    "model_status": "error"
                }), 500
        else:
            # Return error when models are not available
            return jsonify({
                "error": "No trained models available",
                "message": "Please train the models first using the training endpoint",
                "model_status": "no_models"
            }), 503
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return jsonify({"error": str(e), "model_status": "error"}), 500

@app.route('/api/submit-results', methods=['POST'])
def submit_results():
    """Submit actual results and evaluate prediction"""
    try:
        data = request.get_json()
        numbers = data.get('numbers', [])
        bonus_ball = data.get('bonus_ball')
        prediction = data.get('prediction', {})
        
        if not numbers or len(numbers) != 6:
            return jsonify({"error": "Invalid numbers provided"}), 400
        
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
        
        # Store result (in production, this would go to a database)
        result = {
            'date': datetime.now().isoformat(),
            'numbers': numbers,
            'bonus_ball': bonus_ball,
            'predicted_numbers': predicted_numbers,
            'matches': matches,
            'status': status
        }
        
        # Update performance metrics
        update_performance_metrics(result)
        
        return jsonify({
            "matches": matches,
            "status": status,
            "message": f"Results submitted successfully. {matches} matches found."
        })
        
    except Exception as e:
        logger.error(f"Error submitting results: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    """Refresh data from lottery websites"""
    try:
        scraper = LotteryScraper()
        results = scraper.scrape_all_sites()
        scraper.cleanup()
        
        if results:
            # Save real data to CSV files
            import pandas as pd
            from datetime import datetime
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Save to multiple files for redundancy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to main data file
            main_file = f"data/ont49_data_{timestamp}.csv"
            os.makedirs('data', exist_ok=True)
            df.to_csv(main_file, index=False)
            
            # Also save to the standard filename
            standard_file = "data/cleaned_ont49.csv"
            df.to_csv(standard_file, index=False)
            
            logger.info(f"Refreshed {len(results)} results and saved to {main_file} and {standard_file}")
            return jsonify({
                "message": f"Refreshed {len(results)} results",
                "files_saved": [main_file, standard_file],
                "timestamp": timestamp
            })
        else:
            return jsonify({"error": "No data retrieved from websites"}), 500
            
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        global trainer, training_thread, training_status
        
        if training_status['is_training']:
            return jsonify({"error": "Training already in progress"}), 400
        
        # Initialize trainer if needed
        if trainer is None:
            initialize_trainer()
        
        # Get training parameters
        auto_scrape = request.form.get('auto_scrape', 'false').lower() == 'true'
        continuous_training = request.form.get('continuous_training', 'false').lower() == 'true'
        
        # Handle file upload
        file = request.files.get('file')
        if file:
            # Save uploaded file
            filename = f"data/uploaded_{int(time.time())}.csv"
            os.makedirs('data', exist_ok=True)
            file.save(filename)
            logger.info(f"Saved uploaded file: {filename}")
        
        # Start training in background thread
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['status'] = 'Initializing...'
        training_status['completed'] = False
        
        training_thread = threading.Thread(
            target=run_training,
            args=(auto_scrape, continuous_training)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({"message": "Training started successfully"})
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({"error": str(e)}), 500

def run_training(auto_scrape, continuous_training):
    """Run training in background thread"""
    global trainer, training_status, models_loaded
    
    try:
        training_status['status'] = 'Loading data...'
        training_status['progress'] = 10
        
        # Load real training data
        data_files = ['data/cleaned_ont49.csv', 'data/fixed_ont49_new.csv', 'data/fixed_ont49.csv', 'data/Ont49.csv']
        training_data = None
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    import pandas as pd
                    training_data = pd.read_csv(file_path)
                    logger.info(f"Loaded training data from {file_path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        if training_data is None:
            raise Exception("No training data found")
        
        training_status['status'] = 'Preprocessing data...'
        training_status['progress'] = 20
        
        # Preprocess the data
        trainer.preprocess_data(training_data)
        
        training_status['status'] = 'Training Q-Learning agent...'
        training_status['progress'] = 40
        
        # Train Q-Learning agent
        trainer.train_ql_agent(training_data)
        
        training_status['status'] = 'Training Pattern Recognition agent...'
        training_status['progress'] = 60
        
        # Train Pattern Recognition agent
        trainer.train_pattern_agent(training_data)
        
        training_status['status'] = 'Training Frequency Analysis agent...'
        training_status['progress'] = 80
        
        # Train Frequency Analysis agent
        trainer.train_frequency_agent(training_data)
        
        training_status['status'] = 'Saving models...'
        training_status['progress'] = 90
        
        # Save all trained models
        trainer.save_models('models')
        
        # Set models as loaded after successful training
        models_loaded = True
        
        training_status['status'] = 'Training completed successfully!'
        training_status['progress'] = 100
        training_status['completed'] = True
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        training_status['status'] = f'Training failed: {str(e)}'
    finally:
        training_status['is_training'] = False

@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    """Stop model training"""
    global training_status
    
    training_status['is_training'] = False
    training_status['status'] = 'Training stopped'
    
    return jsonify({"message": "Training stopped"})

@app.route('/api/training-progress')
def training_progress():
    """Get training progress"""
    global training_status
    
    return jsonify(training_status)

@app.route('/api/save-model', methods=['POST'])
def save_model():
    """Save current model"""
    try:
        global trainer
        if trainer:
            trainer.save_models('models')
            return jsonify({"message": "Model saved successfully"})
        else:
            return jsonify({"error": "No model to save"}), 400
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance-metrics')
def performance_metrics():
    """Get performance metrics"""
    try:
        # Load metrics from file
        metrics_file = 'models/performance_metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            # Return default metrics
            metrics = {
                'total_predictions': 0,
                'exact_matches': 0,
                'partial_matches': 0,
                'total_matches': 0,
                'win_rate': 0.0,
                'exact_match_rate': 0.0,
                'average_matches': 0.0
            }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error loading performance metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recent-results')
def recent_results():
    """Get recent results"""
    try:
        # Load real data from CSV files
        data_files = ['data/cleaned_ont49.csv', 'data/fixed_ont49_new.csv', 'data/fixed_ont49.csv', 'data/Ont49.csv']
        all_results = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    
                    # Get the most recent results (assuming the data has date and number columns)
                    if not df.empty:
                        # Extract the last few rows as recent results
                        recent_data = df.tail(5)  # Get last 5 results
                        
                        for _, row in recent_data.iterrows():
                            # Extract numbers from the row (adjust column names as needed)
                            numbers = []
                            
                            # Check if we have a 'numbers' column with comma-separated values
                            if 'numbers' in df.columns:
                                numbers_str = str(row['numbers'])
                                # Remove quotes and split by comma
                                numbers_str = numbers_str.strip('"').strip("'")
                                try:
                                    numbers = [int(x.strip()) for x in numbers_str.split(',')]
                                except:
                                    logger.warning(f"Could not parse numbers: {numbers_str}")
                                    continue
                            else:
                                # Try to find individual number columns
                                for col in df.columns:
                                    if 'number' in col.lower() or 'ball' in col.lower():
                                        try:
                                            value = row[col]
                                            if pd.notna(value) and str(value).isdigit():
                                                numbers.append(int(value))
                                        except:
                                            continue
                            
                            if len(numbers) >= 6:
                                result = {
                                    'date': str(row.get('date', 'Unknown')),
                                    'numbers': numbers[:6],  # Take first 6 numbers
                                    'status': 'unknown'  # Will be calculated when compared to predictions
                                }
                                all_results.append(result)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue
        
        # If no real data found, return error
        if not all_results:
            return jsonify({
                "error": "No historical data available",
                "message": "Please refresh data or upload training data first",
                "data_status": "no_data"
            }), 404
        
        # Return real results (limit to 10 most recent)
        return jsonify(all_results[-10:])
        
    except Exception as e:
        logger.error(f"Error loading recent results: {e}")
        return jsonify({"error": str(e)}), 500

def update_performance_metrics(result):
    """Update performance metrics with new result"""
    try:
        metrics_file = 'models/performance_metrics.json'
        
        # Load existing metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {
                'total_predictions': 0,
                'exact_matches': 0,
                'partial_matches': 0,
                'total_matches': 0,
                'win_rate': 0.0,
                'exact_match_rate': 0.0,
                'average_matches': 0.0
            }
        
        # Update metrics
        metrics['total_predictions'] += 1
        metrics['total_matches'] += result['matches']
        
        if result['status'] == 'match':
            metrics['exact_matches'] += 1
        elif result['status'] == 'partial':
            metrics['partial_matches'] += 1
        
        # Calculate rates
        total = metrics['total_predictions']
        metrics['win_rate'] = metrics['partial_matches'] / total if total > 0 else 0
        metrics['exact_match_rate'] = metrics['exact_matches'] / total if total > 0 else 0
        metrics['average_matches'] = metrics['total_matches'] / total if total > 0 else 0
        
        # Save updated metrics
        os.makedirs('models', exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating performance metrics: {e}")

@app.route('/api/test-scraping')
def test_scraping():
    """Test web scraping functionality"""
    try:
        scraper = LotteryScraper()
        scraper.test_scraping()
        scraper.cleanup()
        
        return jsonify({
            "message": "Scraping test completed. Check test_primary_site.html and test_backup_site.html for results."
        })
        
    except Exception as e:
        logger.error(f"Error testing scraping: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    global trainer, models_loaded
    
    # Determine model status
    if models_loaded:
        model_status = "trained_model"
    else:
        model_status = "no_models"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "model_status": model_status
    })

if __name__ == '__main__':
    # Initialize trainer
    initialize_trainer()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', system_config.port))
    
    # Add debug logging
    print(f"Starting Flask app on port {port}")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"Debug mode: {os.environ.get('FLASK_DEBUG', 'False')}")
    
    # Run the application
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False  # Disable debug in production
    ) 