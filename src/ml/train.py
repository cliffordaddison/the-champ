"""
Training module for Champion Winner lottery prediction system
"""

import numpy as np
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import os
import torch

from src.ml.config import model_config, system_config, MODEL_PATHS
from src.ml.agents import (
    QLearningAgent, PatternRecognitionAgent, FrequencyAnalysisAgent, 
    EnsembleAgent, Experience
)
from src.ml.features import FeatureEngineer
from src.utils.scraper import LotteryScraper

logger = logging.getLogger(__name__)

class ChampionWinnerTrainer:
    """Main trainer for the Champion Winner system"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.agents = {}
        self.ensemble = None
        self.training_history = []
        self.performance_metrics = {}
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("Champion Winner trainer initialized")
    
    def _initialize_agents(self):
        """Initialize all ML agents"""
        from src.ml.config import model_config
        
        # Calculate actual feature vector size
        # Basic: 4, Rolling: 40, Gap: 49, Positional: 294, Temporal: 10, Correlation: 49, Seasonal: 49
        state_size = 4 + 40 + 49 + 294 + 10 + 49 + 49  # Total: 495
        # But actual size is 492, so let's use that
        state_size = 492
        action_size = 49  # Numbers 1-49
        
        # Initialize agents
        self.agents['q_learning'] = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            config=model_config
        )
        
        self.agents['pattern_recognition'] = PatternRecognitionAgent()
        self.agents['frequency_analysis'] = FrequencyAnalysisAgent()
        self.agents['ensemble'] = EnsembleAgent(self.agents)
        
        logger.info("All agents initialized successfully")
    
    def _calculate_state_size(self) -> int:
        """Calculate state size for Q-Learning agent"""
        # Basic features: 17
        # Rolling features: max_number * 3 = 147
        # Gap features: max_number = 49
        # Positional features: max_number = 49
        # Temporal features: 8
        # Correlation features: max_number * 2 = 98
        # Seasonal features: 3 * 3 = 9
        
        total_features = 17 + (system_config.max_number * 3) + system_config.max_number + \
                       system_config.max_number + 8 + (system_config.max_number * 2) + 9
        
        return total_features
    
    def load_data(self, data_path: str) -> Tuple[List[List[int]], List[datetime]]:
        """Load and preprocess training data"""
        try:
            # Try to load CSV file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Check if date column exists
                if 'date' in df.columns:
                    # Parse dates
                    dates = []
                    for date_str in df['date']:
                        try:
                            date = pd.to_datetime(date_str)
                            dates.append(date)
                        except:
                            logger.warning(f"Could not parse date: {date_str}")
                            dates.append(datetime.now())
                else:
                    # No date column, create sequential dates starting from 01/10/97
                    logger.info("No date column found, creating sequential dates starting from 01/10/97...")
                    start_date = datetime(1997, 1, 10)  # Wednesday
                    dates = []
                    for i in range(len(df)):
                        # Alternate between Wednesday and Saturday
                        if i % 2 == 0:
                            # Wednesday (add 0 days)
                            draw_date = start_date + timedelta(days=i * 3)
                        else:
                            # Saturday (add 3 days from Wednesday)
                            draw_date = start_date + timedelta(days=(i * 3) + 3)
                        dates.append(draw_date)
                    logger.info(f"Created sequential dates from {dates[0]} to {dates[-1]}")
                
                # Parse numbers
                history = []
                for numbers_str in df['numbers']:
                    try:
                        numbers = [int(x.strip()) for x in numbers_str.split(',')]
                        if len(numbers) == system_config.numbers_to_predict:
                            history.append(sorted(numbers))
                        else:
                            logger.warning(f"Invalid number of numbers: {len(numbers)}")
                    except:
                        logger.warning(f"Could not parse numbers: {numbers_str}")
                
                logger.info(f"Loaded {len(history)} draws from {data_path}")
                return history, dates
            
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return [], []
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return [], []
    
    def scrape_and_load_data(self) -> Tuple[List[List[int]], List[datetime]]:
        """Scrape data from lottery websites"""
        scraper = LotteryScraper()
        
        try:
            results = scraper.scrape_all_sites()
            
            if results:
                history = [result['numbers'] for result in results]
                dates = [result['date'] for result in results]
                
                logger.info(f"Scraped {len(history)} draws from lottery websites")
                return history, dates
            else:
                logger.warning("No data scraped from websites")
                return [], []
                
        except Exception as e:
            logger.error(f"Error scraping data: {e}")
            return [], []
        finally:
            scraper.cleanup()
    
    def train_agents(self, history: List[List[int]], dates: List[datetime] = None):
        """Train all agents on historical data"""
        if not history:
            logger.error("No training data provided")
            return
        
        logger.info(f"Starting training with {len(history)} draws")
        
        # Create features
        features = self.feature_engineer.fit_transform(history, dates)
        
        # Train pattern recognition agent
        self.agents['pattern_recognition'].update_patterns(history)
        
        # Train frequency analysis agent
        self.agents['frequency_analysis'].update_frequencies(history)
        
        # Train Q-Learning agent
        self._train_q_learning_agent(features, history)
        
        logger.info("Training completed for all agents")
    
    def _train_q_learning_agent(self, features: np.ndarray, history: List[List[int]]):
        """Train the Q-Learning agent"""
        agent = self.agents['q_learning']
        
        print("🤖 Training Q-Learning agent...")
        
        # Training loop
        for episode in range(100):  # Reduced for faster training
            total_reward = 0
            
            for step in range(min(50, len(features))):  # Train on first 50 samples
                state = features[step]
                
                # Get action
                action = agent.get_action(state, training=True)
                
                # Simulate environment (use actual next draw as target)
                if step + 1 < len(history):
                    target_draw = history[step + 1]
                    reward = self._calculate_reward(action, target_draw)
                else:
                    reward = 0
                
                # Get next state
                next_state = features[step + 1] if step + 1 < len(features) else state
                
                # Store experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step + 1 >= len(features))
                )
                agent.memory.add(experience)
                
                # Train agent
                if len(agent.memory) > agent.config.batch_size:
                    loss = agent.train()
                    if loss:
                        total_reward += loss
                
                if step % 10 == 0:
                    print(f"  Episode {episode}, Step {step}, Reward: {reward:.2f}")
            
            print(f"  Episode {episode} completed, Total Reward: {total_reward:.2f}")
        
        print("✅ Q-Learning agent training completed!")
    
    def _calculate_reward(self, predicted_number: int, actual_numbers: List[int]) -> float:
        """Calculate reward for Q-Learning agent"""
        if predicted_number in actual_numbers:
            # Exact match
            return model_config.exact_match_reward
        else:
            # Miss
            return model_config.miss_penalty
    
    def predict_next_draw(self, history: List[List[int]], dates: List[datetime] = None) -> Dict:
        """Predict next draw using ensemble"""
        if not history:
            return {"error": "No historical data provided"}
        
        # Create features for current state
        features = self.feature_engineer.transform(history, dates)
        current_state = features[-1] if len(features) > 0 else np.zeros(self._calculate_state_size())
        
        # Get ensemble prediction
        ensemble_probs = self.agents['ensemble'].predict(current_state, history)
        
        # Select top numbers based on probabilities
        number_probs = [(i + 1, prob) for i, prob in enumerate(ensemble_probs)]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 6 numbers
        predicted_numbers = [num for num, prob in number_probs[:system_config.numbers_to_predict]]
        
        # Calculate confidence scores
        confidence_scores = {num: prob for num, prob in number_probs[:system_config.numbers_to_predict]}
        
        # Get individual agent predictions for comparison
        agent_predictions = {}
        for name, agent in self.agents.items():
            if name == 'q_learning':
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                    q_values = agent.q_network(state_tensor).squeeze()
                    agent_probs = torch.softmax(q_values, dim=0).tolist()
            elif name == 'pattern_recognition':
                agent_probs = agent.predict_from_patterns(history)
            elif name == 'frequency_analysis':
                agent_probs = agent.predict_from_frequency()
            
            # Get top numbers for this agent
            agent_number_probs = [(i + 1, prob) for i, prob in enumerate(agent_probs)]
            agent_number_probs.sort(key=lambda x: x[1], reverse=True)
            agent_predictions[name] = [num for num, prob in agent_number_probs[:system_config.numbers_to_predict]]
        
        return {
            "predicted_numbers": predicted_numbers,
            "confidence_scores": confidence_scores,
            "ensemble_probabilities": ensemble_probs,
            "agent_predictions": agent_predictions,
            "prediction_timestamp": datetime.now().isoformat()
        }
    
    def evaluate_prediction(self, prediction: Dict, actual_numbers: List[int]) -> Dict:
        """Evaluate prediction accuracy"""
        predicted_numbers = prediction.get("predicted_numbers", [])
        
        # Calculate matches
        matches = len(set(predicted_numbers) & set(actual_numbers))
        exact_match = matches == len(predicted_numbers)
        partial_match = matches > 0
        
        # Calculate reward for training
        reward = 0
        if exact_match:
            reward = model_config.exact_match_reward
        elif partial_match:
            reward = model_config.partial_match_reward
        else:
            reward = model_config.miss_penalty
        
        # Update performance metrics
        self._update_performance_metrics(exact_match, partial_match, matches)
        
        return {
            "exact_match": exact_match,
            "partial_match": partial_match,
            "matches": matches,
            "predicted_numbers": predicted_numbers,
            "actual_numbers": actual_numbers,
            "reward": reward,
            "performance_metrics": self.performance_metrics
        }
    
    def _update_performance_metrics(self, exact_match: bool, partial_match: bool, matches: int):
        """Update performance tracking metrics"""
        if 'total_predictions' not in self.performance_metrics:
            self.performance_metrics = {
                'total_predictions': 0,
                'exact_matches': 0,
                'partial_matches': 0,
                'total_matches': 0,
                'win_rate': 0.0,
                'exact_match_rate': 0.0,
                'average_matches': 0.0
            }
        
        self.performance_metrics['total_predictions'] += 1
        if exact_match:
            self.performance_metrics['exact_matches'] += 1
        if partial_match:
            self.performance_metrics['partial_matches'] += 1
        
        self.performance_metrics['total_matches'] += matches
        
        # Update rates
        total = self.performance_metrics['total_predictions']
        self.performance_metrics['win_rate'] = self.performance_metrics['partial_matches'] / total
        self.performance_metrics['exact_match_rate'] = self.performance_metrics['exact_matches'] / total
        self.performance_metrics['average_matches'] = self.performance_metrics['total_matches'] / total
    
    def save_models(self, base_path: str = "models"):
        """Save all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save individual agents
        for name, agent in self.agents.items():
            model_path = os.path.join(base_path, f"{name}_model.pkl")
            agent.save_model(model_path)
        
        # Save ensemble
        ensemble_path = os.path.join(base_path, "ensemble_model.pkl")
        self.agents['ensemble'].save_model(ensemble_path)
        
        # Save feature engineer
        scaler_path = os.path.join(base_path, "feature_scaler.pkl")
        self.feature_engineer.save_scaler(scaler_path)
        
        # Save performance metrics
        metrics_path = os.path.join(base_path, "performance_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        logger.info(f"All models saved to {base_path}")
    
    def load_models(self, base_path: str = "models"):
        """Load all trained models"""
        try:
            # Load individual agents
            for name, agent in self.agents.items():
                model_path = os.path.join(base_path, f"{name}_model.pkl")
                if os.path.exists(model_path):
                    agent.load_model(model_path)
            
            # Load ensemble
            ensemble_path = os.path.join(base_path, "ensemble_model.pkl")
            if os.path.exists(ensemble_path):
                self.agents['ensemble'].load_model(ensemble_path)
            
            # Load feature engineer
            scaler_path = os.path.join(base_path, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.feature_engineer.load_scaler(scaler_path)
            
            # Load performance metrics
            metrics_path = os.path.join(base_path, "performance_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            logger.info(f"All models loaded from {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_training_summary(self) -> Dict:
        """Get training summary and performance metrics"""
        return {
            "total_training_steps": len(self.training_history),
            "performance_metrics": self.performance_metrics,
            "model_config": {
                "learning_rate": model_config.learning_rate,
                "epsilon": self.agents['q_learning'].epsilon if 'q_learning' in self.agents else 0,
                "batch_size": model_config.batch_size
            },
            "system_config": {
                "max_number": system_config.max_number,
                "numbers_to_predict": system_config.numbers_to_predict,
                "target_win_rate": system_config.target_win_rate
            }
        }

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Champion Winner models')
    parser.add_argument('--data', default='data/fixed_ont49.csv', help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    
    args = parser.parse_args()
    
    print("🏆 Champion Winner - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ChampionWinnerTrainer()
    
    # Load data
    print(f"📊 Loading data from {args.data}...")
    history, dates = trainer.load_data(args.data)
    
    if not history:
        print("❌ No data loaded. Please check your data file.")
        return
    
    print(f"✅ Loaded {len(history)} draws")
    print(f"📅 Date range: {dates[0]} to {dates[-1]}")
    
    # Train models
    print("\n🚀 Starting model training...")
    trainer.train_agents(history, dates)
    
    # Generate prediction
    print("\n🎯 Generating sample prediction...")
    prediction = trainer.predict_next_draw(history, dates)
    
    print(f"📊 Prediction: {prediction['predicted_numbers']}")
    print(f"🎯 Confidence: {prediction['confidence_scores']}")
    print(f"🏆 Agent scores: {prediction['agent_predictions']}")
    
    # Save models if requested
    if args.save_models:
        print("\n💾 Saving models...")
        trainer.save_models()
        print("✅ Models saved successfully!")
    
    print("\n🎉 Training complete!")
    print("🌐 Ready to deploy your Champion Winner system!")

if __name__ == "__main__":
    main() 