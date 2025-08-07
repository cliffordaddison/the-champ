"""
Configuration file for Champion Winner ML system
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    target_update: int = 100
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    
    # Reward system
    exact_match_reward: int = 100
    partial_match_reward: int = 10
    miss_penalty: int = -5
    
    # Training parameters
    max_episodes: int = 1000
    max_steps: int = 100
    save_interval: int = 100

@dataclass
class ScrapingConfig:
    primary_url: str = "https://www.lotto-8.com/canada/listltoCAON49.asp"
    backup_url: str = "https://www.theluckygene.com/LotteryResults?gid=Ontario49Game"
    timeout: int = 30
    retry_attempts: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

@dataclass
class SystemConfig:
    data_dir: str = "data"
    models_dir: str = "models"
    db_path: str = "lottery_data.db"
    max_number: int = 49
    numbers_to_predict: int = 6
    draw_days: List[str] = field(default_factory=lambda: ["Wednesday", "Saturday"])
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    
    # Feature engineering parameters
    rolling_window: int = 10
    feature_window: int = 50
    
    # Performance targets
    target_accuracy: float = 0.6
    target_exact_matches: int = 1  # per 4-6 weeks

# Feature engineering parameters
FEATURE_PARAMS = {
    'rolling_stats': ['mean', 'std', 'min', 'max'],
    'gap_analysis': True,
    'positional_prob': True,
    'temporal_features': True,
    'correlation_features': True,
    'seasonal_features': True
}

# Model file paths
MODEL_PATHS = {
    'q_learning': 'models/q_learning_model.pkl',
    'pattern_recognition': 'models/pattern_model.pkl',
    'frequency_analysis': 'models/frequency_model.pkl',
    'ensemble': 'models/ensemble_model.pkl',
    'feature_scaler': 'models/feature_scaler.pkl'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Create instances
model_config = ModelConfig()
scraping_config = ScrapingConfig()
system_config = SystemConfig() 