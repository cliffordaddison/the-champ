"""
Feature Engineering for Lottery Prediction System
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src.ml.config import FEATURE_PARAMS, system_config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for lottery prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def create_features(self, history: List[List[int]], dates: List[datetime] = None) -> np.ndarray:
        """Create comprehensive feature set from historical data"""
        if not history:
            return np.array([])
        
        features = []
        
        # Basic statistical features
        basic_features = self._create_basic_features(history)
        features.append(basic_features)
        
        # Rolling statistics
        rolling_features = self._create_rolling_features(history)
        features.append(rolling_features)
        
        # Gap analysis
        gap_features = self._create_gap_features(history)
        features.append(gap_features)
        
        # Positional features
        positional_features = self._create_positional_features(history)
        features.append(positional_features)
        
        # Temporal features
        if dates:
            temporal_features = self._create_temporal_features(history, dates)
            features.append(temporal_features)
        
        # Correlation features
        correlation_features = self._create_correlation_features(history)
        features.append(correlation_features)
        
        # Seasonal features
        seasonal_features = self._create_seasonal_features(history, dates)
        features.append(seasonal_features)
        
        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        
        # Store feature names for interpretability
        self.feature_names = self._generate_feature_names()
        
        return combined_features
    
    def _create_basic_features(self, history: List[List[int]]) -> np.ndarray:
        """Create basic statistical features"""
        features = []
        
        for draw in history:
            draw_features = []
            
            # Basic statistics
            draw_features.extend([
                np.mean(draw),
                np.std(draw),
                np.median(draw),
                np.min(draw),
                np.max(draw),
                np.ptp(draw),  # range
                stats.skew(draw),
                stats.kurtosis(draw)
            ])
            
            # Number distribution features
            draw_features.extend([
                len([x for x in draw if x <= 10]),  # low numbers
                len([x for x in draw if 11 <= x <= 20]),  # mid-low numbers
                len([x for x in draw if 21 <= x <= 30]),  # mid-high numbers
                len([x for x in draw if x >= 31]),  # high numbers
            ])
            
            # Parity features
            draw_features.extend([
                len([x for x in draw if x % 2 == 0]),  # even numbers
                len([x for x in draw if x % 2 == 1]),  # odd numbers
            ])
            
            # Sum and product features
            draw_features.extend([
                np.sum(draw),
                np.prod(draw),
                np.sum(draw) / len(draw),  # average
            ])
            
            features.append(draw_features)
        
        return np.array(features)
    
    def _create_rolling_features(self, history: List[List[int]]) -> np.ndarray:
        """Create rolling statistics features"""
        if len(history) < 10:
            return np.zeros((len(history), 0))
        
        # Use rolling window from config
        from src.ml.config import system_config
        window = system_config.rolling_window
        
        features = []
        for i in range(len(history)):
            if i < window:
                # Use available data for early draws
                recent_draws = history[:i+1]
            else:
                recent_draws = history[i-window+1:i+1]
            
            if not recent_draws:
                features.append([0] * 20)  # Default feature vector
                continue
            
            # Flatten recent draws
            all_numbers = [num for draw in recent_draws for num in draw]
            
            # Basic statistics
            mean_val = np.mean(all_numbers)
            std_val = np.std(all_numbers)
            min_val = np.min(all_numbers)
            max_val = np.max(all_numbers)
            
            # Number frequency
            number_counts = {}
            for num in range(1, 50):
                number_counts[num] = all_numbers.count(num)
            
            # Top 10 most frequent numbers
            sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
            top_10_freq = [count for _, count in sorted_numbers[:10]]
            
            # Bottom 10 least frequent numbers
            bottom_10_freq = [count for _, count in sorted_numbers[-10:]]
            
            feature_vector = [mean_val, std_val, min_val, max_val] + top_10_freq + bottom_10_freq
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_gap_features(self, history: List[List[int]]) -> np.ndarray:
        """Create gap analysis features"""
        from src.ml.config import system_config
        
        features = []
        gap_window = system_config.rolling_window  # Use same window as rolling features
        
        for i in range(len(history)):
            if i < gap_window:
                features.append([0] * 49)  # Default for early draws
                continue
            
            # Calculate gaps for each number
            gaps = []
            for num in range(1, 50):
                gap = 0
                for j in range(i-1, max(-1, i-gap_window-1), -1):
                    if num in history[j]:
                        break
                    gap += 1
                gaps.append(gap)
            
            features.append(gaps)
        
        return np.array(features)
    
    def _create_positional_features(self, history: List[List[int]]) -> np.ndarray:
        """Create positional probability features"""
        from src.ml.config import system_config
        
        features = []
        
        for i in range(len(history)):
            if i < 10:  # Need some history for probabilities
                features.append([1/6] * 294)  # 49 numbers * 6 positions
                continue
            
            # Calculate positional probabilities
            positional_probs = []
            for num in range(1, 50):
                # Count how many times this number appeared in each position
                position_counts = [0] * 6
                for j in range(max(0, i-50), i):  # Last 50 draws
                    if num in history[j]:
                        pos = history[j].index(num)
                        if pos < 6:  # Ensure position is valid
                            position_counts[pos] += 1
                
                # Calculate probability for each position
                total_appearances = sum(position_counts)
                if total_appearances == 0:
                    positional_probs.extend([1/6] * 6)  # Uniform
                else:
                    probs = [count / total_appearances for count in position_counts]
                    positional_probs.extend(probs)
            
            features.append(positional_probs)
        
        return np.array(features)
    
    def _create_temporal_features(self, history: List[List[int]], dates: List[datetime]) -> np.ndarray:
        """Create temporal features"""
        features = []
        
        for i in range(len(history)):
            if i >= len(dates):
                features.append([0] * 10)  # Default features
                continue
            
            date = dates[i]
            
            # Day of week (0=Monday, 6=Sunday)
            day_of_week = date.weekday()
            
            # Month (1-12)
            month = date.month
            
            # Day of month (1-31)
            day_of_month = date.day
            
            # Week of year (1-53)
            week_of_year = date.isocalendar()[1]
            
            # Is weekend
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Is Wednesday or Saturday (lottery days)
            is_lottery_day = 1 if day_of_week in [2, 5] else 0  # Wed=2, Sat=5
            
            # Quarter of year
            quarter = (month - 1) // 3 + 1
            
            # Is end of month
            is_end_of_month = 1 if day_of_month >= 25 else 0
            
            # Is beginning of month
            is_beginning_of_month = 1 if day_of_month <= 7 else 0
            
            # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
            if month in [12, 1, 2]:
                season = 1
            elif month in [3, 4, 5]:
                season = 2
            elif month in [6, 7, 8]:
                season = 3
            else:
                season = 4
            
            feature_vector = [
                day_of_week, month, day_of_month, week_of_year,
                is_weekend, is_lottery_day, quarter,
                is_end_of_month, is_beginning_of_month, season
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_correlation_features(self, history: List[List[int]]) -> np.ndarray:
        """Create correlation-based features"""
        features = []
        
        for i in range(len(history)):
            if i < 20:  # Need sufficient history
                features.append([0] * 49)  # Default
                continue
            
            # Calculate correlations between numbers
            recent_draws = history[max(0, i-20):i]
            correlations = []
            
            for num in range(1, 50):
                # Calculate correlation with other numbers
                correlations_with_num = []
                for other_num in range(1, 50):
                    if num == other_num:
                        correlations_with_num.append(1.0)
                    else:
                        # Simple correlation: how often they appear together
                        together_count = 0
                        total_draws = len(recent_draws)
                        
                        for draw in recent_draws:
                            if num in draw and other_num in draw:
                                together_count += 1
                        
                        correlation = together_count / total_draws if total_draws > 0 else 0
                        correlations_with_num.append(correlation)
                
                # Average correlation for this number
                avg_correlation = np.mean(correlations_with_num)
                correlations.append(avg_correlation)
            
            features.append(correlations)
        
        return np.array(features)
    
    def _create_seasonal_features(self, history: List[List[int]], dates: List[datetime]) -> np.ndarray:
        """Create seasonal pattern features"""
        features = []
        
        for i in range(len(history)):
            if i >= len(dates):
                features.append([0] * 49)  # Default
                continue
            
            date = dates[i]
            month = date.month
            
            # Seasonal patterns for each number
            seasonal_patterns = []
            for num in range(1, 50):
                # Calculate seasonal frequency for this number
                seasonal_freq = 0
                seasonal_count = 0
                
                # Look at same month in previous years (if available)
                for j in range(max(0, i-365), i):
                    if j < len(dates) and dates[j].month == month:
                        if num in history[j]:
                            seasonal_freq += 1
                        seasonal_count += 1
                
                seasonal_prob = seasonal_freq / seasonal_count if seasonal_count > 0 else 1/49
                seasonal_patterns.append(seasonal_prob)
            
            features.append(seasonal_patterns)
        
        return np.array(features)
    
    def _is_similar_seasonal_period(self, date1: datetime, date2: datetime, period: int) -> bool:
        """Check if two dates are in similar seasonal periods"""
        if period == 7:  # Weekly
            return date1.weekday() == date2.weekday()
        elif period == 30:  # Monthly
            return date1.month == date2.month
        elif period == 365:  # Yearly
            return date1.month == date2.month and abs(date1.day - date2.day) <= 7
        else:
            return False
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for the engineered features"""
        feature_names = []
        
        # Basic features
        feature_names.extend(['basic_mean', 'basic_std', 'basic_min', 'basic_max'])
        
        # Rolling features
        feature_names.extend([f'rolling_mean_{i}' for i in range(10)])
        feature_names.extend([f'rolling_std_{i}' for i in range(10)])
        feature_names.extend([f'rolling_min_{i}' for i in range(10)])
        feature_names.extend([f'rolling_max_{i}' for i in range(10)])
        
        # Gap features
        feature_names.extend([f'gap_{i}' for i in range(1, 50)])
        
        # Positional features
        for num in range(1, 50):
            for pos in range(6):
                feature_names.append(f'pos_{num}_{pos}')
        
        # Temporal features
        temporal_names = [
            'day_of_week', 'month', 'day_of_month', 'week_of_year',
            'is_weekend', 'is_lottery_day', 'quarter',
            'is_end_of_month', 'is_beginning_of_month', 'season'
        ]
        feature_names.extend(temporal_names)
        
        # Correlation features
        feature_names.extend([f'correlation_{i}' for i in range(1, 50)])
        
        # Seasonal features
        feature_names.extend([f'seasonal_{i}' for i in range(1, 50)])
        
        return feature_names
    
    def fit_transform(self, history: List[List[int]], dates: List[datetime] = None) -> np.ndarray:
        """Fit the feature scaler and transform data"""
        features = self.create_features(history, dates)
        
        if features.size > 0:
            features_scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
            return features_scaled
        else:
            return features
    
    def transform(self, history: List[List[int]], dates: List[datetime] = None) -> np.ndarray:
        """Transform data using fitted scaler"""
        features = self.create_features(history, dates)
        
        if features.size > 0 and self.is_fitted:
            return self.scaler.transform(features)
        else:
            return features
    
    def get_feature_importance(self, model, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        if feature_names is None:
            feature_names = self.feature_names
        
        importance_dict = {}
        for name, importance in zip(feature_names, model.feature_importances_):
            importance_dict[name] = importance
        
        return importance_dict
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved feature scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load fitted scaler"""
        import pickle
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded feature scaler from {filepath}") 