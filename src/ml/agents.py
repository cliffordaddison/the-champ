"""
Reinforcement Learning Agents for Lottery Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
from src.ml.config import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Experience for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class QNetwork(nn.Module):
    """Deep Q-Network for lottery prediction"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, config: ModelConfig):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ExperienceReplayBuffer:
    """Experience replay buffer for stable training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    """Q-Learning agent for lottery number selection"""
    
    def __init__(self, state_size: int, action_size: int, config: ModelConfig):
        """Initialize Q-Learning agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Q-Network
        self.q_network = QNetwork(state_size, action_size, config.hidden_size, config.num_layers, config)
        self.target_network = QNetwork(state_size, action_size, config.hidden_size, config.num_layers, config)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplayBuffer(config.memory_size)
        
        # Training parameters
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.batch_size = config.batch_size
        self.target_update = config.target_update
        self.update_count = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self, batch_size: int = None):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.config.batch_size:
            return
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        experiences = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config.discount_factor * next_q_values * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_epsilon(self, new_epsilon: float):
        """Update exploration rate"""
        self.epsilon = max(new_epsilon, self.epsilon_min)
    
    def save_model(self, filepath: str):
        """Save Q-Learning model"""
        import pickle
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved Q-Learning model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-Learning model"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.epsilon = model_data['epsilon']
        logger.info(f"Loaded Q-Learning model from {filepath}")

class PatternRecognitionAgent:
    """Agent specialized in pattern recognition"""
    
    def __init__(self, max_number: int = 49):
        self.max_number = max_number
        self.pattern_memory = {}
        self.sequence_length = 5
        self.agent_name = "PatternRecognition"
        
        logger.info(f"Initialized {self.agent_name} agent")
    
    def extract_patterns(self, history: List[List[int]]) -> Dict[str, int]:
        """Extract patterns from historical data"""
        patterns = {}
        
        for i in range(len(history) - 1):
            current_draw = history[i]
            next_draw = history[i + 1]
            
            # Create pattern key from current draw
            pattern_key = tuple(sorted(current_draw))
            
            # Count transitions to next draw
            if pattern_key not in patterns:
                patterns[pattern_key] = {}
            
            next_key = tuple(sorted(next_draw))
            if next_key not in patterns[pattern_key]:
                patterns[pattern_key][next_key] = 0
            
            patterns[pattern_key][next_key] += 1
        
        return patterns
    
    def predict_from_patterns(self, history: List[List[int]]) -> List[float]:
        """Predict based on pattern analysis"""
        if not history or not self.pattern_memory:
            return [1/49] * 49  # Uniform probability
        
        # Get the most recent draw
        recent_draw = history[-1]
        pattern_key = tuple(sorted(recent_draw))
        
        # Find similar patterns
        probabilities = [0.0] * 49
        
        for pattern, transitions in self.pattern_memory.items():
            if pattern == pattern_key:
                # Direct match
                total_transitions = sum(transitions.values())
                for next_pattern, count in transitions.items():
                    weight = count / total_transitions
                    for num in next_pattern:
                        probabilities[num - 1] += weight
            else:
                # Similar pattern (overlap)
                overlap = len(set(pattern) & set(pattern_key))
                if overlap >= 3:  # At least 3 numbers in common
                    similarity = overlap / 6
                    total_transitions = sum(transitions.values())
                    for next_pattern, count in transitions.items():
                        weight = (count / total_transitions) * similarity
                        for num in next_pattern:
                            probabilities[num - 1] += weight
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1/49] * 49
        
        return probabilities
    
    def update_patterns(self, history: List[List[int]]):
        """Update pattern memory with new data"""
        self.pattern_memory = self.extract_patterns(history)
    
    def save_model(self, filepath: str):
        """Save pattern memory"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.pattern_memory, f)
        logger.info(f"Saved {self.agent_name} model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pattern memory"""
        import pickle
        with open(filepath, 'rb') as f:
            self.pattern_memory = pickle.load(f)
        logger.info(f"Loaded {self.agent_name} model from {filepath}")

class FrequencyAnalysisAgent:
    """Agent specialized in frequency analysis"""
    
    def __init__(self, max_number: int = 49):
        self.max_number = max_number
        self.frequency_history = {}
        self.hot_cold_threshold = 0.1
        self.agent_name = "FrequencyAnalysis"
        
        logger.info(f"Initialized {self.agent_name} agent")
    
    def update_frequencies(self, history: List[List[int]]):
        """Update frequency analysis with new data"""
        # Count occurrences of each number
        number_counts = [0] * self.max_number
        
        for draw in history:
            for num in draw:
                if 1 <= num <= self.max_number:
                    number_counts[num - 1] += 1
        
        total_draws = len(history)
        frequencies = [count / total_draws for count in number_counts]
        
        # Store frequency history
        self.frequency_history[total_draws] = frequencies
    
    def get_hot_numbers(self) -> List[int]:
        """Get numbers that appear more frequently than average"""
        if not self.frequency_history:
            return []
        
        latest_frequencies = list(self.frequency_history.values())[-1]
        avg_frequency = sum(latest_frequencies) / len(latest_frequencies)
        
        hot_numbers = []
        for i, freq in enumerate(latest_frequencies):
            if freq > avg_frequency + self.hot_cold_threshold:
                hot_numbers.append(i + 1)
        
        return hot_numbers
    
    def get_cold_numbers(self) -> List[int]:
        """Get numbers that appear less frequently than average"""
        if not self.frequency_history:
            return []
        
        latest_frequencies = list(self.frequency_history.values())[-1]
        avg_frequency = sum(latest_frequencies) / len(latest_frequencies)
        
        cold_numbers = []
        for i, freq in enumerate(latest_frequencies):
            if freq < avg_frequency - self.hot_cold_threshold:
                cold_numbers.append(i + 1)
        
        return cold_numbers
    
    def predict_from_frequency(self) -> List[float]:
        """Predict based on frequency analysis"""
        if not self.frequency_history:
            return [1.0 / self.max_number] * self.max_number
        
        latest_frequencies = list(self.frequency_history.values())[-1]
        
        # Normalize to probabilities
        total = sum(latest_frequencies)
        if total > 0:
            return [freq / total for freq in latest_frequencies]
        
        return [1.0 / self.max_number] * self.max_number
    
    def save_model(self, filepath: str):
        """Save frequency history"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.frequency_history, f)
        logger.info(f"Saved {self.agent_name} model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load frequency history"""
        import pickle
        with open(filepath, 'rb') as f:
            self.frequency_history = pickle.load(f)
        logger.info(f"Loaded {self.agent_name} model from {filepath}")

class EnsembleAgent:
    """Ensemble agent that combines predictions from multiple agents"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.performance_history = {name: [] for name in agents.keys()}
        self.weights = {name: 1.0 / len(agents) for name in agents.keys()}
        
        logger.info(f"Initialized ensemble with {len(agents)} agents")
    
    def predict(self, state: np.ndarray, history: List[List[int]]) -> List[float]:
        """Generate ensemble prediction"""
        predictions = {}
        
        for name, agent in self.agents.items():
            if name == 'q_learning':
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = agent.q_network(state_tensor).squeeze()
                    predictions[name] = torch.softmax(q_values, dim=0).tolist()
            elif name == 'pattern_recognition':
                predictions[name] = agent.predict_from_patterns(history)
            elif name == 'frequency_analysis':
                predictions[name] = agent.predict_from_frequency()
        
        # Combine predictions using weighted average
        ensemble_prediction = np.zeros(49)
        total_weight = sum(self.weights.values())
        
        for name, pred in predictions.items():
            weight = self.weights[name] / total_weight
            ensemble_prediction += np.array(pred) * weight
        
        return ensemble_prediction.tolist()
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update agent weights based on performance"""
        total_performance = sum(performance_metrics.values())
        if total_performance > 0:
            for name, performance in performance_metrics.items():
                if name in self.weights:
                    self.weights[name] = performance / total_performance
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        import pickle
        model_data = {
            'weights': self.weights,
            'performance_history': self.performance_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.performance_history = model_data['performance_history'] 