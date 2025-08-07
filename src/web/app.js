/**
 * Champion Winner - Main Application JavaScript
 */

class ChampionWinnerApp {
    constructor() {
        this.currentPrediction = null;
        this.performanceMetrics = {};
        this.recentResults = [];
        this.isTraining = false;
        this.trainingProgress = 0;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeUI();
        this.startTimer();
        this.loadPerformanceMetrics();
        this.loadRecentResults();
        this.updateModelStatus();
    }
    
    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });
        
        // Prediction generation
        document.getElementById('generatePrediction').addEventListener('click', () => {
            this.generatePrediction();
        });
        
        // Data refresh
        document.getElementById('refreshData').addEventListener('click', () => {
            this.refreshData();
        });
        
        // Results submission
        document.getElementById('submitResults').addEventListener('click', () => {
            this.submitResults();
        });
        
        // Training controls
        document.getElementById('startTraining').addEventListener('click', () => {
            this.startTraining();
        });
        
        document.getElementById('stopTraining').addEventListener('click', () => {
            this.stopTraining();
        });
        
        document.getElementById('saveModel').addEventListener('click', () => {
            this.saveModel();
        });
        
        // File input
        document.getElementById('trainingData').addEventListener('change', (e) => {
            this.handleFileUpload(e);
        });
        
        // Initialize number inputs
        this.initializeNumberInputs();
    }
    
    initializeUI() {
        // Set initial theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
        
        // Initialize prediction numbers display
        this.updatePredictionDisplay([]);
        
        // Initialize confidence bars
        this.updateConfidenceBars({});
        
        // Initialize agent predictions
        this.updateAgentPredictions({});
    }
    
    initializeNumberInputs() {
        const container = document.getElementById('actualNumberInputs');
        container.innerHTML = '';
        
        for (let i = 0; i < 6; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'number-input';
            input.min = '1';
            input.max = '49';
            input.placeholder = (i + 1).toString();
            input.dataset.index = i;
            
            input.addEventListener('input', (e) => {
                this.validateNumberInput(e.target);
            });
            
            container.appendChild(input);
        }
    }
    
    validateNumberInput(input) {
        const value = parseInt(input.value);
        const index = parseInt(input.dataset.index);
        
        if (value < 1 || value > 49) {
            input.style.borderColor = 'var(--error-color)';
            return false;
        }
        
        // Check for duplicates
        const inputs = document.querySelectorAll('.number-input');
        const values = Array.from(inputs).map(inp => parseInt(inp.value)).filter(v => !isNaN(v));
        const duplicates = values.filter((v, i) => values.indexOf(v) !== i);
        
        if (duplicates.length > 0) {
            input.style.borderColor = 'var(--error-color)';
            return false;
        }
        
        input.style.borderColor = 'var(--glass-border)';
        return true;
    }
    
    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }
    
    updateThemeIcon(theme) {
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
    
    startTimer() {
        this.updateTimer();
        setInterval(() => {
            this.updateTimer();
        }, 1000);
    }
    
    updateTimer() {
        const now = new Date();
        const nextDraw = this.getNextDrawTime();
        const timeDiff = nextDraw - now;
        
        if (timeDiff > 0) {
            const days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
            const hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);
            
            const timerValue = document.getElementById('timerValue');
            timerValue.textContent = `${days}d ${hours}h ${minutes}m ${seconds}s`;
        } else {
            document.getElementById('timerValue').textContent = 'Draw Time!';
        }
    }
    
    getNextDrawTime() {
        const now = new Date();
        const drawDays = ['Wednesday', 'Saturday'];
        const drawTime = '20:30'; // EST
        
        let nextDraw = new Date();
        nextDraw.setHours(20, 30, 0, 0);
        
        // Find next draw day
        while (!drawDays.includes(nextDraw.toLocaleDateString('en-US', { weekday: 'long' }))) {
            nextDraw.setDate(nextDraw.getDate() + 1);
        }
        
        // If today is a draw day but past draw time, go to next draw day
        if (now.getDay() === nextDraw.getDay() && now.getHours() >= 20) {
            nextDraw.setDate(nextDraw.getDate() + 1);
            while (!drawDays.includes(nextDraw.toLocaleDateString('en-US', { weekday: 'long' }))) {
                nextDraw.setDate(nextDraw.getDate() + 1);
            }
        }
        
        return nextDraw;
    }
    
    async generatePrediction() {
        this.showLoading('Generating prediction...');
        
        try {
            // Call the real backend API
            const response = await fetch('https://champion-winner-api.onrender.com/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const prediction = await response.json();
                this.currentPrediction = prediction;
                
                this.updatePredictionDisplay(prediction.predicted_numbers);
                this.updateConfidenceBars(prediction.confidence_scores);
                this.updateAgentPredictions(prediction.agent_predictions);
                
                // Show model status
                const modelStatus = prediction.model_status || 'unknown';
                if (modelStatus === 'mock_prediction') {
                    this.showNotification('Using mock prediction (models not available)', 'warning');
                } else if (modelStatus === 'trained_model') {
                    this.showNotification('Prediction generated successfully!', 'success');
                } else {
                    this.showNotification('Prediction generated with unknown model status', 'info');
                }
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`API responded with status: ${response.status} - ${errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error generating prediction:', error);
            this.showNotification('Failed to generate prediction. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    updatePredictionDisplay(numbers) {
        const container = document.getElementById('predictionNumbers');
        container.innerHTML = '';
        
        if (numbers.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No prediction available</p>';
            return;
        }
        
        numbers.forEach((number, index) => {
            const ball = document.createElement('div');
            ball.className = 'prediction-number';
            ball.textContent = number;
            ball.style.animationDelay = `${index * 0.1}s`;
            container.appendChild(ball);
        });
    }
    
    updateConfidenceBars(confidenceScores) {
        const container = document.getElementById('confidenceBars');
        container.innerHTML = '';
        
        if (Object.keys(confidenceScores).length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No confidence data available</p>';
            return;
        }
        
        Object.entries(confidenceScores).forEach(([number, confidence]) => {
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            
            const percentage = (confidence * 100).toFixed(1);
            
            bar.innerHTML = `
                <span class="confidence-number">${number}</span>
                <div class="confidence-progress">
                    <div class="confidence-fill" style="width: ${percentage}%"></div>
                </div>
                <span class="confidence-value">${percentage}%</span>
            `;
            
            container.appendChild(bar);
        });
    }
    
    updateAgentPredictions(agentPredictions) {
        const container = document.getElementById('agentPredictions');
        container.innerHTML = '';
        
        if (Object.keys(agentPredictions).length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No agent predictions available</p>';
            return;
        }
        
        Object.entries(agentPredictions).forEach(([agentName, numbers]) => {
            const prediction = document.createElement('div');
            prediction.className = 'agent-prediction';
            
            const agentNameElement = document.createElement('div');
            agentNameElement.className = 'agent-name';
            agentNameElement.textContent = agentName;
            
            const numbersElement = document.createElement('div');
            numbersElement.className = 'agent-numbers';
            
            numbers.forEach(number => {
                const numberBall = document.createElement('div');
                numberBall.className = 'result-number';
                numberBall.textContent = number;
                numbersElement.appendChild(numberBall);
            });
            
            prediction.appendChild(agentNameElement);
            prediction.appendChild(numbersElement);
            container.appendChild(prediction);
        });
    }
    
    async submitResults() {
        const inputs = document.querySelectorAll('.number-input');
        const bonusBall = document.getElementById('bonusBall').value;
        
        // Validate inputs
        const numbers = [];
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateNumberInput(input)) {
                isValid = false;
            } else {
                numbers.push(parseInt(input.value));
            }
        });
        
        if (!isValid || numbers.length !== 6) {
            this.showNotification('Please enter 6 valid numbers (1-49, no duplicates)', 'error');
            return;
        }
        
        if (!bonusBall || bonusBall < 1 || bonusBall > 49) {
            this.showNotification('Please enter a valid bonus ball number (1-49)', 'error');
            return;
        }
        
        this.showLoading('Submitting results...');
        
        try {
            const response = await fetch('https://champion-winner-api.onrender.com/api/submit-results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    numbers: numbers,
                    bonus_ball: parseInt(bonusBall),
                    prediction: this.currentPrediction
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showNotification(`Results submitted! ${result.matches} matches found.`, 'success');
                
                // Clear inputs
                inputs.forEach(input => input.value = '');
                document.getElementById('bonusBall').value = '';
                
                // Update performance metrics
                this.loadPerformanceMetrics();
                this.loadRecentResults();
                
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`Failed to submit results: ${errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error submitting results:', error);
            this.showNotification(`Failed to submit results: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async refreshData() {
        this.showLoading('Refreshing data...');
        
        try {
            const response = await fetch('https://champion-winner-api.onrender.com/api/refresh-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message || 'Data refreshed successfully!', 'success');
                this.loadRecentResults();
                this.loadPerformanceMetrics();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`API responded with status: ${response.status} - ${errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showNotification(`Failed to refresh data: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async startTraining() {
        const fileInput = document.getElementById('trainingData');
        const autoScrape = document.getElementById('autoScrape').checked;
        const continuousTraining = document.getElementById('continuousTraining').checked;
        
        if (!fileInput.files[0] && !autoScrape) {
            this.showNotification('Please select training data or enable auto-scraping', 'error');
            return;
        }
        
        this.isTraining = true;
        this.updateTrainingUI();
        
        const formData = new FormData();
        if (fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        }
        formData.append('auto_scrape', autoScrape);
        formData.append('continuous_training', continuousTraining);
        
        try {
            const response = await fetch('/api/start-training', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                this.showNotification('Training started successfully!', 'success');
                this.monitorTrainingProgress();
            } else {
                throw new Error('Failed to start training');
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.showNotification('Failed to start training', 'error');
            this.isTraining = false;
            this.updateTrainingUI();
        }
    }
    
    async stopTraining() {
        try {
            const response = await fetch('/api/stop-training', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = false;
                this.updateTrainingUI();
                this.showNotification('Training stopped', 'info');
            }
        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }
    
    async saveModel() {
        this.showLoading('Saving model...');
        
        try {
            const response = await fetch('/api/save-model', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showNotification('Model saved successfully!', 'success');
            } else {
                throw new Error('Failed to save model');
            }
        } catch (error) {
            console.error('Error saving model:', error);
            this.showNotification('Failed to save model', 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.showNotification(`File selected: ${file.name}`, 'info');
        }
    }
    
    updateTrainingUI() {
        const startBtn = document.getElementById('startTraining');
        const stopBtn = document.getElementById('stopTraining');
        const progressText = document.getElementById('progressText');
        
        if (this.isTraining) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            progressText.textContent = 'Training in progress...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            progressText.textContent = 'Ready to train';
        }
    }
    
    async monitorTrainingProgress() {
        if (!this.isTraining) return;
        
        try {
            const response = await fetch('/api/training-progress');
            if (response.ok) {
                const progress = await response.json();
                this.updateTrainingProgress(progress.progress, progress.status);
                
                if (progress.completed) {
                    this.isTraining = false;
                    this.updateTrainingUI();
                    this.showNotification('Training completed!', 'success');
                    this.loadPerformanceMetrics();
                } else {
                    setTimeout(() => this.monitorTrainingProgress(), 1000);
                }
            }
        } catch (error) {
            console.error('Error monitoring training progress:', error);
        }
    }
    
    updateTrainingProgress(progress, status) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = status;
    }
    
    async loadPerformanceMetrics() {
        try {
            const response = await fetch('https://champion-winner-api.onrender.com/api/performance-metrics');
            if (response.ok) {
                this.performanceMetrics = await response.json();
                this.updatePerformanceDisplay();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`API responded with status: ${response.status} - ${errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error loading performance metrics:', error);
            this.showNotification(`Failed to load performance metrics: ${error.message}`, 'error');
        }
    }
    
    updatePerformanceDisplay() {
        const metrics = this.performanceMetrics;
        
        document.getElementById('winRate').textContent = 
            metrics.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : '--%';
        document.getElementById('exactMatches').textContent = 
            metrics.exact_matches || '--';
        document.getElementById('avgMatches').textContent = 
            metrics.average_matches ? metrics.average_matches.toFixed(1) : '--';
        document.getElementById('totalPredictions').textContent = 
            metrics.total_predictions || '--';
    }
    
    async loadRecentResults() {
        try {
            const response = await fetch('https://champion-winner-api.onrender.com/api/recent-results');
            if (response.ok) {
                this.recentResults = await response.json();
                this.updateRecentResultsDisplay();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`API responded with status: ${response.status} - ${errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error loading recent results:', error);
            this.showNotification(`Failed to load recent results: ${error.message}`, 'error');
        }
    }
    
    updateRecentResultsDisplay() {
        const container = document.getElementById('recentResults');
        container.innerHTML = '';
        
        if (this.recentResults.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No recent results</p>';
            return;
        }
        
        this.recentResults.forEach(result => {
            const item = document.createElement('div');
            item.className = 'result-item';
            
            const numbers = document.createElement('div');
            numbers.className = 'result-numbers';
            
            result.numbers.forEach(number => {
                const numberBall = document.createElement('div');
                numberBall.className = 'result-number';
                numberBall.textContent = number;
                numbers.appendChild(numberBall);
            });
            
            const status = document.createElement('div');
            status.className = `result-status ${result.status}`;
            status.textContent = result.status === 'match' ? 'Exact Match' : 
                               result.status === 'partial' ? 'Partial Match' : 'Miss';
            
            item.appendChild(numbers);
            item.appendChild(status);
            container.appendChild(item);
        });
    }
    
    updateModelStatus() {
        // This would be updated based on actual model status from backend
        const statuses = {
            'qlStatus': 'Ready',
            'patternStatus': 'Ready',
            'frequencyStatus': 'Ready',
            'ensembleStatus': 'Ready'
        };
        
        Object.entries(statuses).forEach(([id, status]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = status;
            }
        });
    }
    
    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = document.getElementById('loadingText');
        
        text.textContent = message;
        overlay.classList.add('active');
    }
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.remove('active');
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.championWinnerApp = new ChampionWinnerApp();
}); 