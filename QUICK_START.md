# Champion Winner - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Web Scraping (Optional)

Before running the main application, test the web scraping functionality:

```bash
python run.py --test-scraping
```

This will:
- Test both lottery websites
- Save HTML files for analysis
- Provide suggestions for CSS selectors

### 3. Start the Application

```bash
python run.py
```

The application will be available at: **http://localhost:8000**

### 4. Using the Web Interface

1. **Generate Prediction**: Click "Generate Prediction" to get the next draw prediction
2. **Enter Results**: After each draw, enter the actual winning numbers
3. **Monitor Performance**: View your prediction accuracy and performance metrics
4. **Train Model**: Upload CSV data or enable auto-scraping to improve predictions

### 5. Sample Data

A sample CSV file is provided at `data/sample_lottery_data.csv` for testing.

## 📁 Project Structure

```
Champion Winner/
├── src/                    # Source code
│   ├── ml/                # Machine learning components
│   ├── web/               # Frontend files
│   └── utils/             # Utility functions
├── data/                  # CSV files for training
├── models/                # Saved model checkpoints
├── tests/                 # Test scripts
├── requirements.txt       # Python dependencies
├── run.py                # Startup script
└── README.md            # Full documentation
```

## 🎯 Key Features

- **Advanced ML**: Reinforcement learning with ensemble agents
- **Beautiful UI**: Glassmorphism design with smooth animations
- **Real-time Predictions**: Confidence scores for each number
- **Auto-scraping**: Automated data collection from lottery websites
- **Performance Tracking**: Monitor your prediction accuracy

## 🔧 Configuration

Edit `src/ml/config.py` to customize:
- Model parameters
- Website URLs
- Training settings
- Performance targets

## 📊 Data Format

CSV files should contain:
- `date`: Draw date (YYYY-MM-DD)
- `numbers`: Winning numbers (comma-separated)
- `bonus_ball`: Bonus ball number

Example:
```csv
date,numbers,bonus_ball
2024-01-15,3,12,25,31,38,45,7
2024-01-12,7,15,22,29,36,44,12
```

## 🌐 Deployment

### Local Development
```bash
python run.py --debug
```

### Production
```bash
python run.py --host 0.0.0.0 --port 8000
```

### GitHub Pages
The system is configured for automatic deployment to GitHub Pages. Simply push to the main branch and the workflow will handle deployment.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   python run.py --port 8001
   ```

3. **Scraping Issues**
   - Check internet connection
   - Verify website availability
   - Run scraping test: `python run.py --test-scraping`

4. **Model Training Issues**
   - Ensure CSV data is properly formatted
   - Check file permissions
   - Verify sufficient disk space

## 📞 Support

- Check the full README.md for detailed documentation
- Review logs in the `logs/` directory
- Test scraping functionality if websites are unavailable

## ⚠️ Disclaimer

This system is for educational and research purposes. Lottery predictions are inherently uncertain and past performance does not guarantee future results. Please gamble responsibly.

---

**Ready to become a Champion Winner?** 🏆

Start the application and begin your journey to lottery prediction mastery! 