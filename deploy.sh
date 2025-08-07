#!/bin/bash

# Champion Winner - Deployment Script
# This script prepares the project for GitHub upload while hiding confidential files

echo "🚀 Preparing Champion Winner for deployment..."

# Create necessary directories
mkdir -p public api models data

# Copy web files to public
echo "📁 Copying web files..."
cp -r src/web/* public/

# Copy API handler
echo "🔧 Setting up API..."
cp api/index.py api/

# Create vercel.json if it doesn't exist
if [ ! -f vercel.json ]; then
    echo "📄 Creating vercel.json..."
    cat > vercel.json << 'EOF'
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    },
    {
      "src": "public/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index.py"
    },
    {
      "src": "/(.*)",
      "dest": "/public/$1"
    }
  ],
  "functions": {
    "api/index.py": {
      "maxDuration": 30
    }
  },
  "env": {
    "PYTHONPATH": "src"
  }
}
EOF
fi

# Rename README to hide the real purpose
echo "📝 Creating public README..."
cp README_PUBLIC.md README.md

# Create .gitignore to hide confidential files
echo "🔒 Setting up .gitignore..."
cat > .gitignore << 'EOF'
# Champion Winner - Confidential Files
# This project is public but hides sensitive information

# Data files (confidential)
data/*.csv
data/Ont49.csv
data/cleaned_*.csv
data/uploaded_*.csv

# Model files (confidential)
models/
*.pkl
*.h5
*.pth
*.pt

# Logs (may contain sensitive info)
logs/
*.log

# Performance metrics (confidential)
performance_metrics.json
training_history.json

# Environment files
.env
.env.local
.env.production

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Testing
.coverage
.pytest_cache/
htmlcov/

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Scraping test files
test_primary_site.html
test_backup_site.html

# Local configuration
config.local.py
settings.local.py

# Backup files
*.bak
*.backup

# Sensitive documentation
DEPLOYMENT_GUIDE.md
QUICK_START.md
VERCEL_DEPLOYMENT.md
README_PUBLIC.md

# But keep these public files
!README.md
EOF

echo "✅ Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your Ont49.csv file to the data/ folder"
echo "2. Run: python src/utils/data_preprocessor.py data/Ont49.csv --output data/cleaned_ont49.csv"
echo "3. Test locally: python run.py --debug"
echo "4. Deploy to Vercel: vercel"
echo ""
echo "🔒 Confidential files are hidden from GitHub:"
echo "   - All CSV data files"
echo "   - Model files"
echo "   - Performance metrics"
echo "   - Sensitive documentation"
echo ""
echo "🌐 Your repo will appear as an 'Advanced Machine Learning System'"
echo "   without revealing the lottery prediction purpose" 