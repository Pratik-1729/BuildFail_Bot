# BuildFail Bot

An AI-powered CI/CD build log analyzer that predicts build status, identifies root causes, and provides actionable suggestions using fine-tuned Transformer models.

## Features

- 🤖 **AI-Powered Analysis**: Uses fine-tuned DistilBERT model to predict build status (success/failed/skipped)
- 📊 **Interactive Dashboard**: Real-time visualization of build logs, feedback, and model metrics
- 🌐 **Browser Extension**: Chrome extension for instant log analysis on GitHub Actions
- 🔄 **Auto-Retraining**: Model automatically retrains with user feedback for continuous improvement
- 🔍 **Root Cause Detection**: Identifies common failure patterns and suggests fixes
- 📈 **Analytics**: Charts and trends showing build patterns over time

## Project Structure

```
BuildFail_Bot/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── config/         # Configuration files
│   │   ├── ml/             # ML models and inference
│   │   ├── models/         # Data models
│   │   ├── routers/        # API routes
│   │   └── utils/          # Utility functions
│   ├── data/               # Data storage
│   └── results/            # Training results
├── frontend/
│   ├── dashboard/          # React dashboard
│   └── web-extension/      # Chrome extension
├── data/                   # Shared data directory
├── models/                 # Trained models
└── requirements.txt        # Python dependencies
```

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BuildFail_Bot
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# GitHub Configuration (for webhook integration)
GITHUB_TOKEN=your_github_token_here

# Model Configuration
DEFAULT_MODEL_NAME=distilbert-base-uncased
MAX_SEQUENCE_LENGTH=256

# Logging
LOG_LEVEL=INFO
```

### 4. Frontend Dashboard Setup

```bash
cd frontend/dashboard

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env with your backend URL
# REACT_APP_API_URL=http://localhost:8000
```

### 5. Web Extension Setup

The web extension is ready to use. To configure the API URL, edit `frontend/web-extension/utils/config.js` if needed.

## Running the Application

### Start Backend Server

```bash
# From project root
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Start Frontend Dashboard

```bash
# From frontend/dashboard
npm start
```

The dashboard will open at `http://localhost:3000`

### Load Web Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `frontend/web-extension` directory
5. The extension icon will appear in your toolbar

## Usage

### Dashboard

1. Open `http://localhost:3000`
2. View build logs, feedback, and analytics
3. Trigger model retraining from the "Retrain" tab
4. Monitor retraining history and metrics

### Web Extension

1. Navigate to a GitHub Actions run page
2. Click the BuildFail Bot extension icon
3. Click "Fetch Logs" to extract logs from the page
4. Click "Analyze" to get predictions
5. Provide feedback to improve the model

### API Endpoints

- `POST /predict` - Predict build status from log text
- `POST /api/logs/webhook` - GitHub webhook endpoint
- `POST /api/logs/manual` - Manual log analysis
- `GET /api/dashboard/logs` - Get all ingested logs
- `GET /api/feedback` - Get user feedback
- `POST /api/feedback` - Submit feedback
- `POST /api/retrain` - Trigger model retraining
- `GET /api/retrain/history` - Get retraining history

## Model Training

### Initial Training

If you need to train the model from scratch:

```bash
cd backend
python -m app.ml.preprocess  # Preprocess data
python -m app.ml.train_model  # Train model
```

### Retraining

The model can be retrained via:
- Dashboard UI (Retrain tab)
- API endpoint: `POST /api/retrain`

Retraining uses feedback data and ingested logs to improve accuracy.

## Configuration

### Path Configuration

All paths are centralized in `backend/app/config/settings.py`. The system automatically:
- Creates necessary directories
- Resolves paths relative to project root
- Handles cross-platform path issues

### API Configuration

- Backend: Configure via `.env` file
- Frontend Dashboard: Configure via `frontend/dashboard/.env`
- Web Extension: Edit `frontend/web-extension/utils/config.js`

## Troubleshooting

### Backend Issues

1. **Path Errors**: Ensure all directories exist. The system creates them automatically on first run.
2. **Model Not Found**: Place trained models in `models/distilbert_best/` or `models/distilbert_trained/`
3. **Port Already in Use**: Change `API_PORT` in `.env` file

### Frontend Issues

1. **CORS Errors**: Update `CORS_ORIGINS` in backend `.env` to include your frontend URL
2. **API Connection Failed**: Verify `REACT_APP_API_URL` in `frontend/dashboard/.env`
3. **Build Errors**: Run `npm install` again in `frontend/dashboard`

### Web Extension Issues

1. **Logs Not Fetching**: Ensure you're on a GitHub Actions run page
2. **API Errors**: Check `frontend/web-extension/utils/config.js` for correct API URL

## Development

### Project Structure

- **Backend**: FastAPI application with modular routers
- **Frontend**: React dashboard with Tailwind CSS
- **Extension**: Vanilla JavaScript Chrome extension

### Key Files

- `backend/app/config/settings.py` - Centralized configuration
- `backend/app/main.py` - FastAPI application entry point
- `backend/app/ml/inference.py` - Model inference logic
- `frontend/dashboard/src/config/api.js` - API endpoint configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.

