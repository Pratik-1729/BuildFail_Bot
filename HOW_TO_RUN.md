# How to Run BuildFail Bot - Complete Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (with pip)
- Node.js 16+ (with npm)
- Chrome/Edge browser (for extension)
- Git (optional)

---

## 📦 Step 1: Backend Setup

### 1.1 Install Python Dependencies

```bash
# Navigate to project root
cd D:\BuildFail_Bot

# Install dependencies
pip install -r requirements.txt

# If PyTorch installation fails, install separately:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.2 Set Up Environment Variables

Create `.env` file in `backend/` directory:

```bash
# backend/.env
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
GITHUB_TOKEN=your_github_token_here
DEFAULT_MODEL_NAME=distilbert-base-uncased
MAX_SEQUENCE_LENGTH=256
LOG_LEVEL=INFO
```

**Get GitHub Token:**
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` and `actions:read` permissions
3. Copy token to `.env` file

### 1.3 Run Backend Server

```bash
# From project root
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Or use the script:**
```bash
# Windows
python backend/app/main.py

# Or directly
cd backend
uvicorn app.main:app --reload
```

**Verify Backend:**
- Open browser: http://localhost:8000
- Should see: `{"message": "BuildFail Bot API is running successfully!"}`
- API docs: http://localhost:8000/docs

---

## 🎨 Step 2: Frontend Dashboard Setup

### 2.1 Install Node Dependencies

```bash
# Navigate to dashboard directory
cd frontend/dashboard

# Install dependencies
npm install
```

### 2.2 Configure API URL (Optional)

Create `.env` file in `frontend/dashboard/`:

```bash
# frontend/dashboard/.env
REACT_APP_API_URL=http://localhost:8000
```

### 2.3 Run Dashboard

```bash
# From dashboard directory
npm start
```

**Dashboard will open:** http://localhost:3000

---

## 🔌 Step 3: Web Extension Setup

### 3.1 Load Extension in Chrome

1. **Open Chrome Extensions:**
   - Go to `chrome://extensions/`
   - Or: Menu → More tools → Extensions

2. **Enable Developer Mode:**
   - Toggle "Developer mode" (top right)

3. **Load Extension:**
   - Click "Load unpacked"
   - Select folder: `D:\BuildFail_Bot\frontend\web-extension`
   - Extension should appear in your extensions list

4. **Verify Extension:**
   - Pin extension to toolbar (click puzzle icon → pin)
   - Extension icon should appear in toolbar

### 3.2 Configure Extension API URL

Edit `frontend/web-extension/utils/config.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change if needed
```

**Or set via environment** (if using build process)

---

## ✅ Step 4: Verify Everything Works

### 4.1 Test Backend

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "error: module not found"}'
```

### 4.2 Test Dashboard

1. Open http://localhost:3000
2. Should see dashboard with:
   - Stats cards
   - Logs table (may be empty initially)
   - Feedback table
   - Charts section

### 4.3 Test Web Extension

1. **Go to GitHub Actions:**
   - Visit: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
   - Open any workflow run page

2. **Test Auto-Detection:**
   - Extension should detect build status automatically
   - Badge should update (red "!" for failed, green "✓" for success)

3. **Test Analysis:**
   - Click extension icon
   - Click "Auto-Fetch Logs" button
   - Click "Analyze Logs" button
   - Should see prediction + suggestion + fix command

---

## 🧪 Step 5: Test Real-Time Features

### 5.1 Test Auto-Detection

1. Visit a **failed** GitHub Actions run:
   ```
   https://github.com/YOUR_USERNAME/YOUR_REPO/actions/runs/RUN_ID
   ```

2. **Watch for:**
   - Extension badge turns red with "!"
   - Browser notification appears
   - Console shows: `🔍 Build status detected: failed`

3. **Open Extension Popup:**
   - Should auto-load and analyze
   - Shows prediction + fix command

### 5.2 Test Smart Log Extraction

1. On a failed build page
2. Extension automatically extracts only error lines
3. Check console: `📜 Smart extraction: X error-focused lines`

### 5.3 Test Fix Commands

1. Analyze a log with dependency error (e.g., "ModuleNotFoundError: No module named numpy")
2. Should see:
   - Suggestion: "Install missing dependencies..."
   - Fix Command: `pip install numpy`
   - Copy button to copy command

---

## 🔧 Troubleshooting

### Backend Issues

**Port Already in Use:**
```bash
# Change port in .env or use different port
uvicorn app.main:app --reload --port 8001
```

**Module Not Found:**
```bash
# Make sure you're in the right directory
cd backend
python -m uvicorn app.main:app --reload
```

**PyTorch Errors:**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Frontend Dashboard Issues

**Port 3000 in Use:**
```bash
# Use different port
PORT=3001 npm start
```

**API Connection Failed:**
- Check backend is running
- Verify `REACT_APP_API_URL` in `.env`
- Check CORS settings in backend

### Extension Issues

**Extension Not Loading:**
- Check manifest.json syntax
- Check console for errors (F12)
- Try reloading extension

**API Calls Failing:**
- Verify backend is running on http://localhost:8000
- Check `utils/config.js` has correct API URL
- Check browser console for CORS errors

**Auto-Detection Not Working:**
- Check browser console for errors
- Verify you're on GitHub Actions run page
- Check content script is loaded (console should show: `🧩 BuildFailBot content script loaded.`)

---

## 📋 Complete Startup Checklist

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Backend `.env` file created with GitHub token
- [ ] Backend server running (`uvicorn app.main:app --reload`)
- [ ] Backend accessible at http://localhost:8000
- [ ] Node dependencies installed (`npm install` in dashboard)
- [ ] Dashboard running (`npm start`)
- [ ] Dashboard accessible at http://localhost:3000
- [ ] Extension loaded in Chrome
- [ ] Extension icon visible in toolbar
- [ ] Tested on GitHub Actions page

---

## 🎯 Quick Commands Reference

```bash
# Backend
cd backend
python -m uvicorn app.main:app --reload

# Dashboard
cd frontend/dashboard
npm start

# Install dependencies
pip install -r requirements.txt
cd frontend/dashboard && npm install
```

---

## 📝 Next Steps

1. **Set up GitHub Webhook** (optional):
   - Go to repository settings → Webhooks
   - Add webhook: `http://your-server:8000/api/logs/webhook`
   - Select "workflow_run" events

2. **Train Initial Model** (if not done):
   ```bash
   cd backend
   python -m app.ml.train_model
   ```

3. **Start Using:**
   - Visit GitHub Actions pages
   - Extension will auto-detect and analyze
   - Use dashboard to view all logs and feedback

---

## 🆘 Need Help?

- Check logs in backend console
- Check browser console (F12)
- Verify all services are running
- Check `.env` files are configured correctly
- Review API docs at http://localhost:8000/docs

