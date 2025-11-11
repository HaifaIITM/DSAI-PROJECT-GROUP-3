# Frontend - Hybrid Model Predictions Dashboard

Simple, beautiful web interface for viewing stock predictions from the Hybrid ESN-Ridge model API.

## Features

- 📊 **Real-time Predictions**: Shows latest h1, h5, h20 forecasts
- 📈 **30-Day History**: Table view of all predictions
- 📰 **Recent News**: Last 3 days of SPY headlines
- 🔄 **Auto-refresh**: Updates every 5 minutes
- 📱 **Responsive**: Works on desktop, tablet, and mobile

## Quick Start

### 1. Start the Backend API

First, make sure the backend is running:

```bash
cd ../backend
python main.py
```

The API should be running at `http://localhost:8000`

### 2. Open the Frontend

Simply open `index.html` in your browser:

**Option A: Double-click**
- Double-click `index.html`

**Option B: Command line**
```bash
# Mac
open index.html

# Linux
xdg-open index.html

# Windows
start index.html
```

**Option C: Use a local server (recommended)**
```bash
# Python
python -m http.server 3000

# Node.js
npx http-server -p 3000
```

Then open: http://localhost:3000

## Usage

### Main Dashboard

![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

**Top Section - Latest Signals**:
- 3 cards showing h1, h5, h20 predictions
- Green = BUY, Red = SELL
- Shows predicted return percentage
- h20 marked with ⭐ (best model, Sharpe 6.81)

**Middle Section - Prediction History**:
- Table of last 30 days
- Shows date, close price, and signals for all horizons
- Color-coded: green (BUY), red (SELL)
- Scrollable for long history

**Bottom Section - Recent News**:
- Last 3 days of SPY news
- Shows title, date, publisher
- Click "Read more →" to open full article

**Refresh Button**:
- Manual refresh of all data
- Auto-refreshes every 5 minutes

### Interpreting Signals

**Prediction Value**:
- Positive = Expected price increase → BUY
- Negative = Expected price decrease → SELL

**Example**:
```
h20: +1.685% → BUY
```
Means the model predicts SPY will increase by 1.685% over next 20 days.

**Consensus Strategy**:
- If 2+ horizons agree (e.g., all BUY) → Strong signal
- If mixed (1 BUY, 2 SELL) → Weak signal, be cautious

**Best Model**:
- h20 is the star performer (Sharpe 6.81)
- Use h20 for primary trading decisions
- Use h1/h5 for short-term confirmation

## Customization

### Change API URL

Edit `index.html` line 237:

```javascript
const API_URL = 'http://your-server.com:8000';
```

### Change Auto-refresh Interval

Edit `index.html` line 370:

```javascript
// Refresh every 10 minutes instead of 5
setInterval(fetchPredictions, 10 * 60 * 1000);
```

### Change Colors

Edit the `<style>` section:

```css
/* Buy signal color */
.buy { background: #10b981; }  /* Green */

/* Sell signal color */
.sell { background: #ef4444; }  /* Red */

/* Background gradient */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Add More Symbols

The frontend currently shows SPY only. To add more symbols:

1. Update backend to support multiple symbols
2. Add a dropdown selector in HTML:

```html
<select id="symbol-select" onchange="fetchPredictions()">
    <option value="SPY">SPY</option>
    <option value="AAPL">AAPL</option>
    <option value="TSLA">TSLA</option>
</select>
```

3. Update `fetchPredictions()` to use selected symbol

## Deployment

### Static Hosting

Since this is a simple HTML file, you can host it anywhere:

**GitHub Pages**:
1. Push to GitHub
2. Go to Settings → Pages
3. Select branch and folder
4. Access at `https://username.github.io/repo-name/`

**Netlify**:
1. Drag and drop `application/frontend/` folder
2. Get instant URL: `https://your-app.netlify.app`

**Vercel**:
```bash
npm i -g vercel
cd application/frontend
vercel
```

### CORS Issues

If you deploy the frontend separately from the backend, you may encounter CORS errors.

**Solution 1: Update backend CORS settings**

Edit `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-url.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Solution 2: Use a proxy**

Add this to frontend JavaScript:
```javascript
const API_URL = '/api';  // Proxy to backend
```

Configure Netlify proxy in `netlify.toml`:
```toml
[[redirects]]
  from = "/api/*"
  to = "http://your-backend-url:8000/:splat"
  status = 200
  force = true
```

## Mobile-Friendly

The dashboard is fully responsive and works on mobile devices:

- **Desktop**: 3-column signal cards
- **Tablet**: 2-column signal cards
- **Mobile**: 1-column stacked layout

Test it by resizing your browser window!

## Troubleshooting

**"Loading predictions..." never finishes**:
- Check if backend is running: `curl http://localhost:8000/`
- Open browser console (F12) and check for errors
- Verify API URL in `index.html` is correct

**CORS error in console**:
```
Access to fetch at 'http://localhost:8000/predict' has been blocked by CORS policy
```
- Backend should already have CORS enabled
- If not, add CORS middleware (see Deployment section)

**Data looks incorrect**:
- Click the refresh button
- Check backend logs for errors
- Verify models are loaded: `curl http://localhost:8000/models/info`

**News not showing**:
- yfinance news API may have no recent news
- This is normal, the section will show "No recent news available"

## Tech Stack

- **Pure HTML/CSS/JavaScript**: No frameworks needed
- **Fetch API**: For backend requests
- **CSS Grid**: For responsive layout
- **No build step**: Just open and use

## Performance

- **Page load**: < 1s
- **API request**: ~1-2s (depends on yfinance)
- **UI update**: < 100ms
- **Memory**: ~5 MB

## Future Enhancements

- [ ] Charts (line chart of predictions over time)
- [ ] Multi-symbol comparison
- [ ] Historical accuracy tracking
- [ ] Portfolio simulation
- [ ] Dark mode toggle
- [ ] Export predictions to CSV
- [ ] Email/SMS alerts for strong signals
- [ ] WebSocket for real-time updates

## Screenshots

### Desktop View
- Full 3-column layout
- All features visible at once
- Smooth scrolling table

### Mobile View
- Stacked signal cards
- Touch-friendly buttons
- Collapsible sections

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

**Note**: Uses modern JavaScript (async/await, fetch). IE11 not supported.

## License

Same as main project (see root README.md)

---

**Status**: ✅ Production Ready | ✅ Mobile Friendly | ✅ No Build Required

**Last Updated**: November 11, 2025

