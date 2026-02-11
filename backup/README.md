# CPI Analysis Dashboard

A comprehensive web application for analyzing Consumer Price Index (CPI) trends in India. The dashboard provides interactive visualizations, predictions, and insights based on historical CPI data.

## Features

- **Overview Statistics**: Current CPI index, total inflation, COVID-19 impact
- **Interactive Trend Charts**: View CPI trends over time by category and sector
- **Category Comparison**: Compare CPI values across different commodity categories
- **Sector Analysis**: Compare Rural vs Urban vs Combined metrics
- **Volatility Analysis**: Identify most stable and volatile price categories
- **COVID-19 Impact**: Analyze pre vs post-pandemic price changes
- **CPI Prediction**: Predict general CPI index based on key economic indicators
- **Fuel Price Simulation**: Simulate impact of fuel price changes on overall CPI

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Charts**: Chart.js
- **ML Model**: Scikit-learn Linear Regression Pipeline

## Project Structure

```
CPI/
├── app.py                 # Flask backend API
├── requirements.txt       # Python dependencies
├── data/
│   └── datafile.csv      # CPI data (2013-2025)
├── model/
│   └── linear_regression_pipeline.joblib  # Trained ML model
├── notebook/
│   └── Commodities.ipynb # Analysis notebook
└── static/
    └── index.html        # Frontend dashboard
```

## Installation

1. **Clone/Navigate to the project directory**:
   ```bash
   cd CPI
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open browser** and navigate to:
   ```
   http://localhost:5000
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/overview` | GET | Overall CPI statistics and key metrics |
| `/api/trends` | GET | CPI trends over time (supports category/sector filters) |
| `/api/categories` | GET | List of all available categories |
| `/api/sector-comparison` | GET | Compare CPI across sectors |
| `/api/category-comparison` | GET | Compare all categories |
| `/api/volatility` | GET | Volatility analysis for each category |
| `/api/covid-impact` | GET | COVID-19 impact analysis |
| `/api/yearly-summary` | GET | Yearly summary statistics |
| `/api/correlations` | GET | Correlation matrix for key categories |
| `/api/predict` | POST | Predict general index (requires JSON body) |
| `/api/simulate-fuel-impact` | POST | Simulate fuel price impact |
| `/api/insights` | GET | Key insights from analysis |
| `/api/monthly-trends` | GET | Monthly seasonal patterns |
| `/api/multi-trend` | GET | Trends for multiple categories |

## Model Information

The prediction model uses a Linear Regression pipeline with StandardScaler preprocessing. 

**Model Features**:
- Food & Beverages
- Housing
- Fuel & Light
- Transport & Communication
- Health

**Performance Metrics**:
- R² Score: 0.998
- MAE: 0.935
- MSE: 1.386

## Key Insights

1. **Primary Inflation Drivers**: Food & Beverages (0.436 coefficient) and Health (0.382 coefficient)
2. **COVID-19 Impact**: Meat & fish (+54.4%), Spices (+53.2%), Personal care (+51.6%)
3. **Most Volatile**: Spices, Meat & fish, Vegetables
4. **Most Stable**: Sugar & confectionery, Housing, Transport

## Data Source

Consumer Price Index data for India covering:
- **Period**: January 2013 - 2025
- **Sectors**: Rural, Urban, Rural+Urban
- **Categories**: 25+ commodity and service categories

