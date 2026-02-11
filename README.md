# CPI Inflation Analysis Dashboard

A production-grade data analytics and machine learning web application for analyzing India's Consumer Price Index (CPI) data from 2013 to 2025. I built this project to demonstrate expertise in full-stack data engineering, statistical analysis, and deploying ML-powered dashboards.

**Skills Demonstrated:** Python, Flask, REST API Design, Data Analytics, Machine Learning, Time-Series Analysis, Data Visualization, Frontend Development, Cloud Deployment

---

## Live Demo

**Deployed Application:** [View Live Dashboard](https://your-render-url.onrender.com)

---

## Project Overview

### Problem Statement

Inflation directly impacts economic policy decisions, consumer behavior, and business strategy. However, raw CPI data is complex, multi-dimensional, and difficult for stakeholders to interpret without proper visualization and analysis tools.

### Motivation

I designed this dashboard to transform raw government CPI data into actionable insights. The application enables users to explore inflation trends, identify volatile sectors, understand COVID-19 impact on prices, and simulate policy scenarios through an intuitive web interface.

### Goals

- Build a responsive, interactive dashboard for CPI analysis
- Implement RESTful APIs for modular data access
- Apply machine learning for CPI forecasting
- Analyze pre/post-COVID inflation dynamics
- Deploy a production-ready application on cloud infrastructure

---

## Features

### Data Analysis
- Multi-year CPI trend visualization across all sectors
- Sector-wise and category-wise comparison analysis
- Volatility analysis with coefficient of variation metrics
- Year-over-year inflation rate calculations
- Distribution analysis with statistical summaries

### COVID-19 Impact Analysis
- Pre-COVID vs Post-COVID price comparison
- Percentage change analysis across all categories
- Identification of most affected sectors

### Machine Learning
- Ridge Regression model with time-series lag features
- 12-month CPI forecast generation
- Model performance metrics (R-squared, MAE, RMSE)
- Impact simulation for policy scenario testing

### Executive Dashboard
- Auto-generated executive summary with key insights
- KPI cards for quick metric overview
- Interactive year range filtering
- Category and sector filters
- Dark/Light theme toggle

### Visualization
- Time-series trend charts
- Correlation heatmaps
- Volatility bar charts
- Sector comparison visualizations
- Seasonality pattern analysis

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, Flask, Flask-CORS |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Machine Learning** | Scikit-learn, NumPy, Pandas |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Joblib |
| **Deployment** | Render, Gunicorn |
| **Version Control** | Git, GitHub |

---

## Project Structure

```
cpi_inflation_analysis/
├── app.py                      # Flask application with REST APIs
├── requirements.txt            # Python dependencies
├── start.sh                    # Deployment startup script
├── data/
│   └── datafile.csv            # CPI dataset (2013-2025)
├── model/
│   └── linear_regression_pipeline.joblib  # Trained ML model
├── static/
│   └── index.html              # Frontend dashboard
├── notebook/
│   └── Commodities.ipynb       # Exploratory data analysis
├── README.md                   # Project documentation
└── INTERVIEW_PREP.md           # Technical interview notes
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashvasudeva1/cpi_inflation_analysis.git
   cd cpi_inflation_analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Local Development Server

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### Production Server (Gunicorn)

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend dashboard |
| `/api/executive-summary` | GET | Auto-generated insights summary |
| `/api/kpis` | GET | Key performance indicators |
| `/api/overview` | GET | Dataset overview and statistics |
| `/api/trends` | GET | Time-series trend data |
| `/api/multi-trend` | GET | Multi-category trend comparison |
| `/api/categories` | GET | List of all CPI categories |
| `/api/sector-comparison` | GET | Rural/Urban/Combined comparison |
| `/api/category-comparison` | GET | Cross-category analysis |
| `/api/volatility` | GET | Volatility metrics by category |
| `/api/covid-impact` | GET | Pre/Post COVID price changes |
| `/api/seasonality` | GET | Monthly seasonality patterns |
| `/api/correlations` | GET | Category correlation matrix |
| `/api/yearly-summary` | GET | Year-wise aggregated data |
| `/api/predict` | POST | Predict CPI from input features |
| `/api/simulate-impact` | POST | Simulate category price impact |
| `/api/forecast` | GET | Generate 12-month CPI forecast |
| `/api/regression-coefficients` | GET | Model coefficients and metrics |

---

## Machine Learning Model

### Model Architecture

I implemented a Ridge Regression model with time-series feature engineering to forecast CPI values. The model uses lag features and rolling averages to capture temporal patterns.

### Features Used

| Feature | Description |
|---------|-------------|
| `cpi_lag_1` to `cpi_lag_12` | Previous month CPI values (1-12 months) |
| `cpi_ma_3`, `cpi_ma_6`, `cpi_ma_12` | Rolling averages |
| `month`, `quarter` | Seasonality indicators |
| `yoy_change`, `mom_change` | Rate of change features |

### Model Performance

| Metric | Training | Testing |
|--------|----------|---------|
| R-Squared | 0.9987 | 0.8828 |
| MAE | 0.52 | 1.35 |
| RMSE | 0.67 | 1.55 |

### Training Approach

- Chronological train/test split (80/20)
- TimeSeriesSplit cross-validation (5 folds)
- StandardScaler for feature normalization
- Ridge regularization (alpha=1.0) to prevent overfitting

---

## Key Insights

1. **Sustained Inflation Trend:** The General CPI Index increased from 104.6 (January 2013) to 198+ (2025), representing approximately 90% cumulative inflation over 12 years.

2. **Food Price Volatility:** Vegetables and spices exhibit the highest price volatility with coefficient of variation exceeding 35%, driven by seasonal supply fluctuations.

3. **COVID-19 Impact:** Post-2020 data shows accelerated inflation in meat and fish (54%), spices (53%), and health services (43%), indicating supply chain disruptions.

4. **Sector Divergence:** Rural areas consistently show higher food inflation compared to urban areas, while urban housing costs outpace rural equivalents.

5. **Strong Correlations:** Food and beverages shows 0.98 correlation with the general index, confirming food prices as the primary inflation driver in India.

6. **Seasonal Patterns:** Vegetable prices peak during monsoon months (July-September), creating predictable seasonal inflation cycles.

---

## Data Source

- **Dataset:** Consumer Price Index (CPI) - India
- **Time Period:** January 2013 to November 2025
- **Sectors:** Rural, Urban, Rural+Urban (Combined)
- **Categories:** 28 commodity groups including food, housing, fuel, health, education, and transportation
- **Source:** Ministry of Statistics and Programme Implementation, Government of India

---

## Future Improvements

### Technical Enhancements
- Implement ARIMA/Prophet models for improved forecasting
- Add real-time data ingestion from government APIs
- Implement caching layer for API responses
- Add unit tests and integration tests

### Feature Additions
- Export functionality for charts and reports
- User authentication for saved preferences
- Regional state-wise CPI breakdown
- Inflation calculator tool
- Email alerts for significant price changes

### Infrastructure
- Containerization with Docker
- CI/CD pipeline with GitHub Actions
- Database integration for historical queries
- API rate limiting and monitoring

---

## Author

**Yash Vasudeva**

Data and AI Professional with expertise in Data Analytics, Machine Learning, and Full-Stack Development.

- **GitHub:** [github.com/yashvasudeva1](https://github.com/yashvasudeva1)
- **LinkedIn:** [linkedin.com/in/yash-vasudeva](https://www.linkedin.com/in/yash-vasudeva/)
- **Portfolio:** [yashvasudeva.vercel.app](https://yashvasudeva.vercel.app/)

---

## License

This project is open source and available under the MIT License.
