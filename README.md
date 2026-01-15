# HealthScore-Predictor

A predictive data-driven framework to forecast restaurant health inspection scores and identify establishments at higher risk of violations using machine learning models.

## Overview

This project leverages the San Francisco Health Inspection Scores (2024-current) dataset combined with Google restaurant data to build predictive models that can forecast inspection outcomes and help identify restaurants at risk of health violations. By integrating official inspection records with public review data, the system aims to provide early warnings and insights into restaurant health compliance.

## Problem Statement

Health inspections are typically reactive, conducted periodically or in response to complaints. This project explores whether machine learning models can proactively identify restaurants likely to fail inspections by analyzing historical violation patterns, restaurant characteristics, and public perception metrics.

## Research Hypotheses

1. **Violation Patterns**: Restaurants with previous critical violations, lower Google ratings, or specific cuisine categories are more likely to score poorly in future inspections.

2. **Model Performance**: Machine learning models (Random Forest, XGBoost, Gradient Boosting) can outperform traditional statistical baselines in predicting inspection outcomes.

3. **Review Data Correlation**: Publicly available review data can serve as a proxy for hygiene perception and may correlate with actual inspection results.

## Key Features

### Data Integration
- **Multi-source Data Fusion**: Combines San Francisco health inspection data with Google Places restaurant information
- **Advanced Matching Algorithm**: Uses fuzzy string matching, geographic proximity (Haversine distance), and address canonicalization to accurately link datasets
- **Comprehensive Feature Engineering**: Extracts temporal patterns, violation history, and restaurant characteristics

### Machine Learning Pipeline
- **Multiple Model Architectures**: Implements Random Forest, XGBoost, and Gradient Boosting classifiers
- **Robust Evaluation Framework**: Uses stratified cross-validation and comprehensive metrics (ROC-AUC, precision, recall, F1-score)
- **Class Imbalance Handling**: Employs class weighting techniques to address imbalanced datasets
- **Feature Importance Analysis**: Identifies key predictors of health inspection outcomes

### Temporal Analysis
- **Time-based Validation**: Implements proper temporal splits to prevent data leakage
- **Trend Detection**: Analyzes patterns in violations over time
- **Seasonal Effects**: Investigates potential seasonal variations in inspection outcomes

## Technical Stack

- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **XGBoost**: Gradient boosting implementation
- **Imbalanced-learn**: Resampling techniques for class imbalance
- **Matplotlib & Seaborn**: Data visualization
- **FuzzyWuzzy**: Fuzzy string matching for data integration
- **Geopy**: Geographic distance calculations

## Dataset

### San Francisco Health Inspection Data
- Source: San Francisco Department of Public Health
- Time Period: 2024-current
- Features: Inspection dates, violation types, risk categories, scores

### Google Places Data
- Restaurant names, addresses, locations
- User ratings and review counts
- Business categories and attributes

## Installation

```bash
# Clone the repository
git clone https://github.com/LkNeha/HealthScore-Predictor.git
cd HealthScore-Predictor

# Install required packages
pip install -r requirement.txt
```

## Project Structure

```
HealthScore-Predictor/
│
├── data/                  # Raw and processed data files
├── notebooks/             # Jupyter notebooks for analysis
├── backend/               # Backend API and server code
├── dashboardv1/           # Dashboard application files
├── main.ipynb            # Main analysis notebook
├── requirement.txt       # Project dependencies
└── README.md            # Project documentation
```

## Usage

### Data Preparation

The project includes sophisticated data matching algorithms to combine health inspection records with Google restaurant data:

```python
# Example: Loading and matching datasets
# The matching process uses:
# 1. Fuzzy string matching for restaurant names
# 2. Geographic proximity (Haversine distance)
# 3. Address canonicalization
```

### Model Training

Multiple machine learning models are trained and evaluated:

```python
# Models implemented:
# - Random Forest Classifier
# - XGBoost Classifier
# - Gradient Boosting Classifier
```

### Evaluation

Comprehensive evaluation metrics are used to assess model performance:
- ROC-AUC scores and curves
- Precision, Recall, F1-score
- Confusion matrices
- Feature importance rankings

## Methodology

### 1. Data Collection & Integration
- Merge health inspection records with Google restaurant data
- Handle missing values and data quality issues
- Create unified restaurant identifier

### 2. Feature Engineering
- **Temporal Features**: Days since last inspection, inspection frequency
- **Violation History**: Count and types of previous violations
- **Restaurant Attributes**: Cuisine type, location, ratings
- **Risk Indicators**: Critical violation flags, compliance patterns

### 3. Addressing Data Challenges
- **Temporal Leakage Prevention**: Strict time-based train/test splits
- **Class Imbalance**: Class weight adjustment and resampling strategies
- **Feature Selection**: Recursive feature elimination and importance analysis

### 4. Model Development
- Baseline model establishment
- Hyperparameter tuning using grid search
- Cross-validation for robust performance estimates
- Ensemble methods for improved predictions

### 5. Evaluation & Interpretation
- Multi-metric performance assessment
- Feature importance visualization
- Error analysis and model interpretation
- Comparative analysis across models

## Results

The models demonstrate the feasibility of predicting restaurant health inspection outcomes using historical data and public information. Key findings include:

- Identification of most predictive features for health violations
- Comparison of model performance across different algorithms
- Insights into patterns that precede poor inspection results
- Practical applicability for regulatory agencies and restaurant management

## Challenges Addressed

### Data Leakage
Implemented strict temporal validation to ensure models only use information available at prediction time, preventing unrealistic performance estimates.

### Class Imbalance
The dataset exhibits significant imbalance between passing and failing restaurants. Applied class weighting to ensure models learn from minority class examples.

### Data Quality
Restaurant names and addresses often have inconsistencies. Developed robust matching algorithms using multiple signals (name similarity, geographic proximity, address normalization).

## Future Enhancements

- [ ] Incorporate additional data sources (weather, foot traffic, nearby construction)
- [ ] Develop time-series forecasting for inspection scheduling
- [ ] Create interactive dashboard for visualization
- [ ] Implement deep learning models for pattern recognition
- [ ] Add explainability features (SHAP values, LIME)
- [ ] Deploy as web application for real-time predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed changes.

## Authors
- **Contributors** - See [contributors](https://github.com/LkNeha/HealthScore-Predictor/graphs/contributors) list

## Acknowledgments

- San Francisco Department of Public Health for providing inspection data
- Google Places API for restaurant information
- The data science community for tools and techniques

## License

This project is available for educational and research purposes. Please check with the repository owner for specific licensing information.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on this repository.

---

**Note**: This project is developed for educational and research purposes to demonstrate machine learning applications in public health and food safety domains.
