# TechConative-Assignment
This repository contains solutions for the 2nd question asked in the assignment for TechConative winter internship and two Jupyter Notebooks designed to conduct comprehensive analyses and modeling.The question is about Data Analysis and Prediction(Regression) on a House Price Dataset(Kaggle).

Notebook 1 - Descriptive Analysis and Predictive Modeling: Focuses on exploratory data analysis, data cleaning, feature engineering, and predictive model building.

Notebook 2 - Visual, Inferential, and Time Series Analyses: Emphasizes data visualization, inferential statistics, and time series analysis.

# Getting Started

## Requirements

Ensure you have the following libraries installed and imports before running the notebooks:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy sklearn xgboost lightgbm fbprophet category-encoders lifelines
```

```python
# Data Handling and Manipulation
import pandas as pd
import numpy as np

# Data Preprocessing and Feature Engineering
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.decomposition import PCA

# Outlier Detection and Statistical Analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Model Building
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Plotting and Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.structural import UnobservedComponents
```

# Instructions

1.Open the notebooks in any compatible environment like VS Code, Google Colab, or Jupyter Notebook.

2.Upload the required datasets and execute each cell sequentially to perform the analyses.
