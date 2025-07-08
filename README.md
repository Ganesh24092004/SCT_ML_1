# 🏠 House Price Prediction using Random Forest

This project predicts house sale prices using square footage, number of bedrooms, and number of bathrooms. A **Random Forest Regressor** is used for better accuracy and handling of non-linear relationships.

## 📁 Files

- `train.csv`: Dataset containing house features and sale prices (from Kaggle House Prices competition).
- `house_price_model.py`: Python script implementing the prediction model.
- `README.md`: Project overview and usage instructions.

## 🔍 Project Objectives

- Predict house prices using basic features:
  - `GrLivArea` – Above-ground living area (square feet)
  - `BedroomAbvGr` – Number of bedrooms above ground
  - `FullBath` – Number of full bathrooms
- Train and evaluate a **Random Forest** model.
- Visualize model performance and feature importance.

## 🧪 Requirements

- Python 3.7+
- Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

## 🚀 How to Run

1. Place `train.csv` in the project directory.
2. Run the Python script:
   ```bash
   python house_price_model.py
   ```

## 📊 Output

- **Metrics**:
  - R² Score
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- **Plots**:
  - Actual vs. Predicted Sale Prices
  - Feature Importance Bar Chart

## ⚙️ Sample Results

```
--- Model Performance ---
R² Score: 0.85
RMSE: 28500.42
MAE: 20543.17
```

## 📌 Notes

- You can extend the model by adding more features like `GarageArea`, `YearBuilt`, `TotalBsmtSF`, etc.
- For even better performance, consider:
  - Feature engineering
  - Grid search for hyperparameter tuning
  - Cross-validation

## 📚 Data Source

- [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
