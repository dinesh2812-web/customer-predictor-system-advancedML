# Customer Response Prediction - Appian Hackathon

## Project Overview
This project is for the Appian x IIT-Madras Hackathon 2025, focusing on predicting customer responses to marketing campaigns. The goal is to build a machine learning model that can predict whether a customer will respond positively to a marketing campaign based on demographic information, purchase history, and prior campaign responses.

## Competition Results
- Final model accuracy: **85.35%**
- Multiple models ensemble approach combining CatBoost and XGBoost models
- Achieved high performance through extensive feature engineering and model optimization

## Dataset Description
The dataset contains customer information from a retail/marketing company, including:

- Demographic data (age, education, marital status, income)
- Household information (number of kids/teens)
- Purchase behavior (spending in different product categories)
- Campaign history (previous responses to campaigns)
- Website engagement (visits, purchases)
- Customer enrollment data

### Files in the dataset:
- `train.csv`: Training data with target variable
- `test.csv`: Test data for prediction
- `sample_submission.csv`: Sample format for submission

## Approach
The project implements several machine learning approaches:

1. **Data Preprocessing**:
   - Handling missing values
   - Feature engineering (age calculation, spend ratios, purchase behavior metrics)
   - Date-based features extraction
   - Categorical encoding

2. **Models Implemented**:
   - CatBoost Classifier (primary model)
   - XGBoost models
   - Neural Network models (TabInspiredMarketModel, MLPs)
   
3. **Model Optimization**:
   - Hyperparameter tuning with grid search
   - Cross-validation
   - Ensemble methods
   - Class weighting to handle imbalance

4. **Evaluation Metrics**:
   - Accuracy
   - ROC-AUC
   - Precision/Recall
   - F1 Score (binary, macro, weighted)
   - Precision/Recall
   - F1 Score

## Key Features Engineered
- Customer tenure
- Total spending and category-specific spending percentages
- Purchase channel preferences (web/catalog/store ratios)
- Deal acceptance rates
- Campaign response history
- Spending per day metrics
- Household composition features

## Files in the Repository
- `train1.ipynb`: Main notebook with data preprocessing, feature engineering, and model development
- `test.ipynb`: Testing notebook
- `best_model.pt` / `best_state.pth` / `best_params.pth`: Saved model files
- Various submission files (.csv) from different model iterations

## Model Performance
The best performing models achieved the following results:

- **XGBoost**: 85.35% accuracy on test set
- **CatBoost**: 82.17% accuracy on test set 
- **Ensemble Model**: 83.76% accuracy on test set
- **Best Neural Network (MLP)**: 83.07% accuracy on validation set

The final submission was generated using an optimized CatBoost model with custom preprocessing and feature engineering, effectively handling class imbalance through proper weighting strategies.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- CatBoost
- XGBoost
- PyTorch
- matplotlib
- seaborn

## How to Run
1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook train1.ipynb`
4. For predictions on new data, use the trained model in the inference section of the notebook

## Key Insights and Findings
During model development and analysis, several important insights were discovered:

1. **Most Influential Features**:
   - Total spending across product categories was highly predictive
   - Campaign response history strongly correlated with future responses
   - Specific spending categories (wines, meat products) showed strong predictive power

2. **Customer Segmentation**:
   - Customers with higher income and higher spending were more likely to respond positively
   - Customers with children were less responsive to campaigns
   - Long-term customers showed different response patterns than newer customers

3. **Model Comparison**:
   - Tree-based models (XGBoost, CatBoost) consistently outperformed linear models
   - Neural networks showed promising results but required more tuning
   - Ensemble approaches provided the most robust predictions

These insights not only helped with model development but also provide actionable business intelligence for future marketing campaigns.

## Future Improvements
- More advanced feature engineering
- Deep learning architectures exploration
- Stacking multiple models
- Bayesian optimization for hyperparameter tuning
- Time-series based features

## Contributors
- Monark Shah

## License
This project is part of the Appian x IIT-Madras Hackathon 2025.
