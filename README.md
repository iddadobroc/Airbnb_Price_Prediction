In this project, we developed a machine learning model to predict the price of Airbnb properties in Lisbon using a dataset with various features. The following steps detail the process and decisions made to optimize the model's performance:

Project Steps:

1. Dataset Exploration
- We explored the dataset containing features such as bedrooms, room_type, person_capacity, dist, and others.
- Basic preprocessing and data cleaning were performed to ensure consistency in the dataset.

2. Target Transformation
- To handle the skewed distribution of the target variable (realSum, the price), we applied a logarithmic transformation:
- y = np.log1p(df['realSum'])

3. Data Augmentation
- To enhance the dataset, we generated synthetic data by adding random noise to some features (e.g., realSum, dist):
  
synthetic_df = df.copy()
synthetic_df['realSum'] = synthetic_df['realSum'] * (1 + np.random.uniform(-0.1, 0.1, len(synthetic_df)))
synthetic_df['dist'] = synthetic_df['dist'] * (1 + np.random.uniform(-0.1, 0.1, len(synthetic_df)))
augmented_df = pd.concat([df, synthetic_df])

4. Feature Engineering
   
We created new features based on interactions between existing variables to improve model performance:
- capacity_per_bedroom: Person capacity divided by the number of bedrooms.
- cleanliness_satisfaction: Product of cleanliness rating and guest satisfaction.
  
5. Model Training and Evaluation
   
We trained several models to evaluate their performance:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regressor (SVR)
- Advanced models: XGBoost and LightGBM
  
XGBoost emerged as the best model with R^2 = 0.8978 and MSE = 0.0198

6. Hyperparameter Tuning
- Using Optuna, we further optimized the XGBoost model by exploring the best combination of hyperparameters:
- Best parameters found:
{
    'n_estimators': 388,
    'max_depth': 13,
    'learning_rate': 0.0819,
    'subsample': 0.865,
    'colsample_bytree': 0.607
}
- After optimization, the model achieved R^2 = 0.9354 and MSE = 0.0125

7. Residual Analysis
- We analyzed the residuals to assess model performance:
- Residuals were normally distributed and centered around 0.
- Some outliers were identified but retained to preserve the integrity of the dataset.

8. Future Improvements
- Further data augmentation to expand the dataset.
- Exploration of ensemble methods, such as stacking.
- Refining the handling of outliers or using separate models for extreme values.
