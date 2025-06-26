# src/train.py

"""
Script to train and save the best student marks prediction model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from src.data_preprocessing import load_and_clean_data, encode_categorical
from src.utils import save_model

# Load and clean the dataset
df = load_and_clean_data("data/student_habits_performance.csv")

# Define categorical columns to encode
categorical_cols = ["part_time_job"]

# Encode categorical columns
df, le_dict = encode_categorical(df, categorical_cols)

# Select features and target as in the notebook
features = ['study_hours_per_day','attendance_percentage','mental_health_rating','sleep_hours',"part_time_job"]
target = "exam_score"

X = df[features]
y = df[target]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameters
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }
    }
}

# Train and evaluate models, keeping track of the best
best_models = []
for name, config in models.items():
    print(f"Training {name} model...")
    grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)
    y_pred = grid.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_models.append({
        "models": name,
        "best_params": grid.best_params_,
        "rmse": rmse,
        "r2": r2
    })

# Find the best model by RMSE
results_df = pd.DataFrame(best_models)
best_row = results_df.sort_values(by="rmse").iloc[0]
best_model_name = best_row["models"]
best_model_config = models[best_model_name]
final_model = best_model_config["model"]

# Fit the final model on all data
final_model.fit(X, y)

# Save the trained model
save_model(final_model, "models/best_student_performance_model.pkl")
print("Best model saved to models/best_student_performance_model.pkl")